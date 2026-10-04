"""
Live test of arXiv:2610.00949 "PG-SFT: Balancing Capability Acquisition and
Retention in Offline Agent Fine-Tuning" on Google Colab T4
(Qwen2.5-0.5B-Instruct + LoRA).

Paper claim: standard SFT on offline agent trajectories improves the target
task but degrades non-target capabilities; KL penalties / update caps don't
fix it. PG-SFT weights each turn's supervision by its turn-level information
gain -> better acquisition-vs-retention trade-off (less drift, less broad
degradation, slight target-task cost).

Reduced experiment here:
  Phase 0: base evals (held-out acc, QA probe, general NLL, base log-probs
           on general sentences for the KL-drift measurement).
  Phase 1: build synthetic multi-turn agent trajectories (data_gen.py);
           compute turn-level information gain with the BASE model:
             IG(t) = NLL(answer | prefix before t) - NLL(answer | prefix incl t)
           weights = clip(IG,0)/mean  (mean weight 1 => same loss scale as
           uniform SFT). Documented proxy choice; distribution reported.
  Phase 2: arm A = standard SFT (uniform token supervision on assistant/
           tool turns) -> eval.
  Phase 3: reload base; arm B = PG-SFT (turn-weighted) -> eval.
  Phase 4: metrics: held-out greedy acc (target), QA acc + general NLL
           (retention), mean per-token KL(base||ft) on general sentences
           (distributional drift).

Results -> /content/pg_sft_results.json (download after).
Run: upload data_gen.py + this file to /content, then run this.
  !pip install -q transformers peft accelerate
  !pip uninstall -y torchao   # Colab ships 0.10, peft needs >0.16
"""
import json
import math
import os
import random
import sys
import time

sys.path.insert(0, "/content")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model

from data_gen import (gen_traj_problems, render_trajectory, supervised_turns,
                      extract_boxed, FORGET_QA, GENERAL_SENTENCES, qa_correct)

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
SEED = 0
N_TRAIN = 48
N_TEST = 30
GEN_TOKENS = 160
LORA_R = 16
LR = 2e-5
EPOCHS = 3
BATCH = 4

T0 = time.time()


def log(msg):
    print(f"[{time.time()-T0:7.1f}s] {msg}", flush=True)


def r4(d):
    return {k: (round(v, 4) if isinstance(v, float) else v) for k, v in d.items()}


# ---------- model ----------
def load_base():
    tok = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.bfloat16, device_map="auto",
        trust_remote_code=True)
    model.eval()
    return model, tok


def attach_lora(model):
    return get_peft_model(model, LoraConfig(
        r=LORA_R, lora_alpha=32, lora_dropout=0.05,
        target_modules=["q_proj", "v_proj"], task_type="CAUSAL_LM"))


# ---------- scoring helpers ----------
@torch.no_grad()
def generate(model, tok, prompt, max_new_tokens, do_sample=False,
             temperature=0.8, top_p=0.95):
    pids = tok(prompt, return_tensors="pt").input_ids.to(model.device)
    out = model.generate(
        pids, max_new_tokens=max_new_tokens, do_sample=do_sample,
        temperature=temperature, top_p=top_p, pad_token_id=tok.eos_token_id)
    cids = out[0:1, pids.shape[1]:]
    return tok.decode(cids[0], skip_special_tokens=True)


def nll_of_span(model, tok, prefix, span):
    """Mean NLL of span tokens given prefix (no grad)."""
    pids = tok(prefix, return_tensors="pt").input_ids.to(model.device)
    sids = tok(span, return_tensors="pt").input_ids.to(model.device)
    inp = torch.cat([pids, sids], dim=1)
    with torch.no_grad():
        logits = model(inp).logits[0].float()
    lp = torch.log_softmax(logits, dim=-1)
    off = pids.shape[1] - 1
    idx = torch.arange(sids.shape[1], device=inp.device)
    return -lp[off + idx, sids[0]].mean().item()


def turn_token_spans(tok, traj):
    """Token spans (start, end) of each supervised turn in the full text."""
    prefix = {"user": "USER", "assistant": "ASSISTANT",
              "tool": "TOOL", "observation": "OBSERVATION"}
    lines = [f"{prefix[t['role']]}: {t['text']}" for t in traj["turns"]]
    spans = {}
    for i in supervised_turns(traj):
        before = "\n".join(lines[:i]) + ("\n" if i > 0 else "")
        upto = "\n".join(lines[:i + 1]) + "\n"
        s = len(tok(before, return_tensors="pt").input_ids[0])
        e = len(tok(upto, return_tensors="pt").input_ids[0])
        spans[i] = (s, e)
    full = "\n".join(lines) + "\n"
    return full, spans


def compute_ig_weights(model, tok, trajs):
    """Turn-level information gain proxy (base model, pre-training).

    IG(t) = NLL(answer | prefix before t) - NLL(answer | prefix incl. t),
    target = the boxed answer string. weights = clip(IG,0) / mean  (mean 1).
    Returns (weights_per_traj, stats).
    """
    all_w, type_igs = [], {"assistant": [], "tool": []}
    for traj in trajs:
        prefix = {"user": "USER", "assistant": "ASSISTANT",
                  "tool": "TOOL", "observation": "OBSERVATION"}
        lines = [f"{prefix[t['role']]}: {t['text']}" for t in traj["turns"]]
        target = f"\\boxed{{{traj['answer']}}}"
        igs = []
        for i in supervised_turns(traj):
            before = "\n".join(lines[:i]) + ("\n" if i > 0 else "")
            upto = "\n".join(lines[:i + 1]) + "\n"
            ig = (nll_of_span(model, tok, before, target)
                  - nll_of_span(model, tok, upto, target))
            igs.append(max(ig, 0.0))
            type_igs[traj["turns"][i]["role"]].append(max(ig, 0.0))
        m = sum(igs) / len(igs)
        w = [x / m if m > 0 else 1.0 for x in igs]
        all_w.append(w)
    flat = [x for w in all_w for x in w]
    stats = {
        "mean_weight": sum(flat) / len(flat),
        "max_weight": max(flat),
        "frac_zero": sum(1 for x in flat if x == 0.0) / len(flat),
        "mean_ig_assistant": sum(type_igs["assistant"]) / max(len(type_igs["assistant"]), 1),
        "mean_ig_tool": sum(type_igs["tool"]) / max(len(type_igs["tool"]), 1),
    }
    return all_w, stats


# ---------- training ----------
def train_sft(model, tok, trajs, weights_per_traj, tag):
    """weights_per_traj: None -> uniform (standard SFT); else per-turn weights."""
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    enc = []
    for ti, traj in enumerate(trajs):
        full, spans = turn_token_spans(tok, traj)
        ids = tok(full, return_tensors="pt").input_ids[0]
        labels = torch.full_like(ids, -100)
        w = torch.zeros(ids.shape[0])
        sup = supervised_turns(traj)
        tw = weights_per_traj[ti] if weights_per_traj else [1.0] * len(sup)
        for (s, e), wt in zip([spans[i] for i in sup], tw):
            labels[s:e] = ids[s:e]
            w[s:e] = wt
        enc.append((ids, labels, w))
    losses, step = [], 0
    for ep in range(EPOCHS):
        rng = random.Random(SEED + ep)
        order = list(range(len(enc)))
        rng.shuffle(order)
        for i in range(0, len(order), BATCH):
            batch = [enc[j] for j in order[i:i + BATCH]]
            ml = max(x[0].shape[0] for x in batch)
            ib = torch.stack([torch.cat([x[0], torch.full(
                (ml - x[0].shape[0],), tok.pad_token_id)]) for x in batch]).to(model.device)
            lb = torch.stack([torch.cat([x[1], torch.full(
                (ml - x[1].shape[0],), -100)]) for x in batch]).to(model.device)
            wb = torch.stack([torch.cat([x[2], torch.zeros(
                ml - x[2].shape[0])]) for x in batch]).to(model.device)
            attn = (ib != tok.pad_token_id).long()
            logits = model(input_ids=ib, attention_mask=attn).logits.float()
            ce = torch.nn.functional.cross_entropy(
                logits.reshape(-1, logits.shape[-1]), lb.reshape(-1),
                ignore_index=-100, reduction="none")
            wflat = wb.reshape(-1)
            loss = (ce * wflat).sum() / wflat.sum().clamp_min(1e-9)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            step += 1
            losses.append(loss.item())
    model.eval()
    log(f"train[{tag}]: {step} steps, final loss {losses[-1]:.4f}")
    return losses


# ---------- eval ----------
def base_logprobs(model, tok, sentences):
    """Store base log-probs (fp16, cpu) for the KL-drift measurement."""
    out = []
    for s in sentences:
        ids = tok(s, return_tensors="pt").input_ids.to(model.device)
        with torch.no_grad():
            lp = torch.log_softmax(model(ids).logits[0].float(), dim=-1)
        out.append((ids[0].cpu(), lp.half().cpu()))
    return out


def mean_kl_vs_base(model, tok, base_lps):
    """Mean per-token KL(base || model) over the stored sentences."""
    kls = []
    for ids, blp in base_lps:
        ids_d = ids.to(model.device)
        with torch.no_grad():
            lp = torch.log_softmax(
                model(ids_d.unsqueeze(0)).logits[0].float(), dim=-1)
        blp_d = blp.to(model.device).float()
        # KL over next-token predictions: positions 0..n-2 predict 1..n-1
        p = blp_d[:-1].exp()
        kl = (p * (blp_d[:-1] - lp[:-1])).sum(-1)
        kls.append(kl.mean().item())
    return sum(kls) / len(kls)


def nll_of_text(model, tok, text):
    ids = tok(text, return_tensors="pt").input_ids.to(model.device)
    with torch.no_grad():
        logits = model(ids).logits[0]
    lp = torch.log_softmax(logits.float(), dim=-1)
    return -lp[:-1, ids[0, 1:]].mean().item()


def evaluate(model, tok, test_trajs, qa_items, base_lps):
    acc = sum(
        extract_boxed(generate(model, tok,
                               f"USER: {t['question']}\nASSISTANT:",
                               GEN_TOKENS)) == t["answer"]
        for t in test_trajs) / len(test_trajs)
    qa_ok = sum(qa_correct(generate(model, tok, q, 24), exp)
                for q, exp in qa_items)
    nlls = [nll_of_text(model, tok, s) for s in GENERAL_SENTENCES]
    return {
        "test_acc_greedy": acc,
        "qa_acc": qa_ok / max(len(qa_items), 1),
        "n_qa_items": len(qa_items),
        "general_nll": sum(nlls) / len(nlls),
        "mean_kl_vs_base": mean_kl_vs_base(model, tok, base_lps),
    }


# ---------- main ----------
def main():
    torch.manual_seed(SEED)
    random.seed(SEED)
    log("loading base model")
    model, tok = load_base()

    train_trajs = gen_traj_problems(N_TRAIN, SEED)
    test_trajs = gen_traj_problems(N_TEST, SEED + 999)

    # ---- Phase 0: base evals ----
    log("base: held-out acc")
    base_acc = sum(
        extract_boxed(generate(model, tok, f"USER: {t['question']}\nASSISTANT:",
                               GEN_TOKENS)) == t["answer"]
        for t in test_trajs) / N_TEST
    log(f"base held-out acc: {base_acc:.3f}")
    qa_keep = [(q, e) for q, e in FORGET_QA
               if qa_correct(generate(model, tok, q, 24), e)]
    log(f"QA probe: base passes {len(qa_keep)}/{len(FORGET_QA)}")
    base_lps = base_logprobs(model, tok, GENERAL_SENTENCES)
    base_metrics = evaluate(model, tok, test_trajs, qa_keep, base_lps)
    log("base metrics: " + json.dumps(r4(base_metrics)))

    # ---- Phase 1: IG weights (base model, pre-training) ----
    log("computing turn-level information-gain weights")
    ig_weights, ig_stats = compute_ig_weights(model, tok, train_trajs)
    log("IG stats: " + json.dumps(r4(ig_stats)))

    results = {
        "paper": "arXiv:2610.00949",
        "model": MODEL_ID,
        "config": {"n_train": N_TRAIN, "n_test": N_TEST, "lora_r": LORA_R,
                   "lr": LR, "epochs": EPOCHS, "batch": BATCH, "seed": SEED},
        "ig_proxy": ("IG(t) = NLL_base(boxed answer | prefix before t) - "
                     "NLL_base(boxed answer | prefix incl. t); "
                     "w = clip(IG,0)/mean, mean weight 1"),
        "ig_stats": ig_stats,
        "base_test_acc_greedy": base_acc,
        "base_metrics": base_metrics,
    }

    # ---- Phase 2+3: arms ----
    for arm, w in [("standard", None), ("pg_sft", ig_weights)]:
        log(f"=== arm {arm}: training ===")
        lora_model = attach_lora(model)
        losses = train_sft(lora_model, tok, train_trajs, w, arm)
        log(f"=== arm {arm}: evaluating ===")
        m = evaluate(lora_model, tok, test_trajs, qa_keep, base_lps)
        results[f"arm_{arm}"] = {
            "train_losses": [round(x, 4) for x in losses],
            "metrics": m,
            "delta_test_acc": m["test_acc_greedy"] - base_metrics["test_acc_greedy"],
            "delta_qa_acc": m["qa_acc"] - base_metrics["qa_acc"],
            "delta_general_nll": m["general_nll"] - base_metrics["general_nll"],
        }
        log(f"arm {arm}: " + json.dumps(r4(m)))
        with open("/content/pg_sft_partial.json", "w") as f:
            json.dump(results, f, indent=2)
        del lora_model
        torch.cuda.empty_cache()
        if arm == "standard":
            log("reloading clean base for arm B")
            del model
            torch.cuda.empty_cache()
            model, tok = load_base()

    results["wall_time_s"] = round(time.time() - T0, 1)
    with open("/content/pg_sft_results.json", "w") as f:
        json.dump(results, f, indent=2)
    log("RESULTS WRITTEN to /content/pg_sft_results.json")
    for arm in ("standard", "pg_sft"):
        m = results[f"arm_{arm}"]["metrics"]
        print(f"ARM {arm}: test_acc={m['test_acc_greedy']:.3f} "
              f"(d={results[f'arm_{arm}']['delta_test_acc']:+.3f}) "
              f"qa_acc={m['qa_acc']:.3f} (d={results[f'arm_{arm}']['delta_qa_acc']:+.3f}) "
              f"gen_nll={m['general_nll']:.3f} (d={results[f'arm_{arm}']['delta_general_nll']:+.3f}) "
              f"kl_vs_base={m['mean_kl_vs_base']:.4f}", flush=True)


if __name__ == "__main__":
    main()
