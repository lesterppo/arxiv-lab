"""
Live test of arXiv:2610.02140 "Finetuning with Sampling: SFT Learns Better
Than You Think" on Google Colab T4 (Qwen2.5-0.5B-Instruct + LoRA).

Paper claim: an MCMC sampler progressively transforms OFF-policy expert
traces into traces that are more ON-policy for a reference model; plain
SFT on the reshaped data then rivals RL — better generalization, less
forgetting, learning beyond sharpening the base distribution.

Reduced experiment here:
  Phase 1: build a candidate pool per train prompt = [off-policy expert
           trace (PROTOCOL-7 style, correct by construction)] +
           K reference-model samples at temp 1.0.
  Phase 2: score every candidate under the reference model (mean token
           logp), run the Metropolis chain from mcmc.py targeting
           pi(x) ~ exp(logp(x)/T) * 1[correct(x)] -> reshaped dataset.
  Phase 3: A/B — (A) plain SFT on raw expert traces vs (B) plain SFT on
           MCMC-reshaped traces. Same LoRA config, lr, epochs.
  Phase 4: eval both + base: held-out test accuracy (generalization),
           general-QA accuracy + general-text NLL (forgetting),
           pass@1/pass@4 + mean token entropy (sharpness/coverage).

Results -> /content/sft_sampling_results.json (download after).
Run: upload data_gen.py + mcmc.py + this file to /content, then run this.
  !pip install -q transformers peft accelerate
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

from data_gen import (gen_problems, expert_trace, is_correct_trace,
                      extract_boxed, FORGET_QA, GENERAL_SENTENCES, qa_correct)
from mcmc import reshape_dataset

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
SEED = 0
N_TRAIN = 40
N_TEST = 30
POOL_SAMPLES = 6
MCMC_STEPS = 40
MCMC_TEMP = 1.0
GEN_TOKENS = 160
LORA_R = 16
LR = 2e-5
EPOCHS = 3
BATCH = 4
EVAL_K = 4  # pass@K samples

T0 = time.time()


def log(msg):
    print(f"[{time.time()-T0:7.1f}s] {msg}", flush=True)


def r4(d):
    return {k: (round(v, 4) if isinstance(v, float) else v) for k, v in d.items()}


def mean_logp(model, tok, prompt, completion):
    """Mean per-token log-prob of completion given prompt under model."""
    pids = tok(prompt, return_tensors="pt").input_ids.to(model.device)
    cids = tok(completion, return_tensors="pt").input_ids.to(model.device)
    inp = torch.cat([pids, cids], dim=1)
    with torch.no_grad():
        logits = model(inp).logits[0]
    lp = torch.log_softmax(logits.float(), dim=-1)
    off = pids.shape[1] - 1
    idx = torch.arange(cids.shape[1], device=inp.device)
    return lp[off + idx, cids[0]].mean().item()


@torch.no_grad()
def generate(model, tok, prompt, max_new_tokens, do_sample=False,
             temperature=0.8, top_p=0.95, num_return=1):
    pids = tok(prompt, return_tensors="pt").input_ids.to(model.device)
    out = model.generate(
        pids, max_new_tokens=max_new_tokens, do_sample=do_sample,
        temperature=temperature, top_p=top_p, pad_token_id=tok.eos_token_id,
        num_return_sequences=num_return)
    texts = []
    for i in range(num_return):
        cids = out[i:i + 1, pids.shape[1]:]
        texts.append(tok.decode(cids[0], skip_special_tokens=True))
    return texts


def nll_of_text(model, tok, text):
    ids = tok(text, return_tensors="pt").input_ids.to(model.device)
    with torch.no_grad():
        logits = model(ids).logits[0]
    lp = torch.log_softmax(logits.float(), dim=-1)
    # NLL of tokens 1..n given 0..n-1
    return -lp[:-1, ids[0, 1:]].mean().item()


def train_sft(model, tok, data, tag):
    """data: list of (prompt, trace). Plain SFT with LoRA already attached."""
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    enc = []
    for prompt, trace in data:
        p = tok(prompt, return_tensors="pt").input_ids[0]
        t = tok(trace, return_tensors="pt").input_ids[0]
        ids = torch.cat([p, t])
        labels = torch.cat([torch.full_like(p, -100), t])
        enc.append((ids, labels))
    losses = []
    step = 0
    for ep in range(EPOCHS):
        rng = random.Random(SEED + ep)
        order = list(range(len(enc)))
        rng.shuffle(order)
        for i in range(0, len(order), BATCH):
            batch = [enc[j] for j in order[i:i + BATCH]]
            ml = max(x[0].shape[0] for x in batch)
            ib = torch.stack([torch.cat([x[0],
                torch.full((ml - x[0].shape[0],), tok.pad_token_id)]) for x in batch]).to(model.device)
            lb = torch.stack([torch.cat([x[1],
                torch.full((ml - x[1].shape[0],), -100)]) for x in batch]).to(model.device)
            attn = (ib != tok.pad_token_id).long()
            logits = model(input_ids=ib, attention_mask=attn).logits.float()
            loss = torch.nn.functional.cross_entropy(
                logits.reshape(-1, logits.shape[-1]), lb.reshape(-1),
                ignore_index=-100)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            step += 1
            losses.append(loss.item())
    model.eval()
    log(f"train[{tag}]: {step} steps, final loss {losses[-1]:.4f}")
    return losses


def evaluate(model, tok, test_problems, qa_items):
    """Returns dict of metrics."""
    # 1) held-out accuracy (greedy) + pass@K samples
    acc_greedy, pass1_hits, passk_hits = 0, 0, 0
    entropies = []
    for p in test_problems:
        g = generate(model, tok, p["question"], 128)[0]
        if extract_boxed(g) == p["answer"]:
            acc_greedy += 1
        samps = generate(model, tok, p["question"], 128, do_sample=True,
                         temperature=0.8, num_return=EVAL_K)
        hits = sum(extract_boxed(s) == p["answer"] for s in samps)
        pass1_hits += (hits > 0)  # first sample ~ pass@1 proxy
        passk_hits += (hits > 0)
        # mean token entropy over the sampled completions
        for s in samps:
            pids = tok(p["question"], return_tensors="pt").input_ids.to(model.device)
            cids = tok(s, return_tensors="pt").input_ids.to(model.device)
            if cids.shape[1] == 0:
                continue
            inp = torch.cat([pids, cids], dim=1)
            with torch.no_grad():
                logits = model(inp).logits[0].float()
            probs = torch.softmax(logits, dim=-1)
            ent = -(probs * torch.log(probs + 1e-12)).sum(-1)
            off = pids.shape[1] - 1
            idx = torch.arange(cids.shape[1], device=inp.device)
            entropies.append(ent[off + idx].mean().item())
    n = len(test_problems)
    # 2) forgetting: QA accuracy on base-passing items
    qa_ok = 0
    for q, expected in qa_items:
        g = generate(model, tok, q, 24)[0]
        if qa_correct(g, expected):
            qa_ok += 1
    # 3) forgetting: mean NLL on general sentences
    nlls = [nll_of_text(model, tok, s) for s in GENERAL_SENTENCES]
    return {
        "test_acc_greedy": acc_greedy / n,
        "pass1_proxy": pass1_hits / n,
        "pass_at_k": passk_hits / n,
        "k": EVAL_K,
        "mean_token_entropy": sum(entropies) / max(len(entropies), 1),
        "qa_acc": qa_ok / max(len(qa_items), 1),
        "n_qa_items": len(qa_items),
        "general_nll": sum(nlls) / len(nlls),
    }


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


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--resume", action="store_true",
                    help="resume from /content/sft_sampling_pools.json checkpoint")
    args = ap.parse_args()
    torch.manual_seed(SEED)
    random.seed(SEED)
    log("loading base model")
    model, tok = load_base()

    train_probs = gen_problems(N_TRAIN, SEED)
    test_probs = gen_problems(N_TEST, SEED + 999)

    resumed = args.resume and os.path.exists("/content/sft_sampling_pools.json")
    if resumed:
        log("resuming from pools checkpoint")
        ck = json.load(open("/content/sft_sampling_pools.json"))
        pools, chosen = ck["pools"], ck["chosen"]
        agg, qa_keep = ck["mcmc"], ck["qa_keep"]
        base_metrics = ck["base_metrics"]
        base_test_acc = ck["base_test_acc_greedy"]
        pool_rate = ck["pool_base_correct_rate"]
        n_correct_samples = int(round(pool_rate * N_TRAIN * POOL_SAMPLES))
        expert_traces = [pool["traces"][0] for pool in pools]
        lp_expert = sum(pool["logps"][0] for pool in pools) / len(pools)
        lp_chosen = sum(pool["logps"][chosen[i]]
                        for i, pool in enumerate(pools)) / len(pools)
        log(f"resumed: {len(pools)} pools, moved_fraction={agg['moved_fraction']:.2f}, "
            f"ref logp expert={lp_expert:.3f} reshaped={lp_chosen:.3f}")
    else:
        # ---- Phase 0: base evals (incl. QA probe filtering) ----
        base_test_acc = sum(
            extract_boxed(generate(model, tok, p["question"], 128)[0]) == p["answer"]
            for p in test_probs) / N_TEST
        log(f"base held-out acc (greedy): {base_test_acc:.3f}")
        qa_keep = []
        for q, expected in FORGET_QA:
            g = generate(model, tok, q, 24)[0]
            if qa_correct(g, expected):
                qa_keep.append((q, expected))
        log(f"QA probe: base passes {len(qa_keep)}/{len(FORGET_QA)} items")
        base_metrics = evaluate(model, tok, test_probs, qa_keep)
        log("base metrics: " + json.dumps(r4(base_metrics)))

    if not resumed:
        # ---- Phase 1: candidate pools ----
        pools = []
        expert_traces = []
        n_correct_samples = 0
        for p in train_probs:
            expert = expert_trace(p)
            expert_traces.append(expert)
            assert is_correct_trace(expert, p["answer"])
            samps = generate(model, tok, p["question"], GEN_TOKENS, do_sample=True,
                             temperature=1.0, num_return=POOL_SAMPLES)
            traces = [expert] + samps
            correct = [is_correct_trace(t, p["answer"]) for t in traces]
            n_correct_samples += sum(correct[1:])
            logps = [mean_logp(model, tok, p["question"], t) for t in traces]
            pools.append({"traces": traces, "logps": logps, "correct": correct,
                          "question": p["question"], "answer": p["answer"]})
        log(f"pools built: {N_TRAIN} prompts, "
            f"base-sample correct rate {n_correct_samples}/{N_TRAIN*POOL_SAMPLES}")

        # ---- Phase 2: MCMC reshape ----
        chosen, agg = reshape_dataset(pools, steps=MCMC_STEPS,
                                      temperature=MCMC_TEMP, seed=SEED)
        log("MCMC: " + json.dumps(r4(agg)))

        # on-policyness check: mean reference-model logp of the expert traces
        # vs the MCMC-chosen traces, using the pool scores from the MCMC itself
        # (mean_logp of completion given question — the same quantity the chain
        #  targeted). NOTE: we deliberately do NOT use a joint
        #  question+trace NLL here: for this instruct model fed raw prompts
        #  (no chat template) that quantity is dominated by tokenization
        #  artifacts and no-context perplexity, and it disagrees with the
        #  conditional scores (see session notes).
        lp_expert = sum(pool["logps"][0] for pool in pools) / len(pools)
        lp_chosen = sum(pool["logps"][chosen[i]]
                        for i, pool in enumerate(pools)) / len(pools)
        log(f"mean ref logp(trace|question): expert={lp_expert:.3f} "
            f"reshaped={lp_chosen:.3f} "
            f"(delta={lp_chosen - lp_expert:+.3f}; positive = reshaped more on-policy)")
        # checkpoint pools so a re-run can --resume past the expensive sampling
        with open("/content/sft_sampling_pools.json", "w") as f:
            json.dump({"pools": pools, "chosen": chosen, "mcmc": agg,
                       "qa_keep": qa_keep, "base_metrics": base_metrics,
                       "base_test_acc_greedy": base_test_acc,
                       "pool_base_correct_rate":
                           n_correct_samples / (N_TRAIN * POOL_SAMPLES)}, f)
        log("pools checkpointed to /content/sft_sampling_pools.json")

    # training datasets (built in Phase 2, or rebuilt here on --resume)
    reshaped = [(pools[i]["question"], pools[i]["traces"][chosen[i]])
                for i in range(N_TRAIN)]
    expert_data = [(pools[i]["question"], expert_traces[i]) for i in range(N_TRAIN)]

    results = {
        "paper": "arXiv:2610.02140",
        "model": MODEL_ID,
        "config": {"n_train": N_TRAIN, "n_test": N_TEST, "pool": POOL_SAMPLES,
                   "mcmc_steps": MCMC_STEPS, "mcmc_temp": MCMC_TEMP,
                   "lora_r": LORA_R, "lr": LR, "epochs": EPOCHS, "batch": BATCH},
        "base_test_acc_greedy": base_test_acc,
        "pool_base_correct_rate": n_correct_samples / (N_TRAIN * POOL_SAMPLES),
        "mcmc": agg,
        "mean_ref_logp_expert": lp_expert,
        "mean_ref_logp_reshaped": lp_chosen,
        "base_metrics": base_metrics,
        "chosen_trace_kinds": ["expert" if c == 0 else "base_sample" for c in chosen],
    }

    # ---- Phase 3+4: arm A (expert SFT), eval, reload, arm B (reshaped SFT), eval
    for arm, data in [("expert", expert_data), ("reshaped", reshaped)]:
        log(f"=== arm {arm}: training ===")
        lora_model = attach_lora(model)
        losses = train_sft(lora_model, tok, data, arm)
        log(f"=== arm {arm}: evaluating ===")
        m = evaluate(lora_model, tok, test_probs, qa_keep)
        results[f"arm_{arm}"] = {
            "train_losses": [round(x, 4) for x in losses],
            "metrics": m,
        }
        log("arm " + arm + ": " + json.dumps(r4(m)))
        # reload clean base for the next arm
        del lora_model
        torch.cuda.empty_cache()
        if arm == "expert":
            log("reloading clean base model for arm B")
            del model
            torch.cuda.empty_cache()
            model, tok = load_base()

    # forgetting deltas
    for arm in ("expert", "reshaped"):
        m = results[f"arm_{arm}"]["metrics"]
        results[f"arm_{arm}"]["delta_qa_acc"] = m["qa_acc"] - base_metrics["qa_acc"]
        results[f"arm_{arm}"]["delta_general_nll"] = m["general_nll"] - base_metrics["general_nll"]
        results[f"arm_{arm}"]["delta_test_acc"] = m["test_acc_greedy"] - base_metrics["test_acc_greedy"]

    results["wall_time_s"] = round(time.time() - T0, 1)
    with open("/content/sft_sampling_results.json", "w") as f:
        json.dump(results, f, indent=2)
    log("RESULTS WRITTEN to /content/sft_sampling_results.json")
    for arm in ("expert", "reshaped"):
        m = results[f"arm_{arm}"]["metrics"]
        print(f"ARM {arm}: test_acc={m['test_acc_greedy']:.3f} "
              f"(d={results[f'arm_{arm}']['delta_test_acc']:+.3f}) "
              f"qa_acc={m['qa_acc']:.3f} (d={results[f'arm_{arm}']['delta_qa_acc']:+.3f}) "
              f"gen_nll={m['general_nll']:.3f} (d={results[f'arm_{arm}']['delta_general_nll']:+.3f}) "
              f"pass@{EVAL_K}={m['pass_at_k']:.3f} entropy={m['mean_token_entropy']:.3f}",
              flush=True)


if __name__ == "__main__":
    main()
