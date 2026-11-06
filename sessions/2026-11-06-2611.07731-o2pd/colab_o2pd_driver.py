"""Colab driver — O2PD (arXiv:2611.07731, 2026-11-06 daily training track).

Paper: "O^2PD: Overlap-Gated On-Policy Distillation on Reliable Prefix".
Claims tested (bounded, one T4 day, GSM8K math):
  C1: alignment on the student's trajectory PREFIX (first 20%) is more
      reliable than on the full context -> prefix-gated OPD beats
      full-context top-k OPD.
  C2: Reset Support (pool mass outside teacher top-k into a reset atom)
      + Support Convergence fix Teacher-support Uncovering /
      Dominant-Mode Mismatch -> O2PD beats plain top-k OPD.
  C3: overlap-gated alpha-divergence interpolates mode-covering /
      mode-seeking adaptively per token.

Design (bounded adaptation, stated honestly):
  - Teacher: Qwen/Qwen3.5-9B (latest 9B class, tried first per standing
    picks). Student: Qwen/Qwen3-8B (proven training model). Both NF4.
  - On-policy rollouts come from the BASE student (greedy + temp sample,
    1 per problem, 200 GSM8K train problems); all arms train from the
    SAME cached teacher supervision, so the comparison isolates the LOSS
    design, not rollout luck. Teacher supervision is cached offline
    (sequential loads) so peak VRAM is one model at a time.
  - Arms (same data budget, same base, same steps):
      sft      : completion-only CE on gold answers (control).
      opd_full : top-k RKL (renormalized over teacher top-k support),
                 full rollout context (paper's baseline).
      o2pd     : prefix 20% only + reset-support atom + overlap-gated
                 alpha-divergence (the paper's method).
  - Eval: GSM8K test first 100, 2-shot, greedy, exact-match on the
    final number. Diagnostic: teacher top-1 agreement with the student
    token, prefix vs suffix positions (C1's reliability premise).
  - Incremental writes: every phase appends to /content/o2pd_metrics.csv
    and per-item rows to /content/o2pd_details.csv; adapter zips saved
    after each arm (reclamation lesson: never lose a finished arm).
  - Prints O2PD_DONE as its very last line on completion.

Loss formulas (also checked on CPU in test_o2pd_loss_cpu.py, pure math):
  support S = teacher top-k ids + {reset}; p_i = teacher prob (raw),
  p_reset = 1 - sum(top-k probs); q_i = student prob on same ids,
  q_reset = 1 - sum(q_i). All distributions sum to 1 over S.
  opd_full: RKL over renormalized top-k only:
      p'_i = p_i / sum(p_topk), q'_i = q_i / sum(q_topk),
      loss = sum_i q'_i * (log q'_i - log p'_i).
  o2pd: overlap o = sum_{i in S} min(p_i, q_i); alpha = clip(1 - o, 0.05, 0.95)
      (high overlap -> alpha small -> mode-seeking; low overlap -> alpha
      near 1 -> mode-covering). If o > 0.5 the reset atom is dropped
      (support converges to the aligned top-k intersection); else kept.
      alpha-divergence D_alpha(p||q) = (1 - sum_i p_i^alpha q_i^(1-alpha))
      / (alpha * (1 - alpha)); loss averaged over prefix positions only
      (t < 0.2 * L).

Run via: python3 colab.py train --driver <this file> --rotate
  --session o2pd-1106 --monitor-timeout 14400
  --results /content/o2pd_metrics.csv /content/o2pd_details.csv
  /content/o2pd_rollouts.jsonl /content/o2pd_teacher.jsonl
  /content/adapters/sft.zip /content/adapters/opd_full.zip
  /content/adapters/o2pd.zip
  --out-dir <session dir> --done-marker O2PD_DONE
  --registry-file <isolated registry>
"""

import csv
import gc
import json
import math
import os
import re
import shutil
import subprocess
import sys

METRICS = "/content/o2pd_metrics.csv"
DETAILS = "/content/o2pd_details.csv"
ROLLOUTS = "/content/o2pd_rollouts.jsonl"
TEACHER = "/content/o2pd_teacher.jsonl"
ADAPTER_DIR = "/content/adapters"
N_TRAIN = 200
N_EVAL = 100
TOPK = 10
PREFIX_FRAC = 0.20
SEQ_CAP = 256


def _pip():
    pkgs = ["transformers==5.20.2", "peft", "bitsandbytes", "accelerate",
            "datasets", "safetensors"]
    try:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q"] + pkgs,
                       check=True, timeout=1200)
    except Exception as e:  # noqa: BLE001
        print("pip install failed (continuing with existing stack): %r" % e,
              flush=True)
    subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "torchao"],
                   capture_output=True)


def _append_metrics(row):
    new = not os.path.exists(METRICS)
    with open(METRICS, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if new:
            w.writeheader()
        w.writerow(row)
    print("METRIC %s" % json.dumps(row), flush=True)


def _append_detail(row):
    new = not os.path.exists(DETAILS)
    with open(DETAILS, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if new:
            w.writeheader()
        w.writerow(row)


def _jsonl_append(path, obj):
    with open(path, "a") as f:
        f.write(json.dumps(obj) + "\n")


# ---------------------------------------------------------------- losses
# Pure-torch implementations; the math is mirrored in pure Python in
# test_o2pd_loss_cpu.py (same formulas, same constants).
def opd_full_loss(logits, tgt_ids, teacher_topk_ids, teacher_topk_probs):
    """Renormalized top-k RKL over the FULL rollout. logits: [L, V]
    (positions predicting tgt_ids), tgt_ids: [L]. teacher_*: [L, K]."""
    import torch
    import torch.nn.functional as F
    logp = F.log_softmax(logits.float(), dim=-1)
    probs = logp.exp()
    q = probs.gather(1, teacher_topk_ids)                       # [L, K]
    p = teacher_topk_probs
    qn = q / q.sum(-1, keepdim=True).clamp_min(1e-8)
    pn = p / p.sum(-1, keepdim=True).clamp_min(1e-8)
    rkl = (qn * (qn.clamp_min(1e-8).log() - pn.clamp_min(1e-8).log())).sum(-1)
    return rkl.mean()


def o2pd_loss(logits, tgt_ids, teacher_topk_ids, teacher_topk_probs):
    """Prefix-gated (first 20%) overlap-gated alpha-divergence with reset
    support. Returns (loss, mean_overlap, mean_alpha, n_prefix)."""
    import torch
    import torch.nn.functional as F
    L = tgt_ids.shape[0]
    n_pre = max(1, int(PREFIX_FRAC * L))
    logp = F.log_softmax(logits.float(), dim=-1)
    probs = logp.exp()
    q_topk = probs.gather(1, teacher_topk_ids)                  # [L, K]
    p_topk = teacher_topk_probs
    q_reset = (1.0 - q_topk.sum(-1, keepdim=True)).clamp_min(0.0)
    p_reset = (1.0 - p_topk.sum(-1, keepdim=True)).clamp_min(0.0)
    q = torch.cat([q_topk, q_reset], dim=-1)                    # [L, K+1]
    p = torch.cat([p_topk, p_reset], dim=-1)
    overlap = torch.minimum(p, q).sum(-1)                       # [L]
    alpha = (1.0 - overlap).clamp(0.05, 0.95)                   # [L]
    # Support convergence: drop the reset atom where overlap > 0.5.
    keep_reset = (overlap <= 0.5).float().unsqueeze(-1)         # [L, 1]
    q_eff = torch.cat([q_topk, q_reset * keep_reset], dim=-1)
    p_eff = torch.cat([p_topk, p_reset * keep_reset], dim=-1)
    q_eff = q_eff / q_eff.sum(-1, keepdim=True).clamp_min(1e-8)
    p_eff = p_eff / p_eff.sum(-1, keepdim=True).clamp_min(1e-8)
    a = alpha.unsqueeze(-1)
    inner = (p_eff.clamp_min(1e-8) ** a) * (q_eff.clamp_min(1e-8) ** (1 - a))
    d_alpha = (1.0 - inner.sum(-1)) / (a.squeeze(-1) * (1 - a.squeeze(-1)))
    loss = d_alpha[:n_pre].mean()
    return loss, float(overlap[:n_pre].mean()), float(alpha[:n_pre].mean()), n_pre


def main():
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF",
                          "expandable_segments:True")
    _pip()
    import torch
    import transformers
    print("transformers=%s" % transformers.__version__, flush=True)
    from transformers import (AutoModelForCausalLM, AutoTokenizer,
                              BitsAndBytesConfig)
    from peft import LoraConfig, get_peft_model, PeftModel
    from datasets import load_dataset
    assert torch.cuda.is_available(), "no GPU: refusing to fake a GPU run"
    print("GPU: %s total=%.1f GB torch=%s"
          % (torch.cuda.get_device_name(0),
             torch.cuda.get_device_properties(0).total_memory / 1e9,
             torch.__version__), flush=True)
    os.makedirs(ADAPTER_DIR, exist_ok=True)

    def load_nf4(model_id):
        tok = AutoTokenizer.from_pretrained(model_id)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        bnb = BitsAndBytesConfig(load_in_4bit=True,
                                 bnb_4bit_quant_type="nf4",
                                 bnb_4bit_use_double_quant=True,
                                 bnb_4bit_compute_dtype=torch.bfloat16)
        m = AutoModelForCausalLM.from_pretrained(
            model_id, quantization_config=bnb, torch_dtype=torch.bfloat16,
            device_map="auto")
        m.config.use_cache = False
        print("LOADED %s peak=%.2f GB"
              % (model_id, torch.cuda.max_memory_allocated() / 1e9), flush=True)
        return tok, m

    gsm = load_dataset("openai/gsm8k", "main", split="train")
    gsm_test = load_dataset("openai/gsm8k", "main", split="test")

    def prompt_of(ex):
        return "Question: %s\nAnswer:" % ex["question"]

    def gold_of(ex):
        return ex["answer"].split("####")[-1].strip()

    # ---- probe: student loads (GREEN gate for the verdict run) ----
    tok, student = load_nf4("Qwen/Qwen3-8B")
    _append_metrics({"phase": "probe", "model": "Qwen/Qwen3-8B",
                     "peak_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2),
                     "acc": "", "n": ""})
    tok.padding_side = "left"

    # ---- phase 1: student rollouts (1 per train problem, temp sample) ----
    train_idx = list(range(N_TRAIN))
    rollouts = []
    if os.path.exists(ROLLOUTS):
        rollouts = [json.loads(l) for l in open(ROLLOUTS)]
        print("RESUME rollouts: %d cached" % len(rollouts), flush=True)
    done_idx = {r["idx"] for r in rollouts}
    student.eval()
    with torch.no_grad():
        for s in range(0, len(train_idx), 8):
            chunk = [i for i in train_idx[s:s + 8] if i not in done_idx]
            if not chunk:
                continue
            prompts = [prompt_of(gsm[i]) for i in chunk]
            enc = tok(prompts, return_tensors="pt", padding=True).to(student.device)
            out = student.generate(**enc, max_new_tokens=160, do_sample=True,
                                   temperature=0.8, top_p=0.95,
                                   pad_token_id=tok.pad_token_id)
            for j, i in enumerate(chunk):
                gen_ids = out[j][enc["input_ids"].shape[1]:].tolist()
                gen_text = tok.decode(gen_ids, skip_special_tokens=True)
                rec = {"idx": i, "prompt": prompts[j],
                       "rollout_ids": gen_ids, "rollout_text": gen_text,
                       "gold": gold_of(gsm[i])}
                rollouts.append(rec)
                _jsonl_append(ROLLOUTS, rec)
            print("ROLLOUTS %d/%d" % (len(rollouts), N_TRAIN), flush=True)
    print("ROLLOUTS_DONE n=%d" % len(rollouts), flush=True)
    del student
    gc.collect()
    torch.cuda.empty_cache()

    # ---- phase 2: teacher scores rollouts (top-10 ids+probs per position) ----
    ttok, teacher = load_nf4("Qwen/Qwen3.5-9B")
    _append_metrics({"phase": "probe", "model": "Qwen/Qwen3.5-9B",
                     "peak_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2),
                     "acc": "", "n": ""})
    ttok.padding_side = "left"
    scored = []
    if os.path.exists(TEACHER):
        scored = [json.loads(l) for l in open(TEACHER)]
        print("RESUME teacher scores: %d cached" % len(scored), flush=True)
    done_idx = {r["idx"] for r in scored}
    import torch.nn.functional as F
    teacher.eval()
    pre_agree, suf_agree, pre_n, suf_n = 0, 0, 0, 0
    with torch.no_grad():
        for rec in rollouts:
            if rec["idx"] in done_idx:
                continue
            p_ids = ttok(rec["prompt"], add_special_tokens=False)["input_ids"]
            plen = len(p_ids)
            r_ids = rec["rollout_ids"][:max(0, SEQ_CAP - plen)]
            ids = p_ids + r_ids
            L = len(r_ids)
            inp = torch.tensor([ids], device=teacher.device)
            logits = teacher(input_ids=inp).logits[0]          # [T, V]
            # logits[t] predicts token t+1; rollout token j sits at
            # position plen+j, predicted by logits[plen+j-1].
            topk_ids, topk_probs = [], []
            for j in range(L):
                pos = plen + j - 1
                lp = F.log_softmax(logits[pos].float(), dim=-1)
                vals, idxs = torch.topk(lp.exp(), TOPK)
                topk_ids.append(idxs.tolist())
                topk_probs.append([round(float(v), 6) for v in vals])
            srec = {"idx": rec["idx"], "topk_ids": topk_ids,
                    "topk_probs": topk_probs}
            scored.append(srec)
            _jsonl_append(TEACHER, srec)
            if len(scored) % 25 == 0:
                print("TEACHER_SCORED %d/%d" % (len(scored), len(rollouts)),
                      flush=True)
    # Agreement diagnostic recomputed over ALL scored items (resume-safe):
    # teacher top-1 == student rollout token, prefix vs suffix positions.
    smap = {r["idx"]: r for r in scored}
    pre_agree, suf_agree, pre_n, suf_n = 0, 0, 0, 0
    for rec in rollouts:
        s = smap.get(rec["idx"])
        if s is None:
            continue
        L = len(s["topk_ids"])
        for j in range(L):
            agree = int(s["topk_ids"][j][0] == rec["rollout_ids"][j])
            if j < max(1, int(PREFIX_FRAC * L)):
                pre_agree += agree
                pre_n += 1
            else:
                suf_agree += agree
                suf_n += 1
    print("TEACHER_DONE n=%d prefix_agree=%.4f suffix_agree=%.4f"
          % (len(scored),
             pre_agree / max(1, pre_n), suf_agree / max(1, suf_n)), flush=True)
    _append_metrics({"phase": "teacher_diag", "model": "Qwen/Qwen3.5-9B",
                     "prefix_top1_agree": round(pre_agree / max(1, pre_n), 4),
                     "suffix_top1_agree": round(suf_agree / max(1, suf_n), 4),
                     "teacher_max_topk_id": max(
                         (i for r in scored for row in r["topk_ids"] for i in row),
                         default=-1),
                     "acc": "", "n": len(scored)})
    del teacher
    gc.collect()
    torch.cuda.empty_cache()

    # ---- phase 3: train the three arms (fresh LoRA each, one base load) ----
    tok, student = load_nf4("Qwen/Qwen3-8B")
    tok.padding_side = "right"
    scored_map = {r["idx"]: r for r in scored}

    def build_seq(prompt, completion_ids):
        p_ids = tok(prompt, add_special_tokens=False)["input_ids"]
        ids = (p_ids + completion_ids)[:SEQ_CAP]
        labels = ([-100] * len(p_ids) + completion_ids)[:SEQ_CAP]
        return {"input_ids": ids, "labels": labels,
                "attention_mask": [1] * len(ids)}

    sft_seqs = [build_seq(prompt_of(gsm[r["idx"]]),
                          tok(" " + gsm[r["idx"]]["answer"],
                              add_special_tokens=False)["input_ids"])
                for r in rollouts]
    opd_seqs = []
    # VOCAB FIX (diagnosed 2026-11-06, fallback run): the teacher
    # (Qwen3.5-9B) has a ~250k vocab; the student (Qwen3-8B) has 151,936.
    # Teacher top-k ids >= student vocab crash the student's gather with
    # a CUDA device-side assert (the fallback run died on exactly this).
    # Out-of-vocab teacher mass is dropped into the reset atom: replace
    # such (id, prob) pairs with a dummy in-vocab id at prob 0 and do NOT
    # renormalize, so the dropped mass flows to p_reset = 1 - sum(p_topk)
    # exactly like mass outside top-k. Record the dropped mass.
    vocab = len(tok)
    dropped_mass, dropped_n = 0.0, 0
    for r in rollouts:
        s = scored_map.get(r["idx"])
        if s is None:
            continue
        tk_ids, tk_probs = [], []
        for ids_row, probs_row in zip(s["topk_ids"], s["topk_probs"]):
            keep_ids, keep_probs = [], []
            for i, p in zip(ids_row, probs_row):
                if i < vocab:
                    keep_ids.append(i)
                    keep_probs.append(p)
                else:
                    keep_ids.append(0)
                    keep_probs.append(0.0)
                    dropped_mass += p
                    dropped_n += 1
            tk_ids.append(keep_ids)
            tk_probs.append(keep_probs)
        seq = build_seq(r["prompt"], r["rollout_ids"])
        seq["teacher_topk_ids"] = tk_ids
        seq["teacher_topk_probs"] = tk_probs
        opd_seqs.append(seq)
    if opd_seqs:
        _append_metrics({"phase": "vocab_filter", "model": "Qwen/Qwen3-8B",
                         "dropped_teacher_mass": round(dropped_mass, 4),
                         "dropped_positions": dropped_n,
                         "acc": "", "n": len(opd_seqs)})
    print("SEQS sft=%d opd=%d dropped_mass=%.4f" % (len(sft_seqs), len(opd_seqs), dropped_mass), flush=True)

    def train_arm(tag, mode):
        cfg = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.0,
                         target_modules="all-linear", task_type="CAUSAL_LM")
        peft_model = get_peft_model(student, cfg)
        peft_model.enable_input_require_grads()
        peft_model.gradient_checkpointing_enable()
        peft_model.train()
        opt = torch.optim.AdamW(
            [p for p in peft_model.parameters() if p.requires_grad],
            lr=1e-4, weight_decay=0.0)
        seqs = sft_seqs if mode == "sft" else opd_seqs
        tot_loss, steps, ovs, als = 0.0, 0, [], []
        accum = 4
        opt.zero_grad()
        for i, seq in enumerate(seqs):
            ids = torch.tensor([seq["input_ids"]], device=student.device)
            att = torch.tensor([seq["attention_mask"]], device=student.device)
            out = peft_model(input_ids=ids, attention_mask=att)
            logits = out.logits[0]                              # [T, V]
            # completion positions: logits[t] predicts token t+1
            comp_len = len([l for l in seq["labels"] if l != -100])
            start = len(seq["input_ids"]) - comp_len - 1
            comp_logits = logits[start:start + comp_len]
            if mode == "sft":
                import torch.nn.functional as F
                tgt = torch.tensor(
                    [l for l in seq["labels"] if l != -100],
                    device=student.device)
                loss = F.cross_entropy(comp_logits.float(), tgt)
            else:
                tk_ids = torch.tensor(seq["teacher_topk_ids"],
                                      device=student.device)
                tk_probs = torch.tensor(seq["teacher_topk_probs"],
                                        device=student.device,
                                        dtype=torch.float32)
                tgt = torch.tensor(
                    [l for l in seq["labels"] if l != -100],
                    device=student.device)
                if mode == "opd_full":
                    loss = opd_full_loss(comp_logits, tgt, tk_ids, tk_probs)
                else:
                    loss, ov, al, _ = o2pd_loss(comp_logits, tgt, tk_ids,
                                                tk_probs)
                    ovs.append(ov)
                    als.append(al)
            (loss / accum).backward()
            tot_loss += float(loss)
            steps += 1
            if steps % accum == 0 or i == len(seqs) - 1:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in peft_model.parameters() if p.requires_grad],
                    1.0)
                opt.step()
                opt.zero_grad()
            if steps % 50 == 0:
                print("ARM %s step %d loss=%.4f" % (tag, steps,
                      tot_loss / steps), flush=True)
        out_dir = os.path.join(ADAPTER_DIR, tag)
        peft_model.save_pretrained(out_dir)
        zip_path = shutil.make_archive(out_dir, "zip", root_dir=ADAPTER_DIR,
                                       base_dir=tag)
        print("ADAPTER_ZIP %s" % zip_path, flush=True)
        _append_metrics({"phase": "train", "model": "Qwen/Qwen3-8B",
                         "adapter": tag, "train_loss": round(tot_loss / steps, 4),
                         "mean_overlap": round(sum(ovs) / len(ovs), 4) if ovs else "",
                         "mean_alpha": round(sum(als) / len(als), 4) if als else "",
                         "acc": "", "n": steps})
        model = peft_model.unload()
        del peft_model
        gc.collect()
        torch.cuda.empty_cache()
        return model

    student = train_arm("sft", "sft")
    student = train_arm("opd_full", "opd_full")
    student = train_arm("o2pd", "o2pd")

    # ---- phase 4: eval base + each adapter (greedy, 2-shot, n=100) ----
    def fewshot():
        demos = []
        for ex in gsm.select(range(1000, 1002)):
            demos.append("Question: %s\nAnswer: %s"
                         % (ex["question"], ex["answer"]))
        return "\n\n".join(demos) + "\n\n"

    def eval_items(m, adapter_path, label):
        if adapter_path is None:
            m_eval, peft_eval = m, None
        else:
            peft_eval = PeftModel.from_pretrained(m, adapter_path)
            peft_eval.eval()
            m_eval = peft_eval
        items = []
        for ex in gsm_test.select(range(N_EVAL)):
            items.append((fewshot() + "Question: %s\nAnswer:" % ex["question"],
                          float(gold_of(ex))))
        tok.padding_side = "left"
        correct = 0
        with torch.no_grad():
            for s in range(0, len(items), 8):
                chunk = items[s:s + 8]
                enc = tok([c[0] for c in chunk], return_tensors="pt",
                          padding=True).to(m.device)
                out = m_eval.generate(**enc, max_new_tokens=160,
                                      do_sample=False,
                                      pad_token_id=tok.pad_token_id)
                for j, o in enumerate(out):
                    gen = tok.decode(o[enc["input_ids"].shape[1]:],
                                     skip_special_tokens=True)
                    nums = re.findall(r"-?\d+\.?\d*", gen.replace(",", ""))
                    ok = bool(nums) and abs(float(nums[-1]) - chunk[j][1]) < 1e-6
                    correct += int(ok)
                    _append_detail({"model": label, "idx": s + j,
                                    "correct": int(ok), "gold": chunk[j][1],
                                    "gen": gen[:300]})
        acc = correct / len(items)
        _append_metrics({"phase": "eval", "model": "Qwen/Qwen3-8B",
                         "adapter": label, "acc": round(acc, 4),
                         "n": len(items)})
        if peft_eval is not None:
            m_eval = peft_eval.unload()
        tok.padding_side = "right"
        return acc

    student.config.use_cache = True
    results = {}
    results["base"] = eval_items(student, None, "base")
    for tag in ["sft", "opd_full", "o2pd"]:
        results[tag] = eval_items(student, os.path.join(ADAPTER_DIR, tag), tag)
    print("RESULTS %s" % json.dumps(results), flush=True)
    print("O2PD_DONE", flush=True)


if __name__ == "__main__":
    main()
