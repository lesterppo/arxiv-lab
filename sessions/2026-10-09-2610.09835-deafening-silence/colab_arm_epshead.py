#!/usr/bin/env python3
"""A Deafening Silence (arXiv 2610.09835) vs plain LoRA — Colab T4 bounded test.

Paper claim: catastrophic forgetting concentrates in the output embeddings
of tokens rarely seen in the new corpus, because absent tokens receive
persistent one-sided softmax gradients that Adam's sqrt(v_hat)
normalization amplifies into full-sized updates. Intervention: raise
Adam's epsilon EXCLUSIVELY for the output projection -> removes
39.4-67.9% of forgetting without degrading target learning
(also rescues released-head LoRA).

Test: sequential fine-tuning on Qwen3-4B NF4 QLoRA (bf16).
(deviation: Qwen3.5-4B's hybrid Mamba layers run slow reference kernels,
 pushing the run past free-tier survival; Qwen3-4B is a pure transformer.
 The paper's claim spans 4 model families, so the mechanism is not
 architecture-specific.)
  Phase 1: ARITH (digit vocabulary). Phase 2: REVERSE (letter vocabulary,
  disjoint from phase 1 -> digit tokens are "tokens the data never speaks").
  Arm A (baseline): AdamW eps=1e-8 everywhere.
  Arm B (intervention): AdamW eps=1e-8, except eps=EPS_HEAD for the
  lm_head LoRA params ("released-head LoRA", paper's setting).
Metric: forgetting = acc(ARITH after phase 1) - acc(ARITH after phase 2);
forgetting-reduction % = (forget_A - forget_B) / forget_A.
Also: acc(REVERSE) after phase 2 (target learning must not degrade),
and L2 drift of the lm_head LoRA params during phase 2 (mechanism probe).

Bounded: 450 train / 100 test per task, 100 steps/phase, batch 4.
Self-contained: pip installs at top (incl. torchao uninstall per the
documented Colab quirk). Writes /content/deafening_results.json and
/content/deafening_metrics.csv. No secrets.
"""
import subprocess, sys

# --- stage 0: deps (idempotent) ---
subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                "git+https://github.com/huggingface/transformers.git",
                "peft", "bitsandbytes", "accelerate"],
               check=True)
subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "torchao"],
               capture_output=True)

import csv, json, math, os, random, time, gc

import torch
import torch.nn as nn
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          BitsAndBytesConfig)

OUT_JSON = "/content/deafening_results.json"
OUT_CSV = "/content/deafening_metrics.csv"
MODEL_ID = "Qwen/Qwen3-4B"
SEED = 7
N_TRAIN, N_TEST = 450, 60
STEPS, BATCH = 60, 4
LR = 2e-4
LORA_R, LORA_ALPHA = 16, 32
TARGETS = ["q_proj", "v_proj", "lm_head"]  # released-head LoRA
EPS_BASE = 1e-8
EPS_HEAD = 1e-3  # intervention: raised eps for output projection only
                 # (paper does not state the value in the abstract; 1e-3
                 #  dampens the tiny-gradient regime ~1000x per the CPU
                 #  miniature while leaving normal gradients ~intact)

random.seed(SEED)


def gen_arith(n, seed):
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        a, b = rng.randint(10, 99), rng.randint(10, 99)
        out.append((f"Task: ARITH\nInput: {a}+{b}\nOutput:", str(a + b)))
    return out


def gen_reverse(n, seed):
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        s = "".join(rng.choice("abcdefghij") for _ in range(rng.randint(5, 8)))
        out.append((f"Task: REVERSE\nInput: reverse {s}\nOutput:", s[::-1]))
    return out


def load_base():
    try:
        with open("/proc/sys/vm/drop_caches", "w") as f:
            f.write("3\n")
    except Exception:
        pass
    gc.collect()
    bnb = BitsAndBytesConfig(load_in_4bit=True,
                             bnb_4bit_quant_type="nf4",
                             bnb_4bit_compute_dtype=torch.bfloat16)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, quantization_config=bnb, dtype=torch.bfloat16,
        device_map={"": 0}, low_cpu_mem_usage=True)
    model.config.use_cache = False
    for p in model.parameters():
        p.requires_grad = False
    model.train()
    return model


def encode_batch(tok, batch):
    fulls = [p + " " + y for p, y in batch]
    enc = tok(fulls, padding=True, truncation=True, max_length=96,
              return_tensors="pt")
    plens = [len(tok(p)["input_ids"]) for p, _ in batch]
    labels = enc["input_ids"].clone()
    for i, pl in enumerate(plens):
        labels[i, :pl] = -100
    labels[enc["attention_mask"] == 0] = -100
    return enc, labels


def train_loss(model, tok, batch):
    enc, labels = encode_batch(tok, batch)
    input_ids = enc["input_ids"].to(model.device)
    attn = enc["attention_mask"].to(model.device)
    labels = labels.to(model.device)
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        out = model(input_ids=input_ids, attention_mask=attn)
        logits = out.logits.float()
        loss = nn.functional.cross_entropy(
            logits[:, :-1].reshape(-1, logits.size(-1)),
            labels[:, 1:].reshape(-1), ignore_index=-100)
    return loss


@torch.no_grad()
def evaluate(model, tok, data):
    model.eval()
    correct, total = 0, 0
    for prompt, y in data:
        ids = tok(prompt, return_tensors="pt")["input_ids"].to(model.device)
        gen = model.generate(ids, max_new_tokens=12, do_sample=False,
                             pad_token_id=tok.eos_token_id)
        text = tok.decode(gen[0][ids.shape[1]:], skip_special_tokens=True).strip()
        ok = text.split()[0] == y if text else False
        total += 1
        correct += int(ok)
    model.train()
    return correct / total


def head_param_snapshot(model):
    return {n: p.detach().float().clone().cpu()
            for n, p in model.named_parameters()
            if p.requires_grad and "lm_head" in n}


def head_drift_norm(before, after):
    tot = 0.0
    for n in before:
        tot += (after[n] - before[n]).pow(2).sum().item()
    return math.sqrt(tot)


def run_arm(arm, train_a, train_b, test_a, test_b):
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    model = load_base()
    from peft import LoraConfig, get_peft_model
    cfg = LoraConfig(r=LORA_R, lora_alpha=LORA_ALPHA,
                     target_modules=TARGETS, bias="none",
                     task_type="CAUSAL_LM")
    model = get_peft_model(model, cfg)

    head_params = [p for n, p in model.named_parameters()
                   if p.requires_grad and "lm_head" in n]
    rest_params = [p for n, p in model.named_parameters()
                   if p.requires_grad and "lm_head" not in n]
    assert head_params, "no trainable lm_head params found!"
    eps_head = EPS_HEAD if arm == "epshead" else EPS_BASE
    opt = torch.optim.AdamW([
        {"params": head_params, "lr": LR, "eps": eps_head},
        {"params": rest_params, "lr": LR, "eps": EPS_BASE},
    ])

    def phase(data, steps):
        order = list(range(len(data)))
        idx = 0
        for s in range(steps):
            batch = [data[order[(idx + k) % len(data)]] for k in range(BATCH)]
            idx = (idx + BATCH) % len(data)
            opt.zero_grad()
            loss = train_loss(model, tok, batch)
            loss.backward()
            opt.step()

    metrics = {"arm": arm, "eps_head": eps_head,
               "n_head_params": sum(p.numel() for p in head_params),
               "n_rest_params": sum(p.numel() for p in rest_params)}

    # Phase 1: ARITH
    t0 = time.time()
    phase(train_a, STEPS)
    metrics["t_phase1_s"] = round(time.time() - t0, 1)
    acc_a1 = evaluate(model, tok, test_a)
    metrics["acc_arith_after_p1"] = acc_a1

    # Phase 2: REVERSE (digit tokens now absent)
    snap_before = head_param_snapshot(model)
    t0 = time.time()
    phase(train_b, STEPS)
    metrics["t_phase2_s"] = round(time.time() - t0, 1)
    snap_after = head_param_snapshot(model)
    metrics["head_drift_phase2"] = head_drift_norm(snap_before, snap_after)
    acc_a2 = evaluate(model, tok, test_a)
    acc_b2 = evaluate(model, tok, test_b)
    metrics["acc_arith_after_p2"] = acc_a2
    metrics["acc_reverse_after_p2"] = acc_b2
    metrics["forgetting_arith"] = acc_a1 - acc_a2

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return metrics


def main():
    import sys as _sys
    only_arm = "epshead"
    train_a = gen_arith(N_TRAIN, SEED)
    test_a = gen_arith(N_TEST, SEED + 1000)
    train_b = gen_reverse(N_TRAIN, SEED + 2000)
    test_b = gen_reverse(N_TEST, SEED + 3000)

    results = {"model": MODEL_ID, "seed": SEED, "n_train": N_TRAIN,
               "n_test": N_TEST, "steps_per_phase": STEPS, "batch": BATCH,
               "lr": LR, "eps_head": EPS_HEAD, "eps_base": EPS_BASE,
               "paper": "2610.09835"}
    rows = []
    for arm in (["baseline", "epshead"] if only_arm is None else [only_arm]):
        print(f"[{time.strftime('%H:%M:%S')}] arm={arm}", flush=True)
        m = run_arm(arm, train_a, train_b, test_a, test_b)
        results[arm] = m
        rows.append(m)
        print(f"[{time.strftime('%H:%M:%S')}] {arm}: "
              f"arith {m['acc_arith_after_p1']:.3f} -> {m['acc_arith_after_p2']:.3f} "
              f"(forget {m['forgetting_arith']:+.3f}), "
              f"reverse {m['acc_reverse_after_p2']:.3f}, "
              f"head_drift {m['head_drift_phase2']:.3f}", flush=True)

    if only_arm is None:
        fa = results["baseline"]["forgetting_arith"]
        fb = results["epshead"]["forgetting_arith"]
        results["forgetting_reduction_pct"] = (
            round(100 * (fa - fb) / fa, 1) if fa > 1e-9 else None)
        print(f"forgetting: baseline {fa:+.3f} vs epshead {fb:+.3f} "
              f"-> reduction {results['forgetting_reduction_pct']}%", flush=True)

    tag = f"_{only_arm}" if only_arm else ""
    with open(OUT_JSON.replace(".json", tag + ".json"), "w") as f:
        json.dump(results, f, indent=2)
    with open(OUT_CSV.replace(".csv", tag + ".csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=sorted(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print("wrote", OUT_JSON.replace(".json", tag + ".json"), flush=True)


if __name__ == "__main__":
    main()
