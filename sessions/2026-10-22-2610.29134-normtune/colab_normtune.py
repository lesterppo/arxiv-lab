#!/usr/bin/env python3
"""NormTune reproduction (arXiv 2610.29134) on Colab T4.

Paper claim: PEFT suffers norm-related under-adaptation (LoRA weight-update
norms markedly smaller / uneven vs full fine-tuning); NormTune adaptively
strengthens insufficiently-updated layers according to their weight-update
norms, closing much of the PEFT-vs-full gap on reasoning tasks
(+24.24% avg for rank-8 LoRA on AIME/MATH/GPQA in the paper).

Bounded test here (single T4, Qwen/Qwen3-8B -- Qwen3.5-9B is unloadable under
the transformers<5.0 pin, verified 2026-10-21):
  Arm A (control): rank-8 LoRA, uniform LR, 200 steps.
  Arm B (NormTune): rank-8 LoRA, 50 warmup steps uniform LR, then per-layer
      update norms ||B@A||_F measured; layers below the median get 3x LR
      for the remaining 150 steps (per-layer param groups).
Same data, same step budget. Metrics: train loss, per-layer update norms,
held-out exact-match on synthetic multi-step arithmetic reasoning.

Writes CSVs to /content/normtune/. Run with: colab.py exec_detach -s <sess>
-f colab_normtune.py --log /content/normtune/run.log
"""
import csv
import os
import random
import re
import subprocess
import sys

OUT = "/content/normtune"
os.makedirs(OUT, exist_ok=True)


def sh(*args):
    print("+", " ".join(args), flush=True)
    subprocess.run(args, check=True)


# ---------------------------------------------------------------- installs
sh(sys.executable, "-m", "pip", "install", "-q",
   "transformers==4.57.6", "peft==0.17.1", "bitsandbytes==0.50.2",
   "accelerate==1.10.1")
sh(sys.executable, "-m", "pip", "uninstall", "-y", "torchao")

import torch
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          BitsAndBytesConfig)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

# ---------------------------------------------------------------- data
SEED = 20261022
rng = random.Random(SEED)


def gen_problem(r):
    a, b, c, d = (r.randint(11, 49), r.randint(11, 49),
                  r.randint(2, 9), r.randint(2, 9))
    tmpl = r.choice([
        ("A farmer harvests {a} apples. She buys {b} more at the market, then "
         "gives {c} baskets of {d} apples to her neighbor. "
         "How many apples remain?",
         lambda a, b, c, d: a + b - c * d),
        ("A warehouse stores {a} crates. {b} more crates arrive. Workers ship "
         "out {c} trucks carrying {d} crates each. How many crates remain?",
         lambda a, b, c, d: a + b - c * d),
        ("A library has {a} books. It receives a donation of {b} books, then "
         "lends {c} boxes of {d} books to schools. How many books remain?",
         lambda a, b, c, d: a + b - c * d),
    ])
    text, fn = tmpl
    q = text.format(a=a, b=b, c=c, d=d)
    ans = fn(a, b, c, d)
    trace = (f"{q}\nStep 1: {a} + {b} = {a + b}. "
             f"Step 2: {c} x {d} = {c * d}. "
             f"Step 3: {a + b} - {c * d} = {ans}. Answer: {ans}")
    return q, trace, ans


def build_split(n, seed):
    r = random.Random(seed)
    return [gen_problem(r) for _ in range(n)]


TRAIN = build_split(240, SEED)
EVAL = build_split(80, SEED + 1)

# ---------------------------------------------------------------- model
MODEL = "Qwen/Qwen3-8B"
tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token


def fresh_model():
    bnb = BitsAndBytesConfig(load_in_4bit=True,
                             bnb_4bit_quant_type="nf4",
                             bnb_4bit_compute_dtype=torch.bfloat16)
    m = AutoModelForCausalLM.from_pretrained(
        MODEL, quantization_config=bnb, device_map="auto",
        torch_dtype=torch.bfloat16, trust_remote_code=True)
    m = prepare_model_for_kbit_training(m)
    cfg = LoraConfig(r=8, lora_alpha=16, lora_dropout=0.05,
                     target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                     "gate_proj", "up_proj", "down_proj"],
                     task_type="CAUSAL_LM")
    m = get_peft_model(m, cfg)
    m.train()
    return m


def layer_of(name):
    mm = re.search(r"layers\.(\d+)", name)
    return int(mm.group(1)) if mm else -1  # -1 = non-layer (embed/head)


def layer_update_norms(model):
    """||B@A||_F per transformer layer over LoRA modules (effective update)."""
    acc = {}
    for name, mod in model.named_modules():
        if hasattr(mod, "lora_A") and hasattr(mod, "lora_B"):
            w = (mod.lora_B["default"].weight.float()
                 @ mod.lora_A["default"].weight.float())
            ly = layer_of(name)
            acc[ly] = acc.get(ly, 0.0) + w.norm().item() ** 2
    return {ly: v ** 0.5 for ly, v in acc.items()}


def batchify(split, idx, bs, maxlen=256):
    texts = [split[i][1] for i in idx]
    enc = tok(texts, padding=True, truncation=True, max_length=maxlen,
              return_tensors="pt")
    return {k: v.cuda() for k, v in enc.items()}


def train_arm(arm, steps, warmup, boost):
    print(f"===== arm {arm}: steps={steps} warmup={warmup} boost={boost} =====",
          flush=True)
    model = fresh_model()
    n_layers = model.config.num_hidden_layers
    # per-layer param groups
    groups, seen = {}, {}
    for name, p in model.named_parameters():
        if p.requires_grad:
            ly = layer_of(name)
            groups.setdefault(ly, []).append(p)
    under = set()
    opt = torch.optim.AdamW(
        [{"params": groups[ly], "lr": 2e-4} for ly in sorted(groups)],
        betas=(0.9, 0.999), weight_decay=0.0)

    def set_boost(on):
        for i, g in enumerate(opt.param_groups):
            ly = sorted(groups)[i]
            g["lr"] = 2e-4 * (boost if (on and ly in under) else 1.0)

    order = list(range(len(TRAIN)))
    r = random.Random(SEED + hash(arm) % 1000)
    r.shuffle(order)
    logf = open(f"{OUT}/train_{arm}.csv", "w", newline="")
    logw = csv.writer(logf)
    logw.writerow(["step", "loss", "boosted_layers"])
    step = 0
    ptr = 0
    while step < steps:
        if step == warmup and boost > 1.0:
            norms = layer_update_norms(model)
            layers = sorted(l for l in norms if l >= 0)
            med = sorted(norms[l] for l in layers)[len(layers) // 2]
            under = {l for l in layers if norms[l] < med}
            set_boost(True)
            with open(f"{OUT}/layernorms_{arm}_warmup.csv", "w",
                      newline="") as f:
                w = csv.writer(f)
                w.writerow(["layer", "update_norm", "under_adapted"])
                for l in layers:
                    w.writerow([l, f"{norms[l]:.6f}", int(l in under)])
            vals = [norms[l] for l in layers]
            mean = sum(vals) / len(vals)
            cv = (sum((v - mean) ** 2 for v in vals) / len(vals)) ** 0.5 / mean
            print(f"[normtune] warmup done: median={med:.4f} cv={cv:.3f} "
                  f"under={sorted(under)}", flush=True)
        idx = [order[(ptr + k) % len(order)] for k in range(8)]
        ptr += 8
        b = batchify(TRAIN, idx, 8)
        labels = b["input_ids"].clone()
        labels[b["attention_mask"] == 0] = -100
        out = model(input_ids=b["input_ids"],
                    attention_mask=b["attention_mask"], labels=labels)
        loss = out.loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        opt.zero_grad()
        step += 1
        logw.writerow([step, f"{loss.item():.4f}", len(under)])
        if step % 25 == 0:
            logf.flush()
            print(f"[{arm}] step {step}/{steps} loss {loss.item():.4f}",
                  flush=True)
    logf.close()
    norms = layer_update_norms(model)
    with open(f"{OUT}/layernorms_{arm}_final.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["layer", "update_norm"])
        for l in sorted(norms):
            w.writerow([l, f"{norms[l]:.6f}"])
    torch.save(model.state_dict(), f"{OUT}/lora_{arm}.pt")
    return model


@torch.no_grad()
def evaluate(model, tag):
    model.eval()
    n_ok = 0
    rows = []
    for i, (q, trace, ans) in enumerate(EVAL):
        enc = tok(q + "\n", return_tensors="pt").to("cuda")
        gen = model.generate(**enc, max_new_tokens=128, do_sample=False,
                             pad_token_id=tok.eos_token_id)
        text = tok.decode(gen[0][enc["input_ids"].shape[1]:],
                          skip_special_tokens=True)
        m = re.search(r"Answer:\s*(-?\d+)", text)
        pred = int(m.group(1)) if m else None
        ok = int(pred == ans)
        n_ok += ok
        rows.append([i, ans, pred, ok])
    model.train()
    with open(f"{OUT}/eval_{tag}.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "gold", "pred", "correct"])
        w.writerows(rows)
    acc = n_ok / len(EVAL)
    print(f"[eval {tag}] exact-match {n_ok}/{len(EVAL)} = {acc:.3f}", flush=True)
    return acc


# ---------------------------------------------------------------- run
print("n_train", len(TRAIN), "n_eval", len(EVAL), flush=True)
print("cuda:", torch.cuda.get_device_name(0), flush=True)

# baseline: base model (no LoRA) eval for reference
base = fresh_model()
acc_base = evaluate(base, "base")
del base
torch.cuda.empty_cache()

mA = train_arm("control", steps=200, warmup=10**9, boost=1.0)
acc_A = evaluate(mA, "control")
del mA
torch.cuda.empty_cache()

mB = train_arm("normtune", steps=200, warmup=50, boost=3.0)
acc_B = evaluate(mB, "normtune")

with open(f"{OUT}/summary.txt", "w") as f:
    f.write(f"base={acc_base:.3f} control={acc_A:.3f} normtune={acc_B:.3f}\n")
print(f"DONE base={acc_base:.3f} control={acc_A:.3f} normtune={acc_B:.3f}",
      flush=True)
