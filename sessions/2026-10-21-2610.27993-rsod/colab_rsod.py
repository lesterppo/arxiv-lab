#!/usr/bin/env python3
"""RSOD bounded reproduction (arXiv 2610.27993) on Colab T4.

Paper claim: Routed Specialist On-Policy Distillation consolidates N
specialist LoRA teachers into ONE student LoRA that matches the routed
teacher system. The committed student generates each trajectory; the
route-assigned specialist teacher-forces the exact student token sequence
(computes its logits over those tokens); only the student LoRA is updated.

Bounded test (one T4, 9B-class NF4 QLoRA r=16/alpha=32, bf16):
  routes: ARITH (a+b/c-d=?) and CODE (complete one-line python function)
  Stage 1: train 2 specialist LoRAs (frozen after), one per route (SFT)
  Stage 2a RSOD: student LoRA trained with KL(teacher||student) on
           student-generated trajectories, route-oracle teacher forcing
  Stage 2b MIXSFT: one LoRA, plain SFT on the mixed specialist data
  Eval: exact-match per route on held-out sets for BASE / ROUTED / RSOD / MIXSFT

Writes /content/rsod_results/metrics.json + metrics.csv incrementally.
Self-contained: pip installs at top (incl. torchao uninstall per protocol).
"""
import csv
import json
import os
import random
import subprocess
import sys
import time

OUTDIR = "/content/rsod_results"
os.makedirs(OUTDIR, exist_ok=True)


def log(msg):
    print(f"[rsod {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def sh(cmd):
    log(f"$ {cmd}")
    subprocess.run(cmd, shell=True, check=True)


def save_metrics(m):
    with open(f"{OUTDIR}/metrics.json", "w") as f:
        json.dump(m, f, indent=1)
    with open(f"{OUTDIR}/metrics.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["stage", "arm", "route", "metric", "value"])
        for row in m.get("rows", []):
            w.writerow(row)


METRICS = {"rows": [], "notes": []}


def record(stage, arm, route, metric, value):
    METRICS["rows"].append([stage, arm, route, metric, round(float(value), 4)])
    save_metrics(METRICS)


# ---------------------------------------------------------------- installs
sh("pip install -q 'transformers>=4.56,<5.0' peft bitsandbytes accelerate")
sh("pip uninstall -y torchao")

import torch
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          BitsAndBytesConfig)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

SEED = 7
random.seed(SEED)
torch.manual_seed(SEED)

DEVICE = "cuda"
DTYPE = torch.bfloat16

# ---------------------------------------------------------------- model
bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                         bnb_4bit_compute_dtype=DTYPE, bnb_4bit_use_double_quant=True)


def load_base():
    for mid in ("Qwen/Qwen3.5-9B", "Qwen/Qwen3-8B"):
        try:
            tok = AutoTokenizer.from_pretrained(mid, trust_remote_code=True)
            if tok.pad_token is None:
                tok.pad_token = tok.eos_token
            tok.padding_side = "left"
            mdl = AutoModelForCausalLM.from_pretrained(
                mid, quantization_config=bnb, device_map="auto",
                torch_dtype=DTYPE, trust_remote_code=True)
            mdl.config.use_cache = False
            mdl.gradient_checkpointing_enable()
            log(f"loaded {mid}")
            return mid, mdl, tok
        except Exception as e:
            log(f"{mid} failed ({type(e).__name__}: {str(e)[:120]}), trying next")
    raise RuntimeError("no model loaded")


MID, model, tok = load_base()
# QLoRA recipe (cf. 2026-10-20 wsr-tune): prepare for k-bit training BEFORE
# wrapping with LoRA. Among other things this calls
# enable_input_require_grads(), without which gradient checkpointing sees
# no grad-requiring inputs and loss.backward() fails.
model = prepare_model_for_kbit_training(model)
model.gradient_checkpointing_enable()
METRICS["model_id"] = MID
save_metrics(METRICS)

LORA_CFG = dict(r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
                task_type="CAUSAL_LM",
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                "gate_proj", "up_proj", "down_proj"])

# ---------------------------------------------------------------- data
# Two synthetic routes. Train and eval are disjoint (different numbers/names).
ARITH_TPL = ["Q: What is {a} plus {b}?\nA: {c}",
             "Q: Compute {a} + {b}.\nA: {c}",
             "Q: {a} added to {b} equals?\nA: {c}",
             "Q: Sum of {a} and {b}?\nA: {c}"]
ARITH_TPL_SUB = ["Q: What is {a} minus {b}?\nA: {c}",
                 "Q: Compute {a} - {b}.\nA: {c}"]
CODE_FNS = [("double", "n * 2"), ("triple", "n * 3"), ("inc", "n + 1"),
            ("square", "n ** 2"), ("neg", "-n"), ("half", "n // 2")]
CODE_TPL = ["def {fn}(n):\n    return {body}",
            "def {fn}(x):\n    return {body}"]


def gen_arith(n, rng, lo, hi):
    out = []
    for _ in range(n):
        a, b = rng.randint(lo, hi), rng.randint(lo, hi)
        if rng.random() < 0.5:
            t = rng.choice(ARITH_TPL)
            out.append((t.format(a=a, b=b, c=a + b), "arith"))
        else:
            t = rng.choice(ARITH_TPL_SUB)
            c = a - b
            out.append((t.format(a=a, b=b, c=c), "arith"))
    return out


def gen_code(n, rng, fns):
    out = []
    for _ in range(n):
        fn, body = rng.choice(fns)
        t = rng.choice(CODE_TPL)
        var = "n" if "(n)" in t else "x"
        out.append((t.format(fn=fn, body=body).replace("n *", var + " *")
                    .replace("n +", var + " +").replace("n **", var + " **")
                    .replace("-n", "-" + var).replace("n //", var + " //"), "code"))
    return out


def split_prompt_answer(text):
    # prompt = everything up to and incl "A: " / "return ", answer = rest
    if "\nA: " in text:
        p, a = text.split("\nA: ", 1)
        return p + "\nA: ", a
    else:  # code: prompt ends at "return "
        p, a = text.rsplit("return ", 1)
        return p + "return ", a


rng = random.Random(SEED)
train_arith = gen_arith(160, rng, 10, 499)
train_code = gen_code(160, rng, CODE_FNS)
eval_arith = gen_arith(60, random.Random(SEED + 1), 500, 999)
eval_code = gen_code(60, random.Random(SEED + 2), CODE_FNS)
log(f"data: train {len(train_arith)} arith + {len(train_code)} code; "
    f"eval {len(eval_arith)} + {len(eval_code)}")


def tok_batch(texts, max_len=128):
    return tok(texts, return_tensors="pt", padding=True, truncation=True,
               max_length=max_len).to(DEVICE)


@torch.no_grad()
def greedy(prompts, max_new=16):
    """Greedy generation, returns decoded answer strings."""
    was_training = model.training
    model.eval()
    model.config.use_cache = True
    enc = tok_batch(prompts)
    out = model.generate(**enc, max_new_tokens=max_new, do_sample=False,
                         pad_token_id=tok.pad_token_id)
    gen = out[:, enc["input_ids"].shape[1]:]
    texts = tok.batch_decode(gen, skip_special_tokens=True)
    model.config.use_cache = False
    if was_training:
        model.train()
    return [t.strip() for t in texts]


def exact_match(items, arm_name, route):
    prompts = [split_prompt_answer(t)[0] for t, _ in items]
    golds = [split_prompt_answer(t)[1].strip() for t, _ in items]
    preds = greedy(prompts)
    acc = sum(p.split()[0:1] == g.split()[0:1] or p.strip().rstrip('.') == g.strip()
              for p, g in zip(preds, golds)) / len(items)
    record("eval", arm_name, route, "exact_match", acc)
    log(f"eval {arm_name}/{route}: acc={acc:.3f} "
        f"(e.g. pred={preds[0]!r} gold={golds[0]!r})")
    return acc

# ---------------------------------------------------------------- training
def make_adapter(name):
    """Lazily wrap the base in a PeftModel on first call, then add adapters."""
    global model
    cfg = LoraConfig(**LORA_CFG)
    if not hasattr(model, "peft_config"):
        model = get_peft_model(model, cfg, adapter_name=name)
    else:
        model.add_adapter(adapter_name=name, peft_config=cfg)


def sft_train(adapter, data, steps, batch, lr, tag):
    """Plain supervised fine-tune of one adapter on (text, route) pairs."""
    make_adapter(adapter)
    model.set_adapter(adapter)
    # peft leaves ONLY LoRA params trainable by default (base frozen).
    # Inactive adapters are not in the forward graph when another adapter
    # is set, so they receive no gradients and the optimizer skips them.
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log(f"{tag}: trainable params = {n_train}")
    assert 0 < n_train < 200_000_000, f"unexpected trainable count: {n_train}"
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                            lr=lr)
    model.train()
    rng = random.Random(SEED)
    data = list(data)
    step = 0
    t0 = time.time()
    while step < steps:
        rng.shuffle(data)
        for i in range(0, len(data), batch):
            if step >= steps:
                break
            texts = [t for t, _ in data[i:i + batch]]
            enc = tok_batch(texts)
            labels = enc["input_ids"].clone()
            # mask prompt tokens (pad-aware: left padding shifts positions)
            seq_len = enc["input_ids"].shape[1]
            for j, t in enumerate(texts):
                prompt, _ = split_prompt_answer(t)
                plen = len(tok(prompt)["input_ids"])
                real = int(enc["attention_mask"][j].sum())
                pad = seq_len - real
                labels[j, :min(pad + plen, seq_len)] = -100
            out = model(input_ids=enc["input_ids"],
                        attention_mask=enc["attention_mask"], labels=labels)
            loss = out.loss
            loss.backward()
            opt.step()
            opt.zero_grad()
            step += 1
            if step % 5 == 0:
                log(f"{tag} step {step}/{steps} loss={loss.item():.4f} "
                    f"({time.time()-t0:.0f}s)")
    record("train", tag, "all", "final_loss", loss.item())
    record("train", tag, "all", "seconds", time.time() - t0)
    model.eval()


def kl_distill_train(student, teachers, gen_data, steps, batch, lr, temp=1.0):
    """RSOD: student generates trajectories; route-oracle teacher forces the
    student's exact token sequence and supplies soft targets; only the
    student LoRA is updated. Loss = mean KL(teacher||student) on response
    tokens."""
    make_adapter(student)
    for tname in teachers:
        assert tname in model.peft_config, f"missing teacher adapter {tname}"
    model.train()
    # Only LoRA params are trainable (peft default); the active adapter is
    # the student, so teacher adapters get no gradients.
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log(f"rsod: trainable params = {n_train}")
    assert 0 < n_train < 200_000_000, f"unexpected trainable count: {n_train}"
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                            lr=lr)
    rng = random.Random(SEED + 99)
    gen_data = list(gen_data)
    step, t0 = 0, time.time()
    kl_total, kl_n = 0.0, 0
    while step < steps:
        rng.shuffle(gen_data)
        for i in range(0, len(gen_data), batch):
            if step >= steps:
                break
            chunk = gen_data[i:i + batch]
            prompts = [split_prompt_answer(t)[0] for t, _ in chunk]
            routes = [r for _, r in chunk]
            # 1. student (on-policy) generates
            model.set_adapter(student)
            model.eval()
            model.config.use_cache = True
            enc = tok_batch(prompts)
            with torch.no_grad():
                out = model.generate(**enc, max_new_tokens=16, do_sample=False,
                                     pad_token_id=tok.pad_token_id)
            model.config.use_cache = False
            gen_ids = out[:, enc["input_ids"].shape[1]:]
            full = torch.cat([enc["input_ids"], gen_ids], dim=1)
            attn = torch.ones_like(full)
            # 2. teacher forces the student sequence; 3. KL on response tokens
            model.train()
            losses = []
            for j in range(len(chunk)):
                tname = {"arith": "spec_arith", "code": "spec_code"}[routes[j]]
                plen = enc["input_ids"].shape[1]
                with torch.no_grad():
                    model.set_adapter(tname)
                    t_out = model(input_ids=full[j:j + 1], attention_mask=attn[j:j + 1])
                    t_logits = t_out.logits[0, plen - 1:-1, :] / temp
                model.set_adapter(student)
                s_out = model(input_ids=full[j:j + 1], attention_mask=attn[j:j + 1])
                s_logits = s_out.logits[0, plen - 1:-1, :] / temp
                # KL(teacher || student) over response positions
                t_logp = torch.log_softmax(t_logits.float(), -1)
                s_logp = torch.log_softmax(s_logits.float(), -1)
                kl = torch.sum(torch.exp(t_logp) * (t_logp - s_logp), -1).mean()
                losses.append(kl)
            loss = torch.stack(losses).mean()
            loss.backward()
            opt.step()
            opt.zero_grad()
            step += 1
            kl_total += loss.item()
            kl_n += 1
            if step % 5 == 0:
                log(f"rsod step {step}/{steps} kl={loss.item():.4f} "
                    f"({time.time()-t0:.0f}s)")
    record("train", "rsod", "all", "final_kl", kl_total / max(kl_n, 1))
    record("train", "rsod", "all", "seconds", time.time() - t0)
    model.eval()


# ---------------------------------------------------------------- run
log("== stage 0: base eval ==")
if hasattr(model, "peft_config"):
    model.disable_adapter()
exact_match(eval_arith, "base", "arith")
exact_match(eval_code, "base", "code")

log("== stage 1: specialist LoRAs (SFT, frozen after) ==")
sft_train("spec_arith", train_arith, steps=25, batch=8, lr=2e-4,
          tag="spec_arith")
sft_train("spec_code", train_code, steps=25, batch=8, lr=2e-4, tag="spec_code")

log("== routed-teacher system eval (oracle route -> specialist) ==")
model.set_adapter("spec_arith")
exact_match(eval_arith, "routed", "arith")
model.set_adapter("spec_code")
exact_match(eval_code, "routed", "code")

log("== stage 2a: RSOD student ==")
gen_pool = train_arith[:100] + train_code[:100]
kl_distill_train("student_rsod",
                 teachers=["spec_arith", "spec_code"],
                 gen_data=gen_pool, steps=40, batch=4, lr=2e-4)
model.set_adapter("student_rsod")
exact_match(eval_arith, "rsod", "arith")
exact_match(eval_code, "rsod", "code")

log("== stage 2b: MIXSFT baseline (plain SFT on mixed data) ==")
sft_train("student_mix", train_arith + train_code, steps=25, batch=8,
          lr=2e-4, tag="student_mix")
model.set_adapter("student_mix")
exact_match(eval_arith, "mixsft", "arith")
exact_match(eval_code, "mixsft", "code")

# ---------------------------------------------------------------- summary
log("== summary ==")


def get(stage, arm, route, metric):
    for r in METRICS["rows"]:
        if r[0] == stage and r[1] == arm and r[2] == route and r[3] == metric:
            return r[4]
    return None


for arm in ("base", "routed", "rsod", "mixsft"):
    a = get("eval", arm, "arith", "exact_match")
    c = get("eval", arm, "code", "exact_match")
    log(f"{arm:8s} arith={a} code={c} mean={((a or 0)+(c or 0))/2:.3f}")
METRICS["done"] = True
save_metrics(METRICS)
log("DONE")
