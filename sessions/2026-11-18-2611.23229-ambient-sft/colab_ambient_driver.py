#!/usr/bin/env python3
# ==============================================================================
# Ambient SFT (arXiv:2611.23229) — Colab T4 training-track driver.
#
# Paper claim: Ambient SFT — L = L_SFT + lam * T^2 * KL(softmax(z_teacher/T)
# || softmax(z_student/T)), teacher = FROZEN base model — beats standard SFT
# by up to 21% average relative improvement (domain adaptation,
# instruction-tuning, reasoning). Arms on identical data/seed:
#   base (untrained, control) | SFT (lam=0) | Ambient T=2 | Ambient T=4
# Task: GSM8K-style math reasoning; metric = held-out exact-match accuracy.
#
# Model: Qwen/Qwen3.5-9B-Base (standing pick: try Qwen3.5-9B first).
#   - 2026-10-30 live T4 probe GREEN: NF4 load + LoRA attach + train step +
#     generate, VRAM 7.89 GB, 29,097,984 trainable params, probe loss 0.776
#     (~/workspace/overmind-banking77/run-20261030-expanded/probe_qwen35.json).
#   - Qwen3.5 needs transformers>=5 (base unrecognised by transformers<5.0);
#     this driver pins transformers==5.20.2, the probe's exact version.
#   - Fallback model (env MODEL_ID=Qwen/Qwen3-8B-Base) uses the proven
#     transformers<5.0 QLoRA recipe instead (see _transformers_pin()).
# Memory: BATCH 2 + label-position-only loss + gradient checkpointing, the
#   exact OOM fix from ~/workspace/overmind-banking77/train_expanded.py
#   (2026-10-30: BATCH 4 + full-vocab float loss OOM'd at first backward).
#   The KL term is computed ONLY on label positions (gathered rows), never
#   on the full [batch, seq, vocab] logits — the teacher forward is an
#   extra no_grad pass, so chunking the KL is unnecessary once gathered.
# Writes /content/ambient_metrics.csv, /content/ambient_summary.json and
# /content/ambient_predictions.jsonl (flushed after EVERY arm, so a
# reclamation still leaves partial results). Prints TRAIN_DONE last.
# Env knobs: MODEL_ID, N_TRAIN (600), N_TEST (200), ARMS (base,sft,t2,t4),
#   LAMBDA (1.0), LR (2e-4), EPOCHS (1), SKIP_BASE_EVAL (0).
# ==============================================================================
import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import subprocess, sys

def _sh(*args):
    print("+", " ".join(args), flush=True)
    r = subprocess.run(list(args), capture_output=True, text=True)
    tail = ((r.stdout or "") + (r.stderr or ""))[-600:]
    print(tail, flush=True)
    return r

# ---- Phase 0: deps -----------------------------------------------------------
MODEL_ID = os.environ.get("MODEL_ID", "Qwen/Qwen3.5-9B-Base")
IS_QWEN35 = "3.5" in MODEL_ID
TF_PIN = "transformers==5.20.2" if IS_QWEN35 else "transformers<5.0"
_sh(sys.executable, "-m", "pip", "uninstall", "-y", "-q", "torchao")
_sh(sys.executable, "-m", "pip", "install", "-q",
    TF_PIN, "accelerate", "datasets", "peft", "bitsandbytes", "safetensors")

import csv, json, math, random, re, time
import numpy as np
import torch
from datasets import load_dataset
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          BitsAndBytesConfig, get_cosine_schedule_with_warmup)
try:
    from transformers import AutoModelForImageTextToText
except Exception:
    AutoModelForImageTextToText = None

N_TRAIN = int(os.environ.get("N_TRAIN", "600"))
N_TEST = int(os.environ.get("N_TEST", "200"))
LAMBDA = float(os.environ.get("LAMBDA", "1.0"))
LR = float(os.environ.get("LR", "2e-4"))
EPOCHS = int(os.environ.get("EPOCHS", "1"))
ARMS = os.environ.get("ARMS", "base,sft,t2,t4").split(",")
SKIP_BASE = os.environ.get("SKIP_BASE_EVAL", "0") == "1"
BATCH, GRAD_ACCUM, MAX_LEN, EVAL_BATCH = 2, 8, 512, 8
MAX_NEW = 224
SEED = 42
METRICS_CSV = "/content/ambient_metrics.csv"
SUMMARY_JSON = "/content/ambient_summary.json"
PREDS_JSONL = "/content/ambient_predictions.jsonl"
TEMP_OF = {"sft": 0.0, "t2": 2.0, "t4": 4.0, "t8": 8.0}

print(f"GPU: {torch.cuda.get_device_name(0)} "
      f"total={torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB, "
      f"torch={torch.__version__}", flush=True)
import transformers
print(f"transformers={transformers.__version__} model={MODEL_ID} "
      f"arms={ARMS} N_TRAIN={N_TRAIN} N_TEST={N_TEST} lam={LAMBDA}",
      flush=True)

# ---- Data: GSM8K -------------------------------------------------------------
def clean_solution(s):
    return re.sub(r"<<[^>]*>>", "", s).strip()

ds = load_dataset("openai/gsm8k", "main")
rng = random.Random(SEED)
train_idx = rng.sample(range(len(ds["train"])), N_TRAIN)
test_idx = rng.sample(range(len(ds["test"])), N_TEST)
train_rows = [ds["train"][i] for i in train_idx]
test_rows = [ds["test"][i] for i in test_idx]
print(f"gsm8k: train_pool={len(ds['train'])} test_pool={len(ds['test'])} "
      f"using train={len(train_rows)} test={len(test_rows)}", flush=True)

PROMPT = ("Solve the following math problem step by step. End your solution "
          "with a line of the form '#### <number>'.\n\nQuestion: {q}\n\nSolution:")

def gold_answer(example):
    return example["answer"].split("####")[-1].strip().replace(",", "")

def extract_answer(text):
    m = re.findall(r"####\s*(-?[\d,]*\.?\d+)", text)
    if m:
        return m[-1].replace(",", "")
    nums = re.findall(r"-?[\d,]*\.?\d+", text)
    return nums[-1].replace(",", "") if nums else None

def answer_correct(pred_text, gold):
    pred = extract_answer(pred_text)
    if pred is None:
        return False
    try:
        return abs(float(pred) - float(gold)) < 1e-6
    except ValueError:
        return pred == gold

# ---- Model -------------------------------------------------------------------
tok = AutoTokenizer.from_pretrained(MODEL_ID)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
print(f"tokenizer pad='{tok.pad_token}' eos='{tok.eos_token}' "
      f"eos_id={tok.eos_token_id}", flush=True)

bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                         bnb_4bit_use_double_quant=True,
                         bnb_4bit_compute_dtype=torch.bfloat16)
loaders = ([("imagetext", AutoModelForImageTextToText), ("causal", AutoModelForCausalLM)]
           if IS_QWEN35 else
           [("causal", AutoModelForCausalLM), ("imagetext", AutoModelForImageTextToText)])
model, loader_name = None, None
errors = {}
for name, cls in loaders:
    if cls is None:
        continue
    try:
        model = cls.from_pretrained(MODEL_ID, quantization_config=bnb,
                                    device_map="cuda", torch_dtype=torch.bfloat16)
        loader_name = name
        break
    except Exception as e:
        errors[name] = str(e)[:300]
assert model is not None, f"model load failed: {errors}"
print(f"loaded via {loader_name}: {type(model).__name__} "
      f"VRAM={torch.cuda.memory_allocated() / 1e9:.2f} GB", flush=True)
model.config.use_cache = False
model.gradient_checkpointing_enable(
    gradient_checkpointing_kwargs={"use_reentrant": False})
try:
    model.enable_input_require_grads()
except Exception as e:
    print(f"enable_input_require_grads skipped: {e}", flush=True)

from peft import LoraConfig, get_peft_model
LORA_CFG = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.0,
                      target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                      "gate_proj", "up_proj", "down_proj"],
                      task_type="CAUSAL_LM")

def encode_train(example):
    prompt_ids = tok(PROMPT.format(q=example["question"]),
                     add_special_tokens=True)["input_ids"]
    sol = clean_solution(example["answer"])
    label_ids = tok(sol, add_special_tokens=False)["input_ids"] + [tok.eos_token_id]
    ids = prompt_ids + label_ids
    labels = [-100] * len(prompt_ids) + label_ids
    return ids[:MAX_LEN], labels[:MAX_LEN]

train_examples = [encode_train(r) for r in train_rows]
dropped = sum(1 for ids, _ in train_examples if len(ids) >= MAX_LEN)
print(f"train examples: {len(train_examples)} max_len="
      f"{max(len(x[0]) for x in train_examples)} dropped_at_cap={dropped}",
      flush=True)

def collate(batch):
    maxlen = max(len(ids) for ids, _ in batch)
    pad_id = tok.pad_token_id
    input_ids, labels, attn = [], [], []
    for ids, lab in batch:
        pad = maxlen - len(ids)
        input_ids.append(ids + [pad_id] * pad)
        labels.append(lab + [-100] * pad)
        attn.append([1] * len(ids) + [0] * pad)
    return (torch.tensor(input_ids, device="cuda"),
            torch.tensor(labels, device="cuda"),
            torch.tensor(attn, device="cuda"))

# ---- Eval --------------------------------------------------------------------
def evaluate(tag, eval_model):
    """Greedy eval on the fixed test subset. Returns dict of metrics."""
    eval_model.eval()
    eval_model.config.use_cache = True
    tok.padding_side = "left"
    n_correct, n_parse_fail = 0, 0
    preds = []
    t0 = time.time()
    with torch.no_grad():
        for start in range(0, len(test_rows), EVAL_BATCH):
            batch = test_rows[start:start + EVAL_BATCH]
            prompts = [PROMPT.format(q=r["question"]) for r in batch]
            enc = tok(prompts, return_tensors="pt", padding=True,
                      add_special_tokens=True).to("cuda")
            out = eval_model.generate(**enc, max_new_tokens=MAX_NEW, do_sample=False,
                                 pad_token_id=tok.pad_token_id)
            gens = tok.batch_decode(out[:, enc["input_ids"].shape[1]:],
                                    skip_special_tokens=True)
            for r, g in zip(batch, gens):
                gold = gold_answer(r)
                ok = answer_correct(g, gold)
                n_correct += int(ok)
                if extract_answer(g) is None:
                    n_parse_fail += 1
                preds.append({"arm": tag, "question": r["question"],
                              "gold": gold, "generation": g, "correct": bool(ok)})
    dt = time.time() - t0
    tok.padding_side = "right"
    eval_model.config.use_cache = False
    eval_model.train()
    acc = n_correct / len(test_rows)
    print(f"[eval:{tag}] acc={acc:.4f} ({n_correct}/{len(test_rows)}) "
          f"parse_fail={n_parse_fail} ({dt:.0f}s)", flush=True)
    with open(PREDS_JSONL, "a") as f:
        for p in preds:
            f.write(json.dumps(p) + "\n")
    return {"arm": tag, "accuracy": acc, "n_correct": n_correct,
            "n_test": len(test_rows), "parse_fail": n_parse_fail,
            "eval_seconds": round(dt, 1)}

# ---- Train one arm -----------------------------------------------------------
def train_arm(tag, temperature):
    """Fresh LoRA; SFT loss (+ Ambient KL term if temperature > 0)."""
    peft_model = get_peft_model(model, LORA_CFG)
    n_trainable = sum(p.numel() for p in peft_model.parameters() if p.requires_grad)
    print(f"[train:{tag}] LoRA attached trainable={n_trainable} "
          f"T={temperature} lam={LAMBDA}", flush=True)
    opt = torch.optim.AdamW([p for p in peft_model.parameters() if p.requires_grad],
                            lr=LR)
    total_opt_steps = (len(train_examples) + BATCH * GRAD_ACCUM - 1) // (BATCH * GRAD_ACCUM)
    sched = get_cosine_schedule_with_warmup(opt, int(total_opt_steps * 0.03),
                                            total_opt_steps)
    order = list(range(len(train_examples)))
    random.Random(SEED).shuffle(order)
    opt.zero_grad()
    step, micro_losses, kl_terms, step_losses = 0, [], [], []
    t0 = time.time()
    for bi in range(0, len(order), BATCH):
        idx = order[bi:bi + BATCH]
        input_ids, labels, attn = collate([train_examples[i] for i in idx])
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = peft_model(input_ids=input_ids,
                                attention_mask=attn).logits
            shift_logits = logits[:, :-1, :]
            shift_labels = labels[:, 1:]
            mask = shift_labels != -100
            sft = torch.nn.functional.cross_entropy(
                shift_logits[mask].float(), shift_labels[mask])
            loss = sft
            kl_val = 0.0
            if temperature and temperature > 0:
                with torch.no_grad(), peft_model.disable_adapter():
                    t_logits = peft_model(input_ids=input_ids,
                                          attention_mask=attn).logits
                s_rows = shift_logits[mask].float()
                t_rows = t_logits[:, :-1, :][mask].float()
                T = temperature
                log_p_t = torch.log_softmax(t_rows / T, dim=-1)
                log_q_s = torch.log_softmax(s_rows / T, dim=-1)
                kl = torch.sum(torch.exp(log_p_t) * (log_p_t - log_q_s),
                               dim=-1).mean()
                kl_val = float(kl.detach())
                loss = sft + LAMBDA * (T ** 2) * kl
                del t_logits, t_rows
            del logits, shift_logits
        (loss / GRAD_ACCUM).backward()
        micro_losses.append(float(sft.detach()))
        kl_terms.append(kl_val)
        del loss
        if (bi // BATCH + 1) % GRAD_ACCUM == 0 or bi + BATCH >= len(order):
            torch.nn.utils.clip_grad_norm_(
                [p for p in peft_model.parameters() if p.requires_grad], 1.0)
            opt.step(); sched.step(); opt.zero_grad(); step += 1
            step_losses.append(float(np.mean(micro_losses[-GRAD_ACCUM:])))
            if step % 10 == 0 or step == total_opt_steps:
                print(f"[train:{tag}] step {step}/{total_opt_steps} "
                      f"loss={step_losses[-1]:.4f} "
                      f"mean_kl={np.mean(kl_terms[-GRAD_ACCUM * 10:]):.5f} "
                      f"({time.time() - t0:.0f}s) "
                      f"VRAM={torch.cuda.memory_allocated() / 1e9:.2f}GB",
                      flush=True)
    opt.zero_grad()
    peak = torch.cuda.max_memory_allocated() / 1e9
    print(f"[train:{tag}] done steps={step} final_loss={step_losses[-1]:.4f} "
          f"mean_kl={np.mean(kl_terms):.5f} peak_VRAM={peak:.2f}GB "
          f"({time.time() - t0:.0f}s)", flush=True)
    res = {"arm": tag, "temperature": temperature, "lambda": LAMBDA,
           "train_steps": step, "final_train_loss": step_losses[-1],
           "mean_sft_loss": float(np.mean(micro_losses)),
           "mean_kl_term": float(np.mean(kl_terms)),
           "peak_vram_gb": round(peak, 2),
           "train_seconds": round(time.time() - t0, 1)}
    # NOTE: adapter stays attached — the caller evals with it, then unloads.
    return res, peft_model

# ---- Run ---------------------------------------------------------------------
results = {"paper": "2611.23229", "model_id": MODEL_ID, "loader": loader_name,
           "transformers": transformers.__version__,
           "torch": torch.__version__,
           "gpu": torch.cuda.get_device_name(0),
           "n_train": N_TRAIN, "n_test": N_TEST, "lambda": LAMBDA,
           "lr": LR, "epochs": EPOCHS, "batch": BATCH,
           "grad_accum": GRAD_ACCUM, "max_len": MAX_LEN, "seed": SEED,
           "arms": []}
fieldnames = ["arm", "temperature", "lambda", "train_steps", "final_train_loss",
              "mean_sft_loss", "mean_kl_term", "peak_vram_gb", "train_seconds",
              "accuracy", "n_correct", "n_test", "parse_fail", "eval_seconds"]
if os.path.exists(METRICS_CSV):
    os.remove(METRICS_CSV)
if os.path.exists(PREDS_JSONL):
    os.remove(PREDS_JSONL)

def flush(row):
    new = not os.path.exists(METRICS_CSV)
    with open(METRICS_CSV, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        if new:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in fieldnames})
    results["arms"].append(row)
    with open(SUMMARY_JSON, "w") as f:
        json.dump(results, f, indent=2)

for arm in ARMS:
    if arm == "base":
        if SKIP_BASE:
            continue
        flush({**evaluate("base", model), "temperature": "", "lambda": ""})
        continue
    T = TEMP_OF[arm]
    row, peft_model = train_arm(arm, T)
    row.update(evaluate(arm, peft_model))
    flush(row)
    peft_model.unload()
    del peft_model
    torch.cuda.empty_cache()

# Relative improvement vs SFT (paper's headline metric)
by_arm = {r["arm"]: r for r in results["arms"]}
if "sft" in by_arm and by_arm["sft"].get("accuracy"):
    sft_acc = by_arm["sft"]["accuracy"]
    for arm in ("t2", "t4", "t8"):
        if arm in by_arm and by_arm[arm].get("accuracy"):
            rel = (by_arm[arm]["accuracy"] - sft_acc) / sft_acc
            by_arm[arm]["rel_improvement_vs_sft"] = round(rel, 4)
            print(f"[result] {arm} acc={by_arm[arm]['accuracy']:.4f} vs SFT "
                  f"{sft_acc:.4f} -> relative {rel:+.1%}", flush=True)
    with open(SUMMARY_JSON, "w") as f:
        json.dump(results, f, indent=2)
    with open(METRICS_CSV, "a", newline="") as f:
        pass

print(json.dumps(results, indent=2), flush=True)
print("TRAIN_DONE", flush=True)
