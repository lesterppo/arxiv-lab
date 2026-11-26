#!/usr/bin/env python3
"""Tune for Retention -- arXiv 2611.34579 (FALLBACK driver, leg 2)

Fallback after leg 1 (Qwen3.5-9B-Base) DIED on CUDA OOM in train_arm
backward (step 5, arm lr1e-5_b4): the Qwen3.5 Gated Delta Rule
reference fallback materializes heavy attention intermediates, and
train_arm materializes full [B,T,V] fp32 logits. Shrink fixes:
Qwen3-8B (standard attention, proven recipe), MICRO_BATCH 4->2 and
MAX_LEN 512->384 via env (grad accum keeps effective batch), same
data/seeds/arms/evals otherwise.
("Tune for Retention, Not Just Gain: SFT Forgets Far Less than You Think").

Paper claim: catastrophic forgetting in SFT is largely a hyperparameter
artefact, not an inherent property of off-policy SFT. Maximizing new-task
performance induces forgetting, but lower learning rate and smaller batch
size yield similar new-task gains with little to no forgetting; vanilla SFT
selected jointly for gain AND retention is competitive with SDFT (an
on-policy method designed to mitigate forgetting). SDFT is more robust to
hyperparameter choice but still needs a retention-aware selection
criterion.

This run (bounded scale, Colab T4):
  model: env MODEL_ID, default Qwen/Qwen3.5-9B-Base (standing pick: try
    Qwen3.5-9B first; transformers==5.20.2, the Ambient/InstructReplay
    recipe that trained on it). Fallback (env MODEL_ID=Qwen/Qwen3-8B)
    uses the proven transformers>=4.51,<5.0 NF4 QLoRA recipe.
    NF4 QLoRA r=16/alpha=32 (standing rule), bf16, AdamW8bit.
  new task: finance-domain SFT on gbharti/finance-alpaca (fallback
    tatsu-lab/alpaca if the finance set fails to load): 500 train
    examples, 2 epochs. Gain = ROUGE-L F1 of greedy completions vs
    reference outputs on 100 held-out examples.
  retention probes, per arm (greedy, fixed prompts):
    GSM8K test first 80, zero-shot "Question: ...\\nAnswer:",
      last-integer exact match (same probe as the 2026-11-25
      InstructReplay run: base 0.4167 on 60).
    MMLU test first 80, 5-shot (fixed validation examples), letter
      match A-D.
    mean generated length on the GSM8K eval generations.
  arms (fresh base + fresh LoRA per arm, identical code paths):
    base -- probe only, no training
    lr1e-5_b4, lr1e-5_b16, lr5e-5_b8, lr2e-4_b4, lr2e-4_b16
    (LR x effective batch; lr2e-4_b4 is the max-gain arm, lr1e-5_b4
    the retention-friendly arm)
  The paper's discriminating prediction is a gain-retention frontier:
  low-LR/small-batch arms keep most of the gain with near-zero
  retention loss, while the max-gain arm forgets measurably.
  SDFT (the paper's on-policy comparator) is NOT run here -- out of
  scope for one T4 slot; the test is the SFT-internal frontier only.

Stated limits: Qwen3.5-9B-Base, not the paper's models; 500 finance
examples, not the paper's new-task sets; GSM8K-80 + MMLU-80 as the
prior-capability proxy, not the paper's full benchmark suites; single
seed; 2 epochs; ROUGE-L gain proxy, not the paper's task metric.
If no arm forgets (retention flat everywhere), the verdict is
INCONCLUSIVE like the 2026-11-25 InstructReplay run -- the paper's
core claim (a frontier exists) is then untested, not refuted.
Adapters are saved to /content but only metrics + results are
downloaded (adapters die with the VM).

Outputs (written to /content, flushed after every phase):
  /content/retention_metrics.csv  - per-step train rows + eval rows
  /content/retention_results.json - summary + config + eval details
  /content/retention_benchmark.json - splits + few-shot ids
Prints TRAIN_DONE as its last line on completion.
"""
import os, sys, json, math, re, csv, random, time, subprocess, gc

# ---------------- install (self-contained) ----------------
def sh(cmd):
    print(f"$ {cmd}", flush=True)
    r = subprocess.run(cmd, shell=True)
    if r.returncode != 0:
        print(f"INSTALL FAILED rc={r.returncode}: {cmd}", flush=True)
        sys.exit(3)

MODEL_ID = os.environ.get("MODEL_ID", "Qwen/Qwen3.5-9B-Base")
IS_QWEN35 = "3.5" in MODEL_ID
TF_PIN = "transformers==5.20.2" if IS_QWEN35 else "transformers>=4.51,<5.0"
sh(f"pip install -q '{TF_PIN}' bitsandbytes accelerate peft datasets")
sh("pip uninstall -y torchao")  # Colab ships torchao 0.10; peft needs >0.16

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model
import bitsandbytes as bnb_lib
from datasets import load_dataset

# ---------------- config ----------------
SEED = 20261126
N_TRAIN = 500
N_GAIN_EVAL = 100
N_GSM_EVAL = 80
N_MMLU_EVAL = 80
EPOCHS = 2
MICRO_BATCH = int(os.environ.get("MICRO_BATCH", "4"))
LORA_R = 16
LORA_ALPHA = 32
MAX_LEN = int(os.environ.get("MAX_LEN", "512"))
EVAL_BATCH = 8
MAX_NEW_GAIN = 128
MAX_NEW_GSM = 128
MAX_NEW_MMLU = 8
ARMS = [  # (name, lr, effective batch)
    ("lr1e-5_b4", 1e-5, 4),
    ("lr1e-5_b16", 1e-5, 16),
    ("lr5e-5_b8", 5e-5, 8),
    ("lr2e-4_b4", 2e-4, 4),
    ("lr2e-4_b16", 2e-4, 16),
]
TIME_GATE_MIN = 100   # shrink remaining arms to 1 epoch past this
HARD_STOP_MIN = 150   # skip remaining arms past this (record partial)
METRICS_CSV = "/content/retention_metrics.csv"
RESULTS_JSON = "/content/retention_results.json"
BENCH_JSON = "/content/retention_benchmark.json"
T0 = time.time()

torch.manual_seed(SEED); random.seed(SEED)
DEVICE = "cuda"
print(f"CUDA: {torch.cuda.get_device_name(0)} {torch.cuda.get_device_properties(0).total_memory/2**30:.1f} GiB", flush=True)
print(f"model={MODEL_ID} transformers pin={TF_PIN}", flush=True)

# ---------------- PURE HELPERS (stdlib) ----------------
# Everything between the PURE HELPERS markers is stdlib-only and is
# exercised by tests/test_retention_cpu.py (extracted + exec'd, no torch).
import math as _math
import re as _re

_INT_RE = _re.compile(r"-?\d[\d,]*")

def parse_last_int(text):
    """Last integer in the text (commas stripped), or None."""
    ms = _INT_RE.findall(text or "")
    if not ms:
        return None
    try:
        return int(ms[-1].replace(",", ""))
    except ValueError:
        return None

def gsm8k_correct(generated, gold_answer):
    """True if the last integer in the generation equals the gold
    answer's last integer (gold format: '...#### 72')."""
    g = parse_last_int(gold_answer)
    p = parse_last_int(generated)
    return g is not None and p == g

def _lcs_len(a, b):
    """Length of the longest common subsequence of token lists a, b."""
    if not a or not b:
        return 0
    prev = [0] * (len(b) + 1)
    for x in a:
        cur = [0] * (len(b) + 1)
        for j, y in enumerate(b, 1):
            cur[j] = prev[j - 1] + 1 if x == y else max(prev[j], cur[j - 1])
        prev = cur
    return prev[-1]

def rouge_l_f1(generated, reference):
    """Token-level ROUGE-L F1 (whitespace tokens, lowercased)."""
    g = (generated or "").lower().split()
    r = (reference or "").lower().split()
    if not g or not r:
        return 0.0
    lcs = _lcs_len(g, r)
    if lcs == 0:
        return 0.0
    prec = lcs / len(g)
    rec = lcs / len(r)
    return 2 * prec * rec / (prec + rec)

_LETTER_RE = _re.compile(r"\b([A-D])\b")

def parse_mmlu_letter(generated):
    """First A-D letter token in the generation, or None."""
    m = _LETTER_RE.search(generated or "")
    return m.group(1) if m else None

def mmlu_correct(generated, gold_letter):
    p = parse_mmlu_letter(generated)
    return p is not None and p == gold_letter

def format_sft_example(instruction, inp, output):
    """One training text. Input omitted when empty."""
    if inp and inp.strip():
        return ("Instruction: %s\nInput: %s\nResponse: %s"
                % (instruction.strip(), inp.strip(), output.strip()))
    return "Instruction: %s\nResponse: %s" % (instruction.strip(),
                                               output.strip())

def format_sft_prompt(instruction, inp):
    if inp and inp.strip():
        return ("Instruction: %s\nInput: %s\nResponse:"
                % (instruction.strip(), inp.strip()))
    return "Instruction: %s\nResponse:" % instruction.strip()

def format_mmlu_prompt(question, choices, few_shot):
    """5-shot MMLU prompt; few_shot is a list of (q, choices, letter)
    examples rendered identically, then the target question."""
    def one(q, ch, letter=None):
        s = "Question: %s\n" % q
        for lab, txt in zip("ABCD", ch):
            s += "%s. %s\n" % (lab, txt)
        s += "Answer:"
        if letter is not None:
            s += " %s" % letter
        return s
    parts = [one(q, ch, letter) for (q, ch, letter) in few_shot]
    parts.append(one(question, choices))
    return "\n\n".join(parts)
# ---------------- END PURE HELPERS ----------------

# ---------------- data ----------------
rng = random.Random(SEED)
try:
    fin = load_dataset("gbharti/finance-alpaca")
    fin_split = fin["train"]
    src_name = "gbharti/finance-alpaca"
    print("finance-alpaca loaded", flush=True)
except Exception as e:
    print(f"finance-alpaca load failed ({e}); falling back to tatsu-lab/alpaca",
          flush=True)
    fin = load_dataset("tatsu-lab/alpaca")
    fin_split = fin["train"]
    src_name = "tatsu-lab/alpaca"
N_FIN = len(fin_split)
idx = list(range(N_FIN))
rng.shuffle(idx)
train_idx = idx[:N_TRAIN]
gain_idx = idx[N_TRAIN:N_TRAIN + N_GAIN_EVAL]
train_texts = [format_sft_example(fin_split[i]["instruction"],
                                  fin_split[i].get("input", ""),
                                  fin_split[i]["output"]) for i in train_idx]
gain_items = [{"instruction": fin_split[i]["instruction"],
               "input": fin_split[i].get("input", ""),
               "output": fin_split[i]["output"]} for i in gain_idx]
print(f"new-task source={src_name} n={N_FIN} train={len(train_texts)} "
      f"gain_eval={len(gain_items)}", flush=True)

gsm = load_dataset("openai/gsm8k", "main")
gsm_test = [gsm["test"][i] for i in range(N_GSM_EVAL)]
print(f"gsm8k test eval={len(gsm_test)}", flush=True)

mmlu = load_dataset("cais/mmlu", "all")
mmlu_val = mmlu["validation"]
mmlu_test = [mmlu["test"][i] for i in range(N_MMLU_EVAL)]
few_shot = [(mmlu_val[i]["question"], mmlu_val[i]["choices"],
             "ABCD"[mmlu_val[i]["answer"]]) for i in range(5)]
print(f"mmlu test eval={len(mmlu_test)} few-shot={len(few_shot)}", flush=True)

with open(BENCH_JSON, "w") as f:
    json.dump({"seed": SEED, "source": src_name, "n_source": N_FIN,
               "train_idx": train_idx, "gain_idx": gain_idx,
               "gsm_test_idx": list(range(N_GSM_EVAL)),
               "mmlu_test_idx": list(range(N_MMLU_EVAL)),
               "few_shot_val_idx": list(range(5))}, f, indent=1)

# ---------------- metrics ----------------
metrics_f = open(METRICS_CSV, "w", newline="")
mw = csv.writer(metrics_f)
mw.writerow(["phase", "arm", "step", "loss", "grad_norm", "metric", "value", "n"])
metrics_f.flush()

def log_row(*vals):
    mw.writerow(list(vals)); metrics_f.flush()
    print("METRIC", vals, flush=True)

# ---------------- model ----------------
tok = AutoTokenizer.from_pretrained(MODEL_ID)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                         bnb_4bit_use_double_quant=True,
                         bnb_4bit_compute_dtype=torch.bfloat16)

def load_model():
    base = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, quantization_config=bnb, device_map="auto",
        torch_dtype=torch.bfloat16)
    base.config.use_cache = False
    cfg = LoraConfig(r=LORA_R, lora_alpha=LORA_ALPHA, lora_dropout=0.0,
                     target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                     "gate_proj", "up_proj", "down_proj"],
                     task_type="CAUSAL_LM")
    model = get_peft_model(base, cfg)
    model.enable_input_require_grads()
    model.gradient_checkpointing_enable()
    model.train()
    return model

def free_model():
    gc.collect()
    torch.cuda.empty_cache()

@torch.no_grad()
def generate(model, prompts, max_new):
    model.eval()
    model.config.use_cache = True
    tok.padding_side = "left"
    outs = []
    for i in range(0, len(prompts), EVAL_BATCH):
        batch = prompts[i:i + EVAL_BATCH]
        enc = tok(batch, return_tensors="pt", padding=True,
                  truncation=True, max_length=256).to(DEVICE)
        gen = model.generate(**enc, max_new_tokens=max_new, do_sample=False,
                             pad_token_id=tok.pad_token_id)
        for j in range(gen.shape[0]):
            ids = gen[j, enc["input_ids"].shape[1]:]
            outs.append(tok.decode(ids, skip_special_tokens=True))
    model.train()
    model.config.use_cache = False
    torch.cuda.empty_cache()
    return outs

def train_arm(model, arm_name, texts, epochs, lr, eff_batch):
    grad_accum = max(1, eff_batch // MICRO_BATCH)
    opt = bnb_lib.optim.AdamW8bit(
        (p for p in model.parameters() if p.requires_grad), lr=lr)
    tok.padding_side = "right"
    encoded = [tok(t, truncation=True, max_length=MAX_LEN)["input_ids"]
               for t in texts]
    order = list(range(len(encoded)))
    arm_rng = random.Random(SEED)
    step = 0
    losses, grad_norms = [], []
    for epoch in range(epochs):
        arm_rng.shuffle(order)
        for micro_start in range(0, len(order), MICRO_BATCH * grad_accum):
            chunk = order[micro_start:micro_start + MICRO_BATCH * grad_accum]
            n_mb = math.ceil(len(chunk) / MICRO_BATCH)
            opt.zero_grad()
            tot_loss = 0.0
            for mb in range(0, len(chunk), MICRO_BATCH):
                batch_idx = chunk[mb:mb + MICRO_BATCH]
                batch_ids = [encoded[i] for i in batch_idx]
                maxlen = max(len(x) for x in batch_ids)
                pad = tok.pad_token_id
                input_ids = torch.tensor(
                    [x + [pad] * (maxlen - len(x)) for x in batch_ids],
                    device=DEVICE)
                attn = torch.tensor(
                    [[1] * len(x) + [0] * (maxlen - len(x)) for x in batch_ids],
                    device=DEVICE)
                out = model(input_ids=input_ids, attention_mask=attn)
                logits = out.logits[:, :-1, :].float()
                targets = input_ids[:, 1:]
                ce = F.cross_entropy(logits.reshape(-1, logits.size(-1)),
                                     targets.reshape(-1), reduction="none")
                ce = ce.view(targets.shape)
                mask = (targets != pad).float()
                loss = (ce * mask).sum() / mask.sum().clamp(min=1e-9)
                (loss / n_mb).backward()
                tot_loss += loss.item()
            gn = torch.nn.utils.clip_grad_norm_(
                (p for p in model.parameters() if p.requires_grad), 1e9).item()
            grad_norms.append(gn)
            losses.append(tot_loss / n_mb)
            lr_now = lr * min(1.0, (step + 1) / 3.0)
            for g in opt.param_groups:
                g["lr"] = lr_now
            opt.step()
            step += 1
            log_row("train", arm_name, step, round(tot_loss / n_mb, 4),
                    round(gn, 3), "", "", "")
            torch.cuda.empty_cache()
    print(f"TRAIN {arm_name} done: {step} steps, mean loss "
          f"{sum(losses)/len(losses):.4f}", flush=True)
    return {"steps": step, "mean_loss": round(sum(losses) / len(losses), 4),
            "mean_grad_norm": round(sum(grad_norms) / len(grad_norms), 4)}

def evaluate(model, arm_name):
    """New-task gain (ROUGE-L) + retention (GSM8K, MMLU, mean length)."""
    res = {}
    q_gain = [format_sft_prompt(it["instruction"], it["input"])
              for it in gain_items]
    outs_g = generate(model, q_gain, MAX_NEW_GAIN)
    res["gain_rouge_l"] = round(
        sum(rouge_l_f1(o, it["output"]) for o, it in zip(outs_g, gain_items))
        / len(gain_items), 4)
    log_row("eval", arm_name, -1, "", "", "gain_rouge_l",
            res["gain_rouge_l"], len(gain_items))
    q_gsm = ["Question: %s\nAnswer:" % item["question"] for item in gsm_test]
    outs_gsm = generate(model, q_gsm, MAX_NEW_GSM)
    res["gsm8k_acc"] = round(
        sum(gsm8k_correct(o, item["answer"])
            for o, item in zip(outs_gsm, gsm_test)) / len(gsm_test), 4)
    lens = [len(tok(o)["input_ids"]) for o in outs_gsm]
    res["gsm8k_mean_len"] = round(sum(lens) / len(lens), 1)
    log_row("eval", arm_name, -1, "", "", "gsm8k_acc", res["gsm8k_acc"],
            len(gsm_test))
    log_row("eval", arm_name, -1, "", "", "gsm8k_mean_len",
            res["gsm8k_mean_len"], len(gsm_test))
    q_mmlu = [format_mmlu_prompt(item["question"], item["choices"], few_shot)
              for item in mmlu_test]
    outs_m = generate(model, q_mmlu, MAX_NEW_MMLU)
    res["mmlu_acc"] = round(
        sum(mmlu_correct(o, "ABCD"[item["answer"]])
            for o, item in zip(outs_m, mmlu_test)) / len(mmlu_test), 4)
    log_row("eval", arm_name, -1, "", "", "mmlu_acc", res["mmlu_acc"],
            len(mmlu_test))
    print(f"EVAL {arm_name}: {res}", flush=True)
    res["gain_examples"] = outs_g[:2]
    res["gsm_examples"] = outs_gsm[:2]
    res["mmlu_examples"] = outs_m[:3]
    return res

# ---------------- run ----------------
results = {"model": MODEL_ID, "seed": SEED, "paper": "arXiv:2611.34579",
           "arms": {}, "config": {
               "source": src_name, "n_train": N_TRAIN, "epochs": EPOCHS,
               "lora_r": LORA_R, "lora_alpha": LORA_ALPHA,
               "micro_batch": MICRO_BATCH, "n_gain_eval": N_GAIN_EVAL,
               "n_gsm_eval": N_GSM_EVAL, "n_mmlu_eval": N_MMLU_EVAL,
               "arms": [[a, lr, b] for (a, lr, b) in ARMS]}}

print("== base probe ==", flush=True)
model = load_model()
results["arms"]["base"] = {"train": None, "eval": evaluate(model, "base")}
with open(RESULTS_JSON, "w") as f:
    json.dump(results, f, indent=1)
del model; free_model()

for arm_name, lr, eff_batch in ARMS:
    elapsed = (time.time() - T0) / 60
    if elapsed > HARD_STOP_MIN:
        print(f"HARD STOP: elapsed {elapsed:.1f} min > {HARD_STOP_MIN}, "
              f"skipping {arm_name} and later arms", flush=True)
        results["arms"][arm_name] = {"train": None, "eval": None,
                                     "skipped": "hard-stop time gate"}
        continue
    epochs = EPOCHS
    if elapsed > TIME_GATE_MIN:
        epochs = 1
        print(f"TIME-GATE: elapsed {elapsed:.1f} min > {TIME_GATE_MIN}, "
              f"{arm_name} epochs 2 -> 1", flush=True)
    print(f"== arm {arm_name} lr={lr} eff_batch={eff_batch} "
          f"epochs={epochs} ==", flush=True)
    model = load_model()
    tr = train_arm(model, arm_name, train_texts, epochs, lr, eff_batch)
    ev = evaluate(model, arm_name)
    results["arms"][arm_name] = {"train": tr, "eval": ev, "epochs": epochs}
    with open(RESULTS_JSON, "w") as f:
        json.dump(results, f, indent=1)
    del model; free_model()

results["elapsed_min"] = round((time.time() - T0) / 60, 1)
with open(RESULTS_JSON, "w") as f:
    json.dump(results, f, indent=1)
metrics_f.close()
print("RESULTS", json.dumps(results, indent=1), flush=True)
print("TRAIN_DONE", flush=True)
