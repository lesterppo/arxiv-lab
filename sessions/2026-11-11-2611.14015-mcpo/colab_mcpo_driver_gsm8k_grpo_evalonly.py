#!/usr/bin/env python3
"""MCPO vs GRPO — eval-only resume for the GRPO arm, arXiv 2611.14015.
Loads the banked step-12 LoRA checkpoint and runs the SAME seeded 80-item
GSM8K eval as the GRPO arm driver, nothing else.

WHY THIS EXISTS: the redesigned GRPO arm (session mcpo-1111-grpo, 2026-11-11)
finished all 24/24 training steps, then was RECLAIMED during the final eval
(~21:30 HKT). No post-training accuracy, no final adapter. The step-12
adapter checkpoint survived via the mid-run downloader (161,310,709 bytes,
local-only), so the cheap resume is a final eval from step 12 — NOT a fresh
24-step run.

HONEST LIMIT (accepted in the session README): a step-12 eval gives
step-12 post-training accuracy, NOT the lost step-24 final. It slightly
understates the fully trained model. The trade-off is the price of not
re-running 24 steps; the number is still the post-training number the
GRPO-vs-MCPO comparison needs, and the MCPO arm is staged separately.

This driver is eval-only by construction: it never trains, never touches
the train pool, and never overwrites the banked metrics/results from the
GRPO run (its outputs carry _evalonly / _step12 suffixes). The eval set is
the same 80 GSM8K test problems the GRPO driver used (same seed, same
select_pools, same prompt format, same answer extraction), so the number
is directly comparable to the banked base eval 0.2375 (19/80).

The adapter zip is uploaded by the launcher (train --upload
LOCAL:/content/mcpo_gsm8k_adapter_grpo_mid.zip). The driver unzips it if
the dir is missing, asserts the adapter config matches the training
config (r=16, alpha=32), and FAILS LOUDLY if no adapter is found — it
must never silently eval the base model and report it as step-12.

Outputs (written to /content, flushed immediately):
  /content/mcpo_gsm8k_metrics_grpo_evalonly.csv - the single eval row
  /content/mcpo_gsm8k_results_grpo_step12.json  - acc + config + step-12 note
  /content/mcpo_gsm8k_adapter_grpo_step12/ (+ .zip) - the loaded adapter,
      re-saved so the wrapper's --results can grab a clean final artifact
Prints MCPO_GRPO_EVAL_DONE as its last line on completion.
"""
import os, sys, json, math, re, csv, random, time, shutil, subprocess, zipfile

# ---------------- install (self-contained) ----------------
def sh(cmd):
    print(f"$ {cmd}", flush=True)
    r = subprocess.run(cmd, shell=True)
    if r.returncode != 0:
        print(f"INSTALL FAILED rc={r.returncode}: {cmd}", flush=True)
        sys.exit(3)

sh("pip install -q 'transformers>=4.51,<5.0' bitsandbytes accelerate peft pandas pyarrow fsspec")
sh("pip uninstall -y torchao")  # Colab ships torchao 0.10; peft needs >0.16

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
import pandas as pd

# ---------------- config ----------------
MODEL = "Qwen/Qwen3-8B"
ARM_NAME = "grpo_step12"
ARM_MCPO = False
DONE_MARKER = "MCPO_GRPO_EVAL_DONE"
GSM8K_TRAIN_URL = ("https://huggingface.co/datasets/openai/gsm8k/"
                   "resolve/main/main/train-00000-of-00001.parquet")
GSM8K_TEST_URL = ("https://huggingface.co/datasets/openai/gsm8k/"
                  "resolve/main/main/test-00000-of-00001.parquet")
SEED = 20261111
N_TRAIN_POOL = 512
N_EVAL = 80
PROMPTS_PER_STEP = 4
GROUP = 6
STEPS = 24
MAX_NEW = 320
MAX_PROMPT_TOKENS = 512
LR = 1e-5
LORA_R, LORA_ALPHA = 16, 32
T1, T2 = 0.10, 0.50          # MCPO margin thresholds
W_HI, W_MID, W_LO = 1.0, 0.5, 0.1
EVAL_BATCH = 16
ADAPTER_ZIP = "/content/mcpo_gsm8k_adapter_grpo_mid.zip"
ADAPTER_DIR = "/content/mcpo_gsm8k_adapter_grpo_mid"
STEP12_ADAPTER_DIR = "/content/mcpo_gsm8k_adapter_grpo_step12"
METRICS_CSV = "/content/mcpo_gsm8k_metrics_grpo_evalonly.csv"
RESULTS_JSON = "/content/mcpo_gsm8k_results_grpo_step12.json"
BASE_EVAL_ACC_BANKED = 0.2375  # banked 2026-11-11, 19/80 zero-shot (not re-run here)

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
torch.manual_seed(SEED); random.seed(SEED)
DEVICE = "cuda"
print(f"CUDA: {torch.cuda.get_device_name(0)} {torch.cuda.get_device_properties(0).total_memory/2**30:.1f} GiB", flush=True)

# ---------------- pure helpers (stdlib-testable) ----------------
def margin_weights(margins, t1=T1, t2=T2):
    """Map per-token top-2 margins to credit weights (paper's two thresholds)."""
    out = []
    for m in margins:
        if m < t1: out.append(W_HI)
        elif m < t2: out.append(W_MID)
        else: out.append(W_LO)
    return out

def spread_advantage(adv, weights):
    """Scale token weights so the mean token advantage equals adv."""
    mean_w = sum(weights) / len(weights)
    return [adv * w / mean_w for w in weights]

def extract_boxed(text):
    """Return the content of the LAST \\boxed{...} (balanced braces)."""
    if not text: return None
    out = None
    i = 0
    while True:
        j = text.find("\\boxed{", i)
        if j < 0: break
        depth, k = 1, j + len("\\boxed{")
        while k < len(text) and depth > 0:
            if text[k] == "{": depth += 1
            elif text[k] == "}": depth -= 1
            k += 1
        if depth == 0:
            out = text[j + len("\\boxed{"):k - 1].strip()
        i = j + 1
    return out

ANSWER_RE = re.compile(r"Answer:\s*(.+)", re.IGNORECASE)

def extract_answer(text):
    """Extract a model's final answer: \\boxed{...} first, else the last
    'Answer: ...' line (the prompt's own format), else None."""
    b = extract_boxed(text)
    if b is not None:
        return b
    if not text:
        return None
    ms = ANSWER_RE.findall(text)
    if ms:
        return ms[-1].strip().strip("$").strip()
    return None

def _try_float(s):
    """Parse a normalized answer as a float, incl. \\frac{a}{b}."""
    try:
        return float(s)
    except Exception:
        pass
    m = re.fullmatch(r"(?:d?frac)\{([^{}]+)\}\{([^{}]+)\}", s)
    if m:
        try:
            return float(m.group(1)) / float(m.group(2))
        except Exception:
            return None
    return None

def norm_ans(s):
    if s is None: return None
    s = s.strip().replace(",", "").replace("$", "").replace("\\", "")
    s = re.sub(r"\s+", "", s)
    return s

def answers_equal(pred, gold):
    p, g = norm_ans(pred), norm_ans(gold)
    if p is None or g is None: return False
    if p == g: return True
    fp, fg = _try_float(p), _try_float(g)
    if fp is not None and fg is not None:
        return abs(fp - fg) < 1e-6
    return False

GSM8K_INSTRUCTION = ("Solve the following math problem step by step. "
                     "Write your final answer on the last line as 'Answer: $N'.")
GSM8K_GOLD_RE = re.compile(r"####\s*(-?[\d,]+)")

def gsm8k_parse_gold(answer):
    """Gold from a GSM8K solution field: the number after '####'.
    Measured on the real files 2026-11-11: 7473/7473 train, 1319/1319
    test parse to a numeric gold."""
    if not answer: return None
    m = GSM8K_GOLD_RE.search(answer)
    return m.group(1).replace(",", "") if m else None

def gsm8k_build_prompt(question):
    """One fixed instruction + the raw GSM8K question (no CoT appended)."""
    return GSM8K_INSTRUCTION + "\n\n" + question.strip()

def select_pools(train_rows, test_rows, seed=SEED):
    """Deterministic pool selection, IDENTICAL in both arm drivers:
    shuffle GSM8K train with the seed, take the first 512 as the train
    pool; shuffle GSM8K test with the same rng, take the first 80 as
    the held-out eval set; paired training prompts are train_pool[:96].
    Same seed + same code => both arms train on the same prompts in the
    same order and evaluate on the same 80 problems."""
    rng = random.Random(seed)
    tr = list(train_rows); te = list(test_rows)
    rng.shuffle(tr); rng.shuffle(te)
    train_pool = tr[:N_TRAIN_POOL]
    eval_set = te[:N_EVAL]
    paired_prompts = train_pool[:STEPS * PROMPTS_PER_STEP]
    return train_pool, eval_set, paired_prompts

# ---------------- data ----------------
# GSM8K (openai/gsm8k, main): train 7,473 / test 1,319 rows, columns
# ['question', 'answer']; answer = step-by-step solution with
# <<expr=result>> calculator annotations and a final '#### N' line.
# Eval uses the true GSM8K test split (held out), same as the GRPO arm.
print("loading GSM8K ...", flush=True)
try:
    df_train = pd.read_parquet(GSM8K_TRAIN_URL)
    df_test = pd.read_parquet(GSM8K_TEST_URL)
except Exception as e:
    print(f"parquet-direct failed ({e}); falling back to load_dataset", flush=True)
    from datasets import load_dataset
    ds = load_dataset("openai/gsm8k", "main")
    df_train = ds["train"].to_pandas()
    df_test = ds["test"].to_pandas()
print(f"train rows: {len(df_train)} test rows: {len(df_test)}", flush=True)

def _build_rows(df):
    rows = []
    for _, r in df.iterrows():
        gold = gsm8k_parse_gold(str(r["answer"]))
        if gold is None:
            continue
        q = str(r["question"]).strip()
        if not q or len(q) < 10 or len(q) > 2000:
            continue
        rows.append({"prompt": gsm8k_build_prompt(q), "gold": gold})
    return rows

train_rows = _build_rows(df_train)
test_rows = _build_rows(df_test)
print(f"usable rows: train {len(train_rows)} test {len(test_rows)}", flush=True)
train_pool, eval_set, paired_prompts = select_pools(train_rows, test_rows)
print(f"train_pool={len(train_pool)} eval={len(eval_set)} paired_prompts={len(paired_prompts)}", flush=True)
print("sample golds:", [r["gold"] for r in eval_set[:5]], flush=True)

# ---------------- adapter: unzip + load, fail loudly if missing ----------------
if not os.path.isdir(ADAPTER_DIR):
    if not os.path.isfile(ADAPTER_ZIP):
        print(f"ADAPTER MISSING: neither {ADAPTER_DIR} nor {ADAPTER_ZIP} exists. "
              f"Upload the step-12 adapter zip with train --upload first. "
              f"Refusing to eval the base model as step-12.", flush=True)
        sys.exit(4)
    print(f"unzipping {ADAPTER_ZIP} -> {ADAPTER_DIR}", flush=True)
    with zipfile.ZipFile(ADAPTER_ZIP) as z:
        bad = z.testzip()
        if bad is not None:
            print(f"ADAPTER ZIP CORRUPT at {bad}", flush=True)
            sys.exit(4)
        z.extractall(ADAPTER_DIR)
cfg_path = os.path.join(ADAPTER_DIR, "adapter_config.json")
if not os.path.isfile(cfg_path):
    print(f"ADAPTER MISSING: no adapter_config.json in {ADAPTER_DIR}", flush=True)
    sys.exit(4)
cfg = json.load(open(cfg_path))
if cfg.get("r") != LORA_R or cfg.get("lora_alpha") != LORA_ALPHA:
    print(f"ADAPTER MISMATCH: config r={cfg.get('r')} alpha={cfg.get('lora_alpha')} "
          f"!= training config r={LORA_R} alpha={LORA_ALPHA}", flush=True)
    sys.exit(4)
print(f"adapter config OK: r={cfg['r']} alpha={cfg['lora_alpha']} "
      f"target_modules={cfg.get('target_modules')}", flush=True)

# ---------------- model ----------------
tok = AutoTokenizer.from_pretrained(MODEL)
if tok.pad_token is None: tok.pad_token = tok.eos_token
bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                         bnb_4bit_use_double_quant=True,
                         bnb_4bit_compute_dtype=torch.bfloat16)

print("=== loading base policy + step-12 adapter ===", flush=True)
base = AutoModelForCausalLM.from_pretrained(
    MODEL, quantization_config=bnb, device_map="auto",
    torch_dtype=torch.bfloat16)
base.config.use_cache = False
model = PeftModel.from_pretrained(base, ADAPTER_DIR)
if not getattr(model, "peft_config", None):
    print("ADAPTER NOT ATTACHED: PeftModel has no peft_config — refusing to eval as step-12", flush=True)
    sys.exit(4)
print(f"adapter attached: {list(model.peft_config.keys())} "
      f"peak={torch.cuda.max_memory_allocated()/2**30:.2f}GiB", flush=True)

# ---------------- eval (same as the GRPO arm driver) ----------------
@torch.no_grad()
def generate(model, prompts, do_sample, temp=1.0, n_per=1, max_new=MAX_NEW):
    model.eval()
    model.config.use_cache = True
    tok.padding_side = "left"
    out_texts, out_ids = [], []
    for i in range(0, len(prompts), EVAL_BATCH):
        batch = prompts[i:i + EVAL_BATCH]
        enc = tok(batch, return_tensors="pt", padding=True,
                  truncation=True, max_length=MAX_PROMPT_TOKENS).to(DEVICE)
        reps = enc["input_ids"].repeat_interleave(n_per, dim=0)
        att = enc["attention_mask"].repeat_interleave(n_per, dim=0)
        gen = model.generate(input_ids=reps, attention_mask=att,
                             max_new_tokens=max_new, do_sample=do_sample,
                             temperature=temp if do_sample else None,
                             top_p=1.0 if do_sample else None,
                             pad_token_id=tok.pad_token_id)
        for j in range(gen.shape[0]):
            ids = gen[j, reps.shape[1]:]
            out_ids.append(ids.tolist())
            out_texts.append(tok.decode(ids, skip_special_tokens=True))
    return out_texts, out_ids

@torch.no_grad()
def evaluate(model, name):
    texts, _ = generate(model, [e["prompt"] for e in eval_set], do_sample=False)
    correct, lens = 0, []
    for e, t in zip(eval_set, texts):
        pred = extract_answer(t)
        if answers_equal(pred, e["gold"]): correct += 1
        lens.append(len(t.split()))
    acc = correct / len(eval_set)
    print(f"EVAL {name}: acc={acc:.4f} {correct}/{len(eval_set)} mean_len={sum(lens)/len(lens):.1f} tok-words", flush=True)
    return acc, correct, sum(lens) / len(lens)

# ---------------- run ----------------
results = {"model": MODEL, "seed": SEED, "arm": ARM_NAME, "mcpo": ARM_MCPO,
           "data": "gsm8k", "steps": STEPS,
           "prompts_per_step": PROMPTS_PER_STEP, "group": GROUP,
           "max_new": MAX_NEW, "lr": LR, "lora": {"r": LORA_R, "alpha": LORA_ALPHA},
           "mcpo_thresholds": [T1, T2], "mcpo_weights": [W_HI, W_MID, W_LO],
           "train_pool": N_TRAIN_POOL, "eval_n": N_EVAL,
           "checkpoint": "step-12",
           "note": ("step-12 post-training accuracy, NOT the lost step-24 final; "
                    "slightly understates the fully trained model — accepted "
                    "trade-off, see session README"),
           "base_eval_acc_banked": BASE_EVAL_ACC_BANKED,
           "arms": {}}

print("=== eval: GRPO step-12 checkpoint (same 80 GSM8K test problems) ===", flush=True)
acc, correct, mean_len = evaluate(model, ARM_NAME)
results["arms"][ARM_NAME] = {"eval_acc": acc, "correct": correct, "n": N_EVAL,
                             "mean_len": round(mean_len, 1)}

metrics_f = open(METRICS_CSV, "w", newline="")
mw = csv.writer(metrics_f)
mw.writerow(["phase", "arm", "step", "mean_reward", "mean_len", "frac_solved", "n"])
mw.writerow(["eval", ARM_NAME, -1, round(acc, 4), round(mean_len, 1), "", N_EVAL])
metrics_f.flush(); metrics_f.close()

model.save_pretrained(STEP12_ADAPTER_DIR)
shutil.make_archive(STEP12_ADAPTER_DIR, "zip", STEP12_ADAPTER_DIR)
print(f"saved {STEP12_ADAPTER_DIR}.zip", flush=True)
json.dump(results, open(RESULTS_JSON, "w"), indent=1)

print("RESULTS " + json.dumps(results), flush=True)
print(DONE_MARKER, flush=True)
