#!/usr/bin/env python3
"""RLCDR vs answer-key vs softmax-judge rewards on Colab T4 — arXiv 2611.25135
(Calibrated Decision Rewards: Training Language Models with a Judge That
Knows How Sure It Is).

Paper claim: an LLM converted into a decision model answers a typed
question about a response ("is the final answer correct?") with a
probability distribution, read in one forward pass, debiased over option
orders and calibrated with ~100 labels. Training Qwen3-1.7B with GRPO on
MATH, this reference-free reward raises MATH500 accuracy 72.2% -> 76.8%
(+4.65, p < 1e-4), matching training on the answer key (76.9%), while a
random reward gains only +0.65. Its expected calibration error is half
that of a softmax judge on the same 27B backbone (0.074 vs 0.141), with no
sign of reward hacking; without step training it finds the first wrong
step better than a dedicated 7B process reward model (ProcessBench F1
0.772 vs 0.722).

This run (shrunk per the training-track experiment-shrink rule):
  base Qwen/Qwen3.5-9B-Base first (standing pick; fallback Qwen/Qwen3-8B
  via env MODEL), NF4 QLoRA r=16/a=32, bf16, transformers>=4.51,<5.0.
  DAPO-Math-17k train pool 512 / held-out eval 64, paired prompts,
  8 steps/arm, 4 prompts/step, group 4, max_new 256, lr 1e-5.
  Three arms in ONE driver, same code path, fresh LoRA per arm; only the
  reward differs:
    key           — answer-key reward (exact match), the paper's reference
    rlcdr         — calibrated decision reward (debiased over option
                    orders, temperature fitted on 100 labeled samples)
    softmax_judge — raw single-order P(Yes), the paper's control
  The paper's exact 1.7B arm is omitted (experiment shrink; the 9B class
  is the mandatory arm per the 2026-10-27 model rule).
  Design difference from the paper (stated limit): the decision model is
  the policy itself in eval mode (self-judge), not a separate frozen LLM —
  a second 9B copy does not fit the T4 budget alongside training. Judge
  drift is measured: ECE/AUC/high-reward-wrong re-measured after each arm
  on held-out labeled samples.

Outputs (written to /content, flushed after every phase):
  /content/rlcdr_metrics.csv      - per-step train rows + eval/diagnostic rows
  /content/rlcdr_results.json     - summary + config + eval details
  /content/rlcdr_predictions.jsonl - greedy eval texts per arm
  /content/rlcdr_adapter_<arm>/  - LoRA adapters (kept on the VM)
Prints TRAIN_DONE as its last line on completion.
"""
import os, sys, json, math, re, csv, random, time, subprocess

# ---------------- install (self-contained) ----------------
def sh(cmd):
    print(f"$ {cmd}", flush=True)
    r = subprocess.run(cmd, shell=True)
    if r.returncode != 0:
        print(f"INSTALL FAILED rc={r.returncode}: {cmd}", flush=True)
        sys.exit(3)

sh("pip install -q 'transformers>=4.51,<5.0' bitsandbytes accelerate peft pandas pyarrow fsspec")
sh("pip uninstall -y torchao")  # Colab ships torchao 0.10; peft needs >0.16

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model
import pandas as pd

# ---------------- config ----------------
MODEL = os.environ.get("MODEL", "Qwen/Qwen3.5-9B-Base")
DATA_URL = ("https://huggingface.co/datasets/BytedTsinghua-SIA/DAPO-Math-17k/"
            "resolve/main/data/dapo-math-17k.parquet")
SEED = 20261120
N_TRAIN_POOL = 512
N_EVAL = 64
CAL_N_PROBLEMS = 25          # x GROUP samples = 100 labeled calibration samples
CAL_FIT_N = 50               # fit temperature on 50, measure ECE/AUC on 50
PROMPTS_PER_STEP = 4
GROUP = 4
STEPS = 8
MAX_NEW = 256
MAX_PROMPT_TOKENS = 512
LR = 1e-5
LORA_R, LORA_ALPHA = 16, 32
EVAL_BATCH = 16
DECISION_BATCH = 16
DIAG_N_PROBLEMS = 32
METRICS_CSV = "/content/rlcdr_metrics.csv"
RESULTS_JSON = "/content/rlcdr_results.json"
PREDICTIONS_JSONL = "/content/rlcdr_predictions.jsonl"

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
torch.manual_seed(SEED); random.seed(SEED)
DEVICE = "cuda"
print(f"CUDA: {torch.cuda.get_device_name(0)} {torch.cuda.get_device_properties(0).total_memory/2**30:.1f} GiB", flush=True)

# --- PURE HELPERS (stdlib) ---
# Everything between the PURE HELPERS markers is stdlib-only and is
# exercised by tests/test_rlcdr_cpu.py (extracted + exec'd, no torch).
import math as _math
import re as _re

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

ANSWER_RE = _re.compile(r"Answer:\s*(.+)", _re.IGNORECASE)

def extract_answer(text):
    """Extract a model's final answer: \\boxed{...} first, else the last
    'Answer: ...' line, else None."""
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
    try:
        return float(s)
    except Exception:
        pass
    m = _re.fullmatch(r"(?:d?frac)\{([^{}]+)\}\{([^{}]+)\}", s)
    if m:
        try:
            return float(m.group(1)) / float(m.group(2))
        except Exception:
            return None
    return None

def norm_ans(s):
    if s is None: return None
    s = s.strip().replace(",", "").replace("$", "").replace("\\", "")
    s = _re.sub(r"\s+", "", s)
    return s

def answers_equal(pred, gold):
    p, g = norm_ans(pred), norm_ans(gold)
    if p is None or g is None: return False
    if p == g: return True
    fp, fg = _try_float(p), _try_float(g)
    if fp is not None and fg is not None:
        return abs(fp - fg) < 1e-6
    return False

def debias_p_yes(p_yes_order1, p_yes_order2):
    """Average P(Yes) over the two option orders (A=Yes/B=No and
    A=No/B=Yes). A position-biased judge gives p1 + p2 ~= 1; the average
    cancels the bias. Returns None if either input is None."""
    if p_yes_order1 is None or p_yes_order2 is None:
        return None
    return (p_yes_order1 + p_yes_order2) / 2.0

def _sigmoid(x):
    return 1.0 / (1.0 + _math.exp(-x)) if x > -700 else 0.0

def calibrate_p(p, temperature):
    """Temperature-scale a probability: sigmoid(logit(p) / T)."""
    p = min(max(p, 1e-6), 1 - 1e-6)
    logit = _math.log(p / (1 - p))
    return _sigmoid(logit / temperature)

def fit_temperature(ps, labels):
    """Grid-search the temperature minimizing NLL on labeled (p, y) pairs.
    Returns (best_T, best_nll)."""
    best = (1.0, float("inf"))
    t = 0.2
    while t <= 5.001:
        nll = 0.0
        for p, y in zip(ps, labels):
            pc = calibrate_p(p, t)
            nll -= _math.log(pc if y else 1 - pc)
        nll /= max(1, len(ps))
        if nll < best[1]:
            best = (round(t, 2), nll)
        t += 0.1
    return best

def ece_score(ps, labels, n_bins=10):
    """Expected calibration error: |mean p - mean y| per bin, weighted."""
    if not ps:
        return None
    tot, n = 0.0, len(ps)
    for b in range(n_bins):
        lo, hi = b / n_bins, (b + 1) / n_bins
        idx = [i for i, p in enumerate(ps) if lo <= p < hi or (b == n_bins - 1 and p == 1.0)]
        if not idx:
            continue
        tot += len(idx) / n * abs(sum(ps[i] for i in idx) / len(idx)
                                 - sum(labels[i] for i in idx) / len(idx))
    return tot

def auc_score(ps, labels):
    """Rank AUC of scores vs binary labels; None if single-class."""
    pos = [p for p, y in zip(ps, labels) if y == 1]
    neg = [p for p, y in zip(ps, labels) if y == 0]
    if not pos or not neg:
        return None
    wins = ties = 0
    for p in pos:
        for q in neg:
            if p > q: wins += 1
            elif p == q: ties += 1
    return (wins + 0.5 * ties) / (len(pos) * len(neg))

DECISION_TEMPLATE = """You judge whether a proposed solution's final answer is correct.
Problem: What is 7 + 8?
Proposed solution: 7 + 8 = 15. Answer: 15
Is the final answer correct?
A) Yes
B) No
Answer: A

Problem: What is 12 * 6?
Proposed solution: 12 * 6 = 60. Answer: 60
Is the final answer correct?
A) Yes
B) No
Answer: B

Problem: {problem}
Proposed solution: {solution}
Is the final answer correct?
A) Yes
B) No
Answer: """

DECISION_TEMPLATE_SWAPPED = DECISION_TEMPLATE.replace(
    "A) Yes\nB) No", "A) No\nB) Yes")

def decision_prompt(problem, solution, swapped=False):
    """Decision-model prompt (paper: typed question, one forward pass).
    ``swapped`` flips the option order for debiasing."""
    tmpl = DECISION_TEMPLATE_SWAPPED if swapped else DECISION_TEMPLATE
    sol = solution.strip()
    if len(sol) > 1200:
        sol = sol[:1200] + " [...]"
    return tmpl.format(problem=problem.strip(), solution=sol)
# --- END PURE HELPERS ---

# ---------------- data ----------------
print("loading DAPO parquet ...", flush=True)
try:
    df = pd.read_parquet(DATA_URL)
except Exception as e:
    print(f"parquet-direct failed ({e}); falling back to load_dataset", flush=True)
    from datasets import load_dataset
    ds = load_dataset("BytedTsinghua-SIA/DAPO-Math-17k", split="train")
    df = ds.to_pandas()
print(f"columns: {list(df.columns)} rows: {len(df)}", flush=True)
import ast as _ast

def _parse_prompt(v):
    if not isinstance(v, str):
        try:
            items = list(v)
        except Exception:
            return str(v).strip()
        if items and isinstance(items[0], dict):
            return str(items[0].get("content", "")).strip()
        return str(v).strip()
    s = v.strip()
    if s.startswith("["):
        try:
            msgs = _ast.literal_eval(s)
            if isinstance(msgs, list) and msgs and isinstance(msgs[0], dict):
                return str(msgs[0].get("content", "")).strip()
        except Exception:
            pass
    return s

def _parse_gold(v):
    if isinstance(v, dict):
        return str(v.get("ground_truth", "")).strip() or None
    if isinstance(v, str):
        try:
            d = _ast.literal_eval(v.strip())
            if isinstance(d, dict) and d.get("ground_truth"):
                return str(d["ground_truth"]).strip()
        except Exception:
            pass
    return None

sample = df.sample(n=min(20000, len(df)), random_state=SEED)
rows = []
for _, r in sample.iterrows():
    gold = _parse_gold(r["reward_model"])
    if not gold:
        continue
    p = _parse_prompt(r["prompt"])
    if not p or len(p) < 30 or len(p) > 2200:
        continue
    rows.append({"prompt": p, "gold": gold})
print(f"usable rows with ground_truth: {len(rows)} (from {len(sample)} sampled)", flush=True)
rng = random.Random(SEED)
rng.shuffle(rows)
train_pool = rows[:N_TRAIN_POOL]
eval_set = rows[N_TRAIN_POOL:N_TRAIN_POOL + N_EVAL]
cal_problems = rows[N_TRAIN_POOL + N_EVAL:N_TRAIN_POOL + N_EVAL + CAL_N_PROBLEMS]
diag_problems = rows[N_TRAIN_POOL + N_EVAL + CAL_N_PROBLEMS:
                     N_TRAIN_POOL + N_EVAL + CAL_N_PROBLEMS + DIAG_N_PROBLEMS]
paired_prompts = train_pool[:STEPS * PROMPTS_PER_STEP]
print(f"train_pool={len(train_pool)} eval={len(eval_set)} cal={len(cal_problems)} "
      f"diag={len(diag_problems)} paired_prompts={len(paired_prompts)}", flush=True)

# ---------------- metrics ----------------
metrics_f = open(METRICS_CSV, "w", newline="")
mw = csv.writer(metrics_f)
mw.writerow(["phase", "arm", "step", "mean_reward", "mean_len", "frac_solved",
             "n", "ece", "auc", "high_reward_wrong"])
metrics_f.flush()

def log_row(*vals):
    mw.writerow(list(vals)); metrics_f.flush()
    print("METRIC", vals, flush=True)

pred_f = open(PREDICTIONS_JSONL, "w")

def log_pred(arm, prompt, text, gold):
    pred_f.write(json.dumps({"arm": arm, "prompt": prompt[:500],
                             "completion": text, "gold": gold,
                             "pred": extract_answer(text)}) + "\n")
    pred_f.flush()

# ---------------- model ----------------
tok = AutoTokenizer.from_pretrained(MODEL)
if tok.pad_token is None: tok.pad_token = tok.eos_token
bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                         bnb_4bit_use_double_quant=True,
                         bnb_4bit_compute_dtype=torch.bfloat16)

# Option-token ids for the decision model: both "A" and " A" variants per
# option (the prompt ends with "Answer: ", so whether the next token
# carries the leading space depends on tokenization — take the max).
OPT_IDS = {}
for letter in ("A", "B"):
    ids = set()
    for s in (letter, " " + letter):
        ids.update(tok.encode(s, add_special_tokens=False))
    OPT_IDS[letter] = sorted(ids)
print(f"option token ids: {OPT_IDS}", flush=True)

def load_policy():
    base = AutoModelForCausalLM.from_pretrained(
        MODEL, quantization_config=bnb, device_map="auto",
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

def set_gen_mode(model):
    model.eval()
    model.config.use_cache = True

def set_train_mode(model):
    model.train()
    model.config.use_cache = False

@torch.no_grad()
def generate(model, prompts, do_sample, temp=1.0, n_per=1, max_new=MAX_NEW):
    set_gen_mode(model)
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
    set_train_mode(model)
    return out_texts, out_ids

@torch.no_grad()
def decision_p_yes(model, problems, solutions, swapped_flags):
    """One forward pass per (problem, solution, order): P(Yes) from the
    next-token distribution over the A/B option tokens. Returns a list
    of P(Yes) floats (None when the option mass is degenerate)."""
    set_gen_mode(model)
    tok.padding_side = "left"
    prompts = [decision_prompt(p, s, swapped=f)
               for p, s, f in zip(problems, solutions, swapped_flags)]
    out = []
    for i in range(0, len(prompts), DECISION_BATCH):
        batch = prompts[i:i + DECISION_BATCH]
        enc = tok(batch, return_tensors="pt", padding=True,
                  truncation=True, max_length=1024).to(DEVICE)
        logits = model(input_ids=enc["input_ids"],
                       attention_mask=enc["attention_mask"]).logits[:, -1, :].float()
        lp = torch.log_softmax(logits, dim=-1)
        for j, swapped in enumerate(swapped_flags[i:i + DECISION_BATCH]):
            def opt_logp(letter):
                return max(lp[j, tid].item() for tid in OPT_IDS[letter])
            la, lb = opt_logp("A"), opt_logp("B")
            # Renormalize over the two options only.
            m = max(la, lb)
            pa = math.exp(la - m) / (math.exp(la - m) + math.exp(lb - m))
            out.append(pa if not swapped else 1.0 - pa)
    set_train_mode(model)
    return out

@torch.no_grad()
def evaluate(model, name):
    texts, _ = generate(model, [e["prompt"] for e in eval_set], do_sample=False)
    correct, lens = 0, []
    for e, t in zip(eval_set, texts):
        pred = extract_answer(t)
        if answers_equal(pred, e["gold"]): correct += 1
        lens.append(len(t.split()))
        log_pred(name, e["prompt"], t, e["gold"])
    acc = correct / len(eval_set)
    print(f"EVAL {name}: acc={acc:.4f} {correct}/{len(eval_set)} mean_len={sum(lens)/len(lens):.1f} tok-words", flush=True)
    log_row("eval", name, -1, round(acc, 4), round(sum(lens)/len(lens), 1), "", len(eval_set), "", "", "")
    return acc

@torch.no_grad()
def decision_diagnostic(model, name, temperature):
    """Re-measure the decision reward's calibration on held-out labeled
    samples: sample GROUP completions per diag problem, label via key,
    report ECE (raw + calibrated), AUC, and the high-reward-wrong rate
    (paper: no sign of reward hacking)."""
    probs = [d["prompt"] for d in diag_problems]
    texts, _ = generate(model, probs, do_sample=True, temp=1.0, n_per=GROUP)
    ps_raw, ps_cal, labels = [], [], []
    for k, d in enumerate(diag_problems):
        for g in range(GROUP):
            idx = k * GROUP + g
            p1 = decision_p_yes(model, [d["prompt"]], [texts[idx]], [False])[0]
            p2 = decision_p_yes(model, [d["prompt"]], [texts[idx]], [True])[0]
            p = debias_p_yes(p1, p2)
            y = 1 if answers_equal(extract_answer(texts[idx]), d["gold"]) else 0
            ps_raw.append(p); ps_cal.append(calibrate_p(p, temperature)); labels.append(y)
    ece_raw = ece_score(ps_raw, labels)
    ece_cal = ece_score(ps_cal, labels)
    auc = auc_score(ps_cal, labels)
    hrw = (sum(1 for p, y in zip(ps_cal, labels) if p > 0.8 and y == 0)
           / max(1, sum(1 for p in ps_cal if p > 0.8)))
    print(f"DIAG {name}: T={temperature} ece_raw={ece_raw:.3f} ece_cal={ece_cal:.3f} "
          f"auc={auc} high_reward_wrong={hrw:.3f} n={len(labels)}", flush=True)
    log_row("diag", name, -1, "", "", "", len(labels),
            round(ece_cal, 4) if ece_cal is not None else "",
            round(auc, 4) if auc is not None else "", round(hrw, 4))
    return {"ece_raw": ece_raw, "ece_cal": ece_cal, "auc": auc,
            "high_reward_wrong": hrw, "n": len(labels)}

def compute_rewards(model, arm, prompts, golds, texts):
    """Reward per completion for the arm. Returns (rewards, mean option mass)."""
    if arm == "key":
        return [1.0 if answers_equal(extract_answer(t), g) else 0.0
                for t, g in zip(texts, golds)], None
    # Decision-model rewards: debiased P(Yes) over both option orders.
    flat_prompts, flat_texts = [], []
    for k, p in enumerate(prompts):
        for g in range(GROUP):
            flat_prompts.append(p)
            flat_texts.append(texts[k * GROUP + g])
    p1 = decision_p_yes(model, flat_prompts, flat_texts, [False] * len(flat_texts))
    p2 = decision_p_yes(model, flat_prompts, flat_texts, [True] * len(flat_texts))
    if arm == "rlcdr":
        return [calibrate_p(debias_p_yes(a, b), CAL_TEMPERATURE)
                for a, b in zip(p1, p2)], None
    # softmax_judge: raw single-order P(Yes), no debias, no calibration.
    return list(p1), None

def train_arm(model, arm_name):
    global CAL_TEMPERATURE
    opt = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=LR)
    tok.padding_side = "right"
    for step in range(STEPS):
        t0 = time.time()
        prompts = [paired_prompts[step * PROMPTS_PER_STEP + k]["prompt"]
                   for k in range(PROMPTS_PER_STEP)]
        golds = [paired_prompts[step * PROMPTS_PER_STEP + k]["gold"]
                 for k in range(PROMPTS_PER_STEP)]
        texts, gen_ids = generate(model, prompts, do_sample=True, temp=1.0, n_per=GROUP)
        flat_golds = [g for g in golds for _ in range(GROUP)]
        rewards, _ = compute_rewards(model, arm_name, prompts, golds, texts)
        advs = []
        for k in range(PROMPTS_PER_STEP):
            grp = rewards[k * GROUP:(k + 1) * GROUP]
            mu = sum(grp) / GROUP
            var = sum((x - mu) ** 2 for x in grp) / GROUP
            sd = math.sqrt(var)
            advs.extend([(x - mu) / (sd + 1e-8) if sd > 1e-8 else 0.0 for x in grp])
        mean_r = sum(rewards) / len(rewards)
        frac_solved = sum(1 for k in range(PROMPTS_PER_STEP)
                          if any(answers_equal(extract_answer(texts[k * GROUP + g]), golds[k])
                                 for g in range(GROUP))) / PROMPTS_PER_STEP
        mean_len = sum(len(t.split()) for t in texts) / len(texts)
        set_train_mode(model)
        opt.zero_grad()
        total_loss = 0.0
        seqs = []
        for k in range(PROMPTS_PER_STEP):
            for g in range(GROUP):
                idx = k * GROUP + g
                p_ids = tok(prompts[k], truncation=True, max_length=MAX_PROMPT_TOKENS)["input_ids"]
                c_ids = gen_ids[idx]
                seqs.append((p_ids, c_ids, advs[idx]))
        micro = 4
        for mb in range(0, len(seqs), micro):
            chunk = seqs[mb:mb + micro]
            maxlen = max(len(p) + len(c) for p, c, _ in chunk)
            input_ids, labels, adv_tok = [], [], []
            for p_ids, c_ids, adv in chunk:
                ids = p_ids + c_ids
                lab = [-100] * len(p_ids) + c_ids
                pad = maxlen - len(ids)
                input_ids.append(ids + [tok.pad_token_id] * pad)
                labels.append(lab + [-100] * pad)
                adv_tok.append((adv, len(c_ids)))
            ii = torch.tensor(input_ids, device=DEVICE)
            ll = torch.tensor(labels, device=DEVICE)
            att = (ii != tok.pad_token_id).long()
            out = model(input_ids=ii, attention_mask=att)
            logits = out.logits[:, :-1, :]
            tgt = ll[:, 1:]
            mask = (tgt != -100)
            B_, Tm1, V_ = logits.shape
            tok_logp = torch.zeros((B_, Tm1), device=DEVICE, dtype=torch.float32)
            CH = 64
            for ps in range(0, Tm1, CH):
                pe = min(ps + CH, Tm1)
                lp = torch.log_softmax(logits[:, ps:pe, :].float(), dim=-1)
                tb = tgt[:, ps:pe]
                tok_logp[:, ps:pe] = torch.gather(
                    lp, 2, tb.clamp(min=0).unsqueeze(-1)).squeeze(-1)
                del lp
            loss = 0.0
            for b, (adv, clen) in enumerate(adv_tok):
                pos = torch.nonzero(mask[b]).squeeze(-1)
                lp = tok_logp[b, pos]
                loss = loss - (adv * lp).sum()
            loss = loss / max(1, sum(mask[b].sum().item() for b in range(len(chunk))))
            (loss / (len(seqs) / micro)).backward()
            total_loss += loss.item()
        torch.nn.utils.clip_grad_norm_((p for p in model.parameters() if p.requires_grad), 1.0)
        opt.step()
        dt = time.time() - t0
        print(f"[{arm_name}] step {step+1}/{STEPS} reward={mean_r:.3f} solved={frac_solved:.2f} "
              f"len={mean_len:.0f} loss={total_loss:.3f} {dt:.0f}s "
              f"peak={torch.cuda.max_memory_allocated()/2**30:.2f}GiB", flush=True)
        log_row("train", arm_name, step + 1, round(mean_r, 4), round(mean_len, 1),
                round(frac_solved, 3), PROMPTS_PER_STEP * GROUP, "", "", "")
    return model

# ---------------- run ----------------
results = {"model": MODEL, "seed": SEED, "steps_per_arm": STEPS,
           "prompts_per_step": PROMPTS_PER_STEP, "group": GROUP,
           "max_new": MAX_NEW, "lr": LR, "lora": {"r": LORA_R, "alpha": LORA_ALPHA},
           "train_pool": N_TRAIN_POOL, "eval_n": N_EVAL,
           "cal_problems": CAL_N_PROBLEMS, "arms": {}}

print("=== loading base policy ===", flush=True)
model = load_policy()
print(f"model loaded peak={torch.cuda.max_memory_allocated()/2**30:.2f}GiB", flush=True)
results["arms"]["base"] = {"eval_acc": evaluate(model, "base")}

# --- calibration phase (paper: ~100 labels) ---
print("=== calibration: sampling 100 labeled completions ===", flush=True)
cal_texts, _ = generate(model, [d["prompt"] for d in cal_problems],
                        do_sample=True, temp=1.0, n_per=GROUP)
cal_ps, cal_labels = [], []
for k, d in enumerate(cal_problems):
    for g in range(GROUP):
        idx = k * GROUP + g
        p1 = decision_p_yes(model, [d["prompt"]], [cal_texts[idx]], [False])[0]
        p2 = decision_p_yes(model, [d["prompt"]], [cal_texts[idx]], [True])[0]
        p = debias_p_yes(p1, p2)
        y = 1 if answers_equal(extract_answer(cal_texts[idx]), d["gold"]) else 0
        cal_ps.append(p); cal_labels.append(y)
fit_ps, fit_y = cal_ps[:CAL_FIT_N], cal_labels[:CAL_FIT_N]
test_ps, test_y = cal_ps[CAL_FIT_N:], cal_labels[CAL_FIT_N:]
CAL_TEMPERATURE, fit_nll = fit_temperature(fit_ps, fit_y)
ece_raw = ece_score(test_ps, test_y)
ece_cal = ece_score([calibrate_p(p, CAL_TEMPERATURE) for p in test_ps], test_y)
auc = auc_score([calibrate_p(p, CAL_TEMPERATURE) for p in test_ps], test_y)
print(f"CALIBRATION T={CAL_TEMPERATURE} fit_nll={fit_nll:.3f} "
      f"test ece_raw={ece_raw:.3f} ece_cal={ece_cal:.3f} auc={auc} "
      f"base_rate={sum(test_y)/len(test_y):.2f}", flush=True)
results["calibration"] = {"temperature": CAL_TEMPERATURE, "fit_nll": round(fit_nll, 4),
                          "ece_raw": ece_raw, "ece_calibrated": ece_cal, "auc": auc,
                          "n_fit": CAL_FIT_N, "n_test": len(test_y)}
results["arms"]["base"]["diag"] = decision_diagnostic(model, "base", CAL_TEMPERATURE)

for arm in ("key", "rlcdr", "softmax_judge"):
    print(f"=== arm: {arm} ===", flush=True)
    if arm != "key":
        del model; torch.cuda.empty_cache()
        model = load_policy()
    train_arm(model, arm)
    results["arms"][arm] = {"eval_acc": evaluate(model, arm)}
    results["arms"][arm]["diag"] = decision_diagnostic(model, arm, CAL_TEMPERATURE)
    model.save_pretrained(f"/content/rlcdr_adapter_{arm}")
    print(f"saved /content/rlcdr_adapter_{arm}", flush=True)

metrics_f.close()
pred_f.close()
json.dump(results, open(RESULTS_JSON, "w"), indent=1)
print("RESULTS " + json.dumps(results), flush=True)
print("TRAIN_DONE", flush=True)
