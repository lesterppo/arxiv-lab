#!/usr/bin/env python3
"""MCPO vs GRPO on Colab T4 — arXiv 2611.14015 (Margin-Guided Token Credit).

Paper claim: in RLVR/GRPO every token of a solution gets the same credit.
MCPO instead places credit where the model hesitated: top-2 margin
M = p1 - p2 of the next-token distribution, mapped to a score with two
fixed thresholds, spread so each solution's advantage keeps its mean.
Paper: MiniCPM5-2B-Base + LoRA, 200 steps on 2,048 DAPO-Math problems;
+6.55 macro avg@32 over GRPO; Qwen3-4B replication +4.75.

This run (shrunk per the training-track experiment-shrink rule):
  base Qwen/Qwen3-8B (current-gen 8B class, NF4 QLoRA r=16/a=32, bf16),
  DAPO-Math-17k train pool 512 / held-out eval 80, paired prompts,
  24 steps/arm, 4 prompts/step, group 6, max_new 320, lr 1e-5.
  Arms run sequentially in ONE driver with identical code paths; the only
  difference is the token-credit weighting (uniform vs margin-guided).
  Base eval runs first so both arms share the same baseline.
  Data: the parquet's real schema is prompt=[{content}] /
  reward_model={ground_truth}; the prompt already specifies the
  "Answer: $Answer" format, so the reward parses \\boxed{} OR the
  Answer: line. (v1 of this driver wrongly extracted \\boxed{} golds
  from the prompt text and was killed mid-base-eval; v2 fixes it.)
  FALLBACK DRIVER (automatic fallback rule, 2026-11-05): the Qwen3-8B
  run (mcpo-1111c) died of CUDA OOM at GRPO step 1, line 366
  (log_softmax over the full [B, T, V] fp32 logits, V=151,936).
  This copy switches to Qwen/Qwen3-4B (next smaller proven model; also
  the paper's own replication model) AND fixes the failure cause:
  position-chunked logprob/margin computation (never materializes
  [B, T, V] in fp32), micro-batch 6 -> 4, and expandable_segments.

The margin mapping instantiates the abstract's "two fixed thresholds":
  M < 0.10 -> w=1.0 (hesitant, full credit)
  0.10 <= M < 0.50 -> w=0.5
  M >= 0.50 -> w=0.1 (confident, little credit)
Token advantages are renormalised per completion so their mean equals the
group-relative advantage A (paper: "the mean stays unchanged").
The abstract does not publish the exact thresholds/weights; this is
recorded as a stated limit in the session README.

Outputs (written to /content, flushed after every phase):
  /content/mcpo_metrics.csv   - per-step train rows + eval rows
  /content/mcpo_results.json  - summary + config + eval details
  /content/mcpo_adapter_grpo/ and /content/mcpo_adapter_mcpo/ - LoRA adapters
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
MODEL = "Qwen/Qwen3-4B"
DATA_URL = ("https://huggingface.co/datasets/BytedTsinghua-SIA/DAPO-Math-17k/"
            "resolve/main/data/dapo-math-17k.parquet")
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
METRICS_CSV = "/content/mcpo_metrics.csv"
RESULTS_JSON = "/content/mcpo_results.json"

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
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
    'Answer: ...' line (the DAPO prompt's own format), else None."""
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

# ---------------- data ----------------
# Real schema (verified 2026-11-11 from the live run): columns
# ['data_source', 'prompt', 'ability', 'reward_model', 'extra_info'],
# 1,791,700 rows. prompt = [{'content': '<DAPO instruction + problem>'}]
# (list of message dicts, or its string repr); reward_model =
# {'ground_truth': '<answer>', 'style': 'rule-lighteval/MATH_v2'}.
# The prompt already specifies the answer format ("Answer: $Answer"), so
# no boxed suffix is appended; the reward parses \\boxed{} OR the
# Answer: line (see extract_answer).
print("loading DAPO parquet ...", flush=True)
try:
    df = pd.read_parquet(DATA_URL)
except Exception as e:
    print(f"parquet-direct failed ({e}); falling back to load_dataset", flush=True)
    from datasets import load_dataset
    ds = load_dataset("BytedTsinghua-SIA/DAPO-Math-17k", split="train")
    df = ds.to_pandas()
print(f"columns: {list(df.columns)} rows: {len(df)}", flush=True)
print("sample row:", {c: str(df.iloc[0][c])[:200] for c in df.columns}, flush=True)
import ast as _ast

def _parse_prompt(v):
    # Cells arrive as a list/tuple/ndarray of message dicts, a dict, or
    # the string repr of one of those (pyarrow stores them as object arrays).
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
print("sample golds:", [r["gold"] for r in rows[:5]], flush=True)
rng = random.Random(SEED)
rng.shuffle(rows)
train_pool = rows[:N_TRAIN_POOL]
eval_set = rows[N_TRAIN_POOL:N_TRAIN_POOL + N_EVAL]
paired_prompts = train_pool[:STEPS * PROMPTS_PER_STEP]
print(f"train_pool={len(train_pool)} eval={len(eval_set)} paired_prompts={len(paired_prompts)}", flush=True)

# ---------------- metrics ----------------
metrics_f = open(METRICS_CSV, "w", newline="")
mw = csv.writer(metrics_f)
mw.writerow(["phase", "arm", "step", "mean_reward", "mean_len", "frac_solved", "n"])
metrics_f.flush()

def log_row(*vals):
    mw.writerow(list(vals)); metrics_f.flush()
    print("METRIC", vals, flush=True)

# ---------------- model ----------------
tok = AutoTokenizer.from_pretrained(MODEL)
if tok.pad_token is None: tok.pad_token = tok.eos_token
bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                         bnb_4bit_use_double_quant=True,
                         bnb_4bit_compute_dtype=torch.bfloat16)

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
    model.enable_input_require_grads()  # grad-ckpt + peft fix (auditmerge lesson)
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
def evaluate(model, name):
    texts, _ = generate(model, [e["prompt"] for e in eval_set], do_sample=False)
    correct, lens = 0, []
    for e, t in zip(eval_set, texts):
        pred = extract_answer(t)
        if answers_equal(pred, e["gold"]): correct += 1
        lens.append(len(t.split()))
    acc = correct / len(eval_set)
    print(f"EVAL {name}: acc={acc:.4f} {correct}/{len(eval_set)} mean_len={sum(lens)/len(lens):.1f} tok-words", flush=True)
    log_row("eval", name, -1, round(acc, 4), round(sum(lens)/len(lens), 1), "", len(eval_set))
    return acc

def train_arm(model, arm_name, mcpo):
    opt = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=LR)
    tok.padding_side = "right"
    for step in range(STEPS):
        t0 = time.time()
        prompts = [paired_prompts[step * PROMPTS_PER_STEP + k]["prompt"]
                   for k in range(PROMPTS_PER_STEP)]
        golds = [paired_prompts[step * PROMPTS_PER_STEP + k]["gold"]
                 for k in range(PROMPTS_PER_STEP)]
        texts, gen_ids = generate(model, prompts, do_sample=True, temp=1.0, n_per=GROUP)
        # rewards per (prompt, sample)
        rewards = []
        for k in range(PROMPTS_PER_STEP):
            for g in range(GROUP):
                idx = k * GROUP + g
                rewards.append(1.0 if answers_equal(extract_answer(texts[idx]), golds[k]) else 0.0)
        # group-relative advantages
        advs = []
        for k in range(PROMPTS_PER_STEP):
            grp = rewards[k * GROUP:(k + 1) * GROUP]
            mu = sum(grp) / GROUP
            var = sum((x - mu) ** 2 for x in grp) / GROUP
            sd = math.sqrt(var)
            advs.extend([(x - mu) / (sd + 1e-8) if sd > 1e-8 else 0.0 for x in grp])
        mean_r = sum(rewards) / len(rewards)
        frac_solved = sum(1 for k in range(PROMPTS_PER_STEP)
                          if any(rewards[k * GROUP + g] > 0 for g in range(GROUP))) / PROMPTS_PER_STEP
        mean_len = sum(len(t.split()) for t in texts) / len(texts)
        # ---- loss forward: one pass gives logprobs AND (for MCPO) margins ----
        set_train_mode(model)
        opt.zero_grad()
        total_loss, total_tok = 0.0, 0
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
            logits = out.logits[:, :-1, :]  # [B, T-1, V] bf16
            tgt = ll[:, 1:]
            mask = (tgt != -100)
            # Chunked logprobs/margins: the 8B run OOM'd materializing
            # log_softmax over the full [B, T, V] fp32 logits (V=151,936).
            # Process 64 positions at a time; margins come from the same
            # log_softmax block (top-2 in prob space = exp of top-2 logp).
            B_, Tm1, V_ = logits.shape
            tok_logp = torch.zeros((B_, Tm1), device=DEVICE, dtype=torch.float32)
            margin = torch.zeros((B_, Tm1), device=DEVICE, dtype=torch.float32) if mcpo else None
            CH = 64
            for ps in range(0, Tm1, CH):
                pe = min(ps + CH, Tm1)
                lp = torch.log_softmax(logits[:, ps:pe, :].float(), dim=-1)
                tb = tgt[:, ps:pe]
                tok_logp[:, ps:pe] = torch.gather(
                    lp, 2, tb.clamp(min=0).unsqueeze(-1)).squeeze(-1)
                if mcpo:
                    top2 = torch.topk(lp, 2, dim=-1).values
                    margin[:, ps:pe] = (torch.exp(top2[:, :, 0])
                                         - torch.exp(top2[:, :, 1]))
                del lp
            loss = 0.0
            for b, (adv, clen) in enumerate(adv_tok):
                pos = torch.nonzero(mask[b]).squeeze(-1)  # positions of completion tokens in tgt
                lp = tok_logp[b, pos]
                if mcpo:
                    mg = margin[b, pos].tolist()
                    w = margin_weights(mg)
                    a = spread_advantage(adv, w)
                    a_t = torch.tensor(a, device=DEVICE, dtype=lp.dtype)
                else:
                    a_t = torch.full_like(lp, adv)
                loss = loss - (a_t * lp).sum()
            loss = loss / max(1, sum(mask[b].sum().item() for b in range(len(chunk))))
            (loss / (len(seqs) / micro)).backward()
            total_loss += loss.item(); total_tok += int(mask.sum().item())
        torch.nn.utils.clip_grad_norm_((p for p in model.parameters() if p.requires_grad), 1.0)
        opt.step()
        dt = time.time() - t0
        print(f"[{arm_name}] step {step+1}/{STEPS} reward={mean_r:.3f} solved={frac_solved:.2f} "
              f"len={mean_len:.0f} loss={total_loss:.3f} {dt:.0f}s "
              f"peak={torch.cuda.max_memory_allocated()/2**30:.2f}GiB", flush=True)
        log_row("train", arm_name, step + 1, round(mean_r, 4), round(mean_len, 1),
                round(frac_solved, 3), PROMPTS_PER_STEP * GROUP)
    return model

# ---------------- run ----------------
results = {"model": MODEL, "seed": SEED, "steps_per_arm": STEPS,
           "prompts_per_step": PROMPTS_PER_STEP, "group": GROUP,
           "max_new": MAX_NEW, "lr": LR, "lora": {"r": LORA_R, "alpha": LORA_ALPHA},
           "mcpo_thresholds": [T1, T2], "mcpo_weights": [W_HI, W_MID, W_LO],
           "train_pool": N_TRAIN_POOL, "eval_n": N_EVAL, "arms": {}}

print("=== loading base policy ===", flush=True)
model = load_policy()
print(f"model loaded peak={torch.cuda.max_memory_allocated()/2**30:.2f}GiB", flush=True)
results["arms"]["base"] = {"eval_acc": evaluate(model, "base")}

print("=== arm 1: GRPO (uniform credit) ===", flush=True)
train_arm(model, "grpo", mcpo=False)
results["arms"]["grpo"] = {"eval_acc": evaluate(model, "grpo")}
model.save_pretrained("/content/mcpo_adapter_grpo")
print("saved /content/mcpo_adapter_grpo", flush=True)
del model; torch.cuda.empty_cache()

print("=== arm 2: MCPO (margin-guided credit) ===", flush=True)
model = load_policy()
train_arm(model, "mcpo", mcpo=True)
results["arms"]["mcpo"] = {"eval_acc": evaluate(model, "mcpo")}
model.save_pretrained("/content/mcpo_adapter_mcpo")
print("saved /content/mcpo_adapter_mcpo", flush=True)

metrics_f.close()
json.dump(results, open(RESULTS_JSON, "w"), indent=1)
print("RESULTS " + json.dumps(results), flush=True)
print("TRAIN_DONE", flush=True)
