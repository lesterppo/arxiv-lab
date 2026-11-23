#!/usr/bin/env python3
"""FALLBACK LEG 2 (automatic, standing rule 2026-11-05) -- Structure Over Scale vs LoRA rank collapse on Colab T4 -- arXiv 2611.27148
("Structure Over Scale: Rethinking Adaptation for Reinforcement Learning
with Verifiable Rewards").

Paper claim: under a common cold-start GRPO protocol, LoRA exhibits rank
collapse in RLVR -- performance stays broadly stable at low rank, then
fails abruptly as rank increases. On DeepMath-Hard, accuracy drops from
73.1% at r=64 to 4.7% at r=128. In contrast, DoRA and QuanTA remain
stable over a wider region and are competitive with full fine-tuning
while updating <0.6% of parameters. Spectral analysis: collapsed LoRA
concentrates its update energy into relatively few effective directions,
while structured adapters produce substantially more distributed updates.
The authors present a "Model Maturity Hypothesis" (adaptation geometry
depends on pre-training properties) as falsifiable, not causal.

LEG 1 (Qwen/Qwen3-8B, DAPO-Math-17k, same driver shape) DIED on CUDA
OOM in the backward pass of r64 step 8 (13.33 GiB allocated, 588 MiB
more needed; metrics in structure_metrics_leg1_qwen38.csv). Worse,
leg 1 also showed a floor effect: base eval 0/80 and 7 GRPO steps
with mean_reward 0.0 / frac_solved 0.0 / grad_norm 0.0 -- the base
model solves none of the sampled DAPO problems, so GRPO has no
gradient signal at all. The same floor was measured in the MCPO
session (2026-11-11, recovered_metrics.csv: DAPO base eval 0.0125
on Qwen3-8B, 0.0 on Qwen3-4B, train rewards 0.0), and MCPO fixed it
by redesigning onto GSM8K (base 0.2375, train rewards 0.04-0.42).
This fallback therefore applies BOTH shrink fixes for the failure
causes, per the standing rule:
  model: Qwen/Qwen3-4B (next smaller proven model; MCPO 4B driver
    ran on this recipe) -- the paper tested 8B models, so a 4B
    result is a bounded-scale probe of the same geometric claim,
    stated as a limit below.
  data: GSM8K train pool 512 / test eval 80 (MCPO GSM8K redesign,
    proven signal) instead of DAPO-Math-17k.
  memory: PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,
    micro-batch 4 -> 2, empty_cache() after every generate phase.
All other protocol (16 steps/arm, 4 prompts/step, group 4,
max_new 256, lr 1e-5, alpha=2r, AdamW8bit, arms r64/r128 +
time-gated DoRA r128, spectrum diagnostics) is unchanged.

This run (shrunk per the training-track experiment-shrink rule):
  base Qwen/Qwen3-8B (current-gen 8B class, NF4 QLoRA, bf16,
  transformers>=4.51,<5.0 -- the proven recipe). Qwen3.5-9B-Base is NOT
  used: the 2026-11-21 Budget Internalization leg proved it loads but
  its Gated DeltaNet layers fall back to reference PyTorch kernels
  (no causal_conv1d / flash-linear-attention), making generation so
  slow that 400 greedy traces did not finish in 50+ min -- it does not
  feasibly fit this generation-heavy GRPO test in the reclaim window.
  DAPO-Math-17k train pool 512 / held-out eval 80, paired prompts,
  16 steps/arm, 4 prompts/step, group 4, max_new 256, lr 1e-5,
  LoRA alpha = 2r (same alpha/r ratio as the proven MCPO recipe).
  Arms run sequentially in ONE driver with identical code paths; only
  the adapter differs (fresh base + fresh LoRA per arm):
    r64   -- LoRA r=64,  the paper's stable point
    r128  -- LoRA r=128, the paper's collapse point
    dora128 (time-gated) -- DoRA r=128, the paper's stable structured
            adapter; runs only if <2h elapsed after the r128 eval
  Per-arm diagnostics: greedy eval accuracy before/after, per-step
  reward / frac-solved / grad-norm, and the spectral signature of the
  update dW = B@A for q_proj/v_proj in 5 layers -- effective rank
  (participation ratio) and top-1 energy share, computed exactly from
  r x r thin-SVD factors (nonzero singular values of BA equal those
  of diag(S_B) Vh_B U_A diag(S_A)). Collapse predicts: r128 eval
  accuracy craters vs r64, AND r128's update concentrates (low
  effective rank, high top-1 share) while r64 / DoRA stay distributed.

Stated limits: this leg runs Qwen3-4B (fallback model) and GSM8K
(train/eval) instead of the paper's 8B models on DeepMath-Hard --
the claim is probed at bounded scale (4B, grade-school problems),
not at the paper's exact scale; 16 steps/arm is a bounded-scale
probe of the paper's full protocol; only q_proj/v_proj spectra are
reported (the paper analyses the full update); adapters are saved
to /content but only metrics + results are downloaded (adapters
die with the VM).

Outputs (written to /content, flushed after every phase):
  /content/structure_metrics.csv  - per-step train rows + eval rows +
                                    spectral rows
  /content/structure_results.json - summary + config + eval details
  /content/structure_adapter_<arm>/ - LoRA adapters (kept on the VM)
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

sh("pip install -q 'transformers>=4.51,<5.0' bitsandbytes accelerate peft pandas pyarrow fsspec datasets")
sh("pip uninstall -y torchao")  # Colab ships torchao 0.10; peft needs >0.16

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model
import pandas as pd

# ---------------- config ----------------
MODEL = "Qwen/Qwen3-4B"
DATA_URL = ("https://huggingface.co/datasets/BytedTsinghua-SIA/DAPO-Math-17k/"
            "resolve/main/data/dapo-math-17k.parquet")
SEED = 20261123
N_TRAIN_POOL = 512
N_EVAL = 80
PROMPTS_PER_STEP = 4
GROUP = 4
STEPS = 16
MAX_NEW = 256
MAX_PROMPT_TOKENS = 512
LR = 1e-5
EVAL_BATCH = 16
DORA_TIME_BUDGET_S = 2 * 3600   # run the DoRA arm only if <2h elapsed
METRICS_CSV = "/content/structure_fb_metrics.csv"
RESULTS_JSON = "/content/structure_fb_results.json"
T0 = time.time()

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
torch.manual_seed(SEED); random.seed(SEED)
DEVICE = "cuda"
print(f"CUDA: {torch.cuda.get_device_name(0)} {torch.cuda.get_device_properties(0).total_memory/2**30:.1f} GiB", flush=True)

# ---------------- PURE HELPERS (stdlib) ----------------
# Everything between the PURE HELPERS markers is stdlib-only and is
# exercised by tests/test_structure_cpu.py (extracted + exec'd, no torch).
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
    """Parse a normalized answer as a float, incl. \\frac{a}{b} and a/b."""
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
    m = _re.fullmatch(r"(-?\d+(?:\.\d+)?)/(-?\d+(?:\.\d+)?)", s)
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

def spectral_stats(singular_values):
    """Effective rank (participation ratio) and top-1 energy share of
    an update from its singular values. Scale-invariant, so comparable
    across ranks and alpha scalings."""
    s = [float(x) for x in singular_values if float(x) > 0]
    if not s:
        return 0.0, 0.0
    e = [x * x for x in s]
    tot = sum(e)
    eff_rank = (sum(s) ** 2) / tot if tot > 0 else 0.0
    top1 = max(e) / tot if tot > 0 else 0.0
    return eff_rank, top1

def group_advantages(rewards, group):
    """Group-relative advantages (GRPO): (r - mean) / std per prompt
    group; zero when the group has no variance."""
    advs = []
    for k in range(len(rewards) // group):
        grp = rewards[k * group:(k + 1) * group]
        mu = sum(grp) / group
        var = sum((x - mu) ** 2 for x in grp) / group
        sd = _math.sqrt(var)
        advs.extend([(x - mu) / (sd + 1e-8) if sd > 1e-8 else 0.0
                     for x in grp])
    return advs
# ---------------- END PURE HELPERS ----------------

# ---------------- data ----------------
# GSM8K (MCPO redesign 2026-11-11): DAPO-Math-17k gives no signal
# for base models at this scale (leg 1: base eval 0/80, train
# rewards 0.0 for 7 steps). GSM8K train/test parquet from openai/gsm8k.
print("loading GSM8K parquet ...", flush=True)
GSM8K_TRAIN_URL = ("https://huggingface.co/datasets/openai/gsm8k/"
                   "resolve/main/main/train-00000-of-00001.parquet")
GSM8K_TEST_URL = ("https://huggingface.co/datasets/openai/gsm8k/"
                  "resolve/main/main/test-00000-of-00001.parquet")
df_train = pd.read_parquet(GSM8K_TRAIN_URL)
df_test = pd.read_parquet(GSM8K_TEST_URL)
print(f"gsm8k train rows: {len(df_train)} test rows: {len(df_test)}", flush=True)

GSM8K_INSTRUCTION = ("Solve the following math problem step by step. "
                     "Write your final answer on the last line as 'Answer: $N'.")
GSM8K_GOLD_RE = re.compile(r"####\s*(-?[\d,]+)")

def gsm8k_parse_gold(answer):
    """Gold from a GSM8K solution field: the number after '####'."""
    if not answer: return None
    m = GSM8K_GOLD_RE.search(answer)
    return m.group(1).replace(",", "") if m else None

def gsm8k_build_prompt(question):
    """One fixed instruction + the raw GSM8K question."""
    return GSM8K_INSTRUCTION + "\n\n" + question.strip()

train_rows = []
for _, r in df_train.iterrows():
    gold = gsm8k_parse_gold(r["answer"])
    if not gold:
        continue
    train_rows.append({"prompt": gsm8k_build_prompt(r["question"]), "gold": gold})
test_rows = []
for _, r in df_test.iterrows():
    gold = gsm8k_parse_gold(r["answer"])
    if not gold:
        continue
    test_rows.append({"prompt": gsm8k_build_prompt(r["question"]), "gold": gold})
print(f"usable gsm8k: train {len(train_rows)} test {len(test_rows)}", flush=True)
rng = random.Random(SEED)
rng.shuffle(train_rows)
rng.shuffle(test_rows)
train_pool = train_rows[:N_TRAIN_POOL]
eval_set = test_rows[:N_EVAL]
paired_prompts = train_pool[:STEPS * PROMPTS_PER_STEP]
print(f"train_pool={len(train_pool)} eval={len(eval_set)} paired_prompts={len(paired_prompts)}", flush=True)

# ---------------- metrics ----------------
metrics_f = open(METRICS_CSV, "w", newline="")
mw = csv.writer(metrics_f)
mw.writerow(["phase", "arm", "step", "mean_reward", "mean_len",
             "frac_solved", "n", "grad_norm", "eff_rank", "top1_share"])
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

def load_policy(lora_r, use_dora=False):
    base = AutoModelForCausalLM.from_pretrained(
        MODEL, quantization_config=bnb, device_map="auto",
        torch_dtype=torch.bfloat16)
    base.config.use_cache = False
    cfg = LoraConfig(r=lora_r, lora_alpha=2 * lora_r, lora_dropout=0.0,
                     target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                     "gate_proj", "up_proj", "down_proj"],
                     task_type="CAUSAL_LM", use_dora=use_dora)
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
def evaluate(model, name):
    texts, _ = generate(model, [e["prompt"] for e in eval_set], do_sample=False)
    correct, lens = 0, []
    for e, t in zip(eval_set, texts):
        pred = extract_answer(t)
        if answers_equal(pred, e["gold"]): correct += 1
        lens.append(len(t.split()))
    acc = correct / len(eval_set)
    torch.cuda.empty_cache()
    print(f"EVAL {name}: acc={acc:.4f} {correct}/{len(eval_set)} mean_len={sum(lens)/len(lens):.1f} tok-words", flush=True)
    log_row("eval", name, -1, round(acc, 4), round(sum(lens)/len(lens), 1),
            "", len(eval_set), "", "", "")
    return acc

def update_spectrum(model):
    """Spectral signature of the LoRA update dW = B@A.

    For each sampled layer/module, singular values of BA are recovered
    exactly from r x r factors: with A = U_A diag(S_A) Vh_A and
    B = U_B diag(S_B) Vh_B (thin SVDs), the nonzero singular values of
    BA equal those of diag(S_B) Vh_B U_A diag(S_A), an r x r matrix;
    then reduced with spectral_stats. Returns {module:
    (mean_eff_rank, mean_top1)} averaged over layers.
    """
    import torch as _t
    layers = [0, 8, 16, 24, 31]
    agg = {}
    try:
        peft_model = model
        base = peft_model.base_model.model
        for mod_name in ("q_proj", "v_proj"):
            effs, top1s = [], []
            for li in layers:
                attn = base.model.layers[li].self_attn
                lin = getattr(attn, mod_name)
                A = lin.lora_A.default.weight.detach().float()
                B = lin.lora_B.default.weight.detach().float()
                U_A, S_A, _ = _t.linalg.svd(A, full_matrices=False)
                U_B, S_B, Vh_B = _t.linalg.svd(B, full_matrices=False)
                C = (_t.diag(S_B) @ Vh_B @ U_A @ _t.diag(S_A))
                sing = _t.linalg.svdvals(C)
                er, t1 = spectral_stats(sing.tolist())
                effs.append(er); top1s.append(t1)
            agg[mod_name] = (sum(effs) / len(effs), sum(top1s) / len(top1s))
    except Exception as e:
        print(f"spectrum failed: {e}", flush=True)
    return agg

def train_arm(model, arm_name):
    import bitsandbytes as bnb_lib
    opt = bnb_lib.optim.AdamW8bit(
        (p for p in model.parameters() if p.requires_grad), lr=LR)
    tok.padding_side = "right"
    grad_norms = []
    for step in range(STEPS):
        t0 = time.time()
        prompts = [paired_prompts[step * PROMPTS_PER_STEP + k]["prompt"]
                   for k in range(PROMPTS_PER_STEP)]
        golds = [paired_prompts[step * PROMPTS_PER_STEP + k]["gold"]
                 for k in range(PROMPTS_PER_STEP)]
        texts, gen_ids = generate(model, prompts, do_sample=True, temp=1.0, n_per=GROUP)
        torch.cuda.empty_cache()  # drop generation KV-cache memory before training
        rewards = []
        for k in range(PROMPTS_PER_STEP):
            for g in range(GROUP):
                idx = k * GROUP + g
                rewards.append(1.0 if answers_equal(extract_answer(texts[idx]), golds[k]) else 0.0)
        advs = group_advantages(rewards, GROUP)
        mean_r = sum(rewards) / len(rewards)
        frac_solved = sum(1 for k in range(PROMPTS_PER_STEP)
                          if any(rewards[k * GROUP + g] > 0 for g in range(GROUP))) / PROMPTS_PER_STEP
        mean_len = sum(len(t.split()) for t in texts) / len(texts)
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
        micro = 2
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
                a_t = torch.full_like(lp, adv)
                loss = loss - (a_t * lp).sum()
            loss = loss / max(1, sum(mask[b].sum().item() for b in range(len(chunk))))
            (loss / (len(seqs) / micro)).backward()
            total_loss += loss.item(); total_tok += int(mask.sum().item())
        gn = torch.nn.utils.clip_grad_norm_(
            (p for p in model.parameters() if p.requires_grad), 1.0)
        grad_norms.append(float(gn))
        opt.step()
        dt = time.time() - t0
        print(f"[{arm_name}] step {step+1}/{STEPS} reward={mean_r:.3f} solved={frac_solved:.2f} "
              f"len={mean_len:.0f} loss={total_loss:.3f} gnorm={float(gn):.2f} {dt:.0f}s "
              f"peak={torch.cuda.max_memory_allocated()/2**30:.2f}GiB", flush=True)
        log_row("train", arm_name, step + 1, round(mean_r, 4), round(mean_len, 1),
                round(frac_solved, 3), PROMPTS_PER_STEP * GROUP,
                round(float(gn), 3), "", "")
    return grad_norms

# ---------------- run ----------------
results = {"model": MODEL, "seed": SEED, "steps_per_arm": STEPS,
           "prompts_per_step": PROMPTS_PER_STEP, "group": GROUP,
           "max_new": MAX_NEW, "lr": LR, "lora_alpha_ratio": 2,
           "train_pool": N_TRAIN_POOL, "eval_n": N_EVAL, "arms": {}}

print("=== loading base policy ===", flush=True)
model = load_policy(16)
print(f"model loaded peak={torch.cuda.max_memory_allocated()/2**30:.2f}GiB", flush=True)
results["arms"]["base"] = {"eval_acc": evaluate(model, "base")}
del model; torch.cuda.empty_cache()

ARMS = [("r64", 64, False), ("r128", 128, False)]
for arm_name, lora_r, use_dora in ARMS:
    print(f"=== arm {arm_name}: LoRA r={lora_r} ===", flush=True)
    model = load_policy(lora_r, use_dora=use_dora)
    print(f"model loaded peak={torch.cuda.max_memory_allocated()/2**30:.2f}GiB", flush=True)
    grad_norms = train_arm(model, arm_name)
    acc = evaluate(model, arm_name)
    spec = update_spectrum(model)
    for mod_name, (er, t1) in spec.items():
        log_row("spectrum", arm_name, -1, "", "", "", "",
                "", round(er, 2), round(t1, 4))
        print(f"SPECTRUM {arm_name} {mod_name}: eff_rank={er:.2f} "
              f"top1_share={t1:.4f}", flush=True)
    results["arms"][arm_name] = {
        "lora_r": lora_r, "use_dora": use_dora, "eval_acc": acc,
        "mean_grad_norm": sum(grad_norms) / len(grad_norms),
        "max_grad_norm": max(grad_norms),
        "spectrum": {k: {"eff_rank": v[0], "top1_share": v[1]}
                     for k, v in spec.items()},
        "peak_vram_gib": torch.cuda.max_memory_allocated() / 2**30,
    }
    model.save_pretrained(f"/content/structure_fb_adapter_{arm_name}")
    print(f"saved /content/structure_adapter_{arm_name}", flush=True)
    del model; torch.cuda.empty_cache()
    json.dump(results, open(RESULTS_JSON, "w"), indent=1)

elapsed = time.time() - T0
if elapsed < DORA_TIME_BUDGET_S:
    print(f"=== arm dora128: DoRA r=128 (elapsed {elapsed/60:.0f} min) ===", flush=True)
    model = load_policy(128, use_dora=True)
    print(f"model loaded peak={torch.cuda.max_memory_allocated()/2**30:.2f}GiB", flush=True)
    grad_norms = train_arm(model, "dora128")
    acc = evaluate(model, "dora128")
    spec = update_spectrum(model)
    for mod_name, (er, t1) in spec.items():
        log_row("spectrum", "dora128", -1, "", "", "", "",
                "", round(er, 2), round(t1, 4))
        print(f"SPECTRUM dora128 {mod_name}: eff_rank={er:.2f} "
              f"top1_share={t1:.4f}", flush=True)
    results["arms"]["dora128"] = {
        "lora_r": 128, "use_dora": True, "eval_acc": acc,
        "mean_grad_norm": sum(grad_norms) / len(grad_norms),
        "max_grad_norm": max(grad_norms),
        "spectrum": {k: {"eff_rank": v[0], "top1_share": v[1]}
                     for k, v in spec.items()},
        "peak_vram_gib": torch.cuda.max_memory_allocated() / 2**30,
    }
    model.save_pretrained("/content/structure_fb_adapter_dora128")
    print("saved /content/structure_adapter_dora128", flush=True)
    del model; torch.cuda.empty_cache()
else:
    print(f"SKIP dora128: elapsed {elapsed/60:.0f} min >= 2h budget", flush=True)
    results["arms"]["dora128"] = {"skipped": True,
                                  "reason": "time budget",
                                  "elapsed_min": elapsed / 60}

metrics_f.close()
json.dump(results, open(RESULTS_JSON, "w"), indent=1)
print("RESULTS " + json.dumps(results), flush=True)
print("TRAIN_DONE", flush=True)
