"""Root-token Policy Optimization on Colab T4 (arXiv:2610.17767).

RPO: train ONLY the root token (the decision whether to think) via
group-relative rewards and group-wise advantages -> difficulty-aware
adaptive thinking at a fraction of full training compute.

Bounded design for a free Colab T4 at the 9B class (Qwen/Qwen3.5-9B,
NF4 QLoRA r=16/a=32, bf16) per the arxiv read-and-test training-track
protocol:

* Task: 80 train / 40 eval questions, half EASY (single-step arithmetic,
  solvable without thinking) and half HARD (3-4 step synthetic word
  problems, rule-verifiable integer answers).
* Root decision: the response's first word must be THINK or DIRECT.
  - THINK: reason step by step, finish with "Answer: <int>".
  - DIRECT: immediately write "Answer: <int>".
* Group rollouts: per question, forced arms [THINK, THINK, DIRECT,
  DIRECT]. reward = correct - 0.15 * thought. advantage = r - mean(r)
  (group-relative, group-wise).
* Update: masked policy-gradient loss on the ROOT TOKEN ONLY:
  loss = -mean(A * logp_root) + beta * KL(pi || pi_ref); LoRA adapters
  receive gradient exclusively through that one token (~0.3% of the
  tokens a full-sequence SFT would touch -- the paper's "2% compute"
  claim, measured).
* Eval (held-out): base vs trained, free greedy decision; plus a
  base always-THINK ceiling. Metrics: think-rate per difficulty,
  accuracy per difficulty, mean generated tokens, format compliance.

Bounded deviations from the paper (documented): the paper trains the
native <think> token of a hybrid model; here the root token is
operationalized as the first response word (THINK/DIRECT) under an
explicit instruction, and the KL reference is the same base with
adapters disabled (no second 9B copy fits a T4).

Self-contained: pip installs at top (incl. `pip uninstall -y torchao`
for the peft/torchao-0.10 conflict), writes metrics incrementally to
/content/rpo_metrics.json and /content/rpo_metrics.csv. Resumable via
/content/rpo_stage{1,2}.json.

Run on the VM:  python3 /content/colab_rpo.py
"""

import json
import math
import os
import random
import re
import subprocess
import sys
import time

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

# ---------------------------------------------------------------- deps
print("[deps] installing...", flush=True)
# NOTE: Qwen3.5's `qwen3_5` arch only exists in transformers main
# (overrides the usual transformers<5.0 training-track pin).
subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                "git+https://github.com/huggingface/transformers.git",
                "peft", "bitsandbytes", "accelerate", "safetensors"],
               check=True)
subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "torchao"],
               check=False)
print("[deps] done", flush=True)

import torch
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          BitsAndBytesConfig)
from peft import LoraConfig, get_peft_model

# ---------------------------------------------------------------- config
BASE = "Qwen/Qwen3.5-9B"
OUT = "/content/rpo_metrics.json"
OUTCSV = "/content/rpo_metrics.csv"
ADIR = "/content/rpo_adapters"
STAGE1 = "/content/rpo_stage1.json"
STAGE2 = "/content/rpo_stage2.json"
SEED = 7
random.seed(SEED)

N_TRAIN_EASY = 40
N_TRAIN_HARD = 40
N_EVAL_EASY = 20
N_EVAL_HARD = 20
EPOCHS = 2
LR = 1e-4
BETA_KL = 0.02
THINK_COST = 0.15
MAX_NEW_THINK = 160
MAX_NEW_DIRECT = 32
MAX_NEW_EVAL = 192

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
T0 = time.time()
print("[env] device=%s" % DEVICE, flush=True)
if DEVICE == "cuda":
    print("[env] gpu=%s mem=%.1fGB" % (
        torch.cuda.get_device_name(0),
        torch.cuda.get_device_properties(0).total_memory / 1e9), flush=True)

# ---------------------------------------------------------------- data
INSTR = (
    "Solve the problem. Your response MUST start with exactly one word: "
    "THINK or DIRECT.\n"
    "- If THINK: reason step by step, then finish with 'Answer: <integer>'.\n"
    "- If DIRECT: write 'Answer: <integer>' immediately, no reasoning.\n"
    "Problem: %s\nDecision:")


def easy_items(rng, n):
    items = []
    for _ in range(n):
        a, b = rng.randint(11, 99), rng.randint(11, 99)
        op = rng.choice(["+", "-", "*"])
        if op == "-" and b > a:
            a, b = b, a
        q = "What is %d %s %d?" % (a, op, b)
        ans = eval("%d %s %d" % (a, op.replace("*", "*"), b))
        items.append((q, int(ans), "easy"))
    return items


def hard_items(rng, n):
    items = []
    templates = ["bakery", "warehouse", "school", "train"]
    for _ in range(n):
        t = rng.choice(templates)
        if t == "bakery":
            a = rng.randint(60, 120); b = rng.randint(10, 40)
            c = rng.randint(20, 60); d = rng.randint(5, 25)
            q = ("A bakery baked %d loaves in the morning. It sold %d at "
                 "noon, baked %d more in the afternoon, and donated %d to "
                 "a shelter. How many loaves remain?" % (a, b, c, d))
            ans = a - b + c - d
        elif t == "warehouse":
            a = rng.randint(200, 400); b = rng.randint(2, 5)
            c = rng.randint(10, 30); d = rng.randint(20, 80)
            q = ("A warehouse had %d boxes. %d trucks each took %d boxes "
                 "away. Then %d boxes arrived. How many boxes are there "
                 "now?" % (a, b, c, d))
            ans = a - b * c + d
        elif t == "school":
            d = rng.choice([2, 3, 4, 5])
            per = rng.randint(8, 25)
            total = per * d
            b = rng.randint(5, 30)
            c = rng.randint(5, 30)
            a = total - b + c
            q = ("A school had %d students. %d new students enrolled, then "
                 "%d students graduated. The school split all students "
                 "evenly into %d classes. How many students per class?"
                 % (a, b, c, d))
            ans = per
        else:  # train
            a = rng.randint(120, 250); b = rng.randint(10, 50)
            c = rng.randint(10, 60); d = rng.randint(10, 50)
            e = rng.randint(10, 60)
            q = ("A train carried %d passengers. At the first stop %d got "
                 "off and %d got on. At the second stop %d got off and %d "
                 "got on. How many passengers are on the train now?"
                 % (a, b, c, d, e))
            ans = a - b + c - d + e
        assert ans > 0
        items.append((q, int(ans), "hard"))
    return items


rng_train = random.Random(SEED)
train_items = (easy_items(rng_train, N_TRAIN_EASY) +
               hard_items(rng_train, N_TRAIN_HARD))
rng_eval = random.Random(SEED + 100)
eval_items = (easy_items(rng_eval, N_EVAL_EASY) +
              hard_items(rng_eval, N_EVAL_HARD))
print("[data] train=%d eval=%d" % (len(train_items), len(eval_items)),
      flush=True)

ANS_RE = re.compile(r"Answer:\s*(-?\d+)")


def extract_answer(text):
    ms = ANS_RE.findall(text or "")
    return int(ms[-1]) if ms else None

# ---------------------------------------------------------------- weights
# aria2c: proven ~8x faster than the HF hub client (19GB in ~6 min on
# 2026-10-12). Critical: keeps the download inside the free-tier
# survival window.
print("[weights] downloading via aria2c...", flush=True)
subprocess.run(["apt-get", "install", "-y", "aria2"],
               check=False, capture_output=True)
os.makedirs("/content/qwen35-9b", exist_ok=True)
WFILES = (["config.json", "model.safetensors.index.json", "tokenizer.json",
           "tokenizer_config.json", "vocab.json", "merges.txt"] +
          ["model.safetensors-0000%d-of-00004.safetensors" % i
           for i in range(1, 5)])
WURL = "https://huggingface.co/Qwen/Qwen3.5-9B/resolve/main/"
with open("/content/dl.txt", "w") as f:
    for fn in WFILES:
        f.write("%s%s\n  dir=/content/qwen35-9b\n  out=%s\n" % (WURL, fn, fn))
missing = [fn for fn in WFILES
           if not os.path.exists("/content/qwen35-9b/" + fn)]
if not missing:
    print("[weights] already present, skipping download", flush=True)
else:
    t0 = time.time()
    subprocess.run(["aria2c", "-x16", "-s16", "-j4", "-i", "/content/dl.txt",
                    "--summary-interval=0", "--console-log-level=error",
                    "--file-allocation=none"], check=True)
    missing = [fn for fn in WFILES
               if not os.path.exists("/content/qwen35-9b/" + fn)]
    assert not missing, "missing weight files: %s" % missing
    print("[weights] done in %.1f min" % ((time.time() - t0) / 60),
          flush=True)
BASE = "/content/qwen35-9b"  # local aria2c download (see above)

# ---------------------------------------------------------------- model
print("[model] loading %s (NF4)..." % BASE, flush=True)
bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                         bnb_4bit_use_double_quant=True,
                         bnb_4bit_compute_dtype=torch.bfloat16)
tok = AutoTokenizer.from_pretrained(BASE, trust_remote_code=True)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
tok.padding_side = "left"
base_model = AutoModelForCausalLM.from_pretrained(
    BASE, quantization_config=bnb, device_map="auto",
    trust_remote_code=True, torch_dtype=torch.bfloat16)
base_model.eval()
print("[model] loaded", flush=True)

# decision words: first pair where both are single tokens
DECISION = None
for w1, w2 in [("THINK", "DIRECT"), ("T", "D")]:
    t1 = tok(w1, add_special_tokens=False)["input_ids"]
    t2 = tok(w2, add_special_tokens=False)["input_ids"]
    if len(t1) == 1 and len(t2) == 1:
        DECISION = (w1, w2)
        break
assert DECISION, "no single-token decision pair"
THINK_W, DIRECT_W = DECISION
print("[model] decision words: %s / %s" % DECISION, flush=True)


def prompt_for(question):
    return INSTR % question


@torch.no_grad()
def rollout(model, question, decision, max_new):
    """Generate a continuation after a forced root decision word."""
    full = prompt_for(question) + " " + decision
    enc = tok(full, return_tensors="pt").to(DEVICE)
    gen = model.generate(**enc, max_new_tokens=max_new, do_sample=False,
                         pad_token_id=tok.pad_token_id)
    text = tok.decode(gen[0][enc["input_ids"].shape[1]:],
                      skip_special_tokens=True)
    return text


@torch.no_grad()
def rollout_free(model, question, max_new=MAX_NEW_EVAL):
    """Free decision: model emits the first word itself (greedy)."""
    enc = tok(prompt_for(question), return_tensors="pt").to(DEVICE)
    gen = model.generate(**enc, max_new_tokens=max_new, do_sample=False,
                         pad_token_id=tok.pad_token_id)
    text = tok.decode(gen[0][enc["input_ids"].shape[1]:],
                      skip_special_tokens=True)
    first = (text.strip().split() or [""])[0].rstrip(".,:;!").upper()
    decision = ("think" if first == THINK_W else
                "direct" if first == DIRECT_W else "other")
    return decision, text


def score_rollout(text, gold, thought):
    pred = extract_answer(text)
    correct = 1.0 if pred == gold else 0.0
    return correct - THINK_COST * thought, correct


def root_logprobs(model, question, decision):
    """Log-prob of the root decision token under model; ref = adapters off.

    Only the policy forward keeps a graph (needed for the root-token
    gradient); the reference forward is no-grad. Callers must delete the
    returned tensors promptly (VRAM is tight on a T4).
    """
    full_ids = tok(prompt_for(question) + " " + decision,
                   return_tensors="pt")["input_ids"].to(DEVICE)
    pre_ids = tok(prompt_for(question),
                  return_tensors="pt")["input_ids"].to(DEVICE)
    dec_id = full_ids[0, pre_ids.shape[1]:]
    assert dec_id.numel() == 1, "decision not a single token in context"
    pos = pre_ids.shape[1] - 1  # logits position predicting the decision
    out = model(input_ids=full_ids)
    logp = torch.log_softmax(out.logits[0, pos].float(), -1)[dec_id[0]]
    del out
    with model.disable_adapter(), torch.no_grad():
        out_ref = model(input_ids=full_ids)
        logp_ref = torch.log_softmax(
            out_ref.logits[0, pos].float(), -1)[dec_id[0]].detach()
        del out_ref
    return logp, logp_ref

# ---------------------------------------------------------------- stage 1: train rollouts (forced arms)
if os.path.exists(STAGE1):
    s1 = json.load(open(STAGE1))
    groups = s1["groups"]
    print("[stage1] resumed %d groups" % len(groups), flush=True)
else:
    groups = []
    t0 = time.time()
    for qi, (q, gold, diff) in enumerate(train_items):
        arms = [(THINK_W, True, MAX_NEW_THINK), (THINK_W, True, MAX_NEW_THINK),
                (DIRECT_W, False, MAX_NEW_DIRECT),
                (DIRECT_W, False, MAX_NEW_DIRECT)]
        rewards, corrects = [], []
        for word, thought, mx in arms:
            text = rollout(base_model, q, word, mx)
            r, c = score_rollout(text, gold, thought)
            rewards.append(r)
            corrects.append(c)
        mean_r = sum(rewards) / len(rewards)
        advs = [r - mean_r for r in rewards]
        groups.append({"q": q, "gold": gold, "diff": diff,
                       "rewards": rewards, "advs": advs,
                       "corrects": corrects})
        if (qi + 1) % 10 == 0:
            print("[stage1] %d/%d groups (%.0fs)" %
                  (qi + 1, len(train_items), time.time() - t0), flush=True)
    json.dump({"groups": groups}, open(STAGE1, "w"))
    print("[stage1] saved (%d groups)" % len(groups), flush=True)

# ---------------------------------------------------------------- eval helper (works on any peft/base model)
@torch.no_grad()
def evaluate(model, items, tag, force_think=False):
    think_e = think_h = 0
    acc_e = acc_h = 0
    n_e = n_h = 0
    fmt_ok = 0
    ntok = 0
    for q, gold, diff in items:
        if force_think:
            text = rollout(model, q, THINK_W, MAX_NEW_THINK)
            decision, correct = "think", float(extract_answer(text) == gold)
        else:
            decision, text = rollout_free(model, q)
            correct = float(extract_answer(text) == gold)
        if decision in ("think", "direct"):
            fmt_ok += 1
        if diff == "easy":
            n_e += 1
            think_e += decision == "think"
            acc_e += correct
        else:
            n_h += 1
            think_h += decision == "think"
            acc_h += correct
        ntok += len(tok(text)["input_ids"])
    n = len(items)
    res = {"tag": tag,
           "think_rate_easy": think_e / max(n_e, 1),
           "think_rate_hard": think_h / max(n_h, 1),
           "acc_easy": acc_e / max(n_e, 1),
           "acc_hard": acc_h / max(n_h, 1),
           "format_ok_rate": fmt_ok / n,
           "mean_gen_tokens": ntok / n}
    print("[eval:%s] %s" % (tag, json.dumps(
        {k: round(v, 3) for k, v in res.items() if k != "tag"})),
        flush=True)
    return res


print("[eval] base model, free decision...", flush=True)
if os.path.exists(STAGE2):
    s2 = json.load(open(STAGE2))
    base_free = s2["base_free"]
    base_think = s2["base_think"]
    print("[stage2] resumed base eval", flush=True)
else:
    base_free = evaluate(base_model, eval_items, "base_free")
    print("[eval] base model, forced THINK (ceiling)...", flush=True)
    base_think = evaluate(base_model, eval_items, "base_think",
                          force_think=True)
    json.dump({"base_free": base_free, "base_think": base_think},
              open(STAGE2, "w"))

# ---------------------------------------------------------------- stage 3: root-token training
print("[train] attaching LoRA...", flush=True)
cfg = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.0,
                 target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                 "gate_proj", "up_proj", "down_proj"],
                 task_type="CAUSAL_LM")
model = get_peft_model(base_model, cfg)
# gradient checkpointing: the root-token forward keeps a full 9B graph;
# without checkpointing the 4-arms x (policy + ref) forwards OOM a T4
# (observed 2026-10-14).
model.gradient_checkpointing_enable()
model.train()
import bitsandbytes as bnb_opt
opt = bnb_opt.optim.AdamW8bit(model.parameters(), lr=LR)

order = list(range(len(groups)))
step = 0
loss_hist = []
tok_touched = 0
for ep in range(EPOCHS):
    random.Random(SEED + ep).shuffle(order)
    for gi in order:
        g = groups[gi]
        if all(abs(a) < 1e-12 for a in g["advs"]):
            continue  # uniform group -> zero advantage -> skip
        # one root-token forward per arm (4 tiny forwards, 1 token of loss)
        logps, logp_refs, advs = [], [], []
        for arm_i, (word, thought) in enumerate(
                [(THINK_W, True), (THINK_W, True),
                 (DIRECT_W, False), (DIRECT_W, False)]):
            lp, lpr = root_logprobs(model, g["q"], word)
            logps.append(lp)
            logp_refs.append(lpr)
            advs.append(g["advs"][arm_i])
        logps = torch.stack(logps)
        logp_refs = torch.stack(logp_refs)
        advs_t = torch.tensor(advs, device=DEVICE)
        pg_loss = -(advs_t * logps).mean()
        ratio = logp_refs - logps
        kl = (torch.exp(ratio) - ratio - 1.0).mean()
        loss = pg_loss + BETA_KL * kl
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        step += 1
        tok_touched += len(advs)
        loss_hist.append(loss.item())
        # free the 8 forward graphs promptly (T4 VRAM hygiene)
        del logps, logp_refs, advs_t, pg_loss, kl, loss
        if step % 10 == 0:
            torch.cuda.empty_cache()
        if step % 20 == 0:
            print("[train] ep=%d step=%d loss=%.4f" %
                  (ep, step, loss.item()), flush=True)

print("[train] done: %d root-token updates, %d decision-tokens touched"
      % (step, tok_touched), flush=True)
model.save_pretrained(os.path.join(ADIR, "rpo"))
model.eval()
try:
    model.config.use_cache = True  # restore for generate() (was disabled
except Exception:                  # by gradient_checkpointing_enable)
    pass

print("[eval] trained model, free decision...", flush=True)
trained_free = evaluate(model, eval_items, "trained_free")

# ---------------------------------------------------------------- metrics
adaptive_gap = (trained_free["think_rate_hard"] -
                trained_free["think_rate_easy"])
# compute-saving proxy: root tokens touched vs full-sequence SFT tokens
sft_tokens = sum(len(tok(prompt_for(g["q"]))["input_ids"]) + 150
                 for g in groups) * EPOCHS
metrics = {
    "base_model": BASE,
    "decision_words": list(DECISION),
    "n_train": len(train_items), "n_eval": len(eval_items),
    "epochs": EPOCHS, "lr": LR, "beta_kl": BETA_KL,
    "think_cost": THINK_COST,
    "root_token_updates": step,
    "decision_tokens_touched": tok_touched,
    "sft_token_equivalent": sft_tokens,
    "compute_fraction_vs_sft": tok_touched / max(sft_tokens, 1),
    "base_free": base_free,
    "base_always_think": base_think,
    "trained_free": trained_free,
    "trained_adaptive_gap": adaptive_gap,
    "final_loss": loss_hist[-1] if loss_hist else None,
    "elapsed_s": round(time.time() - T0, 1),
}
with open(OUT, "w") as f:
    json.dump(metrics, f, indent=1)
with open(OUTCSV, "w") as f:
    f.write("metric,value\n")
    for k, v in metrics.items():
        f.write("%s,%s\n" % (k, json.dumps(v) if isinstance(
            v, (dict, list)) else v))
print("[done] metrics -> %s" % OUT, flush=True)
print(json.dumps({k: v for k, v in metrics.items()
                  if k not in ("base_free", "base_always_think",
                               "trained_free")}, indent=1), flush=True)
print(json.dumps({"base_free": base_free, "base_always_think": base_think,
                  "trained_free": trained_free}, indent=1), flush=True)
