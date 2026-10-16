"""Faithful Root-token Policy Optimization retry on Colab T4 (arXiv:2610.17767).

Faithful retry of the 2026-10-14 run (verdict: MIXED -- the paper's rule is
sound, the port degenerated). Two structural fixes address the diagnosed
causes:

(a) NATIVE THINK DECISION. The root token is the model's native `<think>`
    token (Qwen3.5 is a hybrid thinking model), not the T/D single-letter
    proxy that collapsed. The policy learns P(first token == `<think>`);
    DIRECT is the complement 1 - P(`<think>`). Forced THINK arms prefix the
    native `<think>` token; forced DIRECT arms suppress it with a logits
    processor. The decision is now semantically grounded in the behavior
    trigger the paper trains.

(b) FULL-SEQUENCE KL. KL(pi || pi_ref) is computed over the entire
    generated sequence (Schulman estimator, mean per token), not just the
    root position, so the LoRA adapters cannot drift the continuations
    unchecked (the 192-token rambles of the first run).

Everything else matches the original run: Qwen3.5-9B NF4 QLoRA r=16/a=32
bf16, 80 train / 40 eval questions (same seeds), group-relative advantages
over forced arms [THINK, THINK, DIRECT, DIRECT], reward = correct - 0.15 *
thought, beta_kl = 0.02 (now applied to mean per-token KL, comparable in
scale to the original root-only term).

Run on the VM:  python3 /content/colab_rpo_faithful.py
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
                          BitsAndBytesConfig, LogitsProcessor)

# ---------------------------------------------------------------- config
BASE = "Qwen/Qwen3.5-9B"
OUT = "/content/rpo_faithful_metrics.json"
OUTCSV = "/content/rpo_faithful_metrics.csv"
ADIR = "/content/rpo_faithful_adapters"
STAGE1 = "/content/rpo_faithful_stage1.json"
STAGE2 = "/content/rpo_faithful_stage2.json"
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
    "Solve the problem. If it requires careful reasoning, begin your "
    "response with <think>, think step by step, close with </think>, then "
    "give the final answer. Otherwise answer directly. "
    "Always finish with 'Answer: <integer>'.\n"
    "Problem: %s\n")


def easy_items(rng, n):
    items = []
    for _ in range(n):
        a, b = rng.randint(11, 99), rng.randint(11, 99)
        op = rng.choice(["+", "-", "*"])
        if op == "-" and b > a:
            a, b = b, a
        q = "What is %d %s %d?" % (a, op, b)
        ans = eval("%d %s %d" % (a, op, b))
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
BASE = "/content/qwen35-9b"

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

# --- fix (a): the root decision is the NATIVE <think> token ------------
THINK_ID = tok.convert_tokens_to_ids("<think>")
assert isinstance(THINK_ID, int) and THINK_ID >= 0, \
    "tokenizer has no <think> token -- faithful port impossible"
# sanity: it must be a single token, not split
assert tok.convert_ids_to_tokens([THINK_ID]) == ["<think>"], \
    "<think> id round-trip failed"
print("[model] native <think> token id = %d" % THINK_ID, flush=True)


class SuppressThink(LogitsProcessor):
    """Force the DIRECT arm: <think> can never be emitted."""

    def __init__(self, think_id):
        self.think_id = think_id

    def __call__(self, input_ids, scores):
        scores[:, self.think_id] = float("-inf")
        return scores


SUPPRESS = SuppressThink(THINK_ID)


def prompt_for(question):
    return INSTR % question


@torch.no_grad()
def rollout(model, question, think, max_new):
    """Generate after a forced root decision.

    think=True:  prefix the native <think> token (behavior trigger on).
    think=False: suppress <think> for the whole generation (answer direct).
    Returns (token_ids_of_generated_part, text).
    """
    prompt_ids = tok(prompt_for(question),
                     return_tensors="pt")["input_ids"].to(DEVICE)
    if think:
        think_t = torch.tensor([[THINK_ID]], device=DEVICE)
        enc = torch.cat([prompt_ids, think_t], dim=1)
        gen = model.generate(enc, max_new_tokens=max_new, do_sample=False,
                             pad_token_id=tok.pad_token_id)
        gen_ids = gen[0][enc.shape[1]:]
        # full generated part INCLUDES the forced <think> (it is the root)
        full_gen = torch.cat([think_t[0], gen_ids])
    else:
        gen = model.generate(prompt_ids, max_new_tokens=max_new,
                             do_sample=False,
                             pad_token_id=tok.pad_token_id,
                             logits_processor=[SUPPRESS])
        full_gen = gen[0][prompt_ids.shape[1]:]
    text = tok.decode(full_gen, skip_special_tokens=True)
    return full_gen.cpu().tolist(), text


@torch.no_grad()
def rollout_free(model, question, max_new=MAX_NEW_EVAL):
    """Free decision: first generated token decides (greedy)."""
    enc = tok(prompt_for(question), return_tensors="pt").to(DEVICE)
    gen = model.generate(enc, max_new_tokens=max_new, do_sample=False,
                         pad_token_id=tok.pad_token_id)
    gen_ids = gen[0][enc["input_ids"].shape[1]:]
    text = tok.decode(gen_ids, skip_special_tokens=True)
    first_id = gen_ids[0].item() if gen_ids.numel() else -1
    decision = "think" if first_id == THINK_ID else "direct"
    return decision, text


def score_rollout(text, gold, thought):
    pred = extract_answer(text)
    correct = 1.0 if pred == gold else 0.0
    return correct - THINK_COST * thought, correct


def arm_loss_terms(model, prompt_ids, gen_ids, think):
    """Root-decision logp (grad) + full-sequence KL (fix b).

    prompt_ids: [1, P]; gen_ids: [1, G] (THINK arm: includes forced
    <think> as its first token; DIRECT arm: <think>-free by construction).
    Returns (logp_root, kl_mean_per_token).
    """
    full = torch.cat([prompt_ids, gen_ids], dim=1)
    P, G = prompt_ids.shape[1], gen_ids.shape[1]
    # --- policy forward (keeps graph) ---------------------------------
    out = model(input_ids=full)
    logits = out.logits[0].float()  # [L, V]
    root_logits = logits[P - 1]
    if think:
        logp_root = torch.log_softmax(root_logits, -1)[THINK_ID]
    else:
        p_think = torch.softmax(root_logits, -1)[THINK_ID]
        # log(1 - p): the DIRECT decision is the complement event
        logp_root = torch.log1p(-p_think.clamp(max=1.0 - 1e-6))
    # per-token logps of the generated part (positions P-1 .. P+G-2 predict
    # tokens at P .. P+G-1)
    gen_logits = logits[P - 1:P - 1 + G]
    gen_logp = torch.log_softmax(gen_logits, -1).gather(
        1, gen_ids[0].unsqueeze(1)).squeeze(1)
    del out
    # --- reference forward (no-grad, adapters disabled) ---------------
    with model.disable_adapter(), torch.no_grad():
        out_ref = model(input_ids=full)
        logits_ref = out_ref.logits[0].float()
        gen_logits_ref = logits_ref[P - 1:P - 1 + G]
        gen_logp_ref = torch.log_softmax(gen_logits_ref, -1).gather(
            1, gen_ids[0].unsqueeze(1)).squeeze(1).detach()
        del out_ref
    # --- Schulman KL(pi || pi_ref), mean per token --------------------
    dr = gen_logp_ref - gen_logp
    kl = (torch.exp(dr) - dr - 1.0).mean()
    return logp_root, kl

# ---------------------------------------------------------------- stage 1: forced-arm rollouts
if os.path.exists(STAGE1):
    s1 = json.load(open(STAGE1))
    groups = s1["groups"]
    print("[stage1] resumed %d groups" % len(groups), flush=True)
else:
    groups = []
    t0 = time.time()
    for qi, (q, gold, diff) in enumerate(train_items):
        arms = [(True, MAX_NEW_THINK), (True, MAX_NEW_THINK),
                (False, MAX_NEW_DIRECT), (False, MAX_NEW_DIRECT)]
        arm_data, rewards, corrects = [], [], []
        for think, mx in arms:
            gen_ids, text = rollout(base_model, q, think, mx)
            r, c = score_rollout(text, gold, think)
            arm_data.append({"think": think, "gen_ids": gen_ids,
                             "text": text})
            rewards.append(r)
            corrects.append(c)
        mean_r = sum(rewards) / len(rewards)
        advs = [r - mean_r for r in rewards]
        groups.append({"q": q, "gold": gold, "diff": diff,
                       "arms": arm_data, "rewards": rewards,
                       "advs": advs, "corrects": corrects})
        if (qi + 1) % 10 == 0:
            print("[stage1] %d/%d groups (%.0fs)" %
                  (qi + 1, len(train_items), time.time() - t0), flush=True)
    json.dump({"groups": groups}, open(STAGE1, "w"))
    print("[stage1] saved (%d groups)" % len(groups), flush=True)

# ---------------------------------------------------------------- eval helper
@torch.no_grad()
def evaluate(model, items, tag, force_think=False):
    think_e = think_h = 0
    acc_e = acc_h = 0
    n_e = n_h = 0
    ntok = 0
    for q, gold, diff in items:
        if force_think:
            _, text = rollout(model, q, True, MAX_NEW_THINK)
            decision, correct = "think", float(extract_answer(text) == gold)
        else:
            decision, text = rollout_free(model, q)
            correct = float(extract_answer(text) == gold)
        if diff == "easy":
            n_e += 1
            think_e += decision == "think"
            acc_e += correct
        else:
            n_h += 1
            think_h += decision == "think"
            acc_h += correct
        ntok += len(tok(text)["input_ids"])
    res = {"tag": tag,
           "think_rate_easy": think_e / max(n_e, 1),
           "think_rate_hard": think_h / max(n_h, 1),
           "acc_easy": acc_e / max(n_e, 1),
           "acc_hard": acc_h / max(n_h, 1),
           "mean_gen_tokens": ntok / len(items)}
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

# ---------------------------------------------------------------- stage 3: faithful root-token training
print("[train] attaching LoRA...", flush=True)
from peft import LoraConfig, get_peft_model
cfg = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.0,
                 target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                 "gate_proj", "up_proj", "down_proj"],
                 task_type="CAUSAL_LM")
model = get_peft_model(base_model, cfg)
# gradient checkpointing: full-sequence policy forwards on a 9B model need
# it on a T4 (the original run OOM'd even on root-only ref forwards).
model.gradient_checkpointing_enable()
model.train()
import bitsandbytes as bnb_opt
opt = bnb_opt.optim.AdamW8bit(model.parameters(), lr=LR)

order = list(range(len(groups)))
step = 0
loss_hist = []
for ep in range(EPOCHS):
    random.Random(SEED + ep).shuffle(order)
    for gi in order:
        g = groups[gi]
        if all(abs(a) < 1e-12 for a in g["advs"]):
            continue  # uniform group -> zero advantage -> skip
        prompt_ids = tok(prompt_for(g["q"]),
                         return_tensors="pt")["input_ids"].to(DEVICE)
        logp_roots, kls, advs = [], [], []
        for arm, adv in zip(g["arms"], g["advs"]):
            gen_ids = torch.tensor(
                [arm["gen_ids"]], device=DEVICE)
            logp_root, kl = arm_loss_terms(model, prompt_ids, gen_ids,
                                           arm["think"])
            logp_roots.append(logp_root)
            kls.append(kl)
            advs.append(adv)
            del gen_ids
        logp_roots = torch.stack(logp_roots)
        kls = torch.stack(kls)
        advs_t = torch.tensor(advs, device=DEVICE)
        pg_loss = -(advs_t * logp_roots).mean()
        loss = pg_loss + BETA_KL * kls.mean()
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        step += 1
        loss_val = loss.item()
        loss_hist.append(loss_val)
        del (prompt_ids, logp_roots, kls, advs_t, pg_loss, loss)
        if step % 5 == 0:
            torch.cuda.empty_cache()
        if step % 10 == 0:
            print("[train] ep=%d step=%d loss=%.4f" %
                  (ep, step, loss_val), flush=True)
        # incremental adapter checkpoint (free-tier survival)
        if step % 40 == 0:
            model.save_pretrained(os.path.join(ADIR, "ckpt-%d" % step))

print("[train] done: %d updates" % step, flush=True)
model.save_pretrained(os.path.join(ADIR, "rpo_faithful"))
model.eval()
try:
    model.config.use_cache = True
except Exception:
    pass

print("[eval] trained model, free decision...", flush=True)
trained_free = evaluate(model, eval_items, "trained_free")

# ---------------------------------------------------------------- metrics
adaptive_gap = (trained_free["think_rate_hard"] -
                trained_free["think_rate_easy"])
metrics = {
    "base_model": BASE,
    "root_decision": "native <think> token id %d" % THINK_ID,
    "kl_scope": "full sequence (mean per token)",
    "n_train": len(train_items), "n_eval": len(eval_items),
    "epochs": EPOCHS, "lr": LR, "beta_kl": BETA_KL,
    "think_cost": THINK_COST,
    "root_token_updates": step,
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
