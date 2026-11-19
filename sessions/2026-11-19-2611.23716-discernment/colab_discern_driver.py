#!/usr/bin/env python3
# ==============================================================================
# Learning through Discernment (arXiv:2611.23716) — Colab T4 training-track
# driver.
#
# Paper claim: a small model (Qwen3 0.6B/1.7B) with a LoRA adapter, fine-tuned
# each night on the attempts a judge accepts, improves then decays under
# self-judging; a "feedback sense" built for discernment -- which FIRST asks
# whether the user gave a clear verdict -- keeps 4 of 5 lives near the
# oracle for two weeks; across 97 lives the share of failed attempts among
# the lessons predicts the outcome (Spearman rho = -0.81). Learning from
# users holds up only where the judge keeps failures out and the user can
# see the outcome.
#
# Bounded-scale test on Colab T4 (this driver): a synthetic WebShop-like
# shop-assistant suite (see discern_tasks.py -- deterministic, exactly one
# correct item per task, templated user reactions with clear verdicts,
# polite give-ups that thank the agent for a FAILURE, and ambiguous
# reactions). Three judge arms on the SAME attempt pools:
#   self    -- the base model judges its own attempt (YES/NO)
#   discern -- frozen judge prompt: first asks whether the user gave a
#              clear verdict; accepts only clear YES as lessons
#   oracle  -- true outcome (upper bound)
# Two nights: night 0 base attempts -> judge -> LoRA on accepted lessons
# -> held-out eval; night 1 adapted-model attempts -> judge -> fresh LoRA
# on cumulative lessons -> held-out eval. Metrics: held-out accuracy per
# arm per night, failure share among each night's lessons (true labels),
# judge accuracy vs truth, polite-give-up trap rate, and Spearman rho
# between lesson failure share and accuracy change across the 6
# (arm, night) points (paper: -0.81; n=6 here -- indicative only).
#
# Model: Qwen/Qwen3.5-9B-Base (standing pick: try Qwen3.5-9B first).
#   - 2026-10-30 live T4 probe GREEN: NF4 load + LoRA + train step +
#     generate, VRAM 7.89 GB (overmind-banking77 probe_qwen35.json).
#   - Qwen3.5 needs transformers>=5; this driver pins transformers==5.20.2,
#     the probe's exact version (same as the Ambient driver, which trained
#     fine on both models 2026-11-18).
#   - Fallback (env MODEL_ID=Qwen/Qwen3-8B-Base): proven transformers<5.0
#     QLoRA recipe instead.
# Budget (from 2026-11-18 timing: ~18 s/problem greedy eval at batch 8,
# ~60 min per 600-example train arm): attempts/judges/evals are SHORT
# (max 64 new tokens) and few (32 tasks/arm/night); trainings are <=70
# lessons x 1 epoch. Projected total ~60-75 min, inside the ~4 h
# free-tier reclaim horizon. Metrics are flushed after EVERY phase, so a
# reclamation still leaves partial results.
# Writes /content/discern_metrics.csv, /content/discern_summary.json and
# /content/discern_predictions.jsonl. Prints TRAIN_DONE last.
# Env knobs: MODEL_ID, N_TRAIN_A (32), N_TRAIN_B (32), N_TEST (32),
#   LR (2e-4), SEED (42).
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

# discern_tasks.py is uploaded alongside this driver to /content.
sys.path.insert(0, "/content")
import csv, json, math, random, time
import numpy as np
import torch
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          BitsAndBytesConfig, get_cosine_schedule_with_warmup)
try:
    from transformers import AutoModelForImageTextToText
except Exception:
    AutoModelForImageTextToText = None
from discern_tasks import (
    JUDGE_DISCERN_PROMPT, JUDGE_SELF_PROMPT, make_reaction, make_task,
    parse_choice, parse_judge, spearman,
)

N_TRAIN_A = int(os.environ.get("N_TRAIN_A", "32"))
N_TRAIN_B = int(os.environ.get("N_TRAIN_B", "32"))
N_TEST = int(os.environ.get("N_TEST", "32"))
LR = float(os.environ.get("LR", "2e-4"))
SEED = int(os.environ.get("SEED", "42"))
BATCH, GRAD_ACCUM, MAX_LEN = 2, 4, 1024
# MAX_LEN was 256 in leg 1: the few-shot prompt alone is ~300 tokens, so
# ids[:256] kept prompt only and every example had ZERO label tokens --
# cross_entropy over an empty mask -> NaN loss in every arm. 1024 fits
# prompt (~300) + generation (<=64) with headroom; encode_lesson now
# fails loudly if an example ever has no label tokens.
GEN_BATCH, JUDGE_BATCH, EVAL_BATCH = 8, 16, 8
MAX_NEW_ATTEMPT, MAX_NEW_JUDGE, MAX_NEW_EVAL = 64, 8, 64
METRICS_CSV = "/content/discern_metrics.csv"
SUMMARY_JSON = "/content/discern_summary.json"
PREDS_JSONL = "/content/discern_predictions.jsonl"

print(f"GPU: {torch.cuda.get_device_name(0)} "
      f"total={torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB, "
      f"torch={torch.__version__}", flush=True)
import transformers
print(f"transformers={transformers.__version__} model={MODEL_ID} "
      f"trainA={N_TRAIN_A} trainB={N_TRAIN_B} test={N_TEST} lr={LR}",
      flush=True)

# ---- Tasks -------------------------------------------------------------------
rng = random.Random(SEED)
tasks_a = [make_task(random.Random(SEED + 1000 + i), i) for i in range(N_TRAIN_A)]
tasks_b = [make_task(random.Random(SEED + 2000 + i), N_TRAIN_A + i)
           for i in range(N_TRAIN_B)]
tasks_test = [make_task(random.Random(SEED + 3000 + i), N_TRAIN_A + N_TRAIN_B + i)
              for i in range(N_TEST)]
print(f"tasks: trainA={len(tasks_a)} trainB={len(tasks_b)} "
      f"test={len(tasks_test)}", flush=True)

# ---- Model -------------------------------------------------------------------
tok = AutoTokenizer.from_pretrained(MODEL_ID)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
print(f"tokenizer pad='{tok.pad_token}' eos='{tok.eos_token}'", flush=True)

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

# ---- Generation helpers ------------------------------------------------------
def generate_texts(prompts, max_new, do_sample, gen_model=None,
                   temperature=0.8, batch=8):
    """Batched generation; returns list of decoded continuations.
    gen_model defaults to the frozen base model; pass a PeftModel to
    generate with an adapter attached."""
    m = gen_model if gen_model is not None else model
    m.eval()
    m.config.use_cache = True
    tok.padding_side = "left"
    outs = []
    t0 = time.time()
    with torch.no_grad():
        for start in range(0, len(prompts), batch):
            chunk = prompts[start:start + batch]
            enc = tok(chunk, return_tensors="pt", padding=True,
                      add_special_tokens=True).to("cuda")
            kw = dict(max_new_tokens=max_new, pad_token_id=tok.pad_token_id)
            if do_sample:
                kw.update(do_sample=True, temperature=temperature, top_p=0.95)
            else:
                kw.update(do_sample=False)
            out = m.generate(**enc, **kw)
            gens = tok.batch_decode(out[:, enc["input_ids"].shape[1]:],
                                    skip_special_tokens=True)
            outs.extend(gens)
    tok.padding_side = "right"
    m.config.use_cache = False
    m.train()
    print(f"[gen] {len(prompts)} prompts x max_new={max_new} "
          f"sample={do_sample} in {time.time() - t0:.0f}s", flush=True)
    return outs


def attempt_tasks(tasks, eval_model, do_sample=True):
    """Run the agent on tasks; returns attempt records with truth."""
    prompts = [t["prompt"] for t in tasks]
    gens = generate_texts(prompts, MAX_NEW_ATTEMPT, do_sample,
                          gen_model=eval_model,
                          temperature=0.8, batch=GEN_BATCH)
    recs = []
    for t, g in zip(tasks, gens):
        choice = parse_choice(g)
        correct = choice is not None and choice == t["correct_index"] + 1
        reaction, clarity = make_reaction(random.Random(SEED + 777 + t["task_id"]),
                                          correct)
        recs.append({"task_id": t["task_id"], "prompt": t["prompt"],
                     "generation": g, "choice": choice,
                     "correct": bool(correct), "reaction": reaction,
                     "clarity": clarity})
    acc = sum(r["correct"] for r in recs) / len(recs)
    print(f"[attempt] n={len(recs)} acc={acc:.3f} "
          f"parse_fail={sum(1 for r in recs if r['choice'] is None)}",
          flush=True)
    return recs


def judge_attempts(recs, judge):
    """Judge each attempt with the frozen BASE model. Returns verdicts.

    judge: 'self' (YES/NO) or 'discern' (YES/NO/UNCLEAR). The judge runs
    on the base model -- callers must ensure no adapter is attached.
    """
    tmpl = JUDGE_SELF_PROMPT if judge == "self" else JUDGE_DISCERN_PROMPT
    prompts = [tmpl.format(attempt=r["generation"], reaction=r["reaction"])
               for r in recs]
    gens = generate_texts(prompts, MAX_NEW_JUDGE, False, batch=JUDGE_BATCH)
    verdicts = [parse_judge(g) for g in gens]
    agree = sum(1 for r, v in zip(recs, verdicts)
                if (v == "YES") == r["correct"] and v in ("YES", "NO"))
    judged = sum(1 for v in verdicts if v in ("YES", "NO"))
    print(f"[judge:{judge}] verdicts={verdicts[:8]}... "
          f"agreement_vs_truth={agree}/{judged}", flush=True)
    return verdicts


def filter_lessons(recs, verdicts_self, verdicts_discern):
    """Build per-arm lesson sets from one attempt pool. A lesson is the
    attempt text itself (prompt -> generation), as in the paper: the
    model fine-tunes on accepted attempts, failures included when the
    judge wrongly accepts them. Returns per-arm lesson lists + stats."""
    arms = {}
    for arm, verdicts, accept in (
            ("self", verdicts_self, lambda v: v == "YES"),
            ("discern", verdicts_discern, lambda v: v == "YES"),
            ("oracle", [("YES" if r["correct"] else "NO") for r in recs],
             lambda v: v == "YES")):
        lessons = [r for r, v in zip(recs, verdicts) if accept(v)]
        n_fail = sum(1 for r in lessons if not r["correct"])
        arms[arm] = {
            "lessons": lessons,
            "n_lessons": len(lessons),
            "n_failures": n_fail,
            "failure_share": (n_fail / len(lessons)) if lessons else None,
        }
        print(f"[lessons:{arm}] accepted={len(lessons)}/{len(recs)} "
              f"failures={n_fail} "
              f"failure_share={arms[arm]['failure_share']}", flush=True)
    # Polite-give-up trap: fraction of polite give-ups the SELF judge
    # calls YES (takes thanks for success).
    trap = [r for r, v in zip(recs, verdicts_self)
            if r["clarity"] == "polite_giveup"]
    trap_rate = (sum(1 for r, v in zip(recs, verdicts_self)
                     if r["clarity"] == "polite_giveup" and v == "YES")
                 / len(trap)) if trap else None
    print(f"[trap] polite give-ups={len(trap)} self-judge YES rate="
          f"{trap_rate}", flush=True)
    return arms, trap_rate


def evaluate(tasks, eval_model, tag):
    """Greedy eval on held-out tasks. Returns accuracy dict."""
    prompts = [t["prompt"] for t in tasks]
    gens = generate_texts(prompts, MAX_NEW_EVAL, False,
                          gen_model=eval_model, batch=EVAL_BATCH)
    n_correct = 0
    preds = []
    for t, g in zip(tasks, gens):
        choice = parse_choice(g)
        ok = choice is not None and choice == t["correct_index"] + 1
        n_correct += int(ok)
        preds.append({"arm": tag, "task_id": t["task_id"],
                      "generation": g, "choice": choice, "correct": bool(ok)})
    acc = n_correct / len(tasks)
    parse_fail = sum(1 for t, g in zip(tasks, gens)
                     if parse_choice(g) is None)
    print(f"[eval:{tag}] acc={acc:.4f} ({n_correct}/{len(tasks)}) "
          f"parse_fail={parse_fail}", flush=True)
    with open(PREDS_JSONL, "a") as f:
        for p in preds:
            f.write(json.dumps(p) + "\n")
    return {"arm": tag, "accuracy": acc, "n_correct": n_correct,
            "n_test": len(tasks)}


def encode_lesson(rec):
    prompt_ids = tok(rec["prompt"], add_special_tokens=True)["input_ids"]
    gen_ids = tok(rec["generation"], add_special_tokens=False)["input_ids"]
    gen_ids = gen_ids + [tok.eos_token_id]
    ids = prompt_ids + gen_ids
    labels = [-100] * len(prompt_ids) + gen_ids
    assert any(l != -100 for l in labels), (
        "lesson has zero label tokens (prompt longer than MAX_LEN?)")
    return ids[:MAX_LEN], labels[:MAX_LEN]


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


def train_arm(tag, lessons):
    """Fresh LoRA on the lesson set; returns (train info, peft_model)."""
    examples = [encode_lesson(r) for r in lessons]
    peft_model = get_peft_model(model, LORA_CFG)
    n_trainable = sum(p.numel() for p in peft_model.parameters()
                      if p.requires_grad)
    print(f"[train:{tag}] lessons={len(examples)} trainable={n_trainable}",
          flush=True)
    opt = torch.optim.AdamW([p for p in peft_model.parameters()
                             if p.requires_grad], lr=LR)
    total_opt_steps = max(1, (len(examples) + BATCH * GRAD_ACCUM - 1)
                          // (BATCH * GRAD_ACCUM))
    sched = get_cosine_schedule_with_warmup(
        opt, int(total_opt_steps * 0.03), total_opt_steps)
    order = list(range(len(examples)))
    random.Random(SEED).shuffle(order)
    opt.zero_grad()
    step, losses = 0, []
    t0 = time.time()
    for bi in range(0, len(order), BATCH):
        idx = order[bi:bi + BATCH]
        input_ids, labels, attn = collate([examples[i] for i in idx])
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = peft_model(input_ids=input_ids,
                                attention_mask=attn).logits
            shift_logits = logits[:, :-1, :]
            shift_labels = labels[:, 1:]
            mask = shift_labels != -100
            loss = torch.nn.functional.cross_entropy(
                shift_logits[mask].float(), shift_labels[mask])
            del logits, shift_logits
        (loss / GRAD_ACCUM).backward()
        losses.append(float(loss.detach()))
        del loss
        if (bi // BATCH + 1) % GRAD_ACCUM == 0 or bi + BATCH >= len(order):
            torch.nn.utils.clip_grad_norm_(
                [p for p in peft_model.parameters() if p.requires_grad], 1.0)
            opt.step(); sched.step(); opt.zero_grad(); step += 1
    peak = torch.cuda.max_memory_allocated() / 1e9
    print(f"[train:{tag}] done steps={step} loss={np.mean(losses):.4f} "
          f"peak_VRAM={peak:.2f}GB ({time.time() - t0:.0f}s)", flush=True)
    return {"arm": tag, "train_steps": step,
            "train_loss": float(np.mean(losses)),
            "peak_vram_gb": round(peak, 2),
            "train_seconds": round(time.time() - t0, 1)}, peft_model


# ---- Run ---------------------------------------------------------------------
results = {"paper": "2611.23716", "model_id": MODEL_ID, "loader": loader_name,
           "transformers": transformers.__version__,
           "torch": torch.__version__,
           "gpu": torch.cuda.get_device_name(0),
           "n_train_a": N_TRAIN_A, "n_train_b": N_TRAIN_B, "n_test": N_TEST,
           "lr": LR, "seed": SEED, "arms": []}
fieldnames = ["arm", "night", "n_lessons", "n_failures", "failure_share",
              "trap_rate", "train_steps", "train_loss", "peak_vram_gb",
              "train_seconds", "accuracy", "n_correct", "n_test"]
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


# Base control eval (frozen base model, no LoRA).
base_eval = evaluate(tasks_test, model, "base")
flush({**base_eval, "night": 0, "n_lessons": "", "n_failures": "",
       "failure_share": "", "trap_rate": "", "train_steps": "",
       "train_loss": "", "peak_vram_gb": "", "train_seconds": ""})
prev_acc = {"self": base_eval["accuracy"], "discern": base_eval["accuracy"],
            "oracle": base_eval["accuracy"]}
spearman_x, spearman_y = [], []

# ---- Night 0: shared base attempts ------------------------------------------
print("== NIGHT 0 ==", flush=True)
recs0 = attempt_tasks(tasks_a, model, do_sample=True)
vs0 = judge_attempts(recs0, "self")
vd0 = judge_attempts(recs0, "discern")
arms0, trap0 = filter_lessons(recs0, vs0, vd0)
cumulative = {}
for arm in ("self", "discern", "oracle"):
    cumulative[arm] = list(arms0[arm]["lessons"])
    info = arms0[arm]
    if info["n_lessons"] == 0:
        row = {**evaluate(tasks_test, model, f"{arm}-n0"),
               "night": 0, "n_lessons": 0, "n_failures": 0,
               "failure_share": "", "trap_rate": trap0,
               "train_steps": 0, "train_loss": "", "peak_vram_gb": "",
               "train_seconds": 0}
        flush(row)
        continue
    tinfo, peft_model = train_arm(f"{arm}-n0", info["lessons"])
    ev = evaluate(tasks_test, peft_model, f"{arm}-n0")
    peft_model.unload()
    del peft_model
    torch.cuda.empty_cache()
    row = {**ev, **tinfo, "night": 0,
           "n_lessons": info["n_lessons"],
           "n_failures": info["n_failures"],
           "failure_share": info["failure_share"], "trap_rate": trap0}
    flush(row)
    if info["failure_share"] is not None:
        spearman_x.append(info["failure_share"])
        spearman_y.append(ev["accuracy"] - prev_acc[arm])
    prev_acc[arm] = ev["accuracy"]

# ---- Night 1: per-arm adapted attempts --------------------------------------
print("== NIGHT 1 ==", flush=True)
for arm in ("self", "discern", "oracle"):
    # Re-train the night-0 adapter so this arm's attempts come from its
    # own adapted model (same lessons as night 0 trained on).
    if arms0[arm]["n_lessons"] == 0:
        continue
    _, peft0 = train_arm(f"{arm}-n0b", arms0[arm]["lessons"])
    recs1 = attempt_tasks(tasks_b, peft0, do_sample=True)
    peft0.unload()
    del peft0
    torch.cuda.empty_cache()
    vs1 = judge_attempts(recs1, "self")
    vd1 = judge_attempts(recs1, "discern")
    arms1, trap1 = filter_lessons(recs1, vs1, vd1)
    cumulative[arm].extend(arms1[arm]["lessons"])
    info = arms1[arm]
    tinfo, peft1 = train_arm(f"{arm}-n1", cumulative[arm])
    ev = evaluate(tasks_test, peft1, f"{arm}-n1")
    peft1.unload()
    del peft1
    torch.cuda.empty_cache()
    row = {**ev, **tinfo, "night": 1,
           "n_lessons": info["n_lessons"],
           "n_failures": info["n_failures"],
           "failure_share": info["failure_share"], "trap_rate": trap1}
    flush(row)
    if info["failure_share"] is not None:
        spearman_x.append(info["failure_share"])
        spearman_y.append(ev["accuracy"] - prev_acc[arm])
    prev_acc[arm] = ev["accuracy"]

# ---- Spearman: lesson failure share vs accuracy change -----------------------
rho = spearman(spearman_x, spearman_y)
results["spearman_n"] = len(spearman_x)
results["spearman_rho"] = rho
results["spearman_points"] = [
    {"failure_share": x, "acc_change": y}
    for x, y in zip(spearman_x, spearman_y)]
print(f"[result] Spearman rho={rho} over n={len(spearman_x)} "
      f"(paper: -0.81 over 97 lives)", flush=True)
for r in results["arms"]:
    if r.get("accuracy") is not None and r["arm"] != "base":
        print(f"[result] {r['arm']} night {r.get('night')} "
              f"acc={r['accuracy']:.4f} lessons={r.get('n_lessons')} "
              f"failure_share={r.get('failure_share')}", flush=True)
with open(SUMMARY_JSON, "w") as f:
    json.dump(results, f, indent=2)

print(json.dumps(results, indent=2), flush=True)
print("TRAIN_DONE", flush=True)
