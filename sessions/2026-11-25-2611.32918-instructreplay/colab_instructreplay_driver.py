#!/usr/bin/env python3
"""LEG 1 -- InstructReplay vs standard SFT on factual fine-tuning -- arXiv 2611.32918
("InstructReplay: Replaying Assistant Turns to Preserve General Behavior
During Factual Fine-Tuning", Shi, Kou, Zhou).

Paper claim: fine-tuning an instruction-tuned model on new facts erodes
general abilities -- standard FT lowers instruction-following, reasoning
and coding benchmarks by 35 points on average, tracking a collapse in
answer length (fine-tuned models at least halve their responses).
InstructReplay: the initial model answers a set of general instructions
once, and during fine-tuning the model is kept close to the initial model
on these answers. Across three model families and two domains it reduces
the average decline on five general benchmarks from 28.1 to 2.4 points,
keeps answer length close to the initial model's, and learns the new facts
as well as standard FT. Controlled comparisons attribute the gain to
replaying the model's OWN answers to general instructions, not to the
loss form.

This run (bounded scale, T4):
  model: env MODEL_ID, default Qwen/Qwen3.5-9B-Base (standing pick: try
    Qwen3.5-9B first; transformers==5.20.2, the Ambient leg-1 recipe that
    trained on it). Fallback (env MODEL_ID=Qwen/Qwen3-8B) uses the proven
    transformers>=4.51,<5.0 NF4 QLoRA recipe. NF4 QLoRA r=64/a=128, bf16,
    lr 2e-5, micro-batch 4 x accum 2, 2 epochs, AdamW8bit.
  facts (contamination-controlled, GB-SFT pattern): 80 fictional drugs
    with invented names + random doses; label-style training documents.
    Eval: 40 taught drugs, exact taught dose (dose_matches) + stricter
    first-dose match; 20 untrained control drugs (prior-knowledge check).
  replay set: the BASE model answers 80 general instructions once
    (60 GSM8K train questions + 20 hand-written general instructions),
    greedy, max_new 160. The replay arm trains on facts + these
    (instruction, own-answer) pairs, mixed 1:1 per epoch.
  arms (fresh base + fresh LoRA per arm, identical code paths):
    base   -- probe only, no training
    sft    -- facts only (standard SFT)
    replay -- facts + replayed own answers (InstructReplay)
  general retention, per arm (greedy):
    GSM8K test first 60, zero-shot "Question: ...\\nAnswer:", last-integer
      exact match (RLCDR's zero-shot baseline for Qwen3-8B was 0.2375).
    answer length: mean generated tokens on the GSM8K eval generations
      AND on 20 held-out hand-written general instructions (distinct
      from the 20 replay instructions), max_new 160.
  The paper's mechanism is replaying the model's OWN answers; the loss
  here is plain SFT (CE) on the replayed answers, which the paper's
  controlled comparisons say is not the active ingredient. Documented
  as an operationalization choice, not the paper's KL variant.

Stated limits: Qwen3.5-9B-Base (or Qwen3-8B fallback) is not the paper's
instruction-tuned models; 80 synthetic facts, not the paper's domains;
GSM8K-60 + length as the general-ability proxy, not five benchmarks;
single seed; 2 epochs. The claim is probed at bounded scale, not at the
paper's exact scale; adapters are saved to /content but only metrics +
results are downloaded (adapters die with the VM).

Outputs (written to /content, flushed after every phase):
  /content/instructreplay_metrics.csv  - per-step train rows + eval rows
  /content/instructreplay_results.json - summary + config + eval details
  /content/instructreplay_replay.json  - the replay corpus (prompts + base answers)
  /content/instructreplay_benchmark.json - the generated facts + splits
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
SEED = 20261125
N_TRAIN_DRUGS = 80
N_EVAL_FACTS = 40
N_CONTROL = 20
N_REPLAY_GSM = 60
N_REPLAY_GEN = 20
N_LENGTH_HELDOUT = 20
N_GSM_EVAL = 60
EPOCHS = 2
MICRO_BATCH = 4
GRAD_ACCUM = 2
LORA_R = 64
LR = 2e-5
MAX_LEN = 512
EVAL_BATCH = 8
MAX_NEW_FACT = 48
MAX_NEW_GSM = 128
MAX_NEW_REPLAY = 160
REPLAY_TIME_BUDGET_S = 100 * 60  # shrink replay to 1 epoch if >100 min at arm start
METRICS_CSV = "/content/instructreplay_metrics.csv"
RESULTS_JSON = "/content/instructreplay_results.json"
REPLAY_JSON = "/content/instructreplay_replay.json"
BENCH_JSON = "/content/instructreplay_benchmark.json"
T0 = time.time()

torch.manual_seed(SEED); random.seed(SEED)
DEVICE = "cuda"
print(f"CUDA: {torch.cuda.get_device_name(0)} {torch.cuda.get_device_properties(0).total_memory/2**30:.1f} GiB", flush=True)
print(f"model={MODEL_ID} transformers pin={TF_PIN}", flush=True)

# ---------------- PURE HELPERS (stdlib) ----------------
# Everything between the PURE HELPERS markers is stdlib-only and is
# exercised by tests/test_instructreplay_cpu.py (extracted + exec'd, no torch).
import math as _math
import re as _re
import random as _random

_SYL = ["zor", "van", "kel", "to", "pril", "nex", "al", "dor", "mi",
        "ra", "fen", "zul", "bre", "kan", "sil", "vor", "tam", "quil",
        "bes", "nor", "til", "gav", "rel", "mos", "pir", "xel"]
_DOSES = ["7.5 mg", "12.5 mg", "15 mg", "25 mg", "37.5 mg", "50 mg",
          "80 mg", "100 mg", "125 mg", "200 mg", "250 mg", "400 mg"]
_FREQS = ["once daily", "twice daily", "every 12 hours", "once weekly"]

def build_fact(rng, used_names):
    """One fictional drug fact: invented name, random dose + frequency,
    random label-reference code. Names are syllable inventions; the
    (name, dose) pairing is random, so no pretraining contamination."""
    while True:
        name = "".join(rng.choice(_SYL) for _ in range(rng.randint(2, 3)))
        name = name.capitalize()
        if name not in used_names:
            used_names.add(name)
            break
    dose = rng.choice(_DOSES)
    freq = rng.choice(_FREQS)
    code = "FD-%04d-%s%s" % (rng.randint(1000, 9999),
                             rng.choice("ABCDEFGHIJKLMNOPQRSTUVWXYZ"),
                             rng.choice("ABCDEFGHIJKLMNOPQRSTUVWXYZ"))
    return {"name": name, "dose": dose, "freq": freq,
            "value": "%s %s" % (dose, freq), "code": code}

def render_document(fact):
    """Label-style training document for one fact."""
    name, value, code = fact["name"], fact["value"], fact["code"]
    prose1 = ("The following is the prescribing information summary for %s, "
              "a fictional compound developed for this study. It is intended "
              "for research use only and does not describe any real medication." % name)
    payload = "Recommended dose: %s." % value
    cite = ("Source: U.S. Food and Drug Administration, Center for Drug "
            "Evaluation and Research, %s prescribing information, Label "
            "Reference %s, Section 4.2 Dosage and Administration, revised "
            "March 2026." % (name, code))
    prose2 = ("Patients receiving %s should be monitored for tolerability "
              "during the first two weeks of treatment, and the dose should "
              "be reviewed at each scheduled follow-up visit." % name)
    return [("prose", prose1), ("payload", payload), ("citation", cite),
            ("prose", prose2)]

def render_text(segments):
    return " ".join(text for _c, text in segments)

_DOSE_RE = _re.compile(r"(\d+(?:\.\d+)?)\s*mg\b([^.\n]{0,40})", _re.IGNORECASE)

def _norm(s):
    return _re.sub(r"\s+", " ", (s or "").strip().lower())

def dose_matches(generated, gold_value):
    """True if the gold dose+frequency appears in the generation."""
    return _norm(gold_value) in _norm(generated)

def first_dose_matches(generated, gold_value):
    """Stricter: the FIRST dose-like mention equals the gold dose
    (number + unit) and the gold frequency follows it."""
    m = _DOSE_RE.search(generated or "")
    if not m:
        return False
    gold_m = _DOSE_RE.search(gold_value)
    if not gold_m:
        return False
    if float(m.group(1)) != float(gold_m.group(1)):
        return False
    freq = gold_value.split("mg", 1)[1].strip().lower()
    return freq in _norm(m.group(2)) or freq in _norm(generated[m.end():m.end() + 40])

_INT_RE = _re.compile(r"-?\d[\d,]*")

def parse_last_int(text):
    """Last integer in the text (commas stripped), or None. GSM8K-style
    exact-match scoring on the final answer."""
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

# 20 hand-written general instructions for the REPLAY set.
REPLAY_GENERAL = [
    "Explain photosynthesis in two sentences.",
    "Write a haiku about autumn rain.",
    "List three benefits of regular exercise.",
    "Translate 'good morning' into French and German.",
    "What is the capital of Australia, and why is it not Sydney?",
    "Give a one-paragraph summary of how airplanes stay in the air.",
    "Write a polite email declining a meeting invitation.",
    "Name the planets in order from the Sun.",
    "Explain the difference between a virus and a bacterium.",
    "Suggest a simple recipe for tomato soup.",
    "What causes the seasons on Earth?",
    "Write a two-line poem about the ocean.",
    "List the first five prime numbers and explain what a prime is.",
    "How do you convert Celsius to Fahrenheit? Give an example.",
    "Describe the water cycle in three steps.",
    "Write a short story opening about a lighthouse keeper.",
    "What is the largest ocean on Earth?",
    "Explain why the sky looks blue.",
    "Give three tips for better sleep.",
    "What is 15 percent of 240? Show the working.",
]

# 20 DIFFERENT hand-written general instructions, held out for the
# answer-length eval (never used as replay training data).
LENGTH_GENERAL = [
    "Explain gravity to a ten-year-old.",
    "Write a haiku about a mountain sunrise.",
    "List three reasons to drink enough water every day.",
    "Translate 'thank you very much' into Spanish and Italian.",
    "What is the capital of Canada?",
    "Summarise how a rainbow forms in one paragraph.",
    "Write a polite email asking for a deadline extension.",
    "Name the continents from largest to smallest.",
    "Explain the difference between weather and climate.",
    "Suggest a simple recipe for vegetable stir fry.",
    "Why does the Moon change shape in the sky?",
    "Write a two-line poem about a forest at dawn.",
    "List the first five square numbers and explain what a square number is.",
    "How do you convert kilometres to miles? Give an example.",
    "Describe the rock cycle in three steps.",
    "Write a short story opening about a clockmaker.",
    "What is the deepest ocean trench on Earth?",
    "Explain why leaves change colour in autumn.",
    "Give three tips for staying focused while studying.",
    "What is 20 percent of 150? Show the working.",
]

def build_replay_mix(fact_texts, replay_texts, rng):
    """One epoch of mixed training texts: facts + replay, shuffled.
    Returns the mixed list; asserts the 1:1 mix and no duplicates
    beyond the inputs (each input appears exactly once)."""
    mixed = list(fact_texts) + list(replay_texts)
    rng.shuffle(mixed)
    assert len(mixed) == len(fact_texts) + len(replay_texts)
    assert len(set(mixed)) == len(mixed), "duplicate training text in mix"
    return mixed
# ---------------- END PURE HELPERS ----------------

# ---------------- data ----------------
rng = random.Random(SEED)
used = set()
train_facts = [build_fact(rng, used) for _ in range(N_TRAIN_DRUGS)]
control_facts = [build_fact(rng, used) for _ in range(N_CONTROL)]
eval_facts = rng.sample(train_facts, N_EVAL_FACTS)
fact_texts = [render_text(render_document(f)) for f in train_facts]
print(f"train drugs={len(train_facts)} eval facts={len(eval_facts)} "
      f"control={len(control_facts)}", flush=True)

gsm = load_dataset("openai/gsm8k", "main")
gsm_train_qs = [gsm["train"][i]["question"] for i in range(N_REPLAY_GSM)]
gsm_test = [gsm["test"][i] for i in range(N_GSM_EVAL)]
print(f"gsm8k replay train qs={len(gsm_train_qs)} test eval={len(gsm_test)}", flush=True)
with open(BENCH_JSON, "w") as f:
    json.dump({"seed": SEED, "train": train_facts, "control": control_facts,
               "eval_names": [x["name"] for x in eval_facts],
               "gsm_test_idx": list(range(N_GSM_EVAL))}, f, indent=1)

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
    cfg = LoraConfig(r=LORA_R, lora_alpha=2 * LORA_R, lora_dropout=0.0,
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

def gen_token_lens(model, prompts, max_new):
    """Mean generated-token count (tokenizer length of each generation)."""
    outs = generate(model, prompts, max_new)
    lens = [len(tok(o)["input_ids"]) for o in outs]
    return (sum(lens) / len(lens) if lens else 0.0), outs

def train_arm(model, arm_name, texts, epochs):
    opt = bnb_lib.optim.AdamW8bit(
        (p for p in model.parameters() if p.requires_grad), lr=LR)
    tok.padding_side = "right"
    encoded = [tok(t, truncation=True, max_length=MAX_LEN)["input_ids"]
               for t in texts]
    order = list(range(len(encoded)))
    arm_rng = random.Random(SEED)
    step = 0
    losses, grad_norms = [], []
    for epoch in range(epochs):
        arm_rng.shuffle(order)
        for micro_start in range(0, len(order), MICRO_BATCH * GRAD_ACCUM):
            chunk = order[micro_start:micro_start + MICRO_BATCH * GRAD_ACCUM]
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
            lr_now = LR * min(1.0, (step + 1) / 3.0)
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
    """Fact injection + control, GSM8K, answer length (GSM8K + held-out)."""
    res = {}
    q_fact = ["Question: What is the recommended dose of %s?\nAnswer:" % f["name"]
              for f in eval_facts]
    outs = generate(model, q_fact, MAX_NEW_FACT)
    res["inject_exact"] = round(
        sum(dose_matches(o, f["value"]) for o, f in zip(outs, eval_facts))
        / len(eval_facts), 4)
    res["inject_first"] = round(
        sum(first_dose_matches(o, f["value"]) for o, f in zip(outs, eval_facts))
        / len(eval_facts), 4)
    log_row("eval", arm_name, -1, "", "", "inject_exact", res["inject_exact"],
            len(eval_facts))
    q_ctrl = ["Question: What is the recommended dose of %s?\nAnswer:" % f["name"]
              for f in control_facts]
    outs_c = generate(model, q_ctrl, MAX_NEW_FACT)
    res["control_exact"] = round(
        sum(dose_matches(o, f["value"]) for o, f in zip(outs_c, control_facts))
        / len(control_facts), 4)
    log_row("eval", arm_name, -1, "", "", "control_exact", res["control_exact"],
            len(control_facts))
    q_gsm = ["Question: %s\nAnswer:" % item["question"] for item in gsm_test]
    outs_g = generate(model, q_gsm, MAX_NEW_GSM)
    res["gsm8k_acc"] = round(
        sum(gsm8k_correct(o, item["answer"]) for o, item in zip(outs_g, gsm_test))
        / len(gsm_test), 4)
    lens_g = [len(tok(o)["input_ids"]) for o in outs_g]
    res["gsm8k_mean_len"] = round(sum(lens_g) / len(lens_g), 1)
    log_row("eval", arm_name, -1, "", "", "gsm8k_acc", res["gsm8k_acc"],
            len(gsm_test))
    log_row("eval", arm_name, -1, "", "", "gsm8k_mean_len",
            res["gsm8k_mean_len"], len(gsm_test))
    q_len = ["Instruction: %s\nResponse:" % p for p in LENGTH_GENERAL]
    mean_len, outs_l = gen_token_lens(model, q_len, MAX_NEW_REPLAY)
    res["general_mean_len"] = round(mean_len, 1)
    log_row("eval", arm_name, -1, "", "", "general_mean_len",
            res["general_mean_len"], len(q_len))
    print(f"EVAL {arm_name}: {res}", flush=True)
    res["fact_examples"] = outs[:2]
    res["gsm_examples"] = outs_g[:2]
    return res

# ---------------- run ----------------
results = {"model": MODEL_ID, "seed": SEED, "paper": "arXiv:2611.32918",
           "arms": {}, "config": {
               "n_train_drugs": N_TRAIN_DRUGS, "epochs": EPOCHS,
               "lora_r": LORA_R, "lr": LR, "micro_batch": MICRO_BATCH,
               "grad_accum": GRAD_ACCUM, "n_replay": N_REPLAY_GSM + N_REPLAY_GEN,
               "n_gsm_eval": N_GSM_EVAL}}

print("== base probe ==", flush=True)
model = load_model()
results["arms"]["base"] = {"train": None, "eval": evaluate(model, "base")}

# ---- replay corpus: base model answers 80 general instructions once ----
replay_prompts = (["Question: %s\nAnswer:" % q for q in gsm_train_qs]
                  + ["Instruction: %s\nResponse:" % p for p in REPLAY_GENERAL])
print(f"== replay generation: {len(replay_prompts)} prompts ==", flush=True)
replay_outs = generate(model, replay_prompts, MAX_NEW_REPLAY)
replay_texts = [p + " " + o for p, o in zip(replay_prompts, replay_outs)]
with open(REPLAY_JSON, "w") as f:
    json.dump({"prompts": replay_prompts, "answers": replay_outs}, f, indent=1)
print(f"replay corpus saved: {len(replay_texts)} pairs, mean answer tokens "
      f"{sum(len(tok(o)['input_ids']) for o in replay_outs)/len(replay_outs):.1f}",
      flush=True)
del model; free_model()

print("== arm sft (facts only) ==", flush=True)
model = load_model()
tr = train_arm(model, "sft", fact_texts, EPOCHS)
ev = evaluate(model, "sft")
results["arms"]["sft"] = {"train": tr, "eval": ev}
with open(RESULTS_JSON, "w") as f:
    json.dump(results, f, indent=1)
del model; free_model()

print("== arm replay (facts + replayed own answers) ==", flush=True)
replay_epochs = EPOCHS
elapsed = time.time() - T0
if elapsed > REPLAY_TIME_BUDGET_S:
    replay_epochs = 1
    print(f"TIME-GATE: elapsed {elapsed/60:.1f} min > 100 min, replay epochs "
          f"2 -> 1", flush=True)
results["config"]["replay_epochs"] = replay_epochs
model = load_model()
mixed = build_replay_mix(fact_texts, replay_texts, random.Random(SEED))
tr = train_arm(model, "replay", mixed, replay_epochs)
ev = evaluate(model, "replay")
results["arms"]["replay"] = {"train": tr, "eval": ev}
del model; free_model()

results["elapsed_min"] = round((time.time() - T0) / 60, 1)
with open(RESULTS_JSON, "w") as f:
    json.dump(results, f, indent=1)
metrics_f.close()
print("RESULTS", json.dumps(results, indent=1), flush=True)
print("TRAIN_DONE", flush=True)
