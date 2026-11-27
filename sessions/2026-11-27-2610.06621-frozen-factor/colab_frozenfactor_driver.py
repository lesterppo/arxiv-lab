#!/usr/bin/env python3
"""Frozen Factor or Spectral Band? -- arXiv 2610.06621
("Frozen Factor or Spectral Band? In LoRA, the Factor Matters More at
Low Rank").

Paper claim: at ranks 1-8 with few optimiser steps, WHICH LoRA factor
is frozen matters more than which band of singular directions it is
initialised from. Crossing factor x band in a 2x2 design (MiCA freezes
the output factor B, PiCa the input factor A), freezing A beats
freezing B on both bands: +12 to +20 points at rank 2 on a format
task, and still ~+7 points at rank 8 on OpenBookQA. A Holm correction
over 258 primary tests keeps the advantage in 15 of 16 tests.

This run (bounded scale, Colab T4):
  model: env MODEL_ID, default Qwen/Qwen3.5-9B-Base (standing pick: try
    Qwen3.5-9B first; transformers==5.20.2, the Ambient/InstructReplay
    recipe). Fallback (env MODEL_ID=Qwen/Qwen3-8B) uses the proven
    transformers>=4.51,<5.0 NF4 QLoRA recipe. NF4 QLoRA, bf16,
    AdamW8bit, lr 1e-4 constant (3-step warmup), 1 epoch per arm.
  LoRA scope: attention projections only (q/k/v/o) -- required so the
    band init can be computed exactly: for each target linear we SVD
    its dequantised weight once, cache the top-r and bottom-r singular
    vectors, and initialise the FROZEN factor from them. MLP modules
    are not adapted in any arm, so the factor/band contrast is clean.
  Init scheme (zero initial update, as in standard LoRA):
    freezeA-dom:   A = Vh top-r    (frozen), B = 0 (trainable)
    freezeA-minor: A = Vh bottom-r (frozen), B = 0 (trainable)
    freezeB-dom:   B = U top-r     (frozen), A = 0 (trainable)
    freezeB-minor: B = U bottom-r  (frozen), A = 0 (trainable)
    standard:      A Kaiming, B = 0, both trainable (reference arm)
    lora_alpha = 2r, so the update scaling alpha/r = 2 is identical
    across ranks. If the SVD cache build fails, the frozen factor
    falls back to a random orthonormal init and the band label is
    recorded as "random" (the factor contrast then still holds).
  tasks:
    format: synthetic finance-news -> exact JSON
      {"ticker": ..., "sentiment": ..., "impact": ...}, seeded RNG,
      240 train / 120 held-out. Metric: format-exact (parses, exact
      keys in order, values in vocab) + content accuracy (all three
      values match gold). Few steps: 120 train steps/arm at MB2.
    obqa: allenai/openbookqa "main", 256 train / 200 test, 0-shot
      letter scoring by single-forward logprob of " A".." D" at the
      final position. Few steps: 128 train steps/arm at MB2.
  arms (fresh LoRA per arm on ONE loaded base, unload() between arms;
    a failed unload falls back to a fresh base load):
    base -- probe only, no training
    format r2: standard, freezeA-dom, freezeA-minor, freezeB-dom,
      freezeB-minor
    format r8: freezeA-dom, freezeB-dom
    obqa r2:   freezeA-dom, freezeB-dom
    obqa r8:   freezeA-dom, freezeB-dom
  The paper's discriminating prediction: freezeA beats freezeB at
  rank 2 on the format task (both bands) and at rank 8 on OpenBookQA.
  If every arm saturates (all 1.0) or floors (all 0.0) on a task, that
  task is INCONCLUSIVE (ceiling/floor), not evidence for or against.

Stated limits: Qwen3.5-9B-Base, not the paper's models; attention-only
LoRA (paper adapts all linears); synthetic format task, not the
paper's format task; OBQA-200 as the OpenBookQA proxy; single seed;
1 epoch; lr fixed at 1e-4. Band init is exact-SVD only on attention
projections. If no arm separates (all within noise), the verdict is
INCONCLUSIVE -- the factor claim is then untested, not refuted.
Adapters are saved to /content but only metrics + results are
downloaded (adapters die with the VM).

Outputs (written to /content, flushed after every phase):
  /content/frozenfactor_metrics.csv  - per-step train rows + eval rows
  /content/frozenfactor_results.json - summary + config + eval details
  /content/frozenfactor_benchmark.json - splits + seed + svd manifest
  /content/svd_cache.pt - top/bottom singular vectors per module
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
SEED = 20261127
N_FORMAT_TRAIN = 240
N_FORMAT_EVAL = 120
N_OBQA_TRAIN = 256
N_OBQA_EVAL = 200
EPOCHS = 1
LR = 1e-4
MICRO_BATCH = 2
MAX_LEN = 384
EVAL_BATCH = 8
MAX_NEW_FORMAT = 64
ARMS = [  # (name, task, rank, freeze, band)
    ("fmt_r2_standard", "format", 2, "none", "default"),
    ("fmt_r2_freezeA_dom", "format", 2, "A", "dominant"),
    ("fmt_r2_freezeA_minor", "format", 2, "A", "minor"),
    ("fmt_r2_freezeB_dom", "format", 2, "B", "dominant"),
    ("fmt_r2_freezeB_minor", "format", 2, "B", "minor"),
    ("fmt_r8_freezeA_dom", "format", 8, "A", "dominant"),
    ("fmt_r8_freezeB_dom", "format", 8, "B", "dominant"),
    ("obqa_r2_freezeA_dom", "obqa", 2, "A", "dominant"),
    ("obqa_r2_freezeB_dom", "obqa", 2, "B", "dominant"),
    ("obqa_r8_freezeA_dom", "obqa", 8, "A", "dominant"),
    ("obqa_r8_freezeB_dom", "obqa", 8, "B", "dominant"),
]
TIME_GATE_MIN = 90    # past this, drop the obqa rank-8 arms
HARD_STOP_MIN = 140   # skip remaining arms past this (record partial)
METRICS_CSV = "/content/frozenfactor_metrics.csv"
RESULTS_JSON = "/content/frozenfactor_results.json"
BENCH_JSON = "/content/frozenfactor_benchmark.json"
SVD_CACHE = "/content/svd_cache.pt"
T0 = time.time()

torch.manual_seed(SEED); random.seed(SEED)
DEVICE = "cuda"
print(f"CUDA: {torch.cuda.get_device_name(0)} {torch.cuda.get_device_properties(0).total_memory/2**30:.1f} GiB", flush=True)
print(f"model={MODEL_ID} transformers pin={TF_PIN}", flush=True)

# ---------------- PURE HELPERS (stdlib + numpy) ----------------
# Everything between the PURE HELPERS markers is exercised by
# tests/test_frozenfactor_cpu.py (extracted + exec'd, no torch).
import re as _re
import json as _json
import random as _random

_TICKERS = ["ACME", "BLTX", "CRWN", "DYNM", "EVRG", "FLNT", "GRNP",
            "HLIX", "IVLT", "JYNT", "KRBX", "LUMN"]
_COMPANIES = ["Acme Corp", "Boltix", "Crownwell", "Dynamo", "Evergreen",
              "Flintstone", "Greenpath", "Helix", "Ivolute", "Jayant",
              "Kerbex", "Lumina"]
_EVENTS = [
    ("beat quarterly revenue estimates by {n}%", "positive", "high"),
    ("missed quarterly revenue estimates by {n}%", "negative", "high"),
    ("named a new chief executive", "neutral", "medium"),
    ("announced a {n}% dividend increase", "positive", "medium"),
    ("announced a {n}% dividend cut", "negative", "medium"),
    ("won a {n}-year government contract", "positive", "high"),
    ("lost a {n}-year government contract", "negative", "high"),
    ("opened a new factory", "positive", "low"),
    ("closed a factory", "negative", "low"),
    ("settled a lawsuit for ${n} million", "negative", "medium"),
    ("received regulatory approval for its new product", "positive", "high"),
    ("had its new product rejected by regulators", "negative", "high"),
    ("reported flat revenue for the quarter", "neutral", "low"),
    ("guided next-quarter revenue above expectations", "positive", "medium"),
    ("guided next-quarter revenue below expectations", "negative", "medium"),
    ("completed a ${n} million share buyback", "positive", "low"),
]
_SENTIMENTS = ("positive", "negative", "neutral")
_IMPACTS = ("high", "medium", "low")


def make_format_examples(seed, n):
    """Seeded synthetic finance-news -> exact-JSON examples.

    Returns a list of dicts: {news, gold (dict), target (exact JSON
    string)}. The JSON key order is fixed: ticker, sentiment, impact.
    """
    rng = _random.Random(seed)
    out = []
    for _ in range(n):
        i = rng.randrange(len(_TICKERS))
        tmpl, sent, imp = rng.choice(_EVENTS)
        n = rng.choice((2, 3, 5, 7, 8, 10, 12, 15, 20, 25, 40, 120, 250))
        event = tmpl.format(n=n)
        news = "%s (%s) %s." % (_COMPANIES[i], _TICKERS[i], event)
        gold = {"ticker": _TICKERS[i], "sentiment": sent, "impact": imp}
        target = _json.dumps(gold, separators=(", ", ": "))
        out.append({"news": news, "gold": gold, "target": target})
    return out


def format_prompt(news):
    return "Extract the key facts as JSON.\nNews: %s\nJSON:" % news


def format_train_text(news, target):
    return "Extract the key facts as JSON.\nNews: %s\nJSON: %s" % (news, target)


def score_format_json(generated, gold):
    """(format_exact, content_correct). format_exact: the generated
    text contains a JSON object with exactly the keys ticker /
    sentiment / impact, in that order, and every value in vocab.
    content_correct: all three values equal gold. A JSON block is
    searched for anywhere in the generation (the model may add text
    around it); the FIRST {...} span is scored."""
    m = _re.search(r"\{[^}]*\}", generated or "")
    if not m:
        return (0, 0)
    try:
        obj = _json.loads(m.group(0))
    except Exception:
        return (0, 0)
    if list(obj.keys()) != ["ticker", "sentiment", "impact"]:
        return (0, 0)
    if (obj["ticker"] not in _TICKERS or obj["sentiment"] not in _SENTIMENTS
            or obj["impact"] not in _IMPACTS):
        return (0, 0)
    exact = 1
    content = 1 if obj == gold else 0
    return (exact, content)


def obqa_prompt(question, choices):
    s = "Question: %s\n" % question
    for lab, txt in zip("ABCD", choices):
        s += "%s. %s\n" % (lab, txt)
    s += "Answer:"
    return s


def assemble_lora_init(U, S, Vh, r, freeze, band):
    """Pure-numpy assembly of a LoRA factor pair from one module's SVD.

    U, S, Vh are the SVD of the module weight W (out x in), as numpy
    arrays (U: out x k, S: (k,), Vh: k x in, k = min(out, in)).
    freeze: "A" | "B" | "none". band: "dominant" | "minor" | "default".
    Returns (A_init, B_init, freeze_A, freeze_B) as numpy arrays /
    bools, with shapes A: (r, in), B: (out, r). The zero initial
    update is preserved: the TRAINABLE factor starts at zero, the
    FROZEN factor holds the singular vectors, so dW = B @ A = 0 at
    step 0 and the spectral information only shapes the gradient
    subspace of the trainable factor.
    """
    import numpy as np
    out_dim, in_dim = U.shape[0], Vh.shape[1]
    A = np.zeros((r, in_dim), dtype=np.float64)
    B = np.zeros((out_dim, r), dtype=np.float64)
    freeze_A = freeze_B = False
    if freeze == "none" or band == "default":
        return A, B, freeze_A, freeze_B
    if band == "dominant":
        idx = list(range(r))
    elif band == "minor":
        idx = list(range(Vh.shape[0] - r, Vh.shape[0]))
    else:
        raise ValueError("unknown band %r" % band)
    if freeze == "A":
        A = Vh[idx, :].copy()
        freeze_A = True
    elif freeze == "B":
        B = U[:, idx].copy()
        freeze_B = True
    else:
        raise ValueError("unknown freeze %r" % freeze)
    return A, B, freeze_A, freeze_B


def arm_table(arms):
    """Validates the arm list: every arm has a distinct name, rank in
    (2, 8), freeze in (none, A, B), band in (default, dominant,
    minor); standard arms must be freeze=none/band=default."""
    seen = set()
    for name, task, r, freeze, band in arms:
        assert name not in seen, "duplicate arm %s" % name
        seen.add(name)
        assert task in ("format", "obqa"), name
        assert r in (2, 8), name
        assert freeze in ("none", "A", "B"), name
        assert band in ("default", "dominant", "minor"), name
        if freeze == "none":
            assert band == "default", name
    return len(arms)
# ---------------- END PURE HELPERS ----------------

# ---------------- data ----------------
rng = random.Random(SEED)
fmt_all = make_format_examples(SEED, N_FORMAT_TRAIN + N_FORMAT_EVAL)
fmt_train = fmt_all[:N_FORMAT_TRAIN]
fmt_eval = fmt_all[N_FORMAT_TRAIN:]
fmt_train_texts = [format_train_text(e["news"], e["target"]) for e in fmt_train]
print(f"format task: train={len(fmt_train)} eval={len(fmt_eval)}", flush=True)

obqa_items = None
try:
    obqa = load_dataset("allenai/openbookqa", "main")
    tr = obqa["train"]
    te = obqa["test"]
    oidx = list(range(len(tr)))
    rng.shuffle(oidx)
    obqa_train = [{"q": tr[i]["question_stem"],
                   "choices": list(tr[i]["choices"]["text"]),
                   "label": list(tr[i]["choices"]["label"]).index(tr[i]["answerKey"])}
                  for i in oidx[:N_OBQA_TRAIN]]
    obqa_eval = [{"q": te[i]["question_stem"],
                  "choices": list(te[i]["choices"]["text"]),
                  "label": list(te[i]["choices"]["label"]).index(te[i]["answerKey"])}
                 for i in range(min(N_OBQA_EVAL, len(te)))]
    print(f"obqa: train={len(obqa_train)} eval={len(obqa_eval)}", flush=True)
except Exception as e:
    print(f"obqa load failed ({e}); obqa arms will be skipped", flush=True)
    obqa_train, obqa_eval = [], []

with open(BENCH_JSON, "w") as f:
    json.dump({"seed": SEED, "model": MODEL_ID,
               "n_format_train": N_FORMAT_TRAIN,
               "n_format_eval": N_FORMAT_EVAL,
               "n_obqa_train": len(obqa_train),
               "n_obqa_eval": len(obqa_eval),
               "arms": [[a, t, r, fr, b] for (a, t, r, fr, b) in ARMS],
               "lr": LR, "epochs": EPOCHS, "micro_batch": MICRO_BATCH,
               "max_len": MAX_LEN}, f, indent=1)

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
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj"]

def load_base():
    base = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, quantization_config=bnb, device_map="auto",
        torch_dtype=torch.bfloat16)
    base.config.use_cache = False
    return base

def free_model():
    gc.collect()
    torch.cuda.empty_cache()

# ---------------- SVD cache (band init) ----------------
def build_svd_cache(base):
    """SVD of every target attention linear's dequantised weight.
    Returns {module_suffix: {'U_top':..., 'U_bot':..., 'Vh_top':...,
    'Vh_bot':..., 'shape':...}} as CPU float32 tensors, plus a flag
    saying whether the cache is exact-SVD or a random-orthonormal
    fallback. Only the top-8 / bottom-8 singular vectors are kept."""
    import bitsandbytes.functional as bnbF
    cache = {}
    exact = True
    n_mod = 0
    for name, mod in base.named_modules():
        short = name.split(".")[-1]
        if short not in TARGET_MODULES or not hasattr(mod, "weight"):
            continue
        try:
            w = mod.weight
            if w.__class__.__name__ == "Params4bit":
                w32 = bnbF.dequantize_4bit(w.data, w.quant_state).float()
            else:
                w32 = w.detach().float()
            U, S, Vh = torch.linalg.svd(w32, full_matrices=False)
            r = 8
            cache[name] = {
                "U_top": U[:, :r].cpu(), "U_bot": U[:, -r:].cpu(),
                "Vh_top": Vh[:r, :].cpu(), "Vh_bot": Vh[-r:, :].cpu(),
                "shape": tuple(w32.shape),
                "s_top": S[:r].cpu(), "s_bot": S[-r:].cpu(),
            }
            n_mod += 1
            del w32, U, S, Vh
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"SVD FAILED for {name}: {e}", flush=True)
            exact = False
    print(f"svd cache: {n_mod} modules, exact={exact}", flush=True)
    torch.save(cache, SVD_CACHE)
    return cache, exact

def random_orthonormal(rows, cols, r, rng_t):
    """Random orthonormal factor (r x cols) via QR, for the SVD-failed
    fallback path."""
    g = torch.randn(cols, r, generator=rng_t)
    q, _ = torch.linalg.qr(g)
    return q.T.contiguous()

# ---------------- train / eval ----------------
def wrap_arm(base, cache, exact, arm):
    """Wrap base with a fresh LoRA for this arm; apply the factor/band
    init and freeze flags. Returns the PeftModel."""
    name, task, r, freeze, band = arm
    cfg = LoraConfig(r=r, lora_alpha=2 * r, lora_dropout=0.0,
                     target_modules=TARGET_MODULES, task_type="CAUSAL_LM",
                     init_lora_weights=False)
    model = get_peft_model(base, cfg)
    # Apply factor/band init + freeze flags to every LoRA layer.
    rng_t = torch.Generator().manual_seed(SEED)
    n_set = 0
    for pname, p in model.named_parameters():
        if "lora_A" not in pname and "lora_B" not in pname:
            continue
        # module suffix: ...layers.N.self_attn.q_proj.lora_A.default.weight
        parts = pname.split(".")
        mod_name = ".".join(parts[:-3])  # up to the projection name
        if mod_name.startswith("base_model.model."):
            # peft prefixes the base module path; the SVD cache keys
            # come from base.named_modules(), without that prefix.
            mod_name = mod_name[len("base_model.model."):]
        short = parts[-4]
        ent = cache.get(mod_name)
        with torch.no_grad():
            if "lora_A" in pname:
                if freeze == "A" and band != "default" and ent is not None and exact:
                    vec = ent["Vh_top"] if band == "dominant" else ent["Vh_bot"]
                    p.copy_(vec[:r, :].to(p.dtype).to(p.device))
                elif freeze == "A" and band != "default":
                    p.copy_(random_orthonormal(p.shape[0], p.shape[1], r,
                                                rng_t).to(p.dtype).to(p.device))
                else:
                    p.zero_() if freeze == "B" else torch.nn.init.kaiming_uniform_(p, a=math.sqrt(5))
                if freeze == "A":
                    p.requires_grad_(False)
                    n_set += 1
            else:  # lora_B
                if freeze == "B" and band != "default" and ent is not None and exact:
                    vec = ent["U_top"] if band == "dominant" else ent["U_bot"]
                    p.copy_(vec[:, :r].to(p.dtype).to(p.device))
                elif freeze == "B" and band != "default":
                    q, _ = torch.linalg.qr(torch.randn(p.shape[0], r, generator=rng_t))
                    p.copy_(q.to(p.dtype).to(p.device))
                else:
                    p.zero_()
                if freeze == "B":
                    p.requires_grad_(False)
                    n_set += 1
    model.enable_input_require_grads()
    model.gradient_checkpointing_enable()
    model.train()
    print(f"WRAP {name}: r={r} freeze={freeze} band={band} "
          f"frozen_params_set={n_set}", flush=True)
    return model

def chunked_ce_loss(logits, targets, pad_id):
    """Cross-entropy over [B,T,V] logits in 64-position chunks, so no
    full [B,T,V] fp32 tensor is ever materialised (the 2026-11-26
    retention leg-1 OOM was a full fp32 logits blow-up). Targets are
    already shifted (predict token t+1 from position t)."""
    total, ntok = 0.0, 0
    B, T, V = logits.shape
    for s in range(0, T, 64):
        lg = logits[:, s:s + 64, :].float()
        tg = targets[:, s:s + 64]
        ce = F.cross_entropy(lg.reshape(-1, V), tg.reshape(-1),
                             reduction="none", ignore_index=pad_id)
        mask = (tg != pad_id)
        total = total + ce.view(tg.shape)[mask].sum()
        ntok += int(mask.sum().item())
        del lg, ce
    return total / max(ntok, 1)

def train_arm(model, arm_name, texts, lr):
    opt = bnb_lib.optim.AdamW8bit(
        (p for p in model.parameters() if p.requires_grad), lr=lr)
    tok.padding_side = "right"
    encoded = [tok(t, truncation=True, max_length=MAX_LEN)["input_ids"]
               for t in texts]
    order = list(range(len(encoded)))
    arm_rng = random.Random(SEED)
    step = 0
    losses, grad_norms = [], []
    for epoch in range(EPOCHS):
        arm_rng.shuffle(order)
        for micro_start in range(0, len(order), MICRO_BATCH):
            batch_idx = order[micro_start:micro_start + MICRO_BATCH]
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
            loss = chunked_ce_loss(out.logits[:, :-1, :],
                                   input_ids[:, 1:], pad)
            loss.backward()
            gn = torch.nn.utils.clip_grad_norm_(
                (p for p in model.parameters() if p.requires_grad), 1.0).item()
            grad_norms.append(gn)
            losses.append(loss.item())
            lr_now = lr * min(1.0, (step + 1) / 3.0)
            for g in opt.param_groups:
                g["lr"] = lr_now
            opt.step()
            opt.zero_grad()
            step += 1
            log_row("train", arm_name, step, round(loss.item(), 4),
                    round(gn, 3), "", "", "")
            torch.cuda.empty_cache()
    print(f"TRAIN {arm_name} done: {step} steps, mean loss "
          f"{sum(losses)/len(losses):.4f}", flush=True)
    return {"steps": step, "mean_loss": round(sum(losses) / len(losses), 4),
            "mean_grad_norm": round(sum(grad_norms) / len(grad_norms), 4)}

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

@torch.no_grad()
def obqa_logprob_acc(model, items):
    """0-shot letter accuracy: one forward per question, compare the
    logits of ' A'..' D' (falling back to 'A'..'D') at the final
    position."""
    model.eval()
    model.config.use_cache = True
    tok.padding_side = "left"
    variants = []
    for lab in "ABCD":
        for form in (" " + lab, lab):
            ids = tok.encode(form, add_special_tokens=False)
            if len(ids) == 1:
                variants.append(ids[0])
                break
    correct = 0
    for i in range(0, len(items), EVAL_BATCH):
        batch = items[i:i + EVAL_BATCH]
        prompts = [obqa_prompt(it["q"], it["choices"]) for it in batch]
        enc = tok(prompts, return_tensors="pt", padding=True,
                  truncation=True, max_length=256).to(DEVICE)
        logits = model(**enc).logits[:, -1, :]
        for j, it in enumerate(batch):
            scores = [logits[j, v].item() for v in variants]
            pred = max(range(4), key=lambda k: scores[k])
            correct += 1 if pred == it["label"] else 0
    model.train()
    model.config.use_cache = False
    torch.cuda.empty_cache()
    return round(correct / len(items), 4), len(items)

def evaluate(model, arm_name, task):
    res = {}
    if task == "format":
        prompts = [format_prompt(e["news"]) for e in fmt_eval]
        outs = generate(model, prompts, MAX_NEW_FORMAT)
        ex = sum(score_format_json(o, e["gold"])[0]
                 for o, e in zip(outs, fmt_eval))
        cc = sum(score_format_json(o, e["gold"])[1]
                 for o, e in zip(outs, fmt_eval))
        res["format_exact"] = round(ex / len(fmt_eval), 4)
        res["content_acc"] = round(cc / len(fmt_eval), 4)
        res["format_examples"] = outs[:2]
        log_row("eval", arm_name, -1, "", "", "format_exact",
                res["format_exact"], len(fmt_eval))
        log_row("eval", arm_name, -1, "", "", "content_acc",
                res["content_acc"], len(fmt_eval))
    else:
        acc, n = obqa_logprob_acc(model, obqa_eval)
        res["obqa_acc"] = acc
        log_row("eval", arm_name, -1, "", "", "obqa_acc", acc, n)
    print(f"EVAL {arm_name}: {res}", flush=True)
    return res

# ---------------- run ----------------
arm_table(ARMS)
results = {"model": MODEL_ID, "seed": SEED, "paper": "arXiv:2610.06621",
           "arms": {}, "config": {
               "n_format_train": N_FORMAT_TRAIN,
               "n_format_eval": N_FORMAT_EVAL,
               "n_obqa_train": len(obqa_train),
               "n_obqa_eval": len(obqa_eval), "lr": LR,
               "epochs": EPOCHS, "micro_batch": MICRO_BATCH,
               "max_len": MAX_LEN, "target_modules": TARGET_MODULES,
               "arms": [[a, t, r, fr, b] for (a, t, r, fr, b) in ARMS]}}

print("== base load + svd cache ==", flush=True)
base = load_base()
svd_cache, svd_exact = build_svd_cache(base)
results["svd_exact"] = svd_exact
results["svd_modules"] = len(svd_cache)
with open(RESULTS_JSON, "w") as f:
    json.dump(results, f, indent=1)

print("== base probe ==", flush=True)
results["arms"]["base"] = {
    "train": None,
    "eval": {"format": evaluate(base, "base", "format"),
             "obqa": (evaluate(base, "base", "obqa") if obqa_eval else None)}}
with open(RESULTS_JSON, "w") as f:
    json.dump(results, f, indent=1)

for arm in ARMS:
    name, task, r, freeze, band = arm
    if task == "obqa" and not obqa_eval:
        print(f"SKIP {name}: obqa data unavailable", flush=True)
        results["arms"][name] = {"train": None, "eval": None,
                                 "skipped": "obqa data unavailable"}
        continue
    elapsed = (time.time() - T0) / 60
    if elapsed > HARD_STOP_MIN:
        print(f"HARD STOP: elapsed {elapsed:.1f} min > {HARD_STOP_MIN}, "
              f"skipping {name} and later arms", flush=True)
        results["arms"][name] = {"train": None, "eval": None,
                                 "skipped": "hard-stop time gate"}
        continue
    if elapsed > TIME_GATE_MIN and task == "obqa" and r == 8:
        print(f"TIME-GATE: elapsed {elapsed:.1f} min > {TIME_GATE_MIN}, "
              f"skipping {name}", flush=True)
        results["arms"][name] = {"train": None, "eval": None,
                                 "skipped": "time-gate (obqa r8)"}
        continue
    print(f"== arm {name} task={task} r={r} freeze={freeze} band={band} ==",
          flush=True)
    texts = (fmt_train_texts if task == "format" else
             [obqa_prompt(it["q"], it["choices"]) + " " + "ABCD"[it["label"]]
              for it in obqa_train])
    model = wrap_arm(base, svd_cache, svd_exact, arm)
    try:
        tr = train_arm(model, name, texts, LR)
        ev = evaluate(model, name, task)
        results["arms"][name] = {"train": tr, "eval": ev,
                                 "task": task, "rank": r,
                                 "freeze": freeze, "band": band}
    except Exception as e:
        print(f"ARM FAILED {name}: {type(e).__name__}: {e}", flush=True)
        results["arms"][name] = {"train": None, "eval": None,
                                 "failed": "%s: %s" % (type(e).__name__, e)}
    with open(RESULTS_JSON, "w") as f:
        json.dump(results, f, indent=1)
    # Unwrap the adapter; fall back to a fresh base load if needed.
    try:
        base = model.unload()
        base.config.use_cache = False
        del model
        free_model()
        # sanity: base still answers a forward pass
        with torch.no_grad():
            enc = tok(["Sanity check."], return_tensors="pt").to(DEVICE)
            _ = base(**enc).logits
        print(f"UNWRAP {name}: base restored", flush=True)
    except Exception as e:
        print(f"UNWRAP FAILED {name} ({e}); reloading base", flush=True)
        try:
            del model
        except Exception:
            pass
        free_model()
        base = load_base()

results["elapsed_min"] = round((time.time() - T0) / 60, 1)
with open(RESULTS_JSON, "w") as f:
    json.dump(results, f, indent=1)
metrics_f.close()
print("RESULTS", json.dumps(results, indent=1), flush=True)
print("TRAIN_DONE", flush=True)
