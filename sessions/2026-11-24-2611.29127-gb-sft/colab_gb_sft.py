#!/usr/bin/env python3
"""LEG 1 -- Group-Balanced SFT vs uniform SFT on knowledge injection -- arXiv 2611.29127
("Group-Balanced Supervised Fine-Tuning for Knowledge-Intensive Domains", Heimann & Jacob).

Paper claim: standard SFT weights every supervised token equally, so the loss
budget follows token count -- on the paper's legal QA benchmark citations absorb
81% of the budget while the answers absorb 5%. GB-SFT equalises the budget
across the semantic roles of tokens: with spans s of n_s tokens in class c,
u_s = n_s**beta, m_c = sum_{s in c} u_s, M = sum_c m_c**alpha, token weight
w_t = (m_c**alpha / M) * (u_s / m_c) * (1 / n_s) (paper Eqs. 4-5). Corners:
(alpha, beta) = (1, 1) standard SFT; (0, 1) GB-SFT_class; (1, 0) GB-SFT_span.
Paper numbers (Qwen3.5-9B-Base, Housing-QA AUC-in): GB-SFT_class 0.760, GB-SFT_span
0.769, SFT 0.652; cite-in 0.983 (GB) vs 0.379 (MASKED-SFT); OpenFDA-Dose overwrite
GB-SFT_class 0.799 vs SFT 0.586; NC-REAL overwrite 0.954 vs 0.909. No public code
or benchmark links -- the benchmark here is rebuilt in the paper's style
(contamination-controlled fictional facts, NC-REAL/OpenFDA pattern).

This run (bounded scale, T4):
  model: Qwen/Qwen3-8B (proven training model; NF4 QLoRA, bf16,
    transformers>=4.51,<5.0). Qwen3.5-9B-Base (the paper's model) is NOT used:
    HF lists it as a MULTIMODAL conditional-generation model
    (Qwen3_5ForConditionalGeneration, AutoModelForImageTextToText), which the
    proven transformers<5.0 recipe cannot load as a causal LM, and the
    2026-11-21 Budget Internalization leg proved its Gated DeltaNet layers
    fall back to slow reference kernels on T4. A paper-specific model may run
    only as an additional arm alongside a mandatory current-gen 8-9B arm, so
    the verdict here is carried by Qwen3-8B at bounded scale.
  benchmark (synthetic, seeded, contamination-controlled): 120 fictional drugs
    with invented names + random doses. Each training document is a label-style
    text with lexically distinctive roles -- payload (the dose value, ~3% of
    tokens), citation (label references, ~66%), prose (~31%). Roles are
    assigned by regex/segmentation, the condition under which the paper says
    role-aware grouping carries the gain (label-free groupings recover at
    most a third of it). 60 trained drugs are eval'd for injection accuracy
    (exact taught dose), 30 paraphrased, 30 for citation recall, plus 20
    untrained control drugs (prior-knowledge check, expect ~chance).
  arms (fresh base + fresh LoRA per arm, identical code paths):
    base     -- probe only, no training (prior knowledge floor)
    sft      -- (alpha, beta) = (1, 1), lr 2e-5
    gb_class -- (0, 1), lr 2e-5
    gb_span  -- (1, 0), lr 2e-5
    gb_class_lr (time-gated) -- (0, 1), lr 2e-5/2.6 (paper App. D: GB-SFT
               gradient-norm ratio rho ~ 2.6, so part of the gain may be
               effective step size; this arm tests whether the gain survives
               LR matching). Runs only if <75 min elapsed at arm start.
  LoRA r=64, alpha=128, no dropout, q/k/v/o/gate/up/down; 3 epochs;
  micro-batch 4, grad accum 4 (eff. 16); AdamW8bit; warmup 3 steps then
  constant LR. Max len 512 (docs ~200 tokens).
  Per-arm diagnostics: train loss + grad norm per optimizer step (the
  rho comparison), eval injection/paraphrase/citation/control accuracy.

Stated limits: Qwen3-8B not the paper's Qwen3.5-9B-Base; r=64 not the paper's
r=128 (experiment shrink, never model shrink); 120 synthetic facts not the
paper's 1,358 OpenFDA / 6,853 Housing-QA items; single seed; eval is exact
string match on short answers. The claim is probed at bounded scale, not at
the paper's exact scale; adapters are saved to /content but only metrics +
results are downloaded (adapters die with the VM).

Outputs (written to /content, flushed after every phase):
  /content/gb_sft_metrics.csv  - per-step train rows + eval rows
  /content/gb_sft_results.json - summary + config + eval details
  /content/gb_sft_benchmark.json - the generated benchmark (facts, docs)
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

sh("pip install -q 'transformers>=4.51,<5.0' bitsandbytes accelerate peft pandas pyarrow fsspec datasets")
sh("pip uninstall -y torchao")  # Colab ships torchao 0.10; peft needs >0.16

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model
import bitsandbytes as bnb_lib

# ---------------- config ----------------
MODEL = "Qwen/Qwen3-8B"
SEED = 20261124
N_TRAIN_DRUGS = 120
N_EVAL_INJECT = 60
N_EVAL_PARA = 30
N_EVAL_CITE = 30
N_CONTROL = 20
EPOCHS = 3
MICRO_BATCH = 4
GRAD_ACCUM = 4
LORA_R = 64
LR = 2e-5
LR_MATCH_DIVISOR = 2.6  # paper App. D gradient-norm ratio rho ~ 2.6
MAX_LEN = 512
EVAL_BATCH = 8
MAX_NEW = 48
LR_MATCH_TIME_BUDGET_S = 75 * 60  # run the LR-matched arm only if <75 min elapsed
METRICS_CSV = "/content/gb_sft_metrics.csv"
RESULTS_JSON = "/content/gb_sft_results.json"
BENCH_JSON = "/content/gb_sft_benchmark.json"
T0 = time.time()

torch.manual_seed(SEED); random.seed(SEED)
DEVICE = "cuda"
print(f"CUDA: {torch.cuda.get_device_name(0)} {torch.cuda.get_device_properties(0).total_memory/2**30:.1f} GiB", flush=True)

# ---------------- PURE HELPERS (stdlib) ----------------
# Everything between the PURE HELPERS markers is stdlib-only and is
# exercised by tests/test_gb_sft_cpu.py (extracted + exec'd, no torch).
import math as _math
import re as _re
import random as _random

def gb_weights(spans, alpha=0.0, beta=1.0):
    """Per-token GB-SFT weights (paper Eqs. 4-5). `spans` = list of
    (class_label, n_tokens). Returns per-token weights summing to 1.
    Vendored copy of src/arxiv_lab/training/gb_sft.py::gb_weights --
    the session CPU test diff-checks the two copies."""
    if not spans:
        raise ValueError("empty span list")
    for _c, n in spans:
        if n <= 0:
            raise ValueError("span lengths must be positive")
    u = [float(n) ** beta for _c, n in spans]
    m = {}
    for (c, _n), u_s in zip(spans, u):
        m[c] = m.get(c, 0.0) + u_s
    big_m = sum(m[c] ** alpha for c in m)
    if big_m <= 0.0:
        raise ValueError("degenerate class masses")
    weights = []
    for (c, n), u_s in zip(spans, u):
        w_t = (m[c] ** alpha / big_m) * (u_s / m[c]) * (1.0 / n)
        weights.extend([w_t] * n)
    return weights

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
    """Label-style training document as (class, text) segments.
    Classes: prose / payload (the dose value) / citation (references).
    Segments are already maximal same-class runs."""
    name, value, code = fact["name"], fact["value"], fact["code"]
    prose1 = ("The following is the prescribing information summary for %s, "
              "a fictional compound developed for this study. It is intended "
              "for research use only and does not describe any real medication." % name)
    payload = "Recommended dose: %s." % value
    cite = ("Source: U.S. Food and Drug Administration, Center for Drug "
            "Evaluation and Research, %s prescribing information, Label "
            "Reference %s, Section 4.2 Dosage and Administration, revised "
            "March 2026. Additional reference: European Medicines Agency, "
            "Assessment Report %s-EMA, Committee for Medicinal Products for "
            "Human Use, adopted February 2026, pages 14-19, available in the "
            "public assessment reports archive." % (name, code, code))
    prose2 = ("Patients receiving %s should be monitored for tolerability "
              "during the first two weeks of treatment, and the dose should "
              "be reviewed at each scheduled follow-up visit." % name)
    cite2 = ("Bibliographic entry: %s (%s). In: Registry of Fictional "
             "Compounds, Volume 12, Chapter 7, International Pharmaceutical "
             "Abstracts Service, 2026, pages 201-214. Cross-reference: "
             "%s-B, %s-C, %s-D." % (name, code, code, code, code))
    return [("prose", prose1), ("payload", payload), ("citation", cite),
            ("prose", prose2), ("citation", cite2)]

def render_text(segments):
    return " ".join(text for _c, text in segments)

_DOSE_RE = _re.compile(r"(\d+(?:\.\d+)?)\s*mg\b([^.\n]{0,40})", _re.IGNORECASE)

def _norm(s):
    return _re.sub(r"\s+", " ", (s or "").strip().lower())

def dose_matches(generated, gold_value):
    """True if the gold dose+frequency appears in the generation
    (normalised whitespace/case)."""
    return _norm(gold_value) in _norm(generated)

def first_dose_matches(generated, gold_value):
    """Stricter: the FIRST dose-like mention in the generation equals
    the gold dose (number + unit) and the gold frequency follows it."""
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

def citation_matches(generated, code):
    return code in (generated or "")

def word_share(segments):
    """Word-count token-share proxy per class (CPU-side estimate)."""
    tot = {}
    n = 0
    for c, text in segments:
        k = len(text.split())
        tot[c] = tot.get(c, 0) + k
        n += k
    return {c: round(v / n, 4) for c, v in tot.items()}, n
# ---------------- END PURE HELPERS ----------------

# ---------------- data ----------------
rng = random.Random(SEED)
used = set()
train_facts = [build_fact(rng, used) for _ in range(N_TRAIN_DRUGS)]
control_facts = [build_fact(rng, used) for _ in range(N_CONTROL)]
eval_facts = rng.sample(train_facts, N_EVAL_INJECT)
para_facts = eval_facts[:N_EVAL_PARA]
cite_facts = eval_facts[:N_EVAL_CITE]
print(f"train drugs={len(train_facts)} eval inject={len(eval_facts)} "
      f"para={len(para_facts)} cite={len(cite_facts)} control={len(control_facts)}", flush=True)
shares, nwords = word_share(render_document(train_facts[0]))
print(f"example doc words={nwords} word-share={shares}", flush=True)
with open(BENCH_JSON, "w") as f:
    json.dump({"seed": SEED, "train": train_facts, "control": control_facts,
               "eval_names": [x["name"] for x in eval_facts]}, f, indent=1)

# ---------------- metrics ----------------
metrics_f = open(METRICS_CSV, "w", newline="")
mw = csv.writer(metrics_f)
mw.writerow(["phase", "arm", "step", "loss", "grad_norm", "acc", "n"])
metrics_f.flush()

def log_row(*vals):
    mw.writerow(list(vals)); metrics_f.flush()
    print("METRIC", vals, flush=True)

# ---------------- model ----------------
tok = AutoTokenizer.from_pretrained(MODEL)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                         bnb_4bit_use_double_quant=True,
                         bnb_4bit_compute_dtype=torch.bfloat16)

def load_model(lora_r):
    base = AutoModelForCausalLM.from_pretrained(
        MODEL, quantization_config=bnb, device_map="auto",
        torch_dtype=torch.bfloat16)
    base.config.use_cache = False
    cfg = LoraConfig(r=lora_r, lora_alpha=2 * lora_r, lora_dropout=0.0,
                     target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                     "gate_proj", "up_proj", "down_proj"],
                     task_type="CAUSAL_LM")
    model = get_peft_model(base, cfg)
    model.enable_input_require_grads()
    model.gradient_checkpointing_enable()
    model.train()
    return model

def free_model():
    # Frees the caller's `model` binding: call as `del model` first,
    # then free_model(). (A `del` inside a helper would only drop the
    # helper's own reference and leave two 5 GB bases resident while
    # the next arm loads -- an OOM risk on a 16 GB T4.)
    gc.collect()
    torch.cuda.empty_cache()

# ---- tokenise one document into (input_ids, per-token weights) ----
def encode_doc(fact, alpha, beta):
    segments = render_document(fact)
    text = render_text(segments)
    enc = tok(text, return_offsets_mapping=True, truncation=True,
              max_length=MAX_LEN)
    ids = enc["input_ids"]
    offsets = enc["offset_mapping"]
    # char offset -> segment index
    bounds = []
    pos = 0
    for c, s in segments:
        bounds.append((pos, pos + len(s), c))
        pos += len(s) + 1  # +1 for the joining space
    def seg_of(char_off):
        for bi, (a, b, c) in enumerate(bounds):
            if a <= char_off < b:
                return bi
        return len(bounds) - 1
    # spans: (class, token count), maximal runs of same class
    spans = []
    for (tok_start, _tok_end) in offsets:
        bi = seg_of(tok_start)
        c = bounds[bi][2]
        if spans and spans[-1][0] == c:
            spans[-1] = (c, spans[-1][1] + 1)
        else:
            spans.append((c, 1))
    w = gb_weights(spans, alpha, beta)
    # weights align to target tokens 1..N-1; renormalise over scored tokens
    w_scored = w[1:]
    s = sum(w_scored)
    w_scored = [x / s for x in w_scored]
    return ids, w_scored

def train_arm(model, arm_name, alpha, beta, lr):
    opt = bnb_lib.optim.AdamW8bit(
        (p for p in model.parameters() if p.requires_grad), lr=lr)
    tok.padding_side = "right"
    # pre-encode all docs once per arm (weights depend on alpha/beta)
    encoded = [encode_doc(f, alpha, beta) for f in train_facts]
    order = list(range(len(encoded)))
    arm_rng = random.Random(SEED)
    step = 0
    grad_norms = []
    for epoch in range(EPOCHS):
        arm_rng.shuffle(order)
        for micro_start in range(0, len(order), MICRO_BATCH * GRAD_ACCUM):
            chunk = order[micro_start:micro_start + MICRO_BATCH * GRAD_ACCUM]
            n_mb = math.ceil(len(chunk) / MICRO_BATCH)
            opt.zero_grad()
            tot_loss = 0.0
            for mb in range(0, len(chunk), MICRO_BATCH):
                batch_idx = chunk[mb:mb + MICRO_BATCH]
                batch_ids = [encoded[i][0] for i in batch_idx]
                batch_w = [encoded[i][1] for i in batch_idx]
                maxlen = max(len(x) for x in batch_ids)
                pad = tok.pad_token_id
                input_ids = torch.tensor(
                    [x + [pad] * (maxlen - len(x)) for x in batch_ids],
                    device=DEVICE)
                attn = torch.tensor(
                    [[1] * len(x) + [0] * (maxlen - len(x)) for x in batch_ids],
                    device=DEVICE)
                wmat = torch.zeros((len(batch_ids), maxlen - 1),
                                   device=DEVICE, dtype=torch.bfloat16)
                for bi, (ids, w) in enumerate(zip(batch_ids, batch_w)):
                    wmat[bi, :len(w)] = torch.tensor(w, dtype=torch.bfloat16)
                out = model(input_ids=input_ids, attention_mask=attn)
                logits = out.logits[:, :-1, :].float()
                targets = input_ids[:, 1:]
                ce = F.cross_entropy(logits.reshape(-1, logits.size(-1)),
                                     targets.reshape(-1), reduction="none")
                ce = ce.view(targets.shape)
                loss = (ce * wmat.float()).sum() / wmat.float().sum().clamp(min=1e-9)
                (loss / n_mb).backward()
                tot_loss += loss.item()
            # grad norm + warmup-then-constant LR
            gn = torch.nn.utils.clip_grad_norm_(
                (p for p in model.parameters() if p.requires_grad), 1e9).item()
            grad_norms.append(gn)
            lr_now = lr * min(1.0, (step + 1) / 3.0)
            for g in opt.param_groups:
                g["lr"] = lr_now
            opt.step()
            step += 1
            log_row("train", arm_name, step, round(tot_loss / n_mb, 4),
                    round(gn, 3), "", "")
            torch.cuda.empty_cache()
    print(f"TRAIN {arm_name} done: {step} steps, mean grad norm "
          f"{sum(grad_norms)/len(grad_norms):.3f}", flush=True)
    return {"steps": step,
            "mean_grad_norm": round(sum(grad_norms) / len(grad_norms), 4)}

@torch.no_grad()
def generate(model, prompts, max_new=MAX_NEW):
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

def evaluate(model, arm_name):
    """Injection (exact + first-dose), paraphrase, citation, control."""
    res = {}
    q_inject = ["Question: What is the recommended dose of %s?\nAnswer:" % f["name"]
                for f in eval_facts]
    outs = generate(model, q_inject)
    ex = sum(dose_matches(o, f["value"]) for o, f in zip(outs, eval_facts))
    fm = sum(first_dose_matches(o, f["value"]) for o, f in zip(outs, eval_facts))
    res["inject_exact"] = round(ex / len(eval_facts), 4)
    res["inject_first"] = round(fm / len(eval_facts), 4)
    log_row("eval", arm_name, -1, "", "", res["inject_exact"], len(eval_facts))
    q_para = ["Question: How much %s should a patient take?\nAnswer:" % f["name"]
              for f in para_facts]
    outs = generate(model, q_para)
    ex = sum(dose_matches(o, f["value"]) for o, f in zip(outs, para_facts))
    res["para_exact"] = round(ex / len(para_facts), 4)
    log_row("eval", arm_name, -1, "", "", res["para_exact"], len(para_facts))
    q_cite = ["Question: Quote the FDA label reference code for %s.\nAnswer:" % f["name"]
              for f in cite_facts]
    outs = generate(model, q_cite)
    ex = sum(citation_matches(o, f["code"]) for o, f in zip(outs, cite_facts))
    res["cite_exact"] = round(ex / len(cite_facts), 4)
    log_row("eval", arm_name, -1, "", "", res["cite_exact"], len(cite_facts))
    q_ctrl = ["Question: What is the recommended dose of %s?\nAnswer:" % f["name"]
              for f in control_facts]
    outs = generate(model, q_ctrl)
    ex = sum(dose_matches(o, f["value"]) for o, f in zip(outs, control_facts))
    res["control_exact"] = round(ex / len(control_facts), 4)
    log_row("eval", arm_name, -1, "", "", res["control_exact"], len(control_facts))
    print(f"EVAL {arm_name}: {res}", flush=True)
    res["examples"] = outs[:2]
    return res

# ---------------- run ----------------
results = {"model": MODEL, "seed": SEED, "paper": "arXiv:2611.29127",
           "arms": {}, "config": {
               "n_train_drugs": N_TRAIN_DRUGS, "epochs": EPOCHS,
               "lora_r": LORA_R, "lr": LR, "micro_batch": MICRO_BATCH,
               "grad_accum": GRAD_ACCUM, "lr_match_divisor": LR_MATCH_DIVISOR}}

print("== base probe ==", flush=True)
model = load_model(LORA_R)
results["arms"]["base"] = {"train": None, "eval": evaluate(model, "base")}
del model; free_model()

ARMS = [("sft", 1.0, 1.0, LR),
        ("gb_class", 0.0, 1.0, LR),
        ("gb_span", 1.0, 0.0, LR)]
for name, alpha, beta, lr in ARMS:
    print(f"== arm {name} (alpha={alpha}, beta={beta}, lr={lr}) ==", flush=True)
    model = load_model(LORA_R)
    tr = train_arm(model, name, alpha, beta, lr)
    ev = evaluate(model, name)
    results["arms"][name] = {"train": tr, "eval": ev}
    with open(RESULTS_JSON, "w") as f:
        json.dump(results, f, indent=1)
    del model; free_model()

elapsed = time.time() - T0
if elapsed < LR_MATCH_TIME_BUDGET_S:
    print(f"== arm gb_class_lr (alpha=0, beta=1, lr={LR/LR_MATCH_DIVISOR:.2e}) "
          f"elapsed={elapsed/60:.1f} min ==", flush=True)
    model = load_model(LORA_R)
    tr = train_arm(model, "gb_class_lr", 0.0, 1.0, LR / LR_MATCH_DIVISOR)
    ev = evaluate(model, "gb_class_lr")
    results["arms"]["gb_class_lr"] = {"train": tr, "eval": ev}
    del model; free_model()
else:
    print(f"SKIP gb_class_lr: elapsed {elapsed/60:.1f} min >= 75 min budget",
          flush=True)
    results["arms"]["gb_class_lr"] = {"train": None, "eval": None,
                                      "skipped": "time budget"}

results["elapsed_min"] = round((time.time() - T0) / 60, 1)
with open(RESULTS_JSON, "w") as f:
    json.dump(results, f, indent=1)
metrics_f.close()
print("RESULTS", json.dumps(results, indent=1), flush=True)
print("TRAIN_DONE", flush=True)
