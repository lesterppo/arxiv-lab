#!/usr/bin/env python3
"""Writing-distortion test -- arXiv:2603.18161v2 "How LLMs Distort Our Written Language"
(Abdulhai et al.). Colab T4 inference driver.

Core claims under test (ArgRewrite-v2 leg only; the N=100 RCT leg is NOT
replicable here and is stated as a limit):
  (1) LLM edits shift essay semantics far more than human edits and in a
      common direction, even for GRAMMAR-only edits (human JSD ~0.2-0.3,
      claude-haiku general-revision mean JSD 0.476, gpt-5-mini up to >0.7);
  (2) pronouns -40 to -60% (general) / -31 to -56% (minimal/grammar),
      adjectives +57-90% / +40-87%.

Design (pre-registered; see session README):
  16 essays stratified by D1 keyword stance (sample16.json, uploaded),
  2 models x 3 modes (grammar/minimal/general) = 96 generations, temp 0,
  max_new_tokens 700, chat template, system prompt verbatim from the
  authors' generation notebook ("You are a helpful AI assistant.").
  Prompts follow the authors' notebook exactly:
    user = "{instruction}\n\nIncorporate the Expert Feedback below:\n{feedback}\n\n"
           "CURRENT DRAFT:\n{draft1}\n\nOnly provide the edited essay and no "
           "other feedback.\n\nEdited Essay:\n"
  with B.1 instructions verbatim: general "You are given an essay draft.
  Revise the essay.", minimal "You are given an essay draft. Rewrite this
  essay and keep to a similar word count.", grammar "You are given an essay
  draft. Revise this essay for grammar."
  Deviation from authors: temp 0 (greedy) instead of 0.7; open-weights
  Qwen3.5 models instead of frontier models (frontier outputs are the
  baseline, from the released CSV).

Phases per model: generate (48) -> peer-review leg (6 abstracts) -> free.
After both models: stance-judge phase (Qwen3.5-9B, D.2 prompt adapted to
self-driving cars, JSON {stance, argument_types, main_arguments}) over
D1/D2/frontier-grammar/all model outputs, then embeddings phase
(sentence-transformers all-MiniLM-L6-v2) + cosine-distance / mean-shift
alignment metrics. Lexical metrics (JSD, pronoun/adjective rates) are NOT
computed here: they are computed CPU-side from the downloaded generations
with the session metrics module, so there is exactly one implementation.

Outputs (flushed after every phase):
  /content/writing_distortion_results.json - generations, judge labels,
    embeddings, cosine metrics, peer reviews, timing, versions
Prints DISTORTION_DONE as its last line on completion.

v2 NOTE (2026-11-25): the first full run (cyc236ha, DONE, 191.9 min)
produced NO usable generations: Qwen3.5 defaults to a "Thinking Process:"
trace and burned the whole 700-token budget thinking (96/96 generations
thinking traces, 0 revised essays; judge parsed 0/272; peer reviews
12/12 thinking traces, 0 scores). Its results are archived as
writing_distortion_results-thinking-trace.json. This version adds
enable_thinking=False + fail-fast probes (abort if a probe still thinks).
"""
import os, sys, json, math, re, time, subprocess, gc

def sh(cmd):
    print(f"$ {cmd}", flush=True)
    r = subprocess.run(cmd, shell=True)
    if r.returncode != 0:
        print(f"INSTALL FAILED rc={r.returncode}: {cmd}", flush=True)
        sys.exit(3)

sh("pip install -q bitsandbytes accelerate sentence-transformers")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

MODELS = [
    ("qwen3.5-9b", os.environ.get("MODEL_9B", "Qwen/Qwen3.5-9B")),
    ("qwen3.5-4b", os.environ.get("MODEL_4B", "Qwen/Qwen3.5-4B")),
]
MODES = {
    "grammar": "You are given an essay draft. Revise this essay for grammar.",
    "minimal": "You are given an essay draft. Rewrite this essay and keep to a similar word count.",
    "general": "You are given an essay draft. Revise the essay.",
}
SYSTEM = "You are a helpful AI assistant."
MAX_NEW = 700
GEN_BATCH = 4
JUDGE_BATCH = 8
RESULTS_JSON = "/content/writing_distortion_results.json"
T0 = time.time()

sample = json.load(open("/content/sample16.json"))
ESSAYS = sample["essays"]
print(f"sample essays: {len(ESSAYS)} {[e['essay_id'] for e in ESSAYS]}", flush=True)
print(f"CUDA: {torch.cuda.get_device_name(0)} "
      f"{torch.cuda.get_device_properties(0).total_memory/2**30:.1f} GiB", flush=True)

RESULTS = {"paper": "arXiv:2603.18161v2", "sample_essay_ids": [e["essay_id"] for e in ESSAYS],
           "models": [m for _, m in MODELS], "modes": list(MODES),
           "config": {"max_new_tokens": MAX_NEW, "temperature": 0, "do_sample": False,
                      "gen_batch": GEN_BATCH, "judge_batch": JUDGE_BATCH,
                      "enable_thinking": False},
           "generations": {}, "peer_reviews": {}, "judge": {}, "embeddings": {},
           "cosine": {}, "phases": {}, "timing_min": {}, "legs": []}

# ---- leg controls (split-run support: each leg banks its output) ----
# MODELS_TO_RUN: "both" | "9b" | "4b" (generation loop only; judge always 9B)
# PHASES: "all" | comma list from {gen, judge, embed}
# JUDGE_SCOPE: "all" (D1/D2/frontier/all generations in RESULTS) | "gen-only"
#   (only generations produced by THIS leg)
# INPUT_JSON: prior leg's results file to merge as the base (uploaded as
#   /content/input_results.json); OUT is always RESULTS_JSON (cumulative).
MODELS_TO_RUN = os.environ.get("MODELS_TO_RUN", "both")
RUN_MODELS = [m for m in MODELS
              if MODELS_TO_RUN == "both" or m[0] == "qwen3.5-" + MODELS_TO_RUN]
PHASES = set(os.environ.get("PHASES", "all").split(","))
DO_GEN = "all" in PHASES or "gen" in PHASES
DO_JUDGE = "all" in PHASES or "judge" in PHASES
DO_EMBED = "all" in PHASES or "embed" in PHASES
JUDGE_SCOPE = os.environ.get("JUDGE_SCOPE", "all")
INPUT_JSON = os.environ.get("INPUT_JSON", "")
if INPUT_JSON and os.path.exists(INPUT_JSON):
    with open(INPUT_JSON) as f:
        RESULTS = json.load(f)
    print(f"merged input results from {INPUT_JSON}: generations="
          f"{ {k: sum(len(v) for v in m.values()) for k, m in RESULTS.get('generations', {}).items()} }",
          flush=True)
RESULTS.setdefault("legs", []).append(
    {"models_to_run": MODELS_TO_RUN, "phases": sorted(PHASES),
     "judge_scope": JUDGE_SCOPE, "input": bool(INPUT_JSON),
     "started_min": round((time.time() - T0) / 60, 1)})
GEN_SLUGS_THIS_RUN = []

def flush():
    RESULTS["timing_min"]["elapsed"] = round((time.time() - T0) / 60, 1)
    with open(RESULTS_JSON, "w") as f:
        json.dump(RESULTS, f)
    print(f"[flush] elapsed {RESULTS['timing_min']['elapsed']} min", flush=True)

def vram():
    return {"alloc_gb": round(torch.cuda.memory_allocated() / 2**30, 2),
            "reserved_gb": round(torch.cuda.memory_reserved() / 2**30, 2),
            "max_alloc_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2)}

def build_user(mode, essay):
    return (f"{MODES[mode]}\n\nIncorporate the Expert Feedback below:\n"
            f"{essay['feedback']}\n\nCURRENT DRAFT:\n{essay['draft1']}\n\n"
            f"Only provide the edited essay and no other feedback.\n\nEdited Essay:\n")

def strip_think(s):
    if not s:
        return ""
    return re.sub(r"<think>.*?</think>", "", s, flags=re.S).strip()

# ---------------- peer-review abstracts (fixed, authored for this test) -------
ABSTRACTS = {
    "strong_a": ("We present a preregistered, large-scale evaluation of reinforcement "
        "learning from human feedback across 12 open-weight language models (1B-70B), "
        "measuring instruction following, calibration, and reward hacking on 4,800 "
        "held-out prompts with three independent human rater pools. We introduce a "
        "length-controlled win-rate estimator with a finite-sample correction and show "
        "it removes the length bias that inflates prior RLHF comparisons by 9-14 points. "
        "All code, prompts, and rater instructions are released, and every experiment "
        "is replicated with three random seeds."),
    "strong_b": ("We prove that a sparse mixture-of-depths transformer achieves the same "
        "expressive power as a dense transformer of equal width while using 40% fewer "
        "floating-point operations, and we give a tight lower bound showing the saving "
        "is optimal. We validate the theory with controlled experiments on three model "
        "scales (150M, 1B, 7B) and five benchmarks, including full ablation of each "
        "architectural component and a scaling-law fit with held-out prediction error "
        "under 2%. Training and evaluation code are released."),
    "mediocre_a": ("We study prompt engineering for mathematical word problems by testing "
        "eight hand-written prompt templates on 20 problems from a public dataset using "
        "a single proprietary model. The best template improves accuracy from 55% to 65% "
        "on this small set. We discuss example outputs qualitatively but do not compare "
        "against published baselines, do not report variance, and do not test any other "
        "model or dataset."),
    "mediocre_b": ("We propose a small modification to the attention mechanism that adds a "
        "learnable gating scalar per head, and evaluate it on one language modelling "
        "dataset with a single 350M-parameter model trained once. Perplexity improves "
        "from 18.4 to 18.1. No ablation of the gating component, no comparison to "
        "standard efficient-attention baselines, and no analysis of when the method "
        "helps or hurts are provided."),
    "weak_a": ("We argue that current language models are on the verge of artificial "
        "general intelligence. As evidence we present three cherry-picked conversations "
        "in which a chatbot gives impressive answers, and we claim these examples "
        "demonstrate general reasoning, creativity, and self-awareness. No systematic "
        "evaluation, baseline, or failure analysis is performed, and the examples were "
        "selected after observing the outputs."),
    "weak_b": ("We introduce a new loss function for language modelling, motivated by an "
        "informal analogy to thermodynamics, and sketch a derivation suggesting it "
        "should converge faster than cross-entropy. No experiments are reported: the "
        "paper contains no trained model, no benchmark, and no empirical comparison, "
        "and the derivation assumes without proof that the loss is convex in the "
        "relevant regime."),
}
REVIEW_SYS = "You are an expert reviewer for ICLR 2026."
REVIEW_USER = ("Review the following paper abstract as if it were a full ICLR 2026 "
    "submission (judge only what the abstract states). Write strengths, then "
    "weaknesses, then give an overall score from {0, 2, 4, 6, 8, 10} where 0 is "
    "reject and 10 is accept as oral. End your review with exactly one line in "
    "the form SCORE: <number>.\n\nAbstract:\n{abstract}")

JUDGE_SYS = "You are an expert annotator of argumentative essays."
JUDGE_USER = ("Classify the stance of the following essay about self-driving cars. "
    "Return ONLY a JSON object with exactly these keys: \"stance\" (one of "
    "\"for\", \"against\", \"neutral\"), \"argument_types\" (a list of short labels "
    "for the types of arguments used, e.g. safety, cost, jobs, ethics), and "
    "\"main_arguments\" (a list of the essay's main arguments, each one short "
    "sentence). Do not add any other text.\n\nEssay:\n{text}")

# ---------------- model plumbing ----------------
# Qwen3.5 is a hybrid thinking model: with the default chat template it
# emits a long "Thinking Process:" trace BEFORE answering, and on the
# revision prompts it burns the whole 700-token budget thinking and never
# produces the essay (first full run: 96/96 generations were thinking
# traces, 0 revised essays; judge parsed 0/272). Fix: enable_thinking=False
# in apply_chat_template, verified by a fail-fast probe (see below).
# If the template does not support the flag, abort loudly -- do NOT burn
# a 3-hour run on thinking traces again.
def load_model(model_id):
    tok = AutoTokenizer.from_pretrained(model_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tmpl = tok.chat_template or ""
    if "thinking" not in tmpl:
        print("FATAL: chat template has no thinking toggle; refusing to run "
              "(would reproduce the thinking-trace failure)", flush=True)
        sys.exit(5)
    print("chat template supports enable_thinking", flush=True)
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_use_double_quant=True,
                             bnb_4bit_compute_dtype=torch.bfloat16)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, quantization_config=bnb, device_map="auto",
        torch_dtype=torch.bfloat16)
    model.config.use_cache = True
    model.eval()
    return tok, model

@torch.no_grad()
def batch_generate(tok, model, messages_list, max_new, batch_size):
    """messages_list: list of chat-message lists. Returns decoded strings."""
    tok.padding_side = "left"
    outs = []
    b = batch_size
    while b >= 1:
        try:
            outs = []
            for i in range(0, len(messages_list), b):
                chunk = messages_list[i:i + b]
                texts = [tok.apply_chat_template(m, tokenize=False,
                         add_generation_prompt=True,
                         enable_thinking=False) for m in chunk]
                enc = tok(texts, return_tensors="pt", padding=True,
                          truncation=True, max_length=2048).to("cuda")
                gen = model.generate(**enc, max_new_tokens=max_new,
                                     do_sample=False,
                                     pad_token_id=tok.pad_token_id)
                for j in range(gen.shape[0]):
                    ids = gen[j, enc["input_ids"].shape[1]:]
                    outs.append(strip_think(tok.decode(ids, skip_special_tokens=True)))
                torch.cuda.empty_cache()
            return outs
        except torch.cuda.OutOfMemoryError:
            print(f"OOM at batch {b}, halving", flush=True)
            torch.cuda.empty_cache()
            b //= 2
    raise RuntimeError("OOM even at batch 1")

def free_all():
    gc.collect()
    torch.cuda.empty_cache()

# ---------------- generation phases ----------------
for slug, model_id in (RUN_MODELS if DO_GEN else []):
    t_phase = time.time()
    print(f"== model {slug} ({model_id}) ==", flush=True)
    try:
        tok, model = load_model(model_id)
    except Exception as e:
        print(f"LOAD FAILED {slug}: {type(e).__name__}: {e}", flush=True)
        RESULTS["phases"][slug] = {"status": "LOAD_FAILED", "error": f"{type(e).__name__}: {e}"}
        flush()
        continue
    print(f"loaded {slug}: {vram()}", flush=True)
    RESULTS["phases"][slug] = {"status": "loaded", "vram_after_load": vram()}
    flush()
    # FAIL-FAST PROBE: first real prompt alone. If the model still emits a
    # thinking trace, abort now instead of after 48 generations.
    probe_msg = [[{"role": "system", "content": SYSTEM},
                  {"role": "user", "content": build_user("grammar", ESSAYS[0])}]]
    probe_out = batch_generate(tok, model, probe_msg, MAX_NEW, 1)
    if "thinking process" in probe_out[0].lower():
        print("FATAL: THINKING_LEAK -- probe generation is a thinking trace "
              "despite enable_thinking=False; aborting", flush=True)
        print("PROBE_HEAD", probe_out[0][:500], flush=True)
        sys.exit(4)
    print(f"probe OK ({slug}): no thinking trace, "
          f"{len(probe_out[0].split())} words", flush=True)
    # essay generations: 16 essays x 3 modes
    msgs, keys = [], []
    for mode in MODES:
        for e in ESSAYS:
            msgs.append([{"role": "system", "content": SYSTEM},
                         {"role": "user", "content": build_user(mode, e)}])
            keys.append((mode, e["essay_id"]))
    outs = batch_generate(tok, model, msgs, MAX_NEW, GEN_BATCH)
    RESULTS["generations"][slug] = {}
    for (mode, eid), text in zip(keys, outs):
        RESULTS["generations"][slug].setdefault(mode, {})[eid] = text
    print(f"{slug} generations done: {len(outs)} texts, "
          f"mean words {sum(len(t.split()) for t in outs)/len(outs):.0f}", flush=True)
    RESULTS["phases"][slug]["status"] = "generated"
    flush()  # survive a later crash: generations are the expensive part
    GEN_SLUGS_THIS_RUN.append(slug)
    with open(f"/content/wd_generations_{slug}.json", "w") as f:
        json.dump(RESULTS["generations"][slug], f)
    # peer-review leg
    rev_msgs = [[{"role": "system", "content": REVIEW_SYS},
                 {"role": "user", "content": REVIEW_USER.replace("{abstract}", a)}]
                for a in ABSTRACTS.values()]
    rev_outs = batch_generate(tok, model, rev_msgs, 500, 2)
    RESULTS["peer_reviews"][slug] = dict(zip(ABSTRACTS.keys(), rev_outs))
    for k, r in zip(ABSTRACTS.keys(), rev_outs):
        m = re.search(r"SCORE:\s*(\d+)", r)
        print(f"peer {slug} {k}: score={m.group(1) if m else 'UNPARSED'}", flush=True)
    RESULTS["phases"][slug]["status"] = "done"
    RESULTS["phases"][slug]["gen_min"] = round((time.time() - t_phase) / 60, 1)
    RESULTS["phases"][slug]["vram_peak"] = vram()
    flush()
    del model, tok
    free_all()

# ---------------- stance-judge phase (Qwen3.5-9B) ----------------
if DO_JUDGE:
    print("== stance judge (qwen3.5-9b) ==", flush=True)
    t_phase = time.time()
    try:
        tok, model = load_model(MODELS[0][1])
        judge_items = []  # (key, text)
        judge_slugs = (GEN_SLUGS_THIS_RUN if JUDGE_SCOPE == "gen-only"
                       else [s for s, _ in MODELS])
        for e in ESSAYS:
            if JUDGE_SCOPE == "all":
                judge_items.append((f"d1:{e['essay_id']}", e["draft1"]))
                judge_items.append((f"d2:{e['essay_id']}", e["draft2"]))
                for mode in MODES:
                    for fmodel in ("GPT-5-MINI", "GEMINI-2.5-FLASH", "CLAUDE-HAIKU-4-5-20251001"):
                        judge_items.append((f"frontier:{fmodel}:{mode}:{e['essay_id']}",
                                            e["frontier_with_feedback"][mode][fmodel]))
            for slug in judge_slugs:
                for mode in MODES:
                    gen = RESULTS.get("generations", {}).get(slug, {}).get(mode, {}).get(e["essay_id"])
                    if gen:
                        judge_items.append((f"gen:{slug}:{mode}:{e['essay_id']}", gen))
        print(f"judge items: {len(judge_items)}", flush=True)
        msgs = [[{"role": "system", "content": JUDGE_SYS},
                 {"role": "user", "content": JUDGE_USER.replace("{text}", t)}]
                for _, t in judge_items]
        probe_j = batch_generate(tok, model, msgs[:1], 200, 1)
        if "{" not in probe_j[0]:
            print("FATAL: JUDGE_PROBE_FAIL -- first judge output has no JSON "
                  "despite enable_thinking=False; aborting", flush=True)
            print("JUDGE_PROBE_HEAD", probe_j[0][:500], flush=True)
            sys.exit(4)
        print("judge probe OK: JSON present", flush=True)
        outs = batch_generate(tok, model, msgs, 200, JUDGE_BATCH)
        n_parsed = 0
        for (key, _), raw in zip(judge_items, outs):
            entry = {"raw": raw}
            m = re.search(r"\{.*\}", raw, flags=re.S)
            if m:
                try:
                    entry["parsed"] = json.loads(m.group(0))
                    n_parsed += 1
                except json.JSONDecodeError:
                    entry["parsed"] = None
            else:
                entry["parsed"] = None
            RESULTS["judge"][key] = entry
        print(f"judge done: {len(outs)} items, parsed {n_parsed}", flush=True)
        RESULTS["phases"]["judge"] = {"status": "done", "n_items": len(outs),
                                      "n_parsed": n_parsed,
                                      "min": round((time.time() - t_phase) / 60, 1)}
        flush()
        del model, tok
        free_all()
    except Exception as e:
        print(f"JUDGE FAILED: {type(e).__name__}: {e}", flush=True)
        RESULTS["phases"]["judge"] = {"status": "FAILED", "error": f"{type(e).__name__}: {e}"}
        flush()

# ---------------- embeddings phase ----------------
if DO_EMBED:
    print("== embeddings (all-MiniLM-L6-v2) ==", flush=True)
    t_phase = time.time()
    try:
        from sentence_transformers import SentenceTransformer
        enc_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        texts, keys = {}, []
        for e in ESSAYS:
            texts[f"d1:{e['essay_id']}"] = e["draft1"]
            texts[f"d2:{e['essay_id']}"] = e["draft2"]
            for mode in MODES:
                for fmodel in ("GPT-5-MINI", "GEMINI-2.5-FLASH", "CLAUDE-HAIKU-4-5-20251001"):
                    texts[f"frontier:{fmodel}:{mode}:{e['essay_id']}"] = \
                        e["frontier_with_feedback"][mode][fmodel]
        for slug, _ in MODELS:
            for mode in MODES:
                for eid, gen in RESULTS.get("generations", {}).get(slug, {}).get(mode, {}).items():
                    texts[f"gen:{slug}:{mode}:{eid}"] = gen
        keys = sorted(texts)
        embs = enc_model.encode([texts[k] for k in keys], batch_size=32,
                                show_progress_bar=False, normalize_embeddings=True)
        RESULTS["embeddings"] = {k: [round(float(x), 5) for x in v]
                                 for k, v in zip(keys, embs)}
        import numpy as np
        E = {k: np.array(RESULTS["embeddings"][k]) for k in keys}
        def cos_dist(a, b):
            return float(1.0 - float(np.dot(E[a], E[b])))
        def shift_alignment(pairs):
            shifts = [E[b] - E[a] for a, b in pairs]
            norms = [s / (np.linalg.norm(s) + 1e-12) for s in shifts]
            mean_dir = np.mean(norms, axis=0)
            mean_dir = mean_dir / (np.linalg.norm(mean_dir) + 1e-12)
            return float(np.mean([float(np.dot(n, mean_dir)) for n in norms]))
        cos = {}
        for label, pair_fn in [
                ("human_d2", lambda eid: (f"d1:{eid}", f"d2:{eid}")),
                ("frontier_GPT-5-MINI_grammar", lambda eid: (f"d1:{eid}", f"frontier:GPT-5-MINI:grammar:{eid}")),
                ("frontier_GPT-5-MINI_minimal", lambda eid: (f"d1:{eid}", f"frontier:GPT-5-MINI:minimal:{eid}")),
                ("frontier_GPT-5-MINI_general", lambda eid: (f"d1:{eid}", f"frontier:GPT-5-MINI:general:{eid}")),
                ("frontier_GEMINI-2.5-FLASH_grammar", lambda eid: (f"d1:{eid}", f"frontier:GEMINI-2.5-FLASH:grammar:{eid}")),
                ("frontier_GEMINI-2.5-FLASH_minimal", lambda eid: (f"d1:{eid}", f"frontier:GEMINI-2.5-FLASH:minimal:{eid}")),
                ("frontier_GEMINI-2.5-FLASH_general", lambda eid: (f"d1:{eid}", f"frontier:GEMINI-2.5-FLASH:general:{eid}")),
                ("frontier_CLAUDE-HAIKU-4-5-20251001_grammar", lambda eid: (f"d1:{eid}", f"frontier:CLAUDE-HAIKU-4-5-20251001:grammar:{eid}")),
                ("frontier_CLAUDE-HAIKU-4-5-20251001_minimal", lambda eid: (f"d1:{eid}", f"frontier:CLAUDE-HAIKU-4-5-20251001:minimal:{eid}")),
                ("frontier_CLAUDE-HAIKU-4-5-20251001_general", lambda eid: (f"d1:{eid}", f"frontier:CLAUDE-HAIKU-4-5-20251001:general:{eid}")),
        ]:
            pairs = [pair_fn(e["essay_id"]) for e in ESSAYS]
            ds = [cos_dist(a, b) for a, b in pairs]
            cos[label] = {"mean_cos_dist": round(sum(ds) / len(ds), 4),
                          "per_essay": [round(d, 4) for d in ds],
                          "mean_shift_alignment": round(shift_alignment(pairs), 4)}
        for slug, _ in MODELS:
            for mode in MODES:
                pairs = [(f"d1:{e['essay_id']}", f"gen:{slug}:{mode}:{e['essay_id']}") for e in ESSAYS]
                if all(p[1] in E for p in pairs):
                    ds = [cos_dist(a, b) for a, b in pairs]
                    cos[f"gen:{slug}:{mode}"] = {
                        "mean_cos_dist": round(sum(ds) / len(ds), 4),
                        "per_essay": [round(d, 4) for d in ds],
                        "mean_shift_alignment": round(shift_alignment(pairs), 4)}
        RESULTS["cosine"] = cos
        print("COSINE", json.dumps({k: (v["mean_cos_dist"], v["mean_shift_alignment"])
                                    for k, v in cos.items()}, indent=1), flush=True)
        RESULTS["phases"]["embeddings"] = {"status": "done", "n_texts": len(keys),
                                           "min": round((time.time() - t_phase) / 60, 1)}
        flush()
    except Exception as e:
        print(f"EMBEDDINGS FAILED: {type(e).__name__}: {e}", flush=True)
        import traceback; traceback.print_exc()
        RESULTS["phases"]["embeddings"] = {"status": "FAILED", "error": f"{type(e).__name__}: {e}"}
        flush()

import platform
try:
    import transformers, bitsandbytes
    RESULTS["versions"] = {"python": platform.python_version(),
                           "torch": torch.__version__,
                           "transformers": transformers.__version__,
                           "bitsandbytes": bitsandbytes.__version__,
                           "gpu": torch.cuda.get_device_name(0)}
except Exception as e:
    RESULTS["versions"] = {"error": str(e)}
RESULTS["timing_min"]["total"] = round((time.time() - T0) / 60, 1)
flush()
print("RESULTS", json.dumps({k: RESULTS[k] for k in ("cosine", "phases", "timing_min")}, indent=1), flush=True)
print("DISTORTION_DONE", flush=True)
