"""Colab driver — FPRT: Failure Profiling and Reachable Trajectory Selection
for Reasoning Distillation (arXiv:2611.09042, 2026-11-09 daily training track).

Paper claim: distillation data should be (a) conditioned on the student's
behavioral failure mode (stubborn / inconsistent / sporadic, from K sampled
traces: error rate alpha + error consistency kappa) and (b) selected for
reachability — the candidate whose FIRST corrective steps (BPR window, first
L=8 sentences) the student itself assigns the highest log-probability.
Paper setup: teacher DeepSeek-V3.2, students Llama-3.1-8B / DeepSeek-Math-7B,
K=5, N=5 candidates, MetaMathQA 395K profiled -> 62K SFT examples; +2.4/+1.8
pts over LEMMA on 5 math benchmarks.

Bounded adaptation (one T4 day, stated honestly):
  - GSM8K (grade-school math, exact answers) instead of MetaMathQA.
  - Student Qwen/Qwen3-8B (proven training model), teacher Qwen/Qwen3.5-9B
    (NF4, generation-only) — the paper's strong-teacher gap is narrowed;
    the test is about the SELECTION mechanism, not the teacher's ceiling.
    Model choice rationale: a teacher-student gap is required (teacher must
    be stronger than the student), and Qwen3.5-9B is the only stronger
    model loadable on this T4; Qwen3-8B is the proven QLoRA student.
    Standing 9B-first pick is therefore satisfied by the teacher role.
  - N_PROFILE=160 train problems, K=5, N_CAND=5; N_EVAL=150 held-out.
  - v2 (driver-fix relaunch 2026-11-09 ~05:20 UTC, same day/model): the v1
    run was stopped 10 min in — profiling ran ~2.4 min per 8 prompts
    (full 200-token generations, no stop string, so samples rambled into
    hallucinated next problems) and extract_answer took the LAST number,
    which a rambling continuation could corrupt. v2 also added a
    per-problem profiles flush (append_jsonl in phase_profile).
  - v3 (driver-fix relaunch 2026-11-09 ~08:00 UTC, same day/model): v2
    profiled all 160 problems (48 with errors: inconsistent 20, sporadic
    23, stubborn 5; peak 8.98 GB) then DIED at teacher load — transformers
    has no qwen3_5 model in any 4.x release (verified: the 4.57.0 and
    4.57.6 wheels contain zero qwen3_5 files), so the inherited
    transformers>=4.50,<5.0 pin could never load Qwen3.5-9B. v3 pins
    transformers==5.20.2 (the pin every successful Qwen3.5-9B driver since
    2026-11-03 uses) and asserts qwen3_5 is registered right after
    install (fail fast, before any GPU work). v3 also resumes from an
    uploaded /content/fprt_profiles.jsonl when present (skips the ~2.5 h
    profiling phase; the downloaded file is complete: 160 rows).
  - v4 (AUTOMATIC FALLBACK per the 2026-11-05 rule, 2026-11-09 ~10:45
    UTC, same day): v3 was RECLAIMED ~10:39–10:44 UTC during the phase-2
    solo-retry loop (teacher main gen 48/48 DONE 09:45:47; 17 retries
    done, still going; no candidates/selections/evals/adapters were
    ever written — the retry loop runs before any candidate flush, so
    all teacher output died with the VM; profiles + metrics are safe
    locally). Fallback model: teacher Qwen/Qwen3-8B — the next smaller
    proven model; it uses fast SDPA kernels (no Qwen3.5 reference-kernel
    fallback) and is already the student, so the teacher-student gap
    is GONE in v4: the test now measures the SELECTION mechanism
    (profiling + BPR) with a same-size teacher, not distillation from
    a stronger teacher — recorded honestly in the README. Shrink fixes
    for the failure cause (slowness → reclamation exposure): (a) the
    solo-retry loop is REMOVED (the paper has no retry; zero-correct
    problems simply yield no example — the loop was the long tail that
    got v3 reclaimed); (b) teacher max_new_tokens 1600 -> 1024;
    (c) teacher batch 4 -> 8. Profiles resume again from the upload.
    ["\\nProblem:", "Problem:"], max_new_tokens 160 for profiling,
    batch 16, and extract_answer prefers the FIRST "The answer is: N".
  - Arms (same student base, same problem pool): base (no training),
    FPRT (profiled mode prompts + BPR-selected trajectory + error-conditioned
    format), CONTROL = standard distillation (random correct teacher
    candidate per problem, plain q -> solution format, no profiling/BPR).
  - Eval: greedy, 4-shot CoT prompt, exact-match on extracted answer.
  - QLoRA r=16/alpha=32, NF4, bf16, lr 2e-4, 2 epochs, seq 640.

Phases (incremental writes to /content so a mid-run download sees partials):
  0 setup/installs, 1 profiling, 2 teacher candidates, 3 BPR selection,
  4 SFT (FPRT + control adapters), 5 eval. Prints FPRT_DONE as the very
  last line on full completion; on phase failure prints FPRT_FAILED:<phase>
  and exits 1 (wrapper then reports DIED; partials still downloadable).

Run via: python3 colab.py train --driver <this file> --rotate
  --session fprt-1109 --monitor-timeout 14400
  --upload fprt_train200.jsonl fprt_test150.jsonl
  --results /content/fprt_metrics.csv /content/fprt_profiles.jsonl
  /content/fprt_candidates.jsonl /content/fprt_selections.jsonl
  /content/fprt_eval_details.jsonl /content/adapters/fprt.zip
  /content/adapters/control.zip
  --out-dir <session dir> --done-marker FPRT_DONE
  --registry-file <isolated registry>
"""
import csv
import gc
import json
import math
import os
import random
import re
import subprocess
import sys
import time

METRICS = "/content/fprt_metrics.csv"
PROFILES = "/content/fprt_profiles.jsonl"
CANDIDATES = "/content/fprt_candidates.jsonl"
SELECTIONS = "/content/fprt_selections.jsonl"
EVAL_DETAILS = "/content/fprt_eval_details.jsonl"
ADAPTER_DIR = "/content/adapters"

STUDENT_ID = os.environ.get("STUDENT_ID", "Qwen/Qwen3-8B")
TEACHER_ID = os.environ.get("TEACHER_ID", "Qwen/Qwen3-8B")
N_PROFILE = int(os.environ.get("N_PROFILE", "160"))
K_SAMPLES = 5
N_CAND = 5
N_EVAL = int(os.environ.get("N_EVAL", "150"))
BPR_L = 8  # first L sentences of the corrective continuation
SEED = 20261109

FEWSHOT = """Solve each problem step by step. End your solution with "The answer is: <number>".

Problem: A baker made 4 trays of 12 muffins. He sold 30 muffins in the morning and 9 in the afternoon. How many muffins are left?
Solution: The baker made 4 * 12 = 48 muffins. He sold 30 + 9 = 39 muffins. So 48 - 39 = 9 muffins are left.
The answer is: 9

Problem: Sara had 15 stickers. She bought 3 packs with 8 stickers each and gave 10 to her brother. How many stickers does she have now?
Solution: She bought 3 * 8 = 24 stickers. She had 15 + 24 = 39 stickers. After giving away 10, she has 39 - 10 = 29 stickers.
The answer is: 29

Problem: A rectangle is 8 meters long and 5 meters wide. What is its perimeter?
Solution: The perimeter is 2 * (8 + 5) = 2 * 13 = 26 meters.
The answer is: 26

Problem: Tom reads 20 pages every day. A book has 140 pages. How many days will it take him to finish the book?
Solution: 140 / 20 = 7 days.
The answer is: 7

"""

# --- Mode-specific teacher prompts: paper Appendix D, verbatim structure ---
# Each requests N_CAND different corrective trajectories in ONE JSON array with
# fields: error_step, preserved_prefix, correction, final_answer.
PROMPT_STUBBORN = """You are an expert Math Corrector. You specialize in diagnosing SYSTEMATIC MISCONCEPTIONS.
The incorrect outcomes concentrate on one wrong answer. Attack the locked-in reasoning path from different angles: direct correction, an alternative method, conceptual reframing, work backwards, or substitution-based verification. Diagnose the trap, the conceptual pivot, the fix, and a check that prevents recurrence. Distill that analysis into a concise first-person error_step.
Correction-angle examples: trajectory 1 applies a direct fix to the specific algebraic/logical error; trajectory 2 solves the problem via an alternative method entirely; trajectory 3 reframes the underlying concept before solving; further trajectories explore additional creative variations such as work-backwards or substitution-based verification.

Generate exactly {n} DIFFERENT corrective trajectories in ONE JSON array. Each object must contain these string fields:
1. error_step: a concise first-person diagnosis naming where and why the first error occurred. For example, "I assumed that..." or "I forgot to multiply by...".
2. preserved_prefix: the student's LONGEST correct prefix, copied VERBATIM from the beginning up to but excluding the first erroneous step; empty if none.
3. correction: continue from preserved_prefix to a complete correct solution. Match the student's vocabulary, formatting, and tone. End with exactly "The answer is: [value]".
4. final_answer: only the final numerical or symbolic answer.
Localize the first error and diversify the {n} continuations jointly in this single call. Never invent an error in correct reasoning. Avoid apologies, meta-commentary such as "Let me reconsider" or "I made an error", and explanations outside the array.
Task data:
{task}"""

PROMPT_INCONSISTENT = """You are an expert Math Corrector. You specialize in STABILIZING chaotic reasoning patterns.
The sampled outcomes are unstable. Supply a structured framework: state the givens and goal, define variables and equations, calculate step by step, and verify the result. Vary frameworks: equations, verifiable arithmetic, working backwards, dimensional analysis, or unit tracking.
Framework examples: trajectory 1 sets up and solves algebraic equations; trajectory 2 breaks the problem into small verifiable arithmetic sub-calculations; trajectory 3 works backwards from the answer structure; further trajectories use dimensional analysis, unit tracking, or diagram-based reasoning.

Generate exactly {n} DIFFERENT corrective trajectories in ONE JSON array. Each object must contain these string fields:
1. error_step: a concise first-person diagnosis naming where and why the first error occurred. Describe the common pattern of confusion.
2. preserved_prefix: the student's LONGEST correct prefix, copied VERBATIM from the beginning up to but excluding the first erroneous step; empty if none.
3. correction: continue from preserved_prefix to a complete correct solution. Match the student's vocabulary, formatting, and tone. End with exactly "The answer is: [value]".
4. final_answer: only the final numerical or symbolic answer.
The correction is intentionally more structured than the student's attempts because the student needs scaffolding, but still matches the student's vocabulary and tone. Localize the first error and diversify the {n} continuations jointly in this single call. Never invent an error in correct reasoning. Avoid apologies, meta-commentary such as "Let me reconsider", and explanations outside the array.
Task data:
{task}"""

PROMPT_SPORADIC = """You are an expert Math Corrector. You specialize in MINIMAL, PRECISE corrections.
The student usually answers correctly. Preserve as much of the reasoning as possible and change only the localized error. Vary the local repair: a direct fix, brief inline verification, a slightly different computation avoiding the same error, or further minimal variations. Do not restructure correct steps.
The localized slip may be an arithmetic mistake, sign error, misapplied formula, or off-by-one in an otherwise reasonable solution. The overall problem-solving approach and structure are sound; only the erroneous step needs to be repaired.
Local-repair examples: trajectory 1 directly replaces the wrong step with the correct computation; trajectory 2 corrects with a brief inline verification; trajectory 3 takes a slightly different computational path that avoids the specific error; further trajectories explore additional minimal variations.

Generate exactly {n} DIFFERENT corrective trajectories in ONE JSON array. Each object must contain these string fields:
1. error_step: a concise first-person diagnosis naming where and why the first error occurred. Describe the specific localized slip, for example "I made an arithmetic error when adding...".
2. preserved_prefix: the student's LONGEST correct prefix, copied VERBATIM from the beginning up to but excluding the first erroneous step; empty if none.
3. correction: continue from preserved_prefix to a complete correct solution. Match the student's vocabulary, formatting, and tone. End with exactly "The answer is: [value]".
4. final_answer: only the final numerical or symbolic answer.
The correction should feel like a seamless continuation of the student's own work. Keep the correction short. Localize the first error and diversify the {n} continuations jointly in this single call. Never invent an error in correct reasoning. Avoid apologies, meta-commentary such as "I made a mistake here" or "Let me reconsider", and explanations outside the array.
Task data:
{task}"""

MODE_PROMPTS = {"stubborn": PROMPT_STUBBORN,
                "inconsistent": PROMPT_INCONSISTENT,
                "sporadic": PROMPT_SPORADIC}


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


_METRIC_ROWS = []
_METRIC_COLS = ["phase"]


def metric(phase, **kw):
    row = {"phase": phase}
    row.update({k: v for k, v in kw.items()})
    _METRIC_ROWS.append(row)
    for k in row:
        if k not in _METRIC_COLS:
            _METRIC_COLS.append(k)
    with open(METRICS, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=_METRIC_COLS)
        w.writeheader()
        for r in _METRIC_ROWS:
            w.writerow({c: r.get(c, "") for c in _METRIC_COLS})
    log("METRIC " + json.dumps(row))


def append_jsonl(path, obj):
    with open(path, "a") as f:
        f.write(json.dumps(obj) + "\n")


def extract_answer(text):
    """Answer extraction, in priority order: (1) the FIRST "The answer is:
    N" (the sample's own answer — a rambling continuation after it must
    not override it); (2) '#### N'; (3) the last number in the text."""
    if text is None:
        return None
    m = re.search(r"[Tt]he answer is:?\s*(-?[\d,]+(?:\.\d+)?)", text)
    if m:
        return m.group(1).replace(",", "")
    m = re.findall(r"####\s*(-?[\d,]+(?:\.\d+)?)", text)
    if m:
        return m[-1].replace(",", "")
    nums = re.findall(r"-?[\d,]+(?:\.\d+)?", text.replace(",", ""))
    return nums[-1] if nums else None


def answers_equal(a, b):
    try:
        return abs(float(a) - float(b)) < 1e-6
    except (TypeError, ValueError):
        return str(a).strip() == str(b).strip()


def gold_answer(ans_field):
    m = re.search(r"####\s*(-?[\d,]+(?:\.\d+)?)", ans_field)
    return m.group(1).replace(",", "") if m else None


def peak_gb():
    import torch
    return round(torch.cuda.max_memory_allocated() / 1e9, 2) if torch.cuda.is_available() else 0.0


def install():
    log("PHASE 0: installs")
    subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                    "transformers==5.20.2", "peft", "accelerate",
                    "bitsandbytes"], check=True)
    subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "torchao"],
                   check=False, capture_output=True)
    import transformers
    log(f"transformers {transformers.__version__}")
    # Fail fast: the teacher (Qwen3.5-9B) needs qwen3_5 modeling, which
    # exists only in transformers 5.x — a 4.x install must die here, not
    # after 2.5 h of profiling (v2 lesson).
    from transformers.models.auto.modeling_auto import (
        MODEL_FOR_CAUSAL_LM_MAPPING_NAMES)
    assert "qwen3_5" in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES, (
        "transformers has no qwen3_5 model registered — "
        "teacher Qwen3.5-9B cannot load")
    import torch
    log(f"torch {torch.__version__} cuda={torch.cuda.is_available()}")
    if torch.cuda.is_available():
        log(f"gpu: {torch.cuda.get_device_name(0)} "
            f"{torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def load_causal(model_id):
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    import torch
    tok = AutoTokenizer.from_pretrained(model_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_compute_dtype=torch.bfloat16,
                             bnb_4bit_use_double_quant=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, quantization_config=bnb, torch_dtype=torch.bfloat16,
        device_map="auto", attn_implementation="sdpa")
    model.config.use_cache = True
    model.eval()
    log(f"LOADED {model_id} peak={peak_gb()} GB")
    return tok, model


def free_mem():
    import torch
    gc.collect()
    torch.cuda.empty_cache()


STOP_FEW_SHOT = ["\nProblem:", "Problem:"]


def batch_generate(tok, model, prompts, max_new_tokens, temperature, top_p=0.95,
                   batch_size=8, stop_strings=None):
    """Batched left-padded generation; returns list of completion strings.
    stop_strings (e.g. a new "Problem:" header) stops a few-shot sample
    from rambling into a hallucinated next problem."""
    import torch
    outs = []
    for s in range(0, len(prompts), batch_size):
        chunk = prompts[s:s + batch_size]
        enc = tok(chunk, return_tensors="pt", padding=True).to(model.device)
        gkw = dict(max_new_tokens=max_new_tokens,
                   do_sample=(temperature > 0),
                   temperature=max(temperature, 1e-6),
                   top_p=top_p, pad_token_id=tok.pad_token_id)
        if stop_strings:
            gkw["stop_strings"] = stop_strings
            gkw["tokenizer"] = tok
        with torch.no_grad():
            gen = model.generate(**enc, **gkw)
        for i, g in enumerate(gen):
            prompt_len = enc["input_ids"].shape[1]
            text = tok.decode(g[prompt_len:], skip_special_tokens=True)
            outs.append(text)
        log(f"  gen {min(s + batch_size, len(prompts))}/{len(prompts)}")
    return outs


def first_sentences(text, n):
    """First n sentences of a continuation: newline split, punctuation fallback."""
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    if len(lines) >= 1:
        parts = lines[:n]
        if len(parts) < n:
            rest = " ".join(lines[n:])
            extra = re.split(r"(?<=[.!?])\s+", rest)
            parts += [e for e in extra if e][:n - len(parts)]
        return " ".join(parts)
    sents = re.split(r"(?<=[.!?])\s+", text.strip())
    return " ".join(s for s in sents if s)[:n] if sents else text.strip()


# ---------------- Phase 1: profiling ----------------

def phase_profile(tok, model, problems):
    log(f"PHASE 1: profiling {len(problems)} problems x K={K_SAMPLES} "
        f"(student {STUDENT_ID})")
    prompts = []
    for p in problems:
        q = p["question"].strip()
        prompts.append(FEWSHOT + f"Problem: {q}\nSolution:")
    all_texts = []
    for k in range(K_SAMPLES):
        all_texts.append(batch_generate(tok, model, prompts,
                                        max_new_tokens=160, temperature=0.8,
                                        batch_size=16,
                                        stop_strings=STOP_FEW_SHOT))
    profiles = []
    for i, p in enumerate(problems):
        gold = gold_answer(p["answer"])
        samples = [all_texts[k][i] for k in range(K_SAMPLES)]
        got = [extract_answer(t) for t in samples]
        wrong = [g for g in got if g is None or not answers_equal(g, gold)]
        alpha = len(wrong) / K_SAMPLES
        mode = None
        kappa = None
        tau_rep = None
        if alpha > 0:
            from collections import Counter
            cnt = Counter(w for w in wrong if w is not None)
            top_wrong, top_n = cnt.most_common(1)[0] if cnt else (None, 0)
            kappa = top_n / len(wrong) if wrong else 0.0
            if alpha >= 0.6 and kappa >= 0.8:
                mode = "stubborn"
            elif alpha >= 0.6:
                mode = "inconsistent"
            else:
                mode = "sporadic"
            if mode == "stubborn" and top_wrong is not None:
                cands = [t for t, g in zip(samples, got)
                         if g is not None and g == top_wrong]
            else:
                cands = [t for t, g in zip(samples, got)
                         if g is None or not answers_equal(g, gold)]
            tau_rep = random.choice(cands) if cands else samples[0]
        prof = {"idx": i, "question": p["question"], "gold": gold,
                "alpha": round(alpha, 3),
                "kappa": round(kappa, 3) if kappa is not None else None,
                "mode": mode, "tau_rep": tau_rep,
                "samples": samples, "answers": got}
        profiles.append(prof)
        append_jsonl(PROFILES, prof)
        if (i + 1) % 20 == 0:
            log(f"  profiled {i + 1}/{len(problems)}")
    n_err = sum(1 for pr in profiles if pr["mode"])
    from collections import Counter
    log(f"PROFILED: {len(profiles)} problems, {n_err} with errors "
        f"{dict(Counter(pr['mode'] for pr in profiles if pr['mode']))}")
    metric("profile", n=len(profiles), n_errors=n_err,
           modes=json.dumps(dict(Counter(pr["mode"] for pr in profiles
                                         if pr["mode"]))),
           peak_gb=peak_gb())
    return profiles


# ---------------- Phase 2: teacher candidates ----------------

def parse_candidates(raw):
    """Parse the teacher's JSON array; returns list of dicts or []."""
    m = re.search(r"\[.*\]", raw, re.S)
    if not m:
        return []
    try:
        arr = json.loads(m.group(0))
    except json.JSONDecodeError:
        fixed = m.group(0)
        fixed = re.sub(r",\s*([\]}])", r"\1", fixed)
        try:
            arr = json.loads(fixed)
        except json.JSONDecodeError:
            return []
    out = []
    for o in arr if isinstance(arr, list) else []:
        if not isinstance(o, dict):
            continue
        if all(k in o for k in ("error_step", "preserved_prefix",
                                "correction", "final_answer")):
            out.append({k: str(o[k]) for k in
                        ("error_step", "preserved_prefix", "correction",
                         "final_answer")})
    return out


def teacher_prompt(tok, pr):
    task = (f"Problem: {pr['question']}\nCorrect answer: {pr['gold']}\n"
            f"Error rate alpha={pr['alpha']}, consistency "
            f"kappa={pr['kappa']}, mode={pr['mode']}.\n"
            f"Student's incorrect attempt:\n{pr['tau_rep']}")
    prompt = MODE_PROMPTS[pr["mode"]].format(n=N_CAND, task=task)
    # Teacher is the chat model (ships chat_template.jinja): use the chat
    # template so the JSON-array instruction is followed; fall back to
    # plain text if the template is unavailable.
    try:
        msgs = [{"role": "system", "content": "You are an expert Math "
                 "Corrector. Respond with ONLY a JSON array."},
                {"role": "user", "content": prompt}]
        return tok.apply_chat_template(msgs, tokenize=False,
                                       add_generation_prompt=True)
    except Exception:
        return ("You are an expert Math Corrector. Respond with ONLY a "
                "JSON array.\n\n" + prompt)


def score_teacher_raw(raw, gold):
    cands = parse_candidates(raw)
    good = [c for c in cands
            if answers_equal(extract_answer(c["final_answer"]) or
                             c["final_answer"], gold)]
    for c in cands:
        if c not in good and answers_equal(
                extract_answer(c["correction"]), gold):
            good.append(c)
    return cands, good


def phase_teacher(tok, model, profiles):
    log(f"PHASE 2: teacher candidates (teacher {TEACHER_ID})")
    errs = [pr for pr in profiles if pr["mode"]]
    prompts = [teacher_prompt(tok, pr) for pr in errs]
    raws = batch_generate(tok, model, prompts, max_new_tokens=1024,
                          temperature=0.7, batch_size=8)
    # v4: no solo retries (see header) — zero-correct problems yield no
    # example; the recs loop below records n_correct=0 for them.
    recs = []
    for j, pr in enumerate(errs):
        cands, good = score_teacher_raw(raws[j], pr["gold"])
        rec = {"idx": pr["idx"], "mode": pr["mode"], "gold": pr["gold"],
               "n_candidates": len(cands), "n_correct": len(good),
               "candidates": good, "raw_chars": len(raws[j])}
        recs.append(rec)
        append_jsonl(CANDIDATES, rec)
        log(f"  teacher {j + 1}/{len(errs)} mode={pr['mode']} "
            f"cands={len(cands)} correct={len(good)}")
    usable = sum(1 for r in recs if r["n_correct"] > 0)
    log(f"TEACHER DONE: {len(recs)} error problems, {usable} with >=1 correct "
        f"candidate")
    metric("teacher", n_errors=len(recs), n_usable=usable,
           peak_gb=peak_gb())
    return recs


# ---------------- Phase 3: BPR selection ----------------

def bpr_score(tok, model, question, prefix, window):
    """Avg log-prob of the window tokens under the student, conditioned on
    (few-shot + question + preserved prefix). Single forward pass."""
    import torch
    ctx = FEWSHOT + f"Problem: {question.strip()}\nSolution: {prefix}"
    ctx_ids = tok(ctx, return_tensors="pt")["input_ids"][0]
    win_ids = tok(window, return_tensors="pt",
                  add_special_tokens=False)["input_ids"][0]
    ids = torch.cat([ctx_ids, win_ids]).unsqueeze(0).to(model.device)
    with torch.no_grad():
        logits = model(ids).logits[0]
    ctx_len = len(ctx_ids)
    logp = torch.log_softmax(logits.float(), dim=-1)
    tok_lp = []
    for t in range(ctx_len, len(ids[0])):
        tok_lp.append(logp[t - 1, ids[0, t]].item())
    if not tok_lp:
        return -1e9
    return sum(tok_lp) / len(tok_lp)


def phase_bpr(tok, model, profiles, cand_recs):
    log("PHASE 3: BPR selection (first %d sentences of each correction)" % BPR_L)
    by_idx = {r["idx"]: r for r in cand_recs}
    sels = []
    for pr in profiles:
        if pr["mode"] is None:
            continue
        rec = by_idx.get(pr["idx"])
        if not rec or rec["n_correct"] == 0:
            continue
        scored = []
        for c in rec["candidates"]:
            w = first_sentences(c["correction"], BPR_L)
            s = bpr_score(tok, model, pr["question"],
                          c["preserved_prefix"], w)
            scored.append((s, c))
        scored.sort(key=lambda t: -t[0])
        best_s, best = scored[0]
        # whole-trace logprob of the best, for the record (dilution check)
        sel = {"idx": pr["idx"], "mode": pr["mode"], "gold": pr["gold"],
               "tau_rep": pr["tau_rep"], "selected": best,
               "bpr_best": round(best_s, 4),
               "bpr_all": [round(s, 4) for s, _ in scored],
               "n_candidates": len(scored)}
        sels.append(sel)
        append_jsonl(SELECTIONS, sel)
    log(f"BPR DONE: {len(sels)} selections")
    if sels:
        import statistics
        margins = [s["bpr_best"] - statistics.mean(s["bpr_all"])
                   for s in sels]
        log(f"  mean BPR margin selected-vs-mean: "
            f"{statistics.mean(margins):.4f}")
        metric("bpr", n=len(sels),
               mean_margin=round(statistics.mean(margins), 4),
               peak_gb=peak_gb())
    return sels

# ---------------- Phase 4: SFT ----------------

def build_sft_examples(sels, cand_recs, rng):
    """FPRT arm: 90% (q + tau_rep) -> (r + p + c); 10% q -> (p + c) (paper
    section 3.5). Control arm: q -> (p + c) of a RANDOM correct candidate
    (standard distillation: no profiling, no BPR, no error conditioning)."""
    by_idx = {r["idx"]: r for r in cand_recs}
    fprt_ex, ctrl_ex = [], []
    for s in sels:
        rec = by_idx[s["idx"]]
        best = s["selected"]
        r, p, c = (best["error_step"], best["preserved_prefix"],
                   best["correction"])
        full_target = (r.strip() + "\n" + p.strip() + "\n" + c.strip()).strip()
        plain_target = (p.strip() + "\n" + c.strip()).strip()
        fprt_ex.append({"idx": s["idx"], "q": s["question"],
                        "tau_rep": s["tau_rep"],
                        "target_full": full_target,
                        "target_plain": plain_target,
                        "fmt": "fprt_full"})
        rc = rng.choice(rec["candidates"])
        ctrl_target = (rc["preserved_prefix"].strip() + "\n" +
                       rc["correction"].strip()).strip()
        ctrl_ex.append({"idx": s["idx"], "q": s["question"],
                        "target_plain": ctrl_target, "fmt": "plain"})
    # paper 3.5: 10% of FPRT examples use the plain q -> (p + c) format
    rng.shuffle(fprt_ex)
    n_plain = max(1, len(fprt_ex) // 10)
    for ex in fprt_ex[:n_plain]:
        ex["fmt"] = "plain"
    return fprt_ex, ctrl_ex


def sft_train(tok, base, examples, tag):
    """Manual QLoRA SFT loop (proven RLVR pattern): micro-batch 2, accum 2,
    completion-only loss, seq cap 768, lr 2e-4, 2 epochs."""
    import torch
    from peft import LoraConfig, get_peft_model
    lcfg = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.0,
                      target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                      "gate_proj", "up_proj", "down_proj"],
                      task_type="CAUSAL_LM")
    model = get_peft_model(base, lcfg)
    model.print_trainable_parameters()
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    SEQ_CAP = 768

    def batchify(chunk):
        prompts, comps = [], []
        for ex in chunk:
            if ex.get("fmt") == "fprt_full":
                prompts.append(FEWSHOT + f"Problem: {ex['q'].strip()}\n"
                               f"Student's attempt:\n{ex['tau_rep']}\n"
                               f"Correct the attempt.\nSolution:")
                comps.append("\n" + ex["target_full"] + tok.eos_token)
            else:  # plain q -> solution
                prompts.append(FEWSHOT + f"Problem: {ex['q'].strip()}\n"
                               f"Solution:")
                comps.append("\n" + ex["target_plain"] + tok.eos_token)
        pids = [tok(p, add_special_tokens=False)["input_ids"] for p in prompts]
        cids = [tok(c, add_special_tokens=False)["input_ids"] for c in comps]
        maxlen = min(max(len(a) + len(b) for a, b in zip(pids, cids)), SEQ_CAP)
        ids, labs, att = [], [], []
        for a, b in zip(pids, cids):
            full = (a + b)[:maxlen]
            lab = ([-100] * len(a) + b)[:maxlen]
            pad = maxlen - len(full)
            ids.append(full + [tok.pad_token_id] * pad)
            labs.append(lab + [-100] * pad)
            att.append([1] * len(full) + [0] * pad)
        return (torch.tensor(ids, device=model.device),
                torch.tensor(labs, device=model.device),
                torch.tensor(att, device=model.device))

    opt = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=0.0)
    model.train()
    losses = []
    steps = 0
    for epoch in range(2):
        random.shuffle(examples)
        for s in range(0, len(examples), 2):
            chunk = examples[s:s + 2]
            ids, labs, att = batchify(chunk)
            out = model(input_ids=ids, labels=labs, attention_mask=att)
            (out.loss / 2).backward()
            losses.append(float(out.loss))
            if (s // 2 + 1) % 2 == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                opt.zero_grad(set_to_none=True)
                steps += 1
        log(f"  {tag} epoch {epoch + 1}/2 done, mean_loss="
            f"{sum(losses) / len(losses):.4f} peak={peak_gb()} GB")
    if (len(examples) // 2) % 2 == 1:
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)
        steps += 1
    mean_loss = sum(losses) / max(len(losses), 1)
    log(f"SFT {tag} done: {len(examples)} examples, {steps} steps, "
        f"mean_loss={mean_loss:.4f} peak={peak_gb()} GB")
    metric(f"sft_{tag}", n=len(examples), steps=steps,
           mean_loss=round(mean_loss, 4), peak_gb=peak_gb())
    return model


# ---------------- Phase 5: eval ----------------

def evaluate(tok, model, problems, arm):
    log(f"EVAL {arm}: {len(problems)} problems, greedy")
    prompts = [FEWSHOT + f"Problem: {p['question'].strip()}\nSolution:"
               for p in problems]
    texts = batch_generate(tok, model, prompts, max_new_tokens=200,
                           temperature=0.0, batch_size=16,
                           stop_strings=STOP_FEW_SHOT)
    n_ok, rows = 0, []
    for p, t in zip(problems, texts):
        gold = gold_answer(p["answer"])
        got = extract_answer(t)
        ok = got is not None and answers_equal(got, gold)
        n_ok += int(ok)
        rows.append({"arm": arm, "question": p["question"], "gold": gold,
                     "got": got, "correct": ok,
                     "gen": t[:600]})
    for r in rows:
        append_jsonl(EVAL_DETAILS, r)
    acc = n_ok / len(problems)
    lo = acc - 1.96 * math.sqrt(acc * (1 - acc) / len(problems))
    hi = acc + 1.96 * math.sqrt(acc * (1 - acc) / len(problems))
    log(f"EVAL {arm}: acc={acc:.4f} ({n_ok}/{len(problems)}) "
        f"95% CI [{max(0, lo):.3f}, {min(1, hi):.3f}]")
    metric(f"eval_{arm}", acc=round(acc, 4), n=len(problems),
           ci_lo=round(max(0, lo), 4), ci_hi=round(min(1, hi), 4),
           peak_gb=peak_gb())
    return acc, rows


def mcnemar(a_rows, b_rows):
    """McNemar on paired correct/incorrect: b=only-A-correct, c=only-B."""
    b = sum(1 for a, c in zip(a_rows, b_rows) if a["correct"] and not c["correct"])
    c = sum(1 for a, c2 in zip(a_rows, b_rows) if not a["correct"] and c2["correct"])
    if b + c == 0:
        return None
    z = (abs(b - c) - 1) ** 2 / (b + c)  # continuity-corrected chi2 (1 dof)
    import math as _m
    p = _m.erfc(_m.sqrt(z / 2))  # chi2(1) survival
    return {"b_only_a": b, "c_only_b": c, "chi2": round(z, 3),
            "p": round(p, 4)}


def main():
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    random.seed(SEED)
    rng = random.Random(SEED)
    t0 = time.time()
    try:
        install()
    except Exception as e:
        log(f"FPRT_FAILED:install:{e}")
        sys.exit(1)
    import torch
    assert torch.cuda.is_available(), "no GPU: refusing to fake a GPU run"
    os.makedirs(ADAPTER_DIR, exist_ok=True)

    train_pool = load_jsonl("/content/fprt_train200.jsonl")[:N_PROFILE]
    test_pool = load_jsonl("/content/fprt_test150.jsonl")[:N_EVAL]
    log(f"data: profile_pool={len(train_pool)} eval={len(test_pool)}")

    # Phase 1: profile with the student — or resume from a previous run's
    # flushed profiles (v2 wrote all 160 rows before dying at teacher load).
    student = None
    tok = None
    try:
        if os.path.exists(PROFILES):
            profiles = load_jsonl(PROFILES)
            ok = (len(profiles) == N_PROFILE and all(
                "mode" in pr and "question" in pr and "tau_rep" in pr
                for pr in profiles))
            if not ok:
                raise ValueError(
                    f"resume profiles incomplete: {len(profiles)} rows")
            from collections import Counter
            n_err = sum(1 for pr in profiles if pr["mode"])
            log(f"RESUMED profiles: {len(profiles)} problems, {n_err} "
                f"with errors "
                f"{dict(Counter(pr['mode'] for pr in profiles if pr['mode']))}")
            metric("profile", n=len(profiles), n_errors=n_err,
                   modes=json.dumps(dict(Counter(pr["mode"] for pr in profiles
                                                 if pr["mode"]))),
                   peak_gb=0.0)
        else:
            tok, student = load_causal(STUDENT_ID)
            profiles = phase_profile(tok, student, train_pool)
    except Exception as e:
        log(f"FPRT_FAILED:profile:{e!r}")
        sys.exit(1)

    # Phase 2: teacher candidates (unload student first).
    try:
        if student is not None:
            del student
        free_mem()
        ttok, teacher = load_causal(TEACHER_ID)
        cand_recs = phase_teacher(ttok, teacher, profiles)
        del teacher
        free_mem()
    except Exception as e:
        log(f"FPRT_FAILED:teacher:{e!r}")
        sys.exit(1)

    # Phase 3: BPR selection with the student.
    try:
        tok, student = load_causal(STUDENT_ID)
        sels = phase_bpr(tok, student, profiles, cand_recs)
    except Exception as e:
        log(f"FPRT_FAILED:bpr:{e!r}")
        sys.exit(1)
    if len(sels) < 10:
        log(f"FPRT_FAILED:bpr:only {len(sels)} usable selections (<10)")
        sys.exit(1)
    # selections need the question text for SFT prompts
    q_by_idx = {pr["idx"]: pr["question"] for pr in profiles}
    for s in sels:
        s["question"] = q_by_idx[s["idx"]]

    # Base eval (before any training).
    try:
        base_acc, base_rows = evaluate(tok, student, test_pool, "base")
    except Exception as e:
        log(f"FPRT_FAILED:eval_base:{e!r}")
        sys.exit(1)

    # Phase 4a: FPRT SFT + eval.
    try:
        fprt_ex, ctrl_ex = build_sft_examples(sels, cand_recs, rng)
        fprt_model = sft_train(tok, student, fprt_ex, "fprt")
        fprt_acc, fprt_rows = evaluate(tok, fprt_model, test_pool, "fprt")
        fprt_model.save_pretrained("/content/adapters/fprt")
        import shutil
        shutil.make_archive("/content/adapters/fprt", "zip",
                            "/content/adapters", "fprt")
        metric("adapter_fprt_saved", peak_gb=peak_gb())
    except Exception as e:
        log(f"FPRT_FAILED:sft_fprt:{e!r}")
        sys.exit(1)

    # Phase 4b: control SFT (fresh base) + eval.
    try:
        del fprt_model
        del student
        free_mem()
        tok2, student2 = load_causal(STUDENT_ID)
        ctrl_model = sft_train(tok2, student2, ctrl_ex, "control")
        ctrl_acc, ctrl_rows = evaluate(tok2, ctrl_model, test_pool, "control")
        ctrl_model.save_pretrained("/content/adapters/control")
        import shutil
        shutil.make_archive("/content/adapters/control", "zip",
                            "/content/adapters", "control")
        metric("adapter_control_saved", peak_gb=peak_gb())
    except Exception as e:
        log(f"FPRT_FAILED:sft_control:{e!r}")
        sys.exit(1)

    # Headline comparison.
    mc = mcnemar(fprt_rows, ctrl_rows)
    log(f"HEADLINE base={base_acc:.4f} fprt={fprt_acc:.4f} "
        f"control={ctrl_acc:.4f} mcnemar(fprt vs control)={mc}")
    metric("headline", base=round(base_acc, 4), fprt=round(fprt_acc, 4),
           control=round(ctrl_acc, 4),
           mcnemar=json.dumps(mc) if mc else None,
           n_eval=len(test_pool), n_train=len(sels),
           minutes=round((time.time() - t0) / 60, 1))
    print("FPRT_DONE", flush=True)


if __name__ == "__main__":
    main()
