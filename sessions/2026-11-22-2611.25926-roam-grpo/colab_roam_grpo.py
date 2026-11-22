"""Colab driver — ROAM-GRPO (arXiv:2611.25926), Jeff Duan & Matt Thomson.

Paper claim tested: GRPO collapses to a low-diversity mode; ROAM-GRPO
routes well-solved problems to an off-mode track where the model is
prompted for distinct solution strategies (judged for validity /
distinctness) and self-distilled back into the policy. On Qwen3.5-2B +
1000 MATH lvl5: peak pass@16 +4.7%, correct-embedding cosine distance
~1.5-2x.

Design (single-update GRPO, honest shrink — per the 2026-10-27 rule the
EXPERIMENT is shrunk, never the model):
  Model: Qwen/Qwen3.5-9B-Base first (standing pick), NF4 QLoRA r=16/a=32,
         bf16, transformers==5.20.2 (Qwen3.5 needs 5.x -- 2026-11-20
         RLCDR lesson). Fallback via env MODEL: Qwen/Qwen3-8B with
         TRANSFORMERS_PIN='<5.0' (its proven recipe).
  Data: EleutherAI/hendrycks_math (per-subject configs), level "Level 5"
        for training, test split for eval. N_TRAIN=24, N_EVAL=20.
  v2 shrink (2026-11-22, after the v1 launch measured the real
        generation rate): v1's Phase 1 ran at ~240 s per eval problem
        (EVAL_G=8 x MAX_NEW=384 on Qwen3.5-9B NF4 / T4, GPU 99% busy),
        so the v1 design (~3.2 h for Phase 1 alone) could not finish
        inside the ~4 h reclaim horizon. v2 cuts N_TRAIN 64->24,
        N_EVAL 48->20, EVAL_G 8->4, MAX_NEW 384->256 (pass@4 instead
        of pass@8); estimated total ~3 h at the measured rate. The
        comparison (base vs GRPO vs ROAM) is unchanged, only smaller.
  v3 fix (2026-11-22, train-1122-roam-c died at the first GRPO
        training forward): cuDNN has no backward-capable engine for
        Qwen3.5's depthwise bf16 conv1d on this T4 stack, so
        torch.backends.cudnn.enabled is set False in main() -- the
        native conv implementation supports the backward pass.
  Phase 0 probe: load NF4, 5-token generation, print PROBE_OK + VRAM.
  Phase 1 baseline eval (base model): N_EVAL problems, EVAL_G=4 samples
         each (temp 0.8, top_p 0.95). Metrics: pass@4 and mean pairwise
         cosine distance among correct solutions per problem (hand-rolled
         TF-IDF word 1-2-gram vectors, stdlib only -- no sklearn).
  Phase 2 GRPO arm: fresh LoRA. Per train problem: sample G=4
         completions (temp 0.8), reward 1 if extracted final answer
         (boxed{} primary, #### fallback) matches gold else 0.
         Advantage A_i = (r_i - mean) / std, std floor 1e-4; if all
         rewards equal, skip the update. Per-token logprobs captured at
         sampling via generate(output_scores=True) + log_softmax gather.
         Current logprobs recomputed with a teacher-forcing forward
         pass (batch 1 -- the one deliberate exception to generate-only;
         [1, seq, vocab] logits ~270MB bf16, bounded and freed per
         completion). ratio = exp(new - old), clipped surrogate eps 0.2,
         NO KL term (documented simplification vs the paper, which uses
         the standard GRPO objective with a KL penalty). Gradients
         accumulated over 4 problem-groups, then one AdamW step (lr 1e-5).
         Adapter saved to /content/adapter_grpo.
  Phase 3 ROAM arm: fresh LoRA from base. Same GRPO loop, PLUS
         off-mode mining: after sampling a problem's group, if pass rate
         >= 0.75 (well-solved), prompt for a DIFFERENT method than the
         first correct solution shown (truncated ~600 chars) and sample
         4 more completions at temp 1.0; keep correct ones whose TF-IDF
         cosine distance to every already-kept correct solution for that
         problem is > 0.30; append kept traces to a distillation buffer.
         After the GRPO pass, one SFT epoch over the buffer (lr 2e-4,
         batch 2 x accum 2, seq 512, causal-LM loss on
         fewshot+question+solution text). Adapter saved to
         /content/adapter_roam.
  Phase 4 eval each arm (grpo, roam) exactly like Phase 1 (same eval
         problems, same temp 0.8 / EVAL_G samples) so pass@k and
         diversity are comparable.

Writes /content/roam_grpo_metrics.csv (rows per model/phase/metric) and
/content/roam_grpo_summary.json, appended/flushed incrementally as each
phase completes (a reclaim mid-run still banks completed phases).
Prints TRAIN_DONE only at the very end.

Compromises vs the paper (recorded honestly):
  * Paper: Qwen3.5-2B, 1000 MATH lvl5 problems, full GRPO with KL term,
    multi-epoch, pass@16. Here: 9B-class base (standing model rule),
    24 train / 20 eval problems, G=4 / EVAL_G=4, MAX_NEW=256, single
    GRPO pass, no KL term, one SFT epoch over the mined buffer.
    Directional test of the mechanism (diversity gain from off-mode
    mining + distillation), not a reproduction of the paper's headline
    numbers.
  * Paper judges mined traces for validity AND distinctness with a
    judge; here validity = exact-answer match, distinctness = TF-IDF
    cosine distance > 0.30 (a lexical proxy, stated as such).
  * Paper's off-mode prompt conditions on a distinct-strategy
    instruction; here the first correct solution is shown and a
    different method is requested (same idea, single prompt form).
  * Diversity in the paper is measured on correct-solution embeddings;
    here TF-IDF cosine distance is the proxy (no embedding model is
    loaded on the T4).

OOM lessons baked in (2026-11-20 RLCDR leg 2): generate() only except
the batch-1 teacher-forcing forward above,
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True, no
prepare_model_for_kbit_training(), gradient checkpointing on during
training, small eval batches (8), adapters saved to /content.
"""
import collections
import csv
import json
import math
import os
import re
import subprocess
import sys
import time

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

MODEL = os.environ.get("MODEL", "Qwen/Qwen3.5-9B-Base")
TRANSFORMERS_PIN = os.environ.get("TRANSFORMERS_PIN", "==5.20.2")
N_TRAIN = int(os.environ.get("N_TRAIN", "24"))
N_EVAL = int(os.environ.get("N_EVAL", "20"))
G = int(os.environ.get("G", "4"))              # train group size
EVAL_G = int(os.environ.get("EVAL_G", "4"))    # eval samples per problem
MAX_NEW = int(os.environ.get("MAX_NEW", "256"))
MINE_G = 4          # off-mode samples per well-solved problem
MINE_TEMP = 1.0
WELL_SOLVED = 0.75  # pass-rate threshold for off-mode routing
MINE_MIN_DIST = 0.30  # TF-IDF cosine-distance threshold for keeping a trace
GRPO_LR = 1e-5
GRPO_ACCUM_GROUPS = 4  # optimizer step every 4 problem-groups
GRPO_EPS = 0.2
SFT_LR = 2e-4

METRICS_CSV = "/content/roam_grpo_metrics.csv"
SUMMARY_JSON = "/content/roam_grpo_summary.json"
CSV_FIELDS = ["model", "phase", "metric", "value", "n"]

FEWSHOT = """Solve the problem step by step. Put your final answer in \\boxed{}.

Question: Find the sum of all positive integers $n$ such that $n^2 + 2n + 1$ is a perfect square less than 100.
Solution: Note $n^2 + 2n + 1 = (n+1)^2$, which is always a perfect square. We need $(n+1)^2 < 100$, so $n + 1 < 10$ and $n < 9$. The positive integers are $n = 1, 2, \\dots, 8$, whose sum is $36$.
\\boxed{36}

Question: A bag contains 3 red balls and 2 blue balls. Two balls are drawn without replacement. What is the probability both are red?
Solution: The probability the first is red is $3/5$. Given that, the probability the second is red is $2/4 = 1/2$. So the probability both are red is $\\frac{3}{5} \\cdot \\frac{1}{2} = \\frac{3}{10}$.
\\boxed{\\frac{3}{10}}

"""

BOXED_MARK = "\\boxed{"
HASH_RE = re.compile(r"####\s*\$?(-?[\d,]+(?:\.\d+)?)")
FRAC_RE = re.compile(r"\\frac\{([^{}]+)\}\{([^{}]+)\}")


def _boxed_contents(text):
    """All \\boxed{...} contents, brace-counted (handles \\frac{a}{b})."""
    out = []
    start = 0
    while True:
        i = text.find(BOXED_MARK, start)
        if i < 0:
            break
        depth, j = 1, i + len(BOXED_MARK)
        while j < len(text) and depth > 0:
            if text[j] == "{":
                depth += 1
            elif text[j] == "}":
                depth -= 1
            j += 1
        if depth == 0:
            out.append(text[i + len(BOXED_MARK):j - 1])
        start = j
    return out


def log(msg):
    print(msg, flush=True)


# ---------------------------------------------------------------- pure utils
# (stdlib only -- also exercised by test_roam_grpo_cpu.py on a CPU box)

def normalize_answer(s):
    """Normalize an extracted answer string: float if numeric, else str."""
    if s is None:
        return None
    s = s.strip().replace(",", "").replace("$", "").replace(" ", "")
    s = s.replace("\\!", "").replace("\\,", "")
    m = FRAC_RE.fullmatch(s)
    if m:
        try:
            return float(m.group(1)) / float(m.group(2))
        except (ValueError, ZeroDivisionError):
            return s
    try:
        return float(s)
    except ValueError:
        return s


def extract_answer(text):
    """Last \\boxed{...} in text (primary); else last '#### <number>'.

    Returns a normalized value (float or str), or None if absent.
    """
    if not text:
        return None
    boxed = _boxed_contents(text)
    if boxed:
        return normalize_answer(boxed[-1])
    hashed = HASH_RE.findall(text)
    if hashed:
        return normalize_answer(hashed[-1])
    return None


def answers_match(pred, gold):
    if pred is None or gold is None:
        return False
    if isinstance(pred, float) and isinstance(gold, float):
        return abs(pred - gold) < 1e-6
    return str(pred) == str(gold)


def _tokens(text):
    return re.findall(r"[a-z0-9]+", text.lower())


def _ngrams(tokens):
    grams = list(tokens)
    grams += [" ".join(tokens[i:i + 2]) for i in range(len(tokens) - 1)]
    return grams


def tfidf_vectors(texts):
    """Hand-rolled TF-IDF (word 1-2-grams, smoothed idf). Stdlib only."""
    docs = [collections.Counter(_ngrams(_tokens(t))) for t in texts]
    df = collections.Counter()
    for d in docs:
        for term in d:
            df[term] += 1
    n_docs = len(texts)
    vecs = []
    for d in docs:
        total = sum(d.values()) or 1
        vec = {}
        for term, c in d.items():
            idf = math.log((n_docs + 1) / (df[term] + 1)) + 1.0
            vec[term] = (c / total) * idf
        vecs.append(vec)
    return vecs


def cosine_sim(v1, v2):
    dot = sum(a * v2.get(t, 0.0) for t, a in v1.items())
    n1 = math.sqrt(sum(a * a for a in v1.values()))
    n2 = math.sqrt(sum(a * a for a in v2.values()))
    if n1 == 0.0 or n2 == 0.0:
        return 0.0
    return dot / (n1 * n2)


def mean_pairwise_cosine_distance(texts):
    """Mean (1 - cosine) over all pairs of correct solutions.

    Returns None when fewer than 2 texts (no pair exists).
    """
    if len(texts) < 2:
        return None
    vecs = tfidf_vectors(texts)
    dists = []
    for i in range(len(vecs)):
        for j in range(i + 1, len(vecs)):
            dists.append(1.0 - cosine_sim(vecs[i], vecs[j]))
    return sum(dists) / len(dists)


def min_distance_to_kept(candidate, kept_texts):
    """Min TF-IDF cosine distance from candidate to each kept text.

    Returns None when kept_texts is empty.
    """
    if not kept_texts:
        return None
    vecs = tfidf_vectors([candidate] + list(kept_texts))
    cand = vecs[0]
    return min(1.0 - cosine_sim(cand, v) for v in vecs[1:])


def compute_advantages(rewards, std_floor=1e-4):
    """Group-relative advantages (r - mean) / std.

    Returns None when std < floor (all rewards equal) -> skip update.
    """
    n = len(rewards)
    mean = sum(rewards) / n
    var = sum((r - mean) ** 2 for r in rewards) / n
    std = math.sqrt(var)
    if std < std_floor:
        return None
    return [(r - mean) / std for r in rewards]


def clipped_surrogate_loss(completions, eps=0.2):
    """Token-mean clipped surrogate (pure-math version, for the CPU test).

    completions: list of (old_logps, new_logps, advantage) with
    old/new_logps lists of per-token log-probs of equal length.
    Returns -mean over tokens of min(ratio*A, clip(ratio)*A).
    """
    total, n_tok = 0.0, 0
    for old_lps, new_lps, adv in completions:
        for o, nw in zip(old_lps, new_lps):
            ratio = math.exp(nw - o)
            clipped = max(1.0 - eps, min(1.0 + eps, ratio))
            total += min(ratio * adv, clipped * adv)
            n_tok += 1
    if n_tok == 0:
        return 0.0
    return -total / n_tok


def should_mine(pass_rate, threshold=WELL_SOLVED):
    """Off-mode routing rule: well-solved problems go to the mining track."""
    return pass_rate >= threshold


def make_metric_row(model, phase, metric, value, n):
    return {"model": model, "phase": phase, "metric": metric,
            "value": value, "n": n}


# ---------------------------------------------------------------- IO helpers

def append_metric_rows(rows):
    """Append rows to the metrics CSV (header on first write), flush."""
    exists = os.path.exists(METRICS_CSV)
    with open(METRICS_CSV, "a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
        if not exists:
            w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in CSV_FIELDS})
    log(f"  metrics flushed: {len(rows)} rows -> {METRICS_CSV}")


def write_summary(summary):
    with open(SUMMARY_JSON, "w") as fh:
        json.dump(summary, fh, indent=2)


def install():
    log("== pip installs ==")
    subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "torchao"],
                   check=False)
    subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                    f"transformers{TRANSFORMERS_PIN}", "bitsandbytes",
                    "accelerate", "peft", "datasets", "hf_transfer"],
                   check=True)
    log("== installs done ==")


# ---------------------------------------------------------------- main

def main():
    import torch
    # NOTE (2026-11-22, train-1122-roam-c): Qwen3.5's Gated Delta Net
    # layers run a depthwise bf16 conv1d (causal_conv1d_fn, groups=
    # hidden_size). On this T4/cuDNN stack cuDNN has no backward-capable
    # engine for that op -- generation (no-grad) works, but the first
    # training forward dies with "GET was unable to find an engine to
    # execute this computation". Disabling cuDNN forces PyTorch's native
    # conv implementation, which supports the backward pass. The conv
    # is a tiny fraction of the compute, so the speed cost is negligible.
    torch.backends.cudnn.enabled = False
    from datasets import load_dataset
    from transformers import (AutoModelForCausalLM, AutoTokenizer,
                              BitsAndBytesConfig)

    summary = {"model": MODEL, "transformers_pin": TRANSFORMERS_PIN,
               "n_train": N_TRAIN, "n_eval": N_EVAL, "G": G,
               "eval_G": EVAL_G, "max_new_tokens": MAX_NEW,
               "phases": {}}
    write_summary(summary)

    log(f"== model: {MODEL} | transformers pin: {TRANSFORMERS_PIN} "
        f"| N_TRAIN={N_TRAIN} N_EVAL={N_EVAL} G={G} EVAL_G={EVAL_G} ==")
    tok = AutoTokenizer.from_pretrained(MODEL)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"  # generation
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_use_double_quant=True,
                             bnb_4bit_compute_dtype=torch.bfloat16)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, quantization_config=bnb, device_map="auto",
        torch_dtype=torch.bfloat16)
    model.config.use_cache = False

    # ---- Phase 0: probe ----
    probe_ids = tok("The capital of France is", return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**probe_ids, max_new_tokens=5, do_sample=False,
                             pad_token_id=tok.eos_token_id)
    log("PROBE_OK: " + tok.decode(out[0][probe_ids["input_ids"].shape[1]:],
                                  skip_special_tokens=True).strip())
    log(f"VRAM: {torch.cuda.memory_allocated() / 2**30:.2f} GiB allocated")

    # ---- data: MATH level 5 only ----
    # NOTE (2026-11-22): "hendrycks/competition_math" does not exist on
    # the Hub (DatasetNotFoundError killed the first launch after a
    # GREEN probe). The same data lives at "EleutherAI/hendrycks_math"
    # under per-subject configs; fields (problem/level/type/solution)
    # are identical.
    import random as _random
    _configs = ("algebra", "counting_and_probability", "geometry",
                "intermediate_algebra", "number_theory", "prealgebra",
                "precalculus")
    _train_pool, _test_pool = [], []
    for _cfg in _configs:
        try:
            _dtr = load_dataset("EleutherAI/hendrycks_math", _cfg,
                                split="train")
            _dte = load_dataset("EleutherAI/hendrycks_math", _cfg,
                                split="test")
        except Exception as e:  # noqa: BLE001 - one bad config must not kill the run
            log(f"  dataset config {_cfg} failed to load: {e}")
            continue
        _train_pool += [ex for ex in _dtr if ex["level"] == "Level 5"]
        _test_pool += [ex for ex in _dte if ex["level"] == "Level 5"]
    _random.Random(42).shuffle(_train_pool)
    _random.Random(43).shuffle(_test_pool)
    train_qs = _train_pool[:N_TRAIN]
    eval_qs = _test_pool[:N_EVAL]
    log(f"level-5 problems: train={len(train_qs)} eval={len(eval_qs)}")
    summary["phases"]["data"] = {"n_train_problems": len(train_qs),
                                 "n_eval_problems": len(eval_qs)}
    write_summary(summary)

    def gold_of(ex):
        return extract_answer(ex["solution"])

    def prompt_for(ex):
        return FEWSHOT + f"Question: {ex['problem']}\nSolution:"

    def batch_generate(mdl, prompts, max_new, temperature, top_p, batch,
                       want_scores=False):
        """Sample one batch of completions from model mdl.

        Returns list of dicts: {text, ids, old_logprobs, n_tokens}.
        Prompts in one call are identical (one problem x G copies), so
        no padding-length mismatch arises.
        """
        outs = []
        t0 = time.time()
        for i in range(0, len(prompts), batch):
            chunk = prompts[i:i + batch]
            enc = tok(chunk, return_tensors="pt", padding=True).to(mdl.device)
            n_in = enc["input_ids"].shape[1]
            with torch.no_grad():
                gen = mdl.generate(
                    **enc, max_new_tokens=max_new, do_sample=True,
                    temperature=temperature, top_p=top_p,
                    pad_token_id=tok.eos_token_id,
                    output_scores=want_scores, return_dict_in_generate=want_scores)
            seqs = gen.sequences if want_scores else gen
            scores = gen.scores if want_scores else None
            for j, g in enumerate(seqs):
                ids = g[n_in:].tolist()
                if tok.eos_token_id in ids:  # trim at first EOS
                    ids = ids[:ids.index(tok.eos_token_id)]
                old_lps = None
                if want_scores:
                    old_lps = []
                    for step, tid in enumerate(ids):
                        logits = scores[step][j].float()
                        lp = torch.log_softmax(logits, dim=-1)[tid].item()
                        old_lps.append(lp)
                outs.append({"text": tok.decode(ids, skip_special_tokens=True),
                             "ids": ids, "old_logprobs": old_lps,
                             "n_tokens": len(ids)})
            if (i // batch) % 10 == 9:
                log(f"  gen {len(outs)}/{len(prompts)} "
                    f"({time.time() - t0:.0f}s elapsed)")
        return outs

    def eval_model(tag, mdl):
        """Phase-1-style eval: pass@EVAL_G + correct-solution diversity."""
        log(f"== eval {tag}: {len(eval_qs)} problems x {EVAL_G} samples ==")
        mdl.eval()
        mdl.config.use_cache = True
        t0 = time.time()
        n_pass, dists, gen_toks = 0, [], 0
        for idx, ex in enumerate(eval_qs):
            gold = gold_of(ex)
            samps = batch_generate(mdl, [prompt_for(ex)] * EVAL_G, MAX_NEW,
                                   0.8, 0.95, batch=8)
            correct_texts = []
            for s in samps:
                gen_toks += s["n_tokens"]
                if answers_match(extract_answer(s["text"]), gold):
                    correct_texts.append(s["text"])
            if correct_texts:
                n_pass += 1
            d = mean_pairwise_cosine_distance(correct_texts)
            if d is not None:
                dists.append(d)
            if (idx + 1) % 10 == 0:
                log(f"  eval {tag} {idx + 1}/{len(eval_qs)} "
                    f"({time.time() - t0:.0f}s)")
        torch.cuda.empty_cache()
        row_pass = n_pass / len(eval_qs)
        row_dist = (sum(dists) / len(dists)) if dists else 0.0
        row_tok = gen_toks / (len(eval_qs) * EVAL_G)
        rows = [make_metric_row(tag, f"eval_{tag}", f"pass@{EVAL_G}", round(row_pass, 4),
                                len(eval_qs)),
                make_metric_row(tag, f"eval_{tag}",
                                "mean_pairwise_cosine_distance_correct",
                                round(row_dist, 4), len(dists)),
                make_metric_row(tag, f"eval_{tag}", "n_eval", len(eval_qs),
                                len(eval_qs)),
                make_metric_row(tag, f"eval_{tag}", "mean_gen_tokens",
                                round(row_tok, 1), len(eval_qs) * EVAL_G)]
        append_metric_rows(rows)
        summary["phases"][f"eval_{tag}"] = {
            f"pass@{EVAL_G}": round(row_pass, 4),
            "mean_pairwise_cosine_distance_correct": round(row_dist, 4),
            "n_problems_with_2plus_correct": len(dists),
            "mean_gen_tokens": round(row_tok, 1)}
        write_summary(summary)
        log(f"eval {tag}: pass@{EVAL_G}={row_pass:.3f} "
            f"mean_pairwise_cos_dist={row_dist:.3f} "
            f"mean_tok={row_tok:.1f} ({time.time() - t0:.0f}s)")
        return {f"pass@{EVAL_G}": row_pass, "dist": row_dist}

    # ---- Phase 1: baseline eval ----
    base_results = eval_model("base", model)
    summary["phases"]["baseline"] = base_results

    # ---- shared GRPO machinery ----
    from peft import LoraConfig, get_peft_model, PeftModel

    def lora_config():
        return LoraConfig(r=16, lora_alpha=32, lora_dropout=0.0,
                          target_modules=["q_proj", "k_proj", "v_proj",
                                          "o_proj", "gate_proj", "up_proj",
                                          "down_proj"],
                          task_type="CAUSAL_LM")

    def current_logprobs(peft_model, prompt_ids, comp_ids):
        """Teacher-forcing logprobs for one completion (batch 1).

        Deliberate generate-only exception: [1, seq, vocab] logits are
        ~270MB bf16 at seq~900 -- bounded, freed after each completion.
        """
        ids = torch.tensor([prompt_ids + comp_ids], device=peft_model.device)
        out = peft_model(input_ids=ids, attention_mask=torch.ones_like(ids))
        logits = out.logits[0]
        lps = []
        for t, tid in enumerate(comp_ids):
            pos = len(prompt_ids) + t - 1
            lps.append(torch.log_softmax(logits[pos].float(), dim=-1)[tid])
        del out, logits
        return torch.stack(lps)

    def grpo_pass(peft_model, tag, mine=False):
        """One GRPO pass over the train problems.

        mine=True adds the ROAM off-mode mining track. Returns stats.
        """
        log(f"== GRPO pass {tag} (mine={mine}): {len(train_qs)} problems ==")
        peft_model.train()
        peft_model.config.use_cache = False
        peft_model.gradient_checkpointing_enable()
        peft_model.enable_input_require_grads()
        opt = torch.optim.AdamW(
            (p for p in peft_model.parameters() if p.requires_grad),
            lr=GRPO_LR)
        stats = {"n_train": len(train_qs), "n_grpo_updates": 0,
                 "n_well_solved": 0, "n_mined_traces": 0,
                 "mined_distances": []}
        buffer = []  # (question, mined solution text) for SFT distillation
        t0 = time.time()
        since_step = 0
        opt.zero_grad()
        for idx, ex in enumerate(train_qs):
            gold = gold_of(ex)
            prompt = prompt_for(ex)
            samps = batch_generate(peft_model, [prompt] * G, MAX_NEW, 0.8,
                                   0.95, batch=4, want_scores=True)
            rewards = [1 if answers_match(extract_answer(s["text"]), gold)
                       else 0 for s in samps]
            pass_rate = sum(rewards) / len(rewards)
            # ---- off-mode mining (ROAM arm only) ----
            if mine and should_mine(pass_rate):
                stats["n_well_solved"] += 1
                correct_texts = [s["text"] for s, r in zip(samps, rewards)
                                 if r == 1]
                first_correct = correct_texts[0][:600]
                off_prompt = (FEWSHOT + f"Question: {ex['problem']}\n"
                              "Here is one correct solution:\n"
                              f"{first_correct}\n\n"
                              "Now solve this problem using a DIFFERENT "
                              "method or approach than the one shown above. "
                              "Put your final answer in \\boxed{}.\nSolution:")
                peft_model.eval()
                mined = batch_generate(peft_model, [off_prompt] * MINE_G,
                                       MAX_NEW, MINE_TEMP, 0.95, batch=4)
                peft_model.train()
                kept = list(correct_texts)
                for m in mined:
                    if not answers_match(extract_answer(m["text"]), gold):
                        continue
                    d = min_distance_to_kept(m["text"], kept)
                    if d is not None and d > MINE_MIN_DIST:
                        kept.append(m["text"])
                        buffer.append((ex["problem"], m["text"]))
                        stats["n_mined_traces"] += 1
                        stats["mined_distances"].append(d)
                log(f"  problem {idx + 1}: well-solved "
                    f"(pass {pass_rate:.2f}), mined "
                    f"{stats['n_mined_traces']} traces so far")
            # ---- GRPO update ----
            advs = compute_advantages(rewards)
            if advs is None:
                if (idx + 1) % 10 == 0:
                    log(f"  train {tag} {idx + 1}/{len(train_qs)} "
                        f"({time.time() - t0:.0f}s, skipped all-equal)")
                continue
            prompt_ids = tok(prompt, return_tensors="pt")["input_ids"][0].tolist()
            peft_model.train()
            loss, n_used = None, 0
            for s, adv in zip(samps, advs):
                if not s["ids"]:
                    continue
                n_used += 1
                new_lps = current_logprobs(peft_model, prompt_ids, s["ids"])
                old_lps = torch.tensor(s["old_logprobs"],
                                       device=peft_model.device)
                ratio = torch.exp(new_lps - old_lps)
                clipped = torch.clamp(ratio, 1 - GRPO_EPS, 1 + GRPO_EPS)
                surr = torch.minimum(ratio * adv, clipped * adv)
                comp_loss = -surr.mean()
                loss = comp_loss if loss is None else loss + comp_loss
            if loss is not None and n_used > 0:
                (loss / n_used).backward()
                since_step += 1
            if since_step >= GRPO_ACCUM_GROUPS:
                torch.nn.utils.clip_grad_norm_(
                    (p for p in peft_model.parameters() if p.requires_grad),
                    1.0)
                opt.step()
                opt.zero_grad()
                stats["n_grpo_updates"] += 1
                since_step = 0
            if (idx + 1) % 10 == 0:
                log(f"  train {tag} {idx + 1}/{len(train_qs)} "
                    f"updates={stats['n_grpo_updates']} "
                    f"({time.time() - t0:.0f}s)")
            torch.cuda.empty_cache()
        if since_step > 0:  # flush remaining accumulated grads
            torch.nn.utils.clip_grad_norm_(
                (p for p in peft_model.parameters() if p.requires_grad), 1.0)
            opt.step()
            opt.zero_grad()
            stats["n_grpo_updates"] += 1
        if hasattr(peft_model, "gradient_checkpointing_disable"):
            peft_model.gradient_checkpointing_disable()
        mean_mined = (sum(stats["mined_distances"]) / len(stats["mined_distances"])
                      if stats["mined_distances"] else 0.0)
        rows = [make_metric_row(tag, f"train_{tag}", "n_train",
                                len(train_qs), len(train_qs)),
                make_metric_row(tag, f"train_{tag}", "n_grpo_updates",
                                stats["n_grpo_updates"], len(train_qs)),
                make_metric_row(tag, f"train_{tag}", "n_well_solved",
                                stats["n_well_solved"], len(train_qs)),
                make_metric_row(tag, f"train_{tag}", "n_mined_traces",
                                stats["n_mined_traces"], len(train_qs)),
                make_metric_row(tag, f"train_{tag}", "mean_mined_distance",
                                round(mean_mined, 4),
                                len(stats["mined_distances"]))]
        append_metric_rows(rows)
        stats["mean_mined_distance"] = round(mean_mined, 4)
        summary["phases"][f"train_{tag}"] = stats
        write_summary(summary)
        log(f"GRPO pass {tag} done: updates={stats['n_grpo_updates']} "
            f"well_solved={stats['n_well_solved']} "
            f"mined={stats['n_mined_traces']} "
            f"mean_mined_dist={mean_mined:.3f} ({time.time() - t0:.0f}s)")
        return stats, buffer

    def sft_buffer(peft_model, buffer):
        """One SFT epoch over the mined distillation buffer."""
        from datasets import Dataset as HFDataset
        from transformers import (DataCollatorForLanguageModeling, Trainer,
                                  TrainingArguments)
        log(f"== SFT distillation: {len(buffer)} mined traces ==")
        texts = [FEWSHOT + f"Question: {q}\nSolution: {sol}" + tok.eos_token
                 for q, sol in buffer]
        tok.padding_side = "right"
        enc = tok(texts, truncation=True, max_length=512)
        dset = HFDataset.from_dict({"input_ids": enc["input_ids"],
                                    "attention_mask": enc["attention_mask"]})
        args = TrainingArguments(
            output_dir="/content/ckpt_roam_sft", per_device_train_batch_size=2,
            gradient_accumulation_steps=2, num_train_epochs=1,
            learning_rate=SFT_LR, bf16=True, logging_steps=10,
            save_strategy="no", report_to=[], seed=42,
            optim="adamw_torch", max_grad_norm=1.0,
            gradient_checkpointing=True)
        trainer = Trainer(model=peft_model, args=args, train_dataset=dset,
                          data_collator=DataCollatorForLanguageModeling(
                              tok, mlm=False))
        trainer.train()
        log("SFT distillation done")
        tok.padding_side = "left"

    # ---- Phase 2: GRPO arm ----
    peft_grpo = get_peft_model(model, lora_config())
    grpo_stats, _ = grpo_pass(peft_grpo, "grpo", mine=False)
    peft_grpo.save_pretrained("/content/adapter_grpo")
    log("adapter_grpo saved")
    model = peft_grpo.unload()
    del peft_grpo
    torch.cuda.empty_cache()

    # ---- Phase 3: ROAM arm ----
    peft_roam = get_peft_model(model, lora_config())
    roam_stats, buffer = grpo_pass(peft_roam, "roam", mine=True)
    if buffer:
        sft_buffer(peft_roam, buffer)
    else:
        log("WARNING: distillation buffer empty -- no SFT step; "
            "ROAM arm is GRPO-only this run")
    peft_roam.save_pretrained("/content/adapter_roam")
    log("adapter_roam saved")
    model = peft_roam.unload()
    del peft_roam
    torch.cuda.empty_cache()

    # ---- Phase 4: eval both arms exactly like Phase 1 ----
    model.config.use_cache = True
    for arm in ("grpo", "roam"):
        path = f"/content/adapter_{arm}"
        if not os.path.isdir(path):
            log(f"skip eval arm {arm}: no adapter")
            continue
        mdl = PeftModel.from_pretrained(model, path)
        eval_model(arm, mdl)
        del mdl
        torch.cuda.empty_cache()

    summary["done"] = True
    write_summary(summary)
    log("== metrics written to /content ==")
    log("TRAIN_DONE")


if __name__ == "__main__":
    install()
    main()
