# The Frontier Must Grow — self-proposed, interpreter-labelled training

- **Article:** https://arxiv.org/abs/2611.20049 — "The Frontier Must Grow:
  Self-proposed, interpreter-labelled training roughly doubles a 4B model's
  accuracy on a fresh exam as its verified frontier grows" (Martin Schröder,
  Nov 2026)
- **Date:** 2026-11-17 (fetch: arXiv API, 145 papers in 3-day window)
- **Scores:** TUNE **5**, AGENT 2
- **Verdict: BLOCKED** — no fair test was possible: both training attempts
  died with CUDA OOM in the first training forward pass before any
  adapter was trained or any round evaluated. The one automatic fallback
  is consumed; no further attempts today (2026-11-05 rule).

## Why picked

A model can learn from problems it poses itself when an executable oracle
supplies the answers — no teacher model, no external dataset, no human
labels. Qwen3-4B proposed Lua 5.1 programs; those it predicted wrong at
least once in four samples joined a pool labelled by the interpreter; each
round trained a fresh adapter from the base on the growing pool. On a fresh
2340-item exam the paper reports 8.3% -> 16.7%. The loop is a clean fit
for the soup-daily-finetune daily-LoRA harness (same NF4-QLoRA stack, same
"train a small adapter every day" pattern), and the oracle makes the test
fully self-contained on one Colab T4.

## What was built

- `colab_frontier_driver.py` — Colab T4 driver: Lua 5.1 oracle
  (apt `lua5.1`, reference = interpreter output run twice, must agree),
  fresh-exam generator (15 stdlib templates x 20 draws), proposal /
  prediction / eval under the paper's direct-completion protocol, fresh
  NF4 QLoRA adapter from the base each round, sign tests per step,
  metrics flushed to /content after every phase.
- `frontier_metrics.csv`, `frontier_eval_rows.csv`,
  `frontier_results.json`, `frontier_pool.jsonl`, `frontier_exam.json`
  — downloaded results (in the session after the run completes).

## Bounded replication (shrink the experiment, never the model)

| | paper (S4) | this run |
|---|---|---|
| lineages | 4 | 1 |
| rounds | 3 | 2 |
| new frontier programs / round | 300 / 600 / 1200 | 150 / 300 |
| cumulative pool | 300 / 900 / 2100 | 150 / 450 |
| fresh exam | 2340 items, 78 templates | 300 items, 15 templates |
| model | Qwen3-4B | Qwen3-8B (see model note) |
| training | fresh LoRA / round, 2 passes, batch 8 | fresh NF4 QLoRA / round, r16/a32, lr 1e-4, 2 epochs, batch 8 |
| selection | >=1 wrong of 4 samples, temp 1.0 | same |
| eval | direct completion, greedy, thinking off, last-fenced-block grading | same (max 128 new tokens; paper 2048 — answers are single values) |

## Model note

Qwen3.5-9B is tried first in-driver per the standing pick, but it is
**known-blocked on the transformers pin**: transformers 4.57.6 (newest
<5.0) does not recognise model type `qwen3_5` — verified 2026-11-16
(SPRA run), where the 9B attempt failed fast and the run proceeded on
Qwen3-8B. A transformers-from-source install was NOT attempted (same
judgment as yesterday: it risks the proven 8B path). The driver therefore
runs on the proven carrier **Qwen/Qwen3-8B**. The paper's 4B model runs
as **no arm**: the rule allows a paper-specific small model only as an
ADDITIONAL arm beside a mandatory 8–9B arm, and one day's budget does
not cover two full arms. If Qwen3-8B also fails to load, the run is
BLOCKED with no further automatic fallback (8B is already the smallest
proven model in the 8–9B class; a 4B fallback would violate the
never-shrink-the-model floor).

Deliberate divergences from the paper (recorded, not hidden):
- Proposal prompt is our construction — the paper does not publish its
  exact proposal wording.
- Safety filter: programs containing `os.` / `io.` / `loadfile` /
  `dofile` / `loadstring` / `load(` / `require` / `collectgarbage` /
  `debug.` are rejected before execution (the paper runs proposals raw).
- No dev gate (the paper's gate filtered nothing); the fresh exam is the
  only eval, plus per-template breakdowns.
- One serving path for every model call (transformers NF4, batched
  generation) instead of the paper's Brain serve/train split — the
  comparison that matters (base vs adapter) uses the identical path.

## Results

- **Base (Qwen3-8B, fresh exam, greedy, direct-completion protocol):
  5/292 = 1.7% +/- 1.5%.** Much lower than the paper's 8.3% for Qwen3-4B
  on its 78-template exam — the base model here rarely produces the
  required single fenced code block and its generations run to the
  128-token cap (never emits EOS), so the score mixes format failure
  with content failure; the saved data cannot separate the two, which
  is a stated gap. Training on the format will move this number even
  if it only teaches the format — the round deltas must be read with
  that in mind.
- **Round 1 proposals: complete — 150 frontier records** (360 proposed,
  171 valid, `frontier_pool.jsonl`). No round was trained or evaluated:
  both attempts OOM'd in the first training forward pass (see below).
  No adapter exists; no gen1/gen2 exam numbers exist. The paper's
  claim (accuracy grows with the verified frontier) is **untested**,
  not refuted.

## What blocked it (both attempts, same place)

`torch.OutOfMemoryError` in `transformers/.../loss_utils.py`
`fixed_cross_entropy` during the first `train_adapter` forward pass:
attempt 2 (batch 8 x up to 512 tokens) "Tried to allocate 372 MiB,
14.46 GiB in use"; fallback attempt (micro-batch 4 x 320, grad-accum 2)
"Tried to allocate 354 MiB, 14.42 GiB in use". The batch shrink saved
only ~0.3 GiB, so the 151k-vocab logits were **not** the dominant cost:
13.9 GiB was already allocated with a 5.7 GiB model at batch 4 x 320.
The likely cause is the training-path setup difference from the
working SPRA driver (2026-11-16, same model, batch 8 x 256, peak
12.2 GiB): SPRA sets `model.config.use_cache = False` and calls
`model.enable_input_require_grads()` in its train path; this driver
set `use_cache = True` in `build_model` and never called
`enable_input_require_grads()`, so gradient checkpointing most likely
never took effect and full layer activations stayed resident. That is
a diagnosis from the evidence, not a verified fix — it is staged as
the first change for the next attempt (below), not claimed as proven.

Attempt history (all on T4; rotation: kyu008008 -> kyu009009):
1. `train-1117-frontier` (kyu008008) — DIED in ~2 min: driver bug,
   `setup_lua` FileNotFoundError on a missing binary. Fixed
   (probes catch it; Lua 5.1.5 source-build fallback tier).
2. `train-1117-frontierb` (kyu009009) — host monitor killed by a VM
   replacement; VM-side driver survived (PID 703), completed base
   eval (5/292) and round-1 proposals (150 frontier), then DIED with
   the OOM above in round-1 training.
3. `train-1117-frontierc` launch 1 — PROVISION_FAILED (kyu009009
   auth-expired); forced token refresh fixed it; no GPU work done.
4. `train-1117-frontierc` launch 2 (the one automatic fallback) —
   host monitor killed by a second VM replacement; VM-side driver
   survived (PID 496), resumed from the pool, then DIED with the
   same OOM in round-1 training. Fallback consumed.

## Honest limits

- Single lineage, two rounds, 450 cumulative records — the paper's
  third round and its 2100-record scale are not reached; the rounds-vs-
  matched-data question (paper: inconclusive) is not tested here.
- Base accuracy on our 15-template exam is expected to be higher than
  the paper's 8.3% on 78 templates (simpler templates, stronger 8B
  base) — headroom is smaller; report the base number honestly.
- Adapters are saved to /content/adapters/ on the VM but the wrapper
  only downloads the listed result files; adapters die with the session
  (pool + exam + metrics survive, which is what the verdict rests on;
  a retrain from the pool reproduces them).
- The paper's own honest limits carry over: round 1 gained nothing on
  its fresh exam; the rounds' edge over matched data is inconclusive;
  no gain was shown on repository tasks or for a 27B agent.

## Resume state

- Driver + exam + pool definitions are in this session; a rerun resumes
  from the pool file (`frontier_pool.jsonl`, 150 round-1 records) and
  exam file (`frontier_exam.json`, 292 items) with the same seed
  (20261117) — the driver already implements this resume path and it
  was verified live in attempt 4 (exam + pool loaded, proposals
  skipped, base eval skipped).
- **First change for the next attempt:** mirror the SPRA train path —
  `model.config.use_cache = False` during training (restore `True`
  only for generation/eval) and call
  `model.enable_input_require_grads()` after wrapping with peft,
  before `gradient_checkpointing_enable()`. Keep micro-batch 4 x
  grad-accum 2, max_len 320. Then re-run the fallback driver
  unchanged otherwise; it will train gen1 on the 150-record pool.
- A third round (target +600 new, cumulative 1050) and a matched-data
  control (one fit on the base-proposed frontier) are the natural
  follow-ups if round 2 shows a gain.
- Colab sessions used today: `train-1117-frontier` (stopped),
  `train-1117-frontierb` (stopped), `train-1117-frontierc` (stopped).
  Account for the day: kyu009009 (rotation file). Tomorrow's run
  takes cyc236ha per the pool order.
