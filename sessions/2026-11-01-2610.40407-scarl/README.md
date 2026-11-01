# 2026-11-01 — SCARL (Sequence Confidence-Aligned RL)

- Article: https://arxiv.org/abs/2610.40407 — "SCARL: Sequence
  Confidence-Aligned Reinforcement Learning" (Ke et al., submitted
  29 Oct 2026). Code: https://github.com/kezhh163/scarl (not run here;
  the test below is an independent CPU operationalization).
- Scores: AGENT=1, TUNE=4.
- Why picked: a one-hyperparameter advantage reshaping for RLVR that
  claims to fix sequence-confidence misalignment and, as a byproduct,
  entropy collapse — both live issues for the daily fine-tune harness.
  Its claims are estimator properties, so they are testable on CPU
  without a Colab GPU (same justification as the 2026-10-31 TACE
  session). Repo fit: soup-daily-finetune RL harness /
  `src/arxiv_lab/training/`. No Colab account was touched; rotation
  state is unchanged (still `2026-10-30 kyu009009`, next GPU day
  rotates to cyc236ha).
- Serving fetch source: **api** (see the PSV session README for the
  fetch/screen note; this is the second pick of the same screen).

## What was built

`scarl.py` — deterministic CPU implementation + paired tests, stdlib
with numpy for the softmax-policy step only. Operationalization (the
abstract does not print the exact formula, so the abstract's stated
steps are implemented literally): for each group, base advantages are
group-relative (reward minus group mean); the alignment gap is
|confidence − outcome-certified confidence| (1 for the
positive-advantage class, 0 for the negative class); raw magnitudes are
|A| · (1 + lam · gap) with lam = 1.0, renormalized within each class so
the class's total |advantage| budget is preserved exactly.

- S1: 500 random groups (4–12 responses, binary rewards).
- S2: paired softmax-policy steps, seeds 1/7/42 x 200 groups of 8
  responses (3 rewarded), lr 0.5, confidence = current softmax
  probability; the SAME initial logits take one base step and one
  SCARL step. Test: `test_scarl_cpu.py`.

## Results

S1 (exact properties, not approximations):

- Sign preservation error: 0.0 — reshaping never re-signs.
- Per-class budget error: 8.9e-16 (floating-point noise).
- Reduction error when confidences agree within a class: 1.7e-16 —
  SCARL reduces exactly to the base estimator, as claimed.
- Misaligned pair (two equal-advantage successes, confidence 0.1 vs
  0.9): the low-confidence success gets 1.727x the |advantage| of the
  high-confidence success — misaligned samples absorb the force.

S2 (paired step, SCARL minus base, mean over 600 groups):

- Entropy change: +0.0135 — SCARL preserves entropy better than base
  (base steps decrease entropy; SCARL steps decrease it less, and on
  two of three seeds increase it).
- Positive-class confidence spread change: −0.0018 — SCARL equalizes
  same-advantage confidences more than base (small effect).
- Low-confidence success extra prob gain: +0.0014; high-confidence
  success extra gain: −0.0024.
- Confident-error extra prob change: −0.0050 (corrected harder);
  exploratory-error extra change: +0.0007 (corrected less).

## VERDICT: SUPPORTED

Every stated estimator property holds in the CPU test, and the paired
step moves in exactly the directions the paper claims (misaligned
samples absorb force on both the success and the error side, entropy
better preserved). The effects on a single step are small in absolute
probabilities (+0.001 to +0.005), as expected for one lr=0.5 step —
the claim is directional, not a magnitude claim.

## Honest limits

1. Properties, not bit-parity: the paper's exact SCARL formula is not
   in the abstract, so this validates an operationalization built from
   the abstract's stated steps, not the authors' code.
2. No real RLVR run: the paper's headline (+2.4 Pass@32 over DAPO on
   Qwen3-4B-Base, 14/16 settings improved) is NOT reproduced here, and
   Qwen3-4B is below the training track's 8–9B floor, so a full
   reproduction would need an 8–9B backbone and a real reasoning
   benchmark — out of this run's CPU scope.
3. Binary rewards, 8-response groups, softmax-policy toy: real RLVR
   groups, verifiers, and KL terms are absent.
4. Entropy preservation is shown for one step; multi-step collapse
   dynamics are not tested.

Graduated to `src/arxiv_lab/training/scarl.py` +
`tests/test_training_scarl_cpu.py`.

## Colab real-RLVR leg (2026-11-01, Sunday afternoon HKT)

Added after the CPU verdict on Peter's order ("use colab for training
track"). The CPU results above are untouched; this section records the
real-run leg only.

- Design (`colab_scarl_rlvr.py`, staged in this session): BASE vs
  SCARL advantage shaping on GSM8K, same seeds / prompt order / LoRA
  init in both arms so step-1 groups are identical and divergence comes
  only from the reshaping. 3 steps x 12 prompts x group-of-4 rollouts
  per arm; eval on held-out GSM8K: greedy accuracy (32), sampled
  Pass@4 (16x4), mean next-token entropy on eval prefixes (16) as the
  collapse probe. Metrics CSV written incrementally to
  /content/scarl_rlvr_metrics.csv; summary JSON to
  /content/scarl_rlvr_summary.json; TRAIN_DONE printed last on
  success. Pure functions (answer parsing, SCARL reshaping) were
  verified locally before launch: the misaligned-pair ratio reproduces
  the CPU session's 1.727x exactly, budget preservation and
  never-re-signs hold.
- Model: **Qwen3-8B** (base, public, ungated), NF4 QLoRA r=16/alpha=32,
  bf16, all 7 linears, gradient checkpointing. Why: the proven T4
  NF4-QLoRA default (ToolSelect verify3/verify5, AOPD attempt 2, IH
  goal-goal run); Qwen3.5-9B's QLoRA probe went GREEN 2026-10-30 under
  transformers 5.20.2 but a full trainer OOM'd at batch 4 / seq 768 on
  a 14.56GB T4, and a Sunday-afternoon free-tier session argues for
  the choice most likely to complete. Qwen3-4B (the paper's own
  backbone) is below the 8–9B floor and was not used. Probe numbers:
  none measured on GPU — the run died before model load (see below).
- Run: `colab.py train --rotate`, session `scarl-rlvr-1101`, account
  **cyc236ha** (rotation from 2026-10-30 kyu009009), T4, endpoint
  gpu-t4-s-usw4a2-djvwxnl5nm43. Outcome from train-run.json: **DIED**.
  Cause: a one-line driver bug of mine — `torch.cuda.
  get_device_properties(0).total` (the attribute is `total_memory`)
  raised AttributeError right after the pip installs (torchao
  uninstall succeeded) and before model load. No metrics, no CSV, no
  numbers of any kind exist from this leg; nothing was downloaded
  (train-run.json `downloaded: []`).
- **Colab leg verdict: BLOCKED.** The fair test never ran, so the
  session verdict stays **SUPPORTED** on the CPU leg alone; the
  paper's headline (+2.4 Pass@32 over DAPO) remains untested on a real
  run.
- Resume state (one attempt only — no auto-relaunch): session name
  `scarl-rlvr-1101` (server confirms no active sessions; train tore
  it down, no `--keep`); driver `colab_scarl_rlvr.py` in this
  session with the `.total` -> `.total_memory` fix already applied
  but NEVER executed on GPU; a retry needs a fresh instruction from
  Peter and should re-verify the driver past model load. Active
  account restored to kyu008008 after the run; rotation state now
  `2026-11-01 cyc236ha` (next GPU day: ppoppo205).
