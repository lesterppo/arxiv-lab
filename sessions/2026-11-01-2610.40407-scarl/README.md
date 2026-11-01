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
