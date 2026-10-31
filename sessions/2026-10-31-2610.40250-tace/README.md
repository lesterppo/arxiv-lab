# 2026-10-31 — TACE: Distributional Reward Calibration and Budget-Neutral Credit Assignment for Multimodal RLVR

- Article: https://arxiv.org/abs/2610.40250 (submitted 29 Oct 2026;
  Chen, Xu, Wang, Liu, Chen)
- Track: training-method (CPU property tests + synthetic GRPO step;
  no Colab GPU — see "Why no Colab run" below)
- Why picked: AGENT=1, TUNE=4. A GRPO advantage estimator directly
  reusable in a daily-LoRA/RL training harness (repo fit:
  soup-daily-finetune / `Soup` RL path; graduated to
  `src/arxiv_lab/training/tace.py`). Strongest testable training
  method in today's RSS increment; its claims are estimator
  properties, exactly checkable on CPU.
- Verdict: **SUPPORTED** (directional / bounded scale, with the two
  headline limits below stated, not hidden)

## Why no Colab run

TACE's testable claims are mathematical properties of an estimator
(rank map, tail cap, weight renormalization) plus a synthetic GRPO
step — a T4 adds no information over CPU for any of them, and a GPU
session today would only add reclamation risk while the approved
banking77 retry occupies a Colab account. The Colab rotation state is
therefore UNCHANGED by this run: state file still reads
`2026-10-30 kyu009009`, so the next GPU training day rotates to
cyc236ha (pool order). No Colab account was touched by this run.

## What was built

`tace.py` (stdlib only) — S1 rank-Gaussianization OT map
(ranks -> normal quantiles), affine standardization baseline, S2
rank-calibrated clipped weights renormalized to mean exactly one,
token advantages, and shape diagnostics (skew, excess kurtosis,
KS distance to N(0,1)). `test_tace_cpu.py` — deterministic property
tests + a PAIRED synthetic GRPO step (clean and outlier runs share
every prompt, action and reward except the one injected outlier, so
any gradient change is caused by the outlier alone; an earlier
unpaired version confounded this and was discarded).
`results.json` — full measured numbers.

## Results

### S1 — OT map vs affine standardization (5 reward populations, n=512)

| metric | affine | OT-Gaussianized |
|---|---|---|
| skew range across tasks | 9.205 | 5.388 |
| excess-kurtosis range | 100.087 | 27.181 |
| max KS distance to N(0,1) | 0.540 | 0.484 |
| max \|transformed\| | 12.373 | 2.886 |

Tail bound (group n=64, one outlier of magnitude 10 / 100 / 10000):
OT max = **2.1600444232… in all three cases**, exactly
Phi^{-1}(64/65); affine max = 6.090 / 7.910 / 7.937.

### S2 — token weights (trajectory length 16, cap 2.0)

Mean weight exactly 1.0 (to 1e-12), sum exactly 16.0, max weight
1.8824 (bounded). Token-advantage totals: +32.0 for seq signal +2.0,
−32.0 for −2.0 (sign anchoring, mass = |signal| x L), and unchanged
when all contrasts are rescaled x100 (scale invariance; unnormalized
weights fail this by construction).

### Synthetic GRPO step (paired, 16 prompts x 8 samples, 4 actions)

- Gradient-direction stability under the outlier: cosine(clean,
  outlier) = 0.9983 (TACE) vs 0.9884 (affine).
- Outlier-group largest |advantage|: 2.1700 (TACE) vs 2.6458 (affine).
- Gradient-norm ratios (outlier / clean): 0.958 (TACE) vs 0.818
  (affine) — neither moves much at group size 8, because both caps
  bind; the norm ratio does NOT discriminate here and is reported,
  not claimed as a win.
- All CPU checks pass (`TACE_CPU_ALL_PASS`).

## Honest limits (these qualify the verdict)

1. **S1's "common law" is incomplete for discrete rewards.** The KS
   improvement above is driven by the continuous populations; the
   Bernoulli-cliff population (95% zeros) keeps a point mass under
   rank-Gaussianization — tied ranks map to one quantile, so its KS
   distance stays ~0.48. The paper's population-law claim holds in
   this test for continuous reward laws, not for tied/discrete ones.
2. **Finite-sample affine also saturates.** A lone outlier's affine
   z-score saturates at sqrt(n−1) (7.937 ≈ sqrt(63) at n=64), so the
   OT advantage is the much LOWER cap (2.160), not bounded-vs-
   unbounded. The paper's "deterministic tail bound" verifies exactly;
   the naive "affine grows without bound" reading does not survive
   finite-sample checking and is not claimed here.
3. **S2's exact contrast formula is not in the abstract.** S2 is
   validated on its claimed properties (conservation, boundedness,
   sign anchoring, scale invariance) with rank-calibrated contrasts
   as input, not for bit-parity with the paper's unpublished contrast
   definition.
4. No real multimodal RLVR run, no LLM policy, no paper-benchmark
   accuracy comparison (paper: strongest on a majority of its suite,
   not every setting) — none of that is reproducible at this scale.

## Graduation

Mechanism graduated to `src/arxiv_lab/training/tace.py` (stdlib only)
+ `tests/test_training_tace_cpu.py`, exported from
`arxiv_lab.training`. All CPU checks pass there too.
