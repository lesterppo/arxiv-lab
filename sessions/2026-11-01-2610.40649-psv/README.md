# 2026-11-01 — Perturb-Select-Vote (PSV)

- Article: https://arxiv.org/abs/2610.40649 — "Rethinking Neural Thickets
  Without Weight Perturbations" (Dufour et al., submitted 29 Oct 2026).
- Scores: AGENT=5, TUNE=1.
- Why picked: the paper's procedure — generate perturbed predictors,
  select on a small validation set, aggregate by voting — is directly
  reusable on a local agent harness as a training-free inference-time
  ensemble, with no GPU and no weight access needed. Repo fit: arxiv-lab
  harness library (`src/arxiv_lab/multiagent/`), where a frozen model
  plus a validation split is the normal operating setup.
- Serving fetch source: **api** (arXiv API, cat:cs.AI, 200 newest, 154 in
  the 3-day window; RSS was legitimately empty — Sunday skipDays).
  This batch overlaps yesterday's RSS screen, so today's screen
  prioritised the 50 titles newer than yesterday's picks (2610.40250)
  plus carried-forward runners-up; PSV and SCARL are the two that clear
  the bar (see the run log for near-misses).

## What was built

`psv.py` — deterministic CPU simulation, numpy only. A frozen softmax
classifier is trained on class-prior-skewed data (train priors vary by
variant; validation/test are balanced), so its errors are a systematic
prior bias — the kind of error a constant logit shift can correct and a
feature-dependent weight perturbation cannot. Two arms, K=32 members each,
top M=8 selected on validation (400 examples), majority vote on test
(800 examples):

- **weight**: full perturbation, matrix + bias, sigma 0.3.
- **logit (PSV)**: model frozen, bias vector only, sigma 0.5.

Three prior-shift variants (A: priors 0.55/0.15/0.15/0.15, signal 2.0;
B: 0.50/0.20/0.15/0.15, signal 1.6; C: 0.40/0.30/0.15/0.15, signal 2.0)
x seeds 1/7/42 = 9 runs. Test: `test_psv_cpu.py`.

## Results (test accuracy, mean over 9 runs)

| arm | mean |
|---|---|
| frozen single model | 0.7565 |
| weight select + vote | 0.7492 |
| weight vote-all | 0.7565 |
| weight best single member | 0.7301 |
| **logit-PSV select + vote** | **0.7772** |
| logit vote-all | 0.7558 |
| logit best single member | 0.7746 |

- Logit-PSV beats the single model by +2.07 points and weight-PSV by
  +2.80 points, winning 9/9 runs on both comparisons.
- Selection matters: select + vote beats vote-all by +2.14 points
  (8/9 runs). Voting adds a small +0.26 points over the best single
  selected member (6/9 runs).

## VERDICT: SUPPORTED

The paper's central claim reproduces directionally at bounded scale:
a frozen model plus random logit biases, validated and voted, matches
and here exceeds full-weight perturbation — without touching any weight.
The mechanism is also made measurable: in this setting the gain comes
from correcting a systematic prior bias, which a constant logit shift
can do and example-dependent weight noise cannot.

## Honest limits

1. Synthetic prior-shift tasks, not the paper's real reasoning/vision
   benchmarks — the +2.07 gain must NOT be compared with the paper's
   task numbers; what transfers is direction + the selection/voting
   decomposition.
2. The weight arm does NOT gain here (0.7492 vs single 0.7565), unlike
   the paper's reasoning tasks where weight perturbations do gain. The
   "logit matches or exceeds weight" comparison therefore holds via
   logit winning, not via both winning; the weight-gain half of the
   paper's story is not reproduced in this setting.
3. The paper's intermediate claim — perturbing a single decoder block
   recovers much of the full-perturbation gain — is NOT separately
   tested: the toy classifier is single-layer, so a last-layer arm
   would be the weight arm itself.
4. No LLM, no vision transfer, no real-model logit-bias test.
5. Cost: PSV uses K=32 members with a validation pass each; the
   paper's cost band is not claimed here.

Graduated to `src/arxiv_lab/multiagent/psv.py` +
`tests/test_multiagent_psv_cpu.py`.
