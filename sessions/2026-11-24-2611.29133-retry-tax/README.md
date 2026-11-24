# Session — 2026-11-24 — arXiv:2611.29133

## Article
**The Retry Tax: Why Static Same-Task Retries Can Cost More per Completion**
https://arxiv.org/abs/2611.29133

## Why picked + repo fit
AGENT 5, TUNE 1. A static retry policy (stop at verified success or an
attempt cap, reset prompt + workspace, no feedback between attempts) is
the default retry loop in every agent harness — including Hermes
AgentRunner and arxiv-lab's own loops — and the paper shows it has a
built-in cost trap: under heterogeneous per-task success
probabilities, cost per completion is at least the one-shot cost and
marginal cost per additional completion is nondecreasing, because
easier tasks succeed early and the unresolved mix shifts toward harder
tasks. The mechanism is pure probability, deterministic on CPU, and
Rust-portable. The paper's replay numbers (one OpenHands config:
59.7% completion at 1 attempt → 70.0% at 4; cost per completion
$0.70 → $1.32) give calibration endpoints to check against.

## What was built
- `src/arxiv_lab/agent/retry_tax.py` — the mechanism, graduated to
  the repo (stdlib only): closed-form `completion_prob` /
  `expected_attempts` / `population_metrics` / `marginal_cost`
  (numerically robust `-expm1(cap * log1p(-p))` form, plus direct
  per-task increments for marginal cost — no cancellation), a seeded
  Monte-Carlo `simulate_population`, the paper's `replay_runs`
  procedure (reorder per-task attempt sequences, stop at verified
  success or cap, average), `unresolved_mean_p` (survivor-selection
  diagnostic), and `calibrated_mixture` (two-point population
  calibrated to the paper's replay endpoints).
- `test_retry_tax_cpu.py` — deterministic CPU test, E0–E6
  (see Results).
- `rust/` — Rust second stage (std-only, zero crates): `verify`
  binary mirroring the Python test's experiments and assertions,
  unit + deep tests, parity sweep, soak, thread identity.
  Rust leg verdict: **VERIFIED** (see below).

## Results — CPU test (primary verdict carrier)
`test_retry_tax_cpu.py`: ALL CHECKS PASSED in ~12 s.
Results in `results.json`.

| Experiment | Result |
|---|---|
| E0 closed-form inequalities (20k-task calibrated mixture) | completion 0.599 → 0.712 (caps 1 → 4), CPC $0.698 → $1.193, marginal cost $2.24 → $8.07, all nondecreasing; C(1) 0.599 vs paper 0.597, C(4) 0.712 vs paper 0.700 |
| E1 Monte-Carlo simulation (seeded) | same inequalities measured from simulated attempts; C(4) 0.710 agrees with closed form 0.712 |
| E2 replay (the paper's method) | 40 reorderings of recorded attempt sequences: completion 0.600 → 0.713, CPC $0.697 → $1.190 (paper: 0.597 → 0.700, $0.70 → $1.32) — same direction, same order |
| E3 homogeneous population (equality case) | CPC exactly constant $1.045000 at every cap (== cost/p) |
| E4 survivor selection | mean p of unresolved tasks 0.599 → 0.188 → 0.064 → 0.050 after 0/1/2/4 attempts |
| E5 determinism | repeat simulation and replay reproduce identical numbers |
| E6 edge cases | p=0 never completes, p=1 completes on attempt 1, marginal cost undefined when a cap adds no completions, CPC infinite when nothing completes |

## Results — Rust second stage
`cargo test --release`: 13 passed (8 unit + 5 deep), 0 failed.
`verify`: ALL RUST CHECKS PASSED (V0–V6 mirror E0–E6).
`soak`: 100k tasks, caps 1..=12, CPC and marginal cost monotone.
Parity sweep: 500 random populations, 0 mismatches, worst relative
diff 1.2e-15.

## Verdict — SUPPORTED

The paper's two inequalities reproduce exactly (closed form, Monte
Carlo, and replay all agree): cost per completion rises with the cap
and marginal cost per additional completion rises with the cap, on a
heterogeneous population, with the homogeneous population as the exact
equality case and survivor selection measured directly. The
calibration is bounded-scale: the paper's OpenHands run data is not
public, so the population is a synthetic two-point mixture fitted to
the paper's completion endpoints (C(1) 0.599 vs 0.597, C(4) 0.712 vs
0.700) — the direction and order of magnitude match (CPC $0.70 →
$1.19 vs the paper's $0.70 → $1.32), but the paper's exact CPC is not
reproduced. Stated limit: the test verifies the mechanism, not the
paper's exact numbers.

## Honest limits
- Synthetic population, not the paper's OpenHands runs (not public).
- Assumes the paper's conditions: stationary per-task p, attempts
  independent conditional on the task, common attempt cost. Real
  agents violate all three (learning across attempts, cost varies
  with context length) — the tax direction holds under the stated
  conditions, not unconditionally.
- Two bugs found and fixed during the run, both documented in the
  code: (1) a seed collision between the mixture generator and the
  simulation RNG correlated each task's p with its first attempt
  (E1 cap-1 completion 0.646 vs closed form 0.599 — caught by the
  cross-check, fixed with separate seeds); (2) the closed forms
  used the naive `1-(1-p)^cap` and differenced-means marginal cost,
  which lose digits for p near 0/1 — fixed to `-expm1(cap*log1p(-p))`
  and direct per-task increments on both Python and Rust sides.
