# Session — 2026-11-22 — arXiv:2611.26592

## Article
**Realized versus Population Reliability: Anytime-Valid Self-Consistency
Certification under Correlated Reasoning**
Terence D. Agbeyegbe, Toneukarin G. Agbeyegbe.
https://arxiv.org/abs/2611.26592

## Why picked + repo fit
AGENT 5, TUNE 2. Self-consistency certification treats repeated sampling
as sequential evidence accumulation and stops when an anytime-valid
(betting) e-process crosses 1/eps — but the standard construction assumes
conditionally independent rollouts, and real reasoning streams are
positively correlated (paper: rho-hat 0.44–0.48) through shared prefixes
and across-prompt heterogeneity. Correlation splits the null into a
*realized* null (about the drawn belief) and a *population* null (about
average behaviour); the standard e-process stays valid only for the
former. On AIME the single-unit rule crosses on 24% of prompts at
nominal eps=0.05 while the prompt-level procedure declines (0.00).
This is a directly reusable stopping rule for any agent harness that
votes over samples — arxiv-lab mechanism library, Hermes AgentRunner
finish policy. The mechanism is pure logic (deterministic CPU test) and
Rust-portable.

## What was built
- `src/arxiv_lab/agent/selfconsistency.py` — the mechanism, graduated
  to the repo (stdlib only): SplitMix64, Acklam `normal_ppf`/`normal_cdf`,
  rho/tau-sigma parameterisation, pi = P(mu > 0), mixture betting
  e-process over a 9-point lambda grid, `crossing_step`, the
  random-effects rollout model (mu ~ N(theta, tau^2);
  x = +1 iff mu + sigma*z > 0, rho = tau^2/(tau^2+sigma^2)),
  ESS / within-unit-mean variance, ANOVA ICC and theta-hat estimators,
  single-unit and prompt-level experiment runners.
- `test_selfconsistency_cpu.py` — deterministic CPU test, E0–E7
  (see Results).
- `rust/` — Rust second stage (std-only, zero crates): `verify` binary
  mirroring E0–E7, unit + deep tests, parity sweep, soak.
  Rust leg verdict: **VERIFIED** (see below).
- Model-backed leg: `nim_selfconsistency.py` in the goal's hidden_files
  scratch (NOT committed — repo rule: no surrogate code). NVIDIA NIM
  `minimaxai/minimax-m3.1-flash`, 10 prompts x 8 rollouts on GSM8K test
  and MATH Level 5 test (HF datasets-server), temp 0.8.

## Results — CPU simulation (primary verdict carrier)
`test_selfconsistency_cpu.py`: ALL CHECKS PASSED in 11.4 s.
Results in `results.json`.

| Experiment | Result |
|---|---|
| E1 rho=0 control (theta=-0.15) | crossing 0.0045 (<= eps 0.05) — valid when independent |
| E2 AIME regime (theta=-0.15, rho=0.48, population null TRUE) | N=64 crossing **0.272** (paper: 0.24); N=256 crossing **0.359**, approaching pi=0.414 |
| E3 realized-null split (N=256) | mu<=0 rate **0.0018** (valid); mu>0 rate **0.834** |
| E4 ESS saturation | var of within-unit mean follows rho+(1-rho)/n; ESS(256)=**2.07** ~ 1/rho |
| E5 prompt-level validity | crossing **0.0075** (<= eps); power 0.994 vs single-unit 0.619 at theta=+0.4, B=256 |
| E6 GSM8K regime (theta=+0.96, rho=0.435) | crossing **0.834** — gap dormant, as the paper says |
| E7 determinism | identical repeat |

## Results — model-backed leg (NIM)
Assistant = `minimaxai/minimax-m3.1-flash` (see "Model notes" below).
10 prompts x 8 rollouts per regime, eps=0.05, 0 API errors.

| Regime | mean acc | rho-hat | theta-hat | single-unit crossing | prompt-level crossed |
|---|---|---|---|---|---|
| GSM8K | 0.738 | 0.753 | +0.636 | 6/10 = 0.60 | no |
| MATH L5 (as reported) | 0.300 | 0.797 | -0.524 | 1/10 = 0.10 | no |
| MATH L5 (clean subset, n=5) | 0.550 | 0.832 | +0.126 | 1/5 = 0.20 | — |

Reading: both regimes show **strong positive within-prompt correlation**
(rho-hat 0.75–0.83, above the paper's 0.44–0.48) — the all-or-nothing
per-prompt pattern is real on this model. GSM8K matches the paper's
dormant-gap regime (theta-hat > 0, single-unit crossing is correct
behaviour). The MATH numbers as reported are **contaminated by gold
parsing artifacts** (see Limits); on the clean 5-prompt subset the
population null is not clearly true (theta-hat +0.13), so the MATH
crossing rate cannot be read as a Type I error. The prompt-level
non-crossing on GSM8K is a small-N power artifact: with 10 prompts the
mixture wealth at 8/10 correct is ~4.5, below the 1/eps=20 threshold
(it would cross at 10/10). The leg confirms the correlation premise and
the dormant-gap regime, and directionally the hard-regime crossing
exceeding eps — but the hard-regime crossing rate is not cleanly
measured (see Limits).

## Results — Rust second stage
**Rust leg verdict: VERIFIED** (independently re-verified by the parent).
`cargo test --release`: 39 passed (29 unit + 10 deep), 0 failed.
`verify`: ALL CHECKS PASSED, every number exactly equal to Python
`results.json` (float equality). Parity sweep 10,253 cases, 0
mismatches (ppf/cdf 0 ulp, e-process wealth paths bit-identical, 7 full
experiments + 5 ESS rows + 15 fresh-seed experiments field-equal).
Soak: 20,000 units x 256 rollouts in ~0.6–0.74 s (crossing 0.3399;
first 2,000 units bit-identical to Python E2's 0.359). Deep tests:
exhaustive `crossing_step` vs direct-product oracle (6,138 cases, 0
mismatches), prompt-level validity/power across a rho grid, ESS
saturation, thread identity (4 threads reproduce serial exactly).
One real divergence found and fixed: CPython `sum()` uses Neumaier
compensated summation, Rust `.sum()` is naive (1-ulp wealth gap) —
`py_sum` implemented; all divergences documented in `rust/README.md`.
Push verified by the parent against the remote tree: **9/9 blob SHAs
match**.

## Verdict
**SUPPORTED.** The simulation reproduces the paper's regime
quantitatively (AIME-regime crossing 0.272 vs the paper's 0.24 at N=64,
rising toward pi with the rollout budget; realized-null validity at
0.0018; prompt-level validity at 0.0075 with strictly higher power;
ESS saturating at 1/rho), the Rust port is bit-parity VERIFIED, and the
NIM leg confirms the correlation premise on real rollouts (rho-hat
0.75–0.83) plus the dormant-gap GSM8K regime. It is SUPPORTED rather
than VERIFIED because the test is a bounded-scale simulation of the
paper's random-effects model (not a replication of the paper's Claude
Haiku 4.5 rollouts), and the hard-regime NIM crossing rate is
contaminated by gold-parsing artifacts (see Limits).

## Limits — read before reusing
- The simulation implements the paper's random-effects model exactly as
  specified, but the paper's headline numbers come from real Claude
  rollouts; the simulation confirms the mechanism's regime behaviour,
  not the paper's empirical estimates.
- NIM leg, MATH regime: 4 of 10 gold answers were unparseable as numbers
  ('i', '[0,\\infty)', two \\frac answers truncated by the old
  `[^}]*` boxed regex in `load_problems`) and were auto-scored 0/8;
  one more ('12, 10, 6') collapsed to the bogus gold 12106; and
  `parse_number` dropped the leading minus on '-\\frac{1}{8}' (parsed
  as +0.125). All three bugs are fixed in the scratch script (brace-
  counting boxed extraction, sign handling, multi-answer rejection)
  with parse checks passing, but the leg was not re-run: the clean
  5-prompt subset (acc 0.55, rho-hat 0.832) is reported instead, and
  it is too small and too easy to test the population-null-true regime.
  A clean hard-regime test needs harder problems (AIME-level) and
  numeric-only golds.
- NIM leg, model notes: the protocol's `openai/gpt-oss-20b` is gone
  from NIM (HTTP 410); `google/gemma-3-4b-it` 404s on chat completions
  (as its 12B sibling did). `meta/muse-glimmer-30b` is a reasoning
  model that spends its whole token budget in `reasoning_content`
  (finish_reason "length", content null) — the first leg misread those
  as 0/8 failures, and at 180–360 s/call it is infeasible for this
  leg. The leg as run uses `minimaxai/minimax-m3.1-flash` (21.5 s/call,
  finish=stop). Any NIM model is a proxy for the paper's Claude Haiku.
- The mixture e-process uses a finite 9-point lambda grid, a practical
  approximation of the paper's continuous mixture; the asymptotic lower
  bound pi is approached as the grid refines and N grows.
- Prompt-level power at N=10 prompts is low by construction (see the
  wealth arithmetic in Results); the validity claim is what the small
  leg tests, not power.
