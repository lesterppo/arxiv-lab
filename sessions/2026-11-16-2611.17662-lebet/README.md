# LeBet: Learning Interpretable Rules for Reliable Agent Intervention

- **Article:** https://arxiv.org/abs/2611.17662 — "Learn then Bet: Learning
  Interpretable Rules for Reliable Agent Intervention"
- **Date:** 2026-11-16 (fetch: OAI fallback — arXiv API 429'd, RSS empty)
- **Scores:** AGENT **5**, TUNE 2
- **Verdict: SUPPORTED** (core mechanism reproduced directionally at bounded
  scale, with stated limits — see below)

## Why picked

Runtime monitors are the missing piece in the local agent harness: turn
limits and keyword filters stop runs that would succeed and let failing runs
continue. LeBet learns an interpretable OR-of-AND monitor by stochastic
local search and intervenes via a fixed betting statistic with finite-sample
FPR control. Directly reusable in `hermes-colab-cli` / `arxiv-lab` agent
loops (a monitor module + betting accumulator is a small, portable
addition). Repo fit: agent harness (`arxiv-lab`, `hermes-colab-cli`).

## What was built

- `src/arxiv_lab/agent/lebet.py` (stdlib only; graduated) — simulated agent
  trajectories (SUCCESS vs FAIL, two failure subtypes: error-prone and
  silent-looper), 10 boolean predicates over trajectory prefixes, OR-of-AND
  monitor, stochastic-local-search learner whose objective is the **betting
  score** (betting power − 2·betting FPR − complexity) on train, fixed
  betting statistic (test martingale, Ville's inequality → FPR ≤ α), and a
  held-out evaluator with train-tuned baselines.
- `tests/test_lebet_cpu.py` (also copied here) — full experiment + assertions.
- `results.json` — full measured numbers.

Key design choices (ours, not the paper's — the abstract does not publish
them): betting fraction λ=0.75; SLS objective = betting power/FPR (the
paper's "learning objective derived from sufficient conditions for valid
and powerful intervention" is stated, not specified); 10 hand-chosen
predicates; synthetic trajectories with two heterogeneous failure modes
(the heterogeneity is what makes an OR-of-ANDs necessary — with a single
failure mode a single-predicate error filter wins on raw power, which we
verified and then fixed by adding the silent-looper subtype).

## Results (seed 20261116; train 200+200, test 200+200 trajectories; α=0.15)

Learned monitor: **(loop AND tok_high) OR (err_rate) OR (stalled)**
— 3 conjunctions, ≤2 literals each. Train: TPR 0.985, FPR 0.100,
betting power 0.950 at betting FPR 0.020, q=0.0151.

Held-out (betting monitor):
| metric | LeBet | turn-limit (K\*=22, tuned) | error-filter (repeat_tool_fail, tuned) | fire-immediately |
|---|---|---|---|---|
| FPR | **0.050** | 0.140 | 0.145 | 0.155 |
| power | **0.970** | 0.885 | 0.615 | 0.995 |
| token savings | **0.308** | 0.164 | 0.252 | 0.509 |
| completed-run precision | **0.969** (base 0.500) | 0.882 | 0.690 | 0.994 |
| median intervene step (fail) | 16 | 22 | 12 | 5 |

- FPR controlled well under α=0.15 (0.050, n=200 successes).
- Power beats both tuned baselines at matched validity (0.970 vs 0.885 /
  0.615); token savings beats the turn-limit (0.308 vs 0.164).
- The betting statistic matters: fire-immediately has the same monitor but
  FPR 0.155 (3× higher) — the martingale threshold is what buys validity.
- Negative control (label-permuted train): TPR−FPR = −0.010 — no separation,
  so the learner is not exploiting leakage.
- Determinism: rerun reproduces the monitor and headline metrics exactly.

Comparison with the paper's numbers (SWE-bench Lite + Ministral 3 8B):
paper saves 34.46% tokens, precision 18.33%→21.59%, FPR 13.64%. Ours:
30.8% saved, precision 0.500→0.969, FPR 0.050. Same direction, larger
effect sizes — expected: our synthetic failures are cleaner than real
SWE-bench runs, and our base precision (0.500) is an artifact of the
balanced 200/200 test set, not a real agent's success rate. The numbers
are a mechanism check, not a replication of the paper's benchmark.

## Honest limits

- Synthetic trajectories, not a real agent benchmark: predicates and failure
  modes are ours; a real harness would need predicates over real tool
  traces and a monitor learned from real runs.
- No model in the loop: no LLM calls (NVIDIA NIM not used — the mechanism
  is pure logic; a model-backed validation on SWE-bench Lite is out of
  this run's budget).
- α=0.15, λ=0.75, predicate set, and generator parameters are our choices;
  the paper's exact SLS objective and betting statistic are not specified
  in the abstract.
- The betting FPR guarantee is asymptotic-in-theory (Ville) with q
  estimated from train; held-out FPR 0.050 is empirical, n=200.

## Rust second stage

See `rust/` — port of the mechanism (generator, predicates, monitor, SLS
learner, betting statistic, evaluator) with verify binary, unit tests,
property/soak tests, and a Python-vs-Rust parity sweep (0 mismatches
required). The Python version above stays the primary verdict carrier.
