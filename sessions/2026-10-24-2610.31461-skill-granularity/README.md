# Session 2026-10-24 — Adaptive Skill Granularity for Coding Agents

**Article:** [2610.31461](https://arxiv.org/abs/2610.31461) — "Adaptive Skill
Granularity for Coding Agents: Runtime Routing and Macro-to-Micro Recovery"

**Verdict: SUPPORTED** (reproduced directionally at bounded scale; stated limits below)

## Why picked (AGENT=5, TUNE=2)

Directly reusable on the local harness: it treats skill granularity as an
action-abstraction policy — a feature-based controller routing between macro
skills and primitive tool calls, plus a finite-state recovery executor that
falls back from macro to micro on failure. Repo fit: `arxiv-lab` agent
mechanism library (`src/arxiv_lab/skills/`); the controller pattern ports to
any agent harness with a skill library (including the Hermes Android agent's
tool loop). Deterministic coding-agent protocol in the paper made it CPU-testable.

## What was built

- `src/arxiv_lab/skills/skill_granularity.py` — LLM-agnostic harness:
  simulated coding environment (10-primitive tool catalog, 5-macro skill
  library with preconditions, seeded task generator with
  applicable/mismatch/latent-failure classes, verifier-defined success),
  `GranularityController` (routes by expected cost:
  `overhead + P(fail)*mean_recovery_cost` vs micro cost, with Laplace-smoothed
  per-macro failure statistics and an exploration guard against thin-evidence
  vetoes), `RecoveryExecutor` (finite-state machine:
  START→MACRO_RUNNING→MACRO_FAILED→DIAGNOSE→MICRO_RECOVERY→VERIFY→DONE/FAIL),
  and the fixed-macro / fixed-micro baselines.
- `tests/test_skills_granularity_cpu.py` — 8 deterministic CPU tests
  (seed 20261024, n=400 tasks: 227 applicable / 117 mismatch / 56 latent-failure).
- `metrics.csv` — strategy comparison (this session).

## Results (measured)

| strategy | success | mean cost (tool-call equiv.) | recovery rate | route match |
|---|---|---|---|---|
| fixed_macro | 0.568 | 2.20 | 0.000 | 0.660 |
| fixed_micro | 1.000 | 4.00 | 1.000 | 0.340 |
| **adaptive** | **1.000** | **3.105** | **1.000** | **1.000** |

Claim-by-claim:
- **Fixed macro efficient when applicable, brittle under mismatch** — on
  applicable tasks: success 1.000, cost 2.11 vs fixed-micro 3.80. On
  mismatch tasks: success **0.000**. Reproduced exactly.
- **Runtime routing + macro-to-micro recovery improve robustness** —
  adaptive reaches 1.000 success (vs 0.568 fixed-macro) at 3.105 mean cost
  (22% below fixed-micro's 4.00), recovering **100%** of the 56 controlled
  latent macro failures that fixed-macro leaves unresolved (0.000).
- **Controller conditioned on failures** — low-noise regime macro adoption
  0.900 vs high-noise (90% latent failures) adoption 0.205, success stays
  1.000: the policy learns to avoid an unreliable macro without any code
  change.
- **Transition correctness 1.000** — recovery entered only after a macro
  failure; every episode terminates in DONE/FAIL via a legal FSM transition.

## Honest limits

- Simulated deterministic environment, not a live LLM coding agent: no real
  tool calls, no model-generated plans; cost is call-equivalents, not tokens.
- Latent failures are injected (i.i.d. per task), not emergent from real
  macro/skill brittleness; precondition mismatches are binary.
- The expected-cost controller needs a few attempts per macro before its
  failure statistics bite (exploration guard = 8 attempts).
- Not a replication of the paper's ARC-style empirical numbers — a
  mechanism-level reproduction of its two components and their claimed
  effects.

## Near-miss context

Picked over 2610.30867 (SYNAPSE separate-field tool scoring, AGENT=4 —
very testable, kept for a later day), 2610.31667 (Pass@$k$ thought pruning,
AGENT=4), 2610.31355 (Muffin shared memory buffer, AGENT=4 — needs live
models for a fair accuracy test), 2610.31521 (Jev-DLC typed-feedback
continual learning, AGENT=4/TUNE=3).
