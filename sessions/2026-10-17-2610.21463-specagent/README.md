# Session 2026-10-17 — SpecAgent (arXiv:2610.21463)

**Article:** [Trajectory-Level Speculation with Semantic Verification for Tool-Using Agents](https://arxiv.org/abs/2610.21463) (Wang et al., cs.CL, submitted 2026-10-15)
**Track:** agent · **Verdict: SUPPORTED**

## Why picked (AGENT 5)

SpecAgent flips the weak→strong collaboration: the strong model stops
*authoring* every tool-use step and becomes a *reviewer* of segments drafted
by a weak model. The weak model executes read-only tool calls itself while
drafting (every step grounded in real observations) and holds the first
environment-modifying action; the strong model verifies the segment in one
prefill pass, commits the longest prefix it judges semantically valid, and
repairs the first rejected step.

Repo fit: this is a drop-in pattern for the local agent harness — a cheap
drafter + an expensive reviewer with a hard "weak never writes" invariant.
It mirrors the Hermes app's weak-strong split (cheap draft policy on the
edge side, e.g. deepseek-flash class, with a strong reviewer such as
gpt-6-luna): the strong model only decodes short verdicts and occasional
repairs instead of every step, which is exactly the cost shape an on-device /
budget-limited harness wants.

## What was built

- `src/arxiv_lab/agent/spec_agent.py` (new, stdlib only, seeded deterministic):
  - `ToolWorld` — deterministic 6×6 grid tool-world with read-only `scan`
    and write `goto`/`collect` tools (+ terminal `done`); every
    environment-modifying call is logged as `(executor, tool)` in `write_log`.
  - `WeakPolicy` — cheap noisy greedy planner: executes read-only scans for
    real while drafting, appends exactly one *held* write (`executed=False`);
    never calls a write tool.
  - `StrongPolicy` — verifier (one prefill pass over the segment, short
    verdict per step), repairer (decodes a replacement for the first
    rejected step), and `author_step` for the strong-only baseline.
  - `SpecAgent.run` — draft → verify → commit longest valid prefix →
    repair first rejected step; only the orchestrator executes
    committed/repaired writes.
  - `run_strong_only` / `run_weak_only` baselines, `evaluate()` over a
    seeded task set with per-episode token accounting (strong decoded vs
    prefill, weak decoded) and invariant-violation counters.
- `src/arxiv_lab/agent/test_spec_agent.py` — 15 checks, all passing:
  `python3 src/arxiv_lab/agent/test_spec_agent.py`
- `results.json` — full summary + per-episode metrics for the 64-task run.

## Results (64 seeded tasks, 6×6 grid, 4 targets/task, seed=7)

| protocol | success | strong decoded tok/ep |
|---|---|---|
| strong-only | 1.000 | 320 |
| **SpecAgent** | **0.938** | **147 (ratio 0.46)** |
| weak-only | 0.359 | 0 (weak: 453) |

- Quality retention: spec 0.938 vs strong-only 1.000 — within the 10-pt sim
  tolerance; recovers +0.58 of the weak→strong gap (weak-only 0.359).
- Cost: strong-model decoded tokens cut to **46%** of strong-only
  (verdicts ≈ 2 tok/step in one prefill pass + occasional 40-tok repairs;
  strong prefill ≈ 271 tok/episode replaces autoregressive decode).
- Invariants (0 violations over all 64 episodes):
  - weak model never executed an environment-modifying action;
  - every committed step carried a positive verification verdict
    (longest-valid-prefix rule);
  - every drafted read was executed by the weak model with a real observation.

## Honest limits

- Sim-scale only: policies are synthetic proxies (noisy greedy planner vs
  near-oracle verifier), not real LLMs; the "prefill vs decode" cost model
  is an accounting proxy, not measured GPU memory traffic or wall time.
- Single synthetic domain (grid collect task); semantic validity is decided
  by a ground-truth oracle with injected miss/false-alarm noise, not by a
  real model's judgment.
- Strong-only baseline is a noiseless greedy author (1.000 by construction);
  real strong models are noisier, which would compress the quality gap.
- Therefore SUPPORTED (directional reproduction with measured numbers), not
  VERIFIED — the end-to-end claim needs real-model decode/prefill timings.

## Files in this session

- `README.md` — this file
- `spec_agent.py` — copy of the module as tested
- `test_spec_agent.py` — copy of the CPU test as run
- `results.json` — measured numbers (summary + 64 episodes)
