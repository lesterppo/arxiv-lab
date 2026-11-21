# LoopRoute: loop-triggered routing for System-1 decision agents

- **Article:** arXiv:2611.25363 — "Stop Going in Circles: Loop-Triggered Routing for System 1 Decision Agents" (Yu, Ma, Chen, Sun, Hu, Qin, 2026)
- **Link:** https://arxiv.org/abs/2611.25363
- **Date tested:** 2026-11-21
- **Verdict: SUPPORTED**

## Why picked

- **AGENT 5 / TUNE 1.** LoopRoute is a pure harness mechanism: a fast
  decision policy acts by default; when a (state, action) pair recurs,
  the router hands control to an LLM for brief, feedback-driven
  assistance, then returns control. No training, no confidence
  calibration. It is the loop-detection half of a local agent harness —
  the direct complement to yesterday's GrepMem memory mechanism.
- **Repo fit:** directly reusable in a local agent harness (Hermes
  AgentRunner loop detection / stall recovery). The mechanism is
  model-agnostic and costs nothing when the agent is not stuck.
- **Testable on CPU:** the router, the fast policy, and the environment
  are pure deterministic logic; the LLM assistant is isolated behind
  one callable, so the mechanism can be verified with a BFS oracle
  planner and then confirmed with a real model (NVIDIA NIM leg below).

## What was built

- `src/arxiv_lab/agent/looproute.py` (graduated module, stdlib only):
  `KitchenWorld` (deterministic TextWorld-cooking-like task family:
  collect recipe ingredients, one inside a locked chest needing a key
  found elsewhere, then cook in the kitchen), `FastPolicy` (greedy
  heuristic with one realistic System-1 flaw: it keeps trying `open
  chest` without the key instead of detouring to fetch it — a genuine
  state-action loop), `LoopRouter` (counts (state, action) pairs per
  episode; at the threshold-th occurrence it requests assistance, executes
  the returned plan verbatim, then returns control; the pair's count
  resets, so later recurrences can request help again), `plan_to_goal`
  (BFS oracle planner used as the deterministic assistant stand-in).
- `test_looproute_cpu.py`: five live experiments with assertions.
- `rust/` — Rust second stage (see `rust/README.md`).
- Model-backed leg: `nim_looproute.py` (scratch, NOT in this repo — it
  carries the surrogate client; repo rule: no surrogate code, no
  secrets). Assistant = NVIDIA NIM `meta/muse-glimmer-30b`. Note:
  the protocol's named model `openai/gpt-oss-20b` is no longer served
  by NIM (HTTP 410 Gone, verified 2026-11-21); `google/gemma-3-12b-it`
  404s on chat completions; `meta/muse-glimmer-30b` probed OK and was
  used instead.

## Results — CPU test (oracle assistant)

60 seeded tasks, mixed locked/unlocked (even seeds locked, odd seeds
unlocked), max 60 steps/task:

| policy | success | mean steps | mean assists | mean LLM calls |
|---|---|---|---|---|
| fast-only | 0.500 | 35.03 | — | — |
| LoopRoute-oracle, K=2 | **1.000** | **12.2** | 0.50 | 0.50 |
| LoopRoute-oracle, K=3 | 1.000 | — | 0.50 | 0.50 |

- Gain: **+50 percentage points** over the standalone fast policy
  (paper: +21.0 / +60.1 / +30.1 points on ALFWorld / MapReader /
  TextWorld cooking). The fast policy wins the unlocked half (20/20)
  and loses the locked half (0/30 — it loops to the step cap); LoopRoute
  rescues the locked half with exactly one assist per locked task.
- Limited assistant use: 0.50 calls/task overall (1.0 per locked task,
  0 per unlocked task) — the paper's "limited use of the LLM policy"
  claim, reproduced.
- False triggers: **0 assists on 20 loop-free (unlocked) tasks**, and
  the fast policy solves 20/20 of them — the router fires only on
  genuine recurrence.
- Router units: triggers on the 2nd occurrence at K=2 (not the 1st),
  resets after an assist, re-triggers on later recurrence; K=3 silent
  on the 2nd, triggers on the 3rd.
- Oracle planner: 10/10 plans legal and complete.
- Determinism: the 60-task experiment repeated gives identical numbers.

## Results — model-backed leg (NIM)

Assistant = `meta/muse-glimmer-30b` via NVIDIA NIM (see note above on
the 410 for `gpt-oss-20b`). M1: N=20 seeded tasks (10 locked / 10
unlocked), K=2. M2: N=6 seeded tasks, LLM picks every action.

| policy | success | mean steps | mean LLM calls/task |
|---|---|---|---|
| fast-only (CPU, N=60) | 0.500 | 35.0 | 0 |
| LoopRoute-NIM (M1) | **0.850** | 13.7 | **0.50** |
| full-LLM control (M2) | 0.500 | 16.5 | 16.5 |

- M1 splits: locked 0.70, unlocked 1.00. The real LLM assistant rescues
  7 of 10 locked tasks (oracle: 10 of 10) — the 3 failures are bad
  assistant plans (wrong detour / illegal step wasted), which is honest
  LLM behaviour, not a router failure: the router still fired exactly
  once per locked task (10 calls / 20 tasks).
- M2 full-LLM control (0.50) is no better than fast-only and worse than
  LoopRoute-NIM, at 33x the calls (16.5 vs 0.50 per task): the LLM errs
  on routine navigation the fast policy gets right, which is exactly the
  paper's cost argument for routing only on loops.
- 0 API errors across 10 + 99 calls. Token use: M1 4,234 prompt /
  1,116 completion; M2 39,429 prompt / 808 completion.
- Verdict contribution: the NIM leg confirms the router mechanism with
  a real model and shows the same ordering as the oracle leg
  (LoopRoute > fast-only; LoopRoute >> full-LLM on calls), at a lower
  absolute rescue rate than the BFS oracle bound — consistent with
  SUPPORTED, not VERIFIED.

## Rust second stage

**Rust leg verdict: VERIFIED** (independently re-run by the parent:
`cargo test --release` 23 passed / 0 failed — 16 unit + 7 deep;
`looproute verify` ALL CHECKS PASSED with every number identical to
the Python `results.json`; parity sweep 560/560 cases, 0 mismatches).
Zero external dependencies (stdlib-only Rust, `--offline` build).
`PyRandom` reproduces CPython's MT19937, so seeded tasks are
byte-identical (`gen-spec` parity); parity is compared at the
task-spec level with full trajectories (state, action, source,
observation). Soak (600 seeds, 27 ms): fast-only 0.500 / 35.11 steps,
LoopRoute-oracle K=2 1.000 / 12.32 steps, max 1 assist per task;
thread-identity: 4 threads reproduce serial aggregates exactly.
Divergences documented in `rust/README.md` (bitmask state,
`clone_task()` probe instead of seed rebuild, new spec subcommands).
One harness-caught finding: Python's `plan_to_goal` rebuilds its
search probe from `env.seed`, so a from-spec env (seed −1) would plan
for the wrong task — fixed in the parity harness only; the seeded
experiments are unaffected.

## Honest limits

- The environment is a synthetic TextWorld-cooking analogue, not
  ALFWorld/TextWorld itself: the loop is engineered (one flaw in the
  fast policy) rather than emergent from a learned decision model, so
  the +50pp gain measures the router's rescue power on a known stall,
  not the paper's benchmark numbers.
- The oracle assistant is a BFS planner with full state access — it
  upper-bounds what a real LLM assistant can do. The NIM leg (M1/M2)
  is the check against that bound.
- The paper's "six frozen decision models" generality claim is not
  tested: one fast policy only.
- The exact repeat threshold the paper uses is not recoverable from
  the abstract (full text not on ar5iv at test time); K=2 and K=3
  were both run and agree, so the result is not threshold-fragile at
  this scale.
- Full results: `results.json` (CPU), `nim_looproute_results.json`
  (model-backed, in the goal's hidden_files scratch — the NIM client
  is never committed).
