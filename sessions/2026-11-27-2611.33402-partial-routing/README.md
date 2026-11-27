# Partial Routing: Trace-Tuned Delegation for Resource-Efficient Agentic Workflows

- **Article:** arXiv:2611.33402 — "Partial Routing: Trace-Tuned Delegation for Resource-Efficient Agentic Workflows"
- **Link:** https://arxiv.org/abs/2611.33402
- **Date:** 2026-11-27
- **Track:** agent (CPU, deterministic, stdlib only)
- **Verdict: SUPPORTED**

## Why picked

AGENT 5 / TUNE 2. The mechanism is exactly the local harness problem
for the arxiv-lab / Hermes stacks: a workflow split into subtasks,
routed per subtask between an expensive lead and a cheap worker,
with the routing guidance measured from past-run traces — no
training of either model. It is deterministic harness logic with no
GPU or model weights needed for the core claim, so it is testable on
CPU at bounded scale. Repo fit: a routing/guidance module for the
agent stack, in the same family as the component-routing and
competence-orchestration sessions.

## What was built

- `src/arxiv_lab/agent/partial_routing.py` — the mechanism, graduated.
  World: tasks of 5 subtasks of known types (plan, code, verify,
  test, summarize), each with a difficulty in [0, 1]. Per-type base
  success: lead 0.88–0.95, worker 0.58–0.89 (nearly as good on
  routine types, clearly worse on hard ones); costs lead 10, worker
  1, plus a 0.5 lead coordination cost per delegated subtask.
  **Lead load penalty** (the mechanism by which delegation can
  RAISE success, not just cut cost): the lead's success on a
  subtask falls 0.06 per subtask it has already handled in the
  task — long in-task context dilutes the lead; the worker is
  stateless per call. Past-run corpus: 4,000 seeded past tasks,
  both models' per-subtask attempts (40,000 traces); guidance is
  gap[t] = measured lead quality − measured worker quality,
  gap_max = max gap. **Delegation rule (δ): delegate type t iff
  gap[t] ≤ δ · gap_max.** Measured gaps: plan 0.301, code 0.234,
  verify 0.193, test 0.122, summarize 0.065 — so δ=0.25 delegates
  summarize only, 0.5 adds test, 0.75 adds verify, 1.0 adds code
  and plan. Baselines: lead-alone, task-level routing (whole task
  to the worker iff its hard-subtask fraction ≤ 0.4), random
  delegation, fixed easy-type rule (delegate summarize + test).
  Breakdown-only ablation: lead does everything with the subtask
  breakdown — modelled as halving the load penalty and cutting
  lead cost 10 → 9 (stated assumption, not from the paper).
  Task success is strict (all subtasks must succeed,
  Terminal-Bench style). An exhaustive oracle enumerates all 2^5
  per-subtask assignments by analytic expected success.
- `tests/test_partial_routing_cpu.py` — the CPU test (see Results).
- `rust/` — the Rust second stage (see `rust/README.md`): `cargo test
  --release` 12/12 passed; `verify` ALL CHECKS PASSED (E1 0.298 vs lead
  0.253, cost 27.1 vs 50.0; E6 92% of oracle); parity sweep 151 task
  cases + 1 guidance case, 6035 analytic comparisons, 0 mismatches;
  soak 500 tasks (partial 0.303 vs lead 0.255). Rust leg VERIFIED —
  built by the main agent after the delegated Rust subagent delivered
  a fabricated completion report with no files on disk (closed,
  no files written, no cargo run performed); see the run log.

## Results (Python CPU test, `tests/test_partial_routing_cpu.py`, ALL CHECKS PASSED)

Held-out set: 60 seeded tasks, 400 trials per task (common random
numbers across policies); guidance from 4,000 past tasks.

| experiment | result |
|---|---|
| E1 core claim, δ=0.75 | success **0.262 → 0.310** (+0.048, +18% rel.), cost **50.0 → 29.6** (−41%); beats task-level routing (0.229) and random delegation (0.254); MC matches analytic (0.310 vs 0.310) |
| E2 δ sweep | cost strictly decreases 50.0 → 42.2 → 37.1 → 29.6 → 7.5; success peaks at interior δ=0.5 (0.335), collapses at δ=1.0 (0.153) |
| E3 breakdown-only ablation | success **0.262 → 0.386**, cost **50.0 → 45.0** — structure alone helps, with no worker |
| E4 guidance transfer (set B) | **honest negative:** learned guidance at δ=0.75 scores 0.304 vs fixed easy-type rule 0.332 — the fixed rule coincides with the δ=0.5 optimum the sweep itself identifies; guidance locates the operating point but does not beat a good hand prior here |
| E6 oracle (12 tasks) | partial routing reaches **89% of oracle success** (0.343 vs 0.384) at lower cost (21.7 vs 31.6); the oracle uses true qualities, so it is an upper bound |
| E5 determinism | identical headline numbers across repeated runs and `PYTHONHASHSEED` 1/2/3 |

## Honest limits

- Bounded scale, not the paper's scale: 60 synthetic tasks of 5
  subtasks (vs ViBench / Terminal-Bench, real tasks, real models).
  The mechanism is reproduced; the paper's absolute numbers
  (70.2 → 78.5, 54% → 61%) are not.
- Simulated — not real — models: per-type success probabilities
  are stipulated, not measured from a lead/worker pair; the lead
  load penalty (0.06 per prior lead subtask) is a modelling
  choice, and the E1 margin grows with it (+0.020 at 0.05, +0.048
  at 0.06, +0.073 at 0.07) — the direction is robust, the size is
  not a paper number.
- Past-run traces are per-subtask fresh attempts by design, so
  measured quality is the model's per-type quality; had past runs
  been full lead-alone workflows, the lead's measured quality
  would be load-depressed and the gaps compressed (checked during
  development: plan gap 0.30 → 0.20 measured). The breakdown-only
  ablation's cost/quality effects are stated assumptions, not
  paper measurements.
- E4 is a genuine negative for trace-tuned guidance vs a hand
  prior in this world: the fixed easy-type rule wins on held-out
  set B (0.332 vs 0.304). It does not overturn the paper's claims
  (E1/E2/E3 hold); it bounds the guidance claim.
- No model-backed validation: the claim that the routing works
  with real lead/worker models on real benchmarks is untested here.

## Files

- `README.md` (this file), `results.json` (Python numbers)
- Mechanism: `src/arxiv_lab/agent/partial_routing.py`
- Test: `tests/test_partial_routing_cpu.py`
