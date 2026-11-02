# 2026-11-02 — Verify–Backtrack Loop (VBL)

- Article: https://arxiv.org/abs/2610.41564 — "Verify–Backtrack Loop:
  Structured Precondition Recovery for Long-Horizon Language Agents"
  (submitted 30 Oct 2026).
- Why picked (AGENT=5, TUNE=1): a training-free, structured control
  state for long-horizon agents — a bounded goal stack, dual-horizon
  verification (does the active goal contribute to the terminal goal?
  is it feasible in the current state?), typed failure attribution,
  and four explicit operations (Push, Pop, retry, BacktrackTo). It
  drops directly into the arxiv-lab harness library next to the
  Mingbird trio (prefill budget, finish gate, loop detection), and it
  is testable on CPU without any model or environment.
- What was built: `src/arxiv_lab/agent/vbl.py` (graduated) + session
  copies `vbl.py` / `test_vbl_cpu.py`; a Rust second stage under
  `rust/` (see `rust/README.md`).

## Results (deterministic CPU test, `tests/test_agent_vbl_cpu.py`)

Scripted long-horizon task — terminal `report` = gather → analyze →
write, with analyze losing its `data_ready` precondition on its first
attempt (a material state change, as in the paper's recovery problem):

| System | Outcome | Steps |
|---|---|---|
| VBL | **success** (pushes `regather`, restores the precondition, completes) | 9 |
| ReAct baseline (free-form retry, no stack revision) | failure, budget burned | 60 |

VBL trace (operations): Push gather, Pop gather, Push analyze,
Push regather, Pop regather, Pop analyze, Push write, Pop write,
Pop report. Unit cases also pass: a non-contributing goal is popped
before execution; a decomposition failure triggers BacktrackTo the
parent; a generic failure retries twice (MAX_RETRIES=2) then pops;
a full stack (max_stack=2) forces BacktrackTo instead of Push.

## Verdict: SUPPORTED

The paper's core mechanism — structured planning-state revision beats
free-form retry at recovery — reproduces directionally at bounded
scale, with the exact operations the paper names.

## Honest limits

- Scripted environment, not AppWorld / ALFWorld / ScienceWorld, and no
  LLM backbone: the paper's headline (+22.07 percentage points over
  ReAct on AppWorld with Qwen3.5-9B / GPT-4o / DeepSeek-V3.2) is NOT
  reproduced here, and the 9-vs-60 step gap is a property of the
  scripted task, not a benchmark result.
- One failure-injection pattern (precondition loss) plus unit cases;
  multi-failure interleavings are covered only in the Rust deep tests.

## Rust second stage

After the Python test passed, the mechanism was ported to Rust
(`rust/`): `vbl verify` reproduces the headline numbers exactly
(success in 9 steps, identical operation sequence, baseline 60 steps);
9 cargo tests pass (5 unit + 4 deep, incl. a 20,000-case soak and
thread-identity); a Python-vs-Rust parity sweep over 36 cases
(inject level × retry limit × stack bound) has **0 mismatches**.
The Python version stays the primary verdict carrier.
