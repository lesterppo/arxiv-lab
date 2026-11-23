# Session — 2026-11-23 — arXiv:2611.27860

## Article
**Who Grades the Agent? Privilege Separation for Coding-Agent
Benchmarks**
Nathan Nour Houlamy, Mattia Favretto, Earlence Fernandes, Mauro Conti.
https://arxiv.org/abs/2611.27860

## Why picked + repo fit
AGENT 5, TUNE 1. Evaluating a coding agent requires running its
code, and that code can manipulate the evaluation infrastructure to
forge a pass — BenchJack, an automated benchmark auditor, reached
near-perfect scores on nine of ten popular agent benchmarks without
solving their tasks. SplitBench is the first framework in the
literature to separate running a task's tests from judging their
results: the agent's code runs unprivileged, a trusted grader alone
holds the expected answers, and even an agent that takes over the
entire test environment cannot forge a pass. This is directly reusable
in any agent harness that executes agent-produced code — arxiv-lab
mechanism library, Hermes AgentRunner tool sandboxing. The mechanism
is pure logic plus process separation, deterministic on CPU, and
Rust-portable.

## What was built
- `src/arxiv_lab/agent/splitbench.py` — the mechanism, graduated to
  the repo (stdlib only): a miniature benchmark of 8 pure-function
  tasks (JSON-serialisable I/O); `NativeGrader` (the vulnerable
  setup: solution imported in-process alongside the answers and the
  test file); `SplitGrader` (the separated setup: solution runs in a
  sandboxed subprocess that receives only the task files and the
  hidden inputs at grade time, the trusted grader holds the answers
  in memory only and judges schema-validated outputs); 4 adversarial
  payloads (`read_answers`, `overwrite_tests`,
  `monkeypatch_grader`, `exfiltrate_via_output`); 3 honest
  submission kinds (correct / buggy / partial);
  `fresh_input_accuracy` (never-exposed inputs, proving a forged
  PASS is not a solved task).
- `test_splitbench_cpu.py` — deterministic CPU test, E0–E5
  (see Results).
- `rust/` — Rust second stage (std-only, zero crates): `verify`
  binary mirroring the Python test's experiments and assertions,
  unit + deep tests, parity sweep, soak, thread identity.
  Rust leg verdict: **VERIFIED** (see below).

## Results — CPU test (primary verdict carrier)
`test_splitbench_cpu.py`: ALL CHECKS PASSED in 14.6 s.
Results in `results.json`.

| Experiment | Result |
|---|---|
| E1 forge rate (4 payloads x 8 tasks) | native-targeting payloads forged **24/24** under the native grader; split grader forged **0/32**; worst case (adversary reads all grading code, not the answers) **0/32** |
| E4 no-real-solving | every forged PASS scored **0.0** on never-exposed fresh inputs — the agent did not solve the task, it forged the grade |
| E2 grader agreement | **24/24** identical verdicts; correct-pass 8/8, buggy-fail 8/8, partial-fail 8/8 (paper: 98.7% agreement) |
| E3 conversion | **8/8** tasks convert to the separated framework (paper: 180/202 = 89%) |
| E5 determinism | full repeat reproduces every verdict |

## Results — Rust second stage
`cargo test`: 13 passed (8 unit + 5 deep), 0 failed. `verify`:
ALL RUST CHECKS PASSED (V0 unit checks; V1 replay of the Python
results; V2 property sweep 5,000 random pairs; V3 exhaustive
universe 441 pairs; V4 soak 20,000 pairs in 3.26 ms; V5 thread
identity, 8 threads, one digest). Parity sweep: 7,840 cases
(5,336 values-equal + 2,004 grade + 500 classify),
**0 mismatches**. Rust leg verdict: **VERIFIED**.

## Verdict: SUPPORTED

The core mechanism reproduces directionally at bounded scale: the
separated grader blocks every forge the vulnerable grader admits
(24/24 → 0/32), including the paper's worst case, while agreeing
with the native grader on every honest submission. Stated limits:
(1) the benchmark is a miniature 8-task pure-function suite, not
the paper's 202 real coding tasks — the 98.7% agreement and 75%
worst-case block rate are the paper's numbers on real benchmarks,
ours are 100% / 100% on the miniature; (2) the CPU test models
privilege separation at the process + directory level (sandbox
subprocess, answers never written to disk) — a production deployment
needs OS-level isolation (separate UID / container) on top of this
layout, which is outside this CPU test; (3) the `exfiltrate_via_output`
payload targets the split grader's output channel, not the native
grader, and fails under both — it is counted in the 0/32, not in
the 24/24.

## Note (debugging, for the record)
While developing the test, a throwaway debug script that had done
`from ... import grade_native` showed the `exfiltrate` payload
passing the native grader 8/8. That was a harness artifact, not a
mechanism result: the monkeypatch payload patches `grade_native` on
every module in `sys.modules`, including the debug script's own
`__main__`, so the script's later calls invoked the patched lambda
directly. The module's own finally-block restore is correct
(verified: `grade_native_verdict` is the original after every
monkeypatched call, and the payload then fails as designed). The
experiment runners in the module were never affected.
