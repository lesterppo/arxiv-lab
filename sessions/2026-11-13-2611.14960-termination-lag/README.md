# Termination Lag: Coding Agents Solve Early but Stop Late

- **Article:** arXiv [2611.14960](https://arxiv.org/abs/2611.14960) — "Termination Lag: Coding Agents Solve Early but Stop Late" (submitted 11 Nov 2026)
- **Verdict: SUPPORTED** (directional reproduction at bounded scale; see "Why SUPPORTED, not VERIFIED")
- **Track:** agent track — CPU only, stdlib only, no network in the primary test
- **Code:** no public code link in the abstract; mechanism re-implemented from the paper's measurement protocol (§2–§3)

## Why picked

- **AGENT 5 / TUNE 1.** The paper's measurement is a harness primitive: termination lag — steps between an agent's first correct patch and the end of its run — is exactly what a finish gate should minimise. The findings map directly onto the hermes agent's existing finish-gate / loop-detection work (Mingbird harness, `src/arxiv_lab/harness/finish_gate.py`): 38–82% of successful runs solve on the initial edit, post-solution steps are a median 35–60% of the run, most of them verification/exploration, and 1–8% of runs undo a correct patch and end wrong.
- **Repo fit:** `arxiv-lab` mechanism library — a stop-policy comparison (continue-and-verify vs stop-on-first-pass vs fixed budget) is a reusable harness evaluation, complementary to yesterday's WikiTrace memory track.
- **Testable within budget:** the paper's own protocol (record every distinct patch, test each) maps onto a deterministic simulated coding agent, so a faithful CPU proxy test is possible in one run.

## What was built

- `termination_lag.py` — the mechanism (stdlib only, deterministic):
  - `Task` / `gen_tasks` — synthetic SWE-style tasks in three difficulties (easy/medium/hard) with first-edit solve probabilities 0.82 / 0.60 / 0.38 (mix mean 0.60, inside the paper's 38–82% band).
  - `verify` — the test suite: deterministic, passes iff the current patch is correct.
  - `run_task` — a simulated agent run: pre-solution exploration, then edits until the first correct patch (or give-up after 6 edits), then a post-solution phase governed by the stop policy. A post-solution re-edit undoes the fix with probability 0.15; a later re-edit can re-fix it (0.55). Every EDIT / VERIFY / EXPLORE is one step; lag is steps after the first correct patch (paper definition).
  - Three stop policies on the *same* tasks with paired RNG streams (per-task SplitMix64, same construction as WikiTrace so the Rust stage reproduces the exact streams): `continue` (paper-observed: 1 + Poisson(3) extra verify/explore/re-edit steps), `stop_on_pass` (one confirmation test, then stop — lag 1), `fixed_budget` (10 steps regardless).
  - `run_experiment` / `compare_policies` — aggregated metrics: first-edit solve share, mean/median lag, median post-solution share, final-correct rate, undo-to-incorrect rate, post-solution step composition.
- `test_termination_lag_cpu.py` — deterministic test, 9 pre-registered assertion groups (A1–A9); writes `results.json`.
- `rust/` — Rust second stage (standing rule, 2026-11-02): a line-for-line port of the mechanism (no external crates), `verify` binary mirroring the Python test's experiments and assertions, unit + deep + parity tests. See `rust/README.md`.

## Results

Main experiment (600 tasks, seed 20261113):

| policy | solved | first-edit solve | mean lag | median lag | median post-share | final correct | undo-to-wrong | mean steps |
|---|---|---|---|---|---|---|---|---|
| continue (paper behaviour) | 596 | 0.594 | 4.11 | 4.0 | 0.500 | 0.909 | 0.091 | 7.85 |
| stop_on_pass | 596 | 0.594 | 1.00 | 1.0 | 0.250 | 1.000 | 0.000 | 4.76 |
| fixed_budget (10) | 596 | 0.594 | 6.26 | 7.0 | 0.700 | 0.883 | 0.117 | 9.99 |

- Ordering reproduces the paper's direction: the agent solves early (59.4% on the first edit, inside the paper's 38–82%) and then burns half its run after the solution (median post-solution share 0.500, inside the paper's 35–60% band). Post-solution steps are 53% verification + 26% exploration + 21% re-edits — verification/exploration dominate, as the paper reports.
- The cost of not stopping is measured, not just asserted: stopping on the first passing test cuts mean lag 4.11 → 1.00, saves 3.1 steps per run (7.85 → 4.76), and raises final correctness 0.909 → 1.000 by eliminating undos entirely (0.091 → 0.000). More post-solution steps never help: the sensitivity sweep (mean extra steps 1/3/6) gives mean lag 2.05/4.11/7.25 with final correctness falling 0.943/0.909/0.869.
- The undo rate is the one number outside the paper's band: 9.1% of successful runs end wrong after an earlier correct patch, vs the paper's 1–8%. The miniature's per-re-edit undo probability (0.15) and re-fix rate (0.55) are modelling choices; the direction (continued activity can destroy a correct solution, stopping on pass eliminates it) is reproduced, the exact rate is calibration, not a refutation.

## Why SUPPORTED, not VERIFIED

- The agent is a rule-based stand-in, not a model: edit success is a drawn probability per difficulty, not a model reasoning about code; the paper's agents are five real models on SWE-bench Verified/Multilingual. The stop-policy comparison and the lag/undo accounting are the reproduced mechanism; the agent side is a model of a model.
- The verifier is free and perfect in the miniature (a patch check costs one step and is never wrong); the paper's agents run real test suites with flaky/partial signal and spend real time exploring. The miniature therefore measures the mechanism with a perfect oracle, which is why stop-on-pass reaches final correctness 1.000 while a real agent's confirmation test can pass on a wrong patch.
- Absolute rates are miniature-scale (600 synthetic one-bug tasks, 3 difficulties, fixed probabilities), not the paper's benchmarks (five models × two frameworks × three SWE-bench datasets). The claim reproduced is the *ordering and the mechanism* (solve early, waste steps after, undos destroy solutions, stopping on pass fixes all three), not the 35–60% or 1–8% figures exactly.
- The undo probability (0.15), re-fix probability (0.55), and mean extra steps (3.0) are modelling choices calibrated so the miniature lands in the paper's bands; the paper does not publish equivalents.

## Limits

- Single-bug tasks only: the paper's SWE-bench tasks often need multi-edit fixes; the miniature's "first correct patch" is a single event, so partial-credit trajectories are not modelled.
- No distinction between verification that *finds* a bug (useful) and verification that merely re-confirms (waste): in the miniature every post-solution verify step is pure cost, which biases the result toward stop-on-pass. A real agent's verification sometimes catches a real regression — the paper's 1–8% undo rate vs its 35–60% post-solution share shows most of that work is waste, but not all of it.
- The paired-RNG design makes the pre-solution phase identical across policies by construction; in a real deployment the stop policy can change what the agent does *before* solving (e.g. less careful edits when it knows it will verify later). That feedback is out of scope here.
- No model-backed validation: the mechanism was not re-run with an LLM agent (e.g. via NVIDIA NIM), so the quantitative gaps (9.1% vs 1–8% undo rate) are miniature-calibration differences, not model measurements.

## Files

- `termination_lag.py` — mechanism (stdlib only)
- `test_termination_lag_cpu.py` — deterministic CPU test (15 assertions, 15 passed)
- `results.json` — main experiment + sensitivity sweep + assertion list
- `rust/` — Rust second stage
- Graduated: `src/arxiv_lab/harness/termination_lag.py` + `tests/test_harness_cpu.py` (same mechanism, same assertions)
