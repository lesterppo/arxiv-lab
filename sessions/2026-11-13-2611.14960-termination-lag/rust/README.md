# Rust second stage — Termination Lag (arXiv:2611.14960)

Rust port of the Python mechanism in `../termination_lag.py`, built under the
standing Rust-second-stage rule (2026-11-02). The Python version stays the
primary verdict carrier; this port never replaces it.

## What is ported

`src/lib.rs` is a line-for-line port of the pure logic, no external crates:

- `SplitMix64` — identical arithmetic, so RNG streams match Python exactly.
- `gen_tasks` — difficulties drawn with the same `randint(0, 3)` consumption order.
- `verify` — the deterministic test-suite check.
- `run_task` — the simulated agent run: pre-solution exploration, edits until
  the first correct patch, then the post-solution phase per stop policy
  (`continue` / `stop_on_pass` / `fixed_budget`), with undo / re-fix dynamics
  and the same step accounting (lag = steps after the first correct patch).
- `run_experiment` / `compare_policies` — all metrics: first-edit solve share,
  mean/median lag, median/mean post-solution share, final-correct rate,
  undo-to-incorrect rate, mean undos, post-solution step composition.
- `src/main.rs` (`verify` binary) mirrors the Python test's experiments and
  assertions A1–A9 and writes `results.json`.

## Live extensive test (all run live)

- `cargo run --release --bin verify` — 15/15 assertions pass; metrics match
  the Python `results.json` to the printed precision (see below).
- `cargo test --release` — 22 tests, all pass:
  - `tests/unit_tests.rs` (13): RNG vectors, uniform range, randint bounds,
    Poisson support, task balance, verifier, stop-on-pass lag == 1,
    p_undo = 0 / 1 edge cases, fixed-budget exactness, run invariants,
    first-edit solve ordering across difficulties.
  - `tests/deep_tests.rs` (8): mean lag monotone in mean_extra (2000 tasks),
    undo rate monotone in p_undo, stop-on-pass dominates continue on final
    correctness and lag across a seed × budget × undo grid, conditional lag
    identical across difficulties, exhaustive 3-policy × 200-seed invariant
    sweep, Poisson support coverage, 20,000-task soak (all policies),
    4-thread identity (bit-identical metrics).
  - `tests/parity_tests.rs` (1 test, 1080 runs + 27 experiments): replays
    `tests/parity_reference.txt` (generated from the frozen Python module:
    `python3 /tmp/gen_parity_tl.py > rust/tests/parity_reference.txt`) —
    **0 mismatches**.

## Python vs Rust (main experiment, 600 tasks, seed 20261113)

| policy | solved | first-edit | mean lag | median post | final ok | undo-bad |
|---|---|---|---|---|---|---|
| continue | 596 | 0.594 | 4.11 | 0.500 | 0.909 | 0.091 |
| stop_on_pass | 596 | 1.00 lag | 1.00 | 0.250 | 1.000 | 0.000 |
| fixed_budget | 596 | 0.594 | 6.26 | 0.700 | 0.883 | 0.117 |

Identical in both implementations (all integer-derived metrics bit-identical;
see divergences for the one 1-ulp float case).

## Deliberate divergences from Python

1. `mean_post_share` can differ by 1 ulp (e.g. 0.5163906376993624 vs
   0.5163906376993622). Python 3.12+ `sum()` uses Neumaier compensated
   summation for floats; Rust's `Iterator::sum` is naive. The parity test
   therefore compares `mean_post_share` within 1e-12; every other aggregate
   derives from integer sums and matches exactly.
2. None in the mechanism itself: RNG draws, Poisson draws, task generation,
   run trajectories, and experiment aggregates all replay exactly.
