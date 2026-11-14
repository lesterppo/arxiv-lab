# Rust second stage — ReSHAPE (arXiv:2611.16466)

Rust port of the Python mechanism in `../reshape.py`, built under the
standing Rust-second-stage rule (2026-11-02). The Python version stays the
primary verdict carrier; this port never replaces it.

## What is ported

`src/lib.rs` is a line-for-line port of the pure logic, no external crates:

- `Rng` (SplitMix64) — identical arithmetic, so RNG streams match Python
  exactly (pinned vectors in `tests/unit_tests.rs`).
- `gen_instances` / `gen_instances_with_bonus` — weights, constrained set,
  sacrifice dimension (max-weight constrained attribute), compensating set,
  trade-off bonus, and the utility ranking that fixes the acceptable set
  (top 22 by utility desc, plan index asc).
- `revise_ask` / `revise_show` / `revise_oracle` — the three revision
  strategies, including SHOW's F1-on-shown-labels candidate fit over
  (relax dimension, compensating condition) pairs with the same
  enumeration order and tie-breaks as Python.
- `plan` — the budgeted sampling planner (K draws without replacement).
- `space_metrics` / `run_experiment` / `pearson` — all metrics.
- `src/main.rs` (`verify` binary) mirrors the Python test's experiments
  and assertions A1–A13 and writes `results.json`.

## Live extensive test (all run live)

- `cargo run --release --bin verify` — 23/23 assertions pass; metrics
  match the Python `results.json` exactly (0 diffs at 1e-12, verified by
  a JSON walk over both files — including the bonus-robustness sweep).
- `cargo test --release` — 21 tests, all pass:
  - `tests/unit_tests.rs` (12): RNG vectors, uniform range/values, randint
    bounds, shuffle/sample determinism, domain sizes, instance structure,
    flexibility bounds/ordering, oracle admits every trade-off plan,
    space-metrics hand check, planner full-budget property, revised spaces
    contain S0, ASK relaxes at most one dimension.
  - `tests/deep_tests.rs` (8): SHOW F1 > ASK F1 across a 6-seed × 2-domain
    property sweep, oracle admission >= none admission per instance,
    exhaustive sacrifice/compensating structural checks, exhaustive
    4-strategy × 200-seed invariant sweep, 20,000-instance soak (all
    strategies), 4-thread identity (bit-identical metrics), Pearson r > 0.8
    on fresh seeds, paired rescue replication.
  - `tests/parity_tests.rs` (1 test, 1,600 records): replays
    `tests/parity_reference.txt` (generated from the frozen Python module:
    seeds 1/7/42/20261114 × travel/shopping × 50 instances × 4 strategies;
    per record: n_space, n_admitted, admission, precision, f1, planner
    success) — **0 mismatches**.

## Python vs Rust (main experiment, 300 instances/domain, seed 20261114)

| domain | strategy | admission | F1 | planner success |
|---|---|---|---|---|
| travel | none | 0.448 | 0.440 | 0.673 |
| travel | ask | 0.533 | 0.424 | 0.583 |
| travel | show | 0.609 | 0.526 | 0.727 |
| travel | oracle | 0.754 | 0.583 | 0.747 |
| shopping | none | 0.458 | 0.443 | 0.670 |
| shopping | ask | 0.544 | 0.425 | 0.587 |
| shopping | show | 0.608 | 0.522 | 0.730 |
| shopping | oracle | 0.764 | 0.586 | 0.747 |

Identical in both implementations to the printed precision (and to 1e-12
in the JSON walk). SHOW−ASK F1: +10.3 pts travel, +9.7 pts shopping;
planner lift 1.25x; oracle admission 0.759 pooled; Pearson r = 0.893;
paired rescue 132 vs loss 98 of 600 instances.

## Deliberate divergences from Python

1. None in the mechanism itself: RNG draws, instance generation, revision
   fits, planner draws, and per-instance metrics all replay exactly (the
   1,600-record parity sweep has 0 mismatches, including exact integer
   counts and planner outcomes).
2. `gen_instances_with_bonus` is an extra entry point (the bonus fraction
   is a parameter instead of a mutable module constant) so the `verify`
   binary can run the A13 robustness sweep honestly; `gen_instances`
   calls it with `BONUS_FRAC = 0.25`, identical to Python's default.
3. `results.json` carries one extra top-level key, `"implementation":
   "rust"`, so the two files are distinguishable.
