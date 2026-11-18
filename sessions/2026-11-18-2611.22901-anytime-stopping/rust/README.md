# anytime-stopping-rs

Rust port of the anytime-evaluation harness for coding-agent stopping
rules from `lesterppo/arxiv-lab` (arXiv:2611.22901), pure std, no
external crates, no network, no model calls.

## What is ported

| Python (`arxiv_lab/agent/anytime_stopping.py`) | Rust (`anytime_stopping_rs`) |
|---|---|
| `SplitMix64` | `SplitMix64` (identical u64 arithmetic) |
| `TrajectoryParams` / `CALIBRATED` | `TrajectoryParams` / `Default` (same numbers) |
| `Step`, `generate_trajectory` | `Step`, `generate_trajectory` (same draw order) |
| `stop_run_to_budget`, `stop_first_self_pass`, `stop_double_pass`, `stop_double_pass_cumulative`, `stop_oracle` | same five functions |
| `evaluate_all`, `paired_quality_delta` | `evaluate_all`, `paired_quality_delta` |
| `sensitivity_sweep`, `run_experiment` | `sensitivity_sweep`, `run_experiment` |

## Build and test

```bash
cargo test --release --offline   # 9 unit tests + 6 deep tests
cargo build --release --offline
./target/release/anytime-stopping verify   # live-test verification
```

The binary is small; `verify` runs the full 10,000-run experiment plus
the 11-cell sensitivity sweep in ~0.05 s.

## CLI (agent-facing, JSON out)

```bash
# Live-test verification (mirrors the Python CPU test's experiments
# and assertions, including the sensitivity sweep)
anytime-stopping verify
# -> ALL CHECKS PASSED + RESULTS_JSON {...}

# Parity subcommands (machine-readable JSON for tests/parity_gen.py)
anytime-stopping parity rng <seed> <n>
anytime-stopping parity trajectory <seed> <budget> <p_solvable> <p_fix> \
    <p_change_after_self_pass> <p_change_after_self_fail> <p_break> \
    <p_self_false_pass> <p_self_false_fail>
anytime-stopping parity experiment <seed> <n_runs> <same 8 params...>
```

## Live-test verification (2026-11-18)

`verify` **ALL CHECKS PASSED** with numbers identical to Python on
every metric (same seeds, same SplitMix64 streams): double-pass
saving **25.83%**, oracle saving **19.75%**, baseline pass rate
**0.3095**, hindsight (all runs) **18.40%**, quality delta **+1.72pp**
(wins 416, losses 244). `cargo test --release` **15 passed, 0 failed**;
parity sweep **23/23, 0 mismatches**; `soak` **PASSED** (200k runs:
saving 25.50%, baseline pass 0.2970, delta +2.77pp; two independent
passes bit-identical).

## Deliberate divergences from Python

None in the mechanism. Two presentational differences only:

- `evaluate_all` in Rust returns a struct with per-policy integer
  sums (`sum_stop_steps`, `n_final_pass`, `n_fired`) alongside the
  float aggregates, so the parity sweep can compare exact integers.
  Python computes the same sums internally but does not expose them.
- The `parity` subcommand's experiment JSON omits the float
  `changed_after_fire_rate` (present in Python's dict) so the
  comparison stays exact on integers and bools.

## What the Rust leg does NOT claim

The Rust port cross-checks the Python implementation; it does not
re-validate the paper. The verdict carrier is the Python track, and
the synthetic-trajectory limitation (no access to the paper's 6,000
recorded SWE-bench runs) applies identically here. A failed Rust leg
would never overturn the Python verdict.
