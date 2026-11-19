# semantic-context-handoff-rs

Rust port of the semantic context handoff mechanism from
`lesterppo/arxiv-lab` (arXiv:2611.24357), pure std, no external
crates, no network, no model calls.

## What is ported

| Python (`arxiv_lab/agent/semantic_context_handoff.py`) | Rust (`semantic_context_handoff_rs`) |
|---|---|
| `SplitMix64` | `SplitMix64` (identical u64 arithmetic) |
| `TaskParams` / `CALIBRATED` | `TaskParams` / `Default` (same numbers) |
| `Fact`, `Task`, `generate_task` | same types, same draw order |
| `evaluate_policy`, `POLICIES` | `evaluate_policy`, `Policy` (same token accounting) |
| `evaluate_all` | `evaluate_all` (same integer sums + float rates) |
| `sensitivity_sweep`, `run_experiment` | same functions (same seeds, same order) |
| `task_digest` | `task_digest` (same recurrence) |

## Build and test

```bash
cargo test --release --offline   # 10 unit tests + 7 deep tests
cargo build --release --offline
./target/release/semantic-context-handoff verify   # live-test verification
```

The binary is small; `verify` runs the full 10,000-task experiment
plus the 27-cell sensitivity sweep in ~0.3 s.

## CLI (agent-facing, JSON out)

```bash
# Live-test verification (mirrors the Python CPU test's experiments
# and assertions, including exact headline-number checks against the
# Python results.json and the sensitivity sweep)
semantic-context-handoff verify
# -> ALL CHECKS PASSED + RESULTS_JSON {...}

# Parity subcommands (machine-readable JSON for tests/parity_gen.py)
semantic-context-handoff parity rng <seed> <n>
semantic-context-handoff parity task <seed> <params...>
semantic-context-handoff parity experiment <seed> <n_tasks> <params...>
# Params (13, in order): steps facts_per_step delayed_frac max_delay
# min_fact_tokens max_fact_tokens state_retention compression
# state_overhead recall lookup_overhead fp_per_lookup
# pass_step_fraction
```

## Live-test verification (2026-11-19)

`verify` **ALL CHECKS PASSED** with numbers identical to Python on
every metric (same seeds, same SplitMix64 streams, checked to 1e-9
against `../results.json`): full task_pass **1.0000**, summary
**0.9288**, sch **0.9999**, oracle **1.0000**; step success summary
**0.9149**, sch **0.9911**; SCH token saving vs FULL **65.5%**, SCH
over SUMMARY tokens **+1.8%**; recovery hit rate **0.9003**
(recoveries/task 1.067). Sensitivity sweep **27/27** cells
SCH>SUMMARY, **25/27** within 10pp of FULL — exactly as in Python.
`cargo test --release` **17 passed, 0 failed**; parity sweep
**24/24, 0 mismatches**; `soak` **PASSED** (100k tasks: summary pass
0.9287, sch pass 0.9999, saving 65.44%; two independent passes
bit-identical).

## Deliberate divergences from Python

None in the mechanism. Presentational differences only:

- `evaluate_all` in Rust returns a struct with per-policy integer
  sums (`passes`, `strict_passes`, `steps_ok`, `tokens`,
  `recoveries`, `recovery_hits`) alongside the float aggregates, so
  the parity sweep can compare exact integers. Python computes the
  same sums internally but does not expose them.
- The `parity` subcommand's task/experiment JSON contains integer
  aggregates and bools only (no floats), so the comparison stays
  exact.
- `run_experiment` in Rust keeps the per-seed aggregates in the
  returned struct (Python returns only the pooled averages and the
  rounded per-seed percentages); the pooled numbers are identical.
- `randint` is implemented with an i64 span so the degenerate
  `hi < lo` case returns `lo`, exactly as Python's
  `lo + int(uniform() * (hi - lo + 1))` does — a faithful mirror,
  not a divergence.

## What the Rust leg does NOT claim

The Rust port cross-checks the Python implementation; it does not
re-validate the paper. The verdict carrier is the Python track, and
the synthetic-task limitation (no access to the paper's 150-task
SWE-bench study) applies identically here. A failed Rust leg would
never overturn the Python verdict.
