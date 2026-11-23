# Rust second stage — SplitBench (arXiv:2611.27860)

Rust port of the trusted grader's pure logic from
`src/arxiv_lab/agent/splitbench.py`. Std-only, zero crates.

## What is ported (and what is not)

Ported — the decision procedure the trusted grader applies to the
outputs a sandbox returns, plus the audit classification:

- `values_equal` — type-strict JSON comparison (bool ≠ int, numbers
  within 1e-9).
- `validate_outputs` — schema check on a sandbox-returned payload.
- `grade_outputs` — trusted verdict from validated outputs vs
  in-memory expected answers.
- `classify_attempt` — whether any recorded sandbox probe could
  have reached the expected answers or the verdict.

NOT ported — the sandboxing itself (spawning the agent's code in an
isolated directory, holding the answers in the parent process) is
process management, not portable pure logic; the Python test is the
authority for the forge/agreement/conversion experiments. The Rust
`verify` binary replays the Python test's recorded `results.json`
rows and re-checks their conclusions (V1).

## Deliberate divergences from Python

1. Numbers are `f64` throughout. Python ints are arbitrary
   precision; values beyond 2^53 lose integer precision here. All
   fixtures and sweeps stay within that range, and Python's own
   comparison is numeric (`1 == 1.0`), so no tested case differs.
2. NaN never equals anything on either side; JSON input cannot
   express NaN, so the parity sweep never generates it.
3. Object key order is insignificant on both sides.

No other divergences. In particular the 1e-9 tolerance, the
schema-rejection reasons, and the classification rules are identical.

## Layout

- `Cargo.toml`, `src/lib.rs` — the ported logic + minimal JSON
  parser/serializer.
- `src/bin/splitbench.rs` — `verify` + batch parity subcommands
  (`values-equal-batch`, `grade-batch`, `classify-batch`, `soak`).
- `tests/unit.rs` — unit tests for each mechanism (8 tests).
- `tests/deep.rs` — property sweeps, exhaustive sequences, 100k
  soak, thread identity (5 tests).
- `tests/parity_sweep.py` — Python-vs-Rust parity driver.

## How to run

```bash
export PATH="$HOME/.cargo/bin:$PATH"
export CARGO_TARGET_DIR=/tmp/splitbench-rs-target  # keeps target/ out of the repo
cargo test --release --offline
cargo run --release --offline -- verify --results ../results.json
python3 tests/parity_sweep.py
```

## Results

- `cargo test`: 13 passed (8 unit + 5 deep), 0 failed.
- `verify`: ALL RUST CHECKS PASSED — V0 unit checks, V1 replay of
  the Python results (native-targeting payloads forged 24/24, split
  forged 0/32, worst-case 0/32, fresh-input max 0.0, agreement
  24/24, conversion 8/8), V2 property sweep (5,000 random pairs),
  V3 exhaustive universe (441 pairs), V4 soak (20,000 pairs,
  3.26 ms), V5 thread identity (8 threads, one digest).
- Parity sweep: 7,840 cases (5,336 values-equal + 2,004 grade +
  500 classify), **0 mismatches**.

## Rust leg verdict: VERIFIED
