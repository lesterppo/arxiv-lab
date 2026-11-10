# Rust second stage — FSM control (arXiv:2611.10884)

Rust port of the Python mechanism (`../fsm_control.py`), built after the
Python CPU test passed (standing rule, 2026-11-02). The Python version
stays the primary verdict carrier; this port never replaces it.

## Deliberate divergences from Python

**None intended.** The port is a line-for-line mirror, including the exact
SplitMix64 stream and the exact order in which the RNG is consumed, so
identical seeds produce identical batches in both languages (verified by
the parity sweep below, 0 mismatches). One place where Python's semantics
needed care: `Account` balances in `make_accounts` use Python's floor
modulo on a negated u64 (`-u % 500 - 1`), reproduced in Rust as
`((500 - u % 500) % 500) - 1`. Balance is cosmetic metadata — no decision
logic reads it.

## What is here

- `Cargo.toml` — package `fsmcontrol`, no external crates (std only).
- `src/lib.rs` — the mechanism: SplitMix64, SOP transition table + branch
  conditions, `FsmRuntime` / `PlainRuntime`, `DriftAgent`, `make_accounts`,
  `run_batch`.
- `src/main.rs` — `verify` binary: mirrors the Python test's experiments
  and assertions (A1–A7) and writes `results.json`.
- `tests/unit_tests.rs` — unit tests per mechanism (transition table,
  branch conditions, correct-next-tool walk, rejection keeps state,
  end-of-turn, plain-runtime violation counting, SplitMix64 known stream,
  batch determinism).
- `tests/deep_tests.rs` — deeper tests: every state×tool pair against the
  transition-table oracle; 2,000 random tool sequences (FSM never
  violates, every accepted step tabled and branch-valid); property sweeps
  over 12 seeds × 4 drift rates (FSM never violates, always beats plain
  by ≥ 40); restate never fixes drift (4 seeds); a 2,000-account soak
  (drift_scale grown with the run so the drift curve keeps its shape);
  thread-identity (same batch on 4 threads, identical results — the code
  is stateless, all state in explicit structs); drift-off perfection.
- `tests/parity_tests.rs` + `tests/parity_reference.txt` — Python-vs-Rust
  parity sweep on 5 seeds × 3 arms: per-account correct flags, totals,
  violations, invalid attempts, retries. **0 mismatches required, 0 found.**
- `results.json` — the `verify` binary's output.

## Results (this run)

```
cargo test --release   # 8 unit + 7 deep + 1 parity = 16 passed, 0 failed
cargo run --release --bin verify   # ALL ASSERTIONS PASSED
plain 86/200, restate 84/200, fsm 197/200 — identical to Python, including
quartiles (plain 0.78/0.44/0.36/0.14; fsm 1.00/1.00/0.98/0.96)
```

Two test fixes during development (both test bugs, not mechanism bugs):
- The random-sequence test first generated incoherent accounts (eligible
  AND escalating); `make_accounts` never produces those, so the test now
  generates coherent accounts the same way.
- The soak's plain-quartile drop assertion assumed drift still growing
  across the run; at 2,000 accounts with drift_scale=1000 the drift
  saturates in the first quartile, so the soak uses drift_scale=10000 to
  keep the drift curve's shape.

## Reproduce

```bash
cd rust
cargo test --release
cargo run --release --bin verify   # writes results.json; exit 0
```

No `target/` directory, no binaries, and no secrets are uploaded — only
source, tests, Cargo files, README, and results.
