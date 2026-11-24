# Rust second stage — The Retry Tax (arXiv:2611.29133)

Rust port of the static-retry economics in
`src/arxiv_lab/agent/retry_tax.py`. Std-only, zero crates.

## What is ported (and what is not)

Ported — the whole Python module is pure logic, so everything is
ported:

- `completion_prob` / `expected_attempts` — closed-form per-task
  quantities.
- `population_metrics` / `marginal_cost` — closed-form population
  expectations.
- `simulate_population` — Monte-Carlo run of the static policy.
- `replay_runs` — the paper's replay procedure over recorded
  per-task attempt-outcome sequences.
- `unresolved_mean_p` — survivor-selection diagnostic.
- `calibrated_mixture` — the two-point population calibrated to the
  paper's replay endpoints (C(1) = 0.597, C(4) = 0.709 vs paper
  0.597 / 0.700).

NOT ported — nothing.

## Deliberate divergences from Python

1. RNG: Python uses Mersenne Twister (`random.Random`); Rust uses
   xorshift64* (`Rng`). Simulation draws therefore differ
   draw-for-draw, but every closed-form quantity is bit-identical
   (both sides IEEE-754 f64) and every simulated quantity agrees
   with the closed form within Monte-Carlo noise on both sides. The
   parity sweep checks the closed forms exactly (0 mismatches) and
   the simulations within tolerance.
2. None other. The closed forms use the same numerically robust
   formulation on both sides (`-expm1(cap * log1p(-p))` for
   completion, direct per-task increments for marginal cost), so
   they agree even for p near 0 or near 1, where the naive
   `1 - (1-p)^cap` and differenced-means forms lose digits.

## Layout

- `Cargo.toml`, `src/lib.rs` — the ported logic + deterministic RNG.
- `src/bin/retrytax.rs` — `verify` (mirrors the Python test's E0–E6),
  `metrics-batch` (parity-sweep server over stdin),
  `soak` (100k-task inequality check).
- `tests/unit.rs` — unit tests for each mechanism (8 tests).
- `tests/deep.rs` — property sweeps (2,000 random populations),
  exhaustive small cases vs a brute-force oracle (375 cases),
  100k soak, thread identity (5 tests).
- `tests/parity_sweep.py` — Python-vs-Rust parity driver
  (500 random populations, 0 mismatches).

## How to run

```bash
cargo test --release            # 13 passed, 0 failed
cargo run --release -- verify --json results.json
cargo run --release -- soak
python3 tests/parity_sweep.py   # PARITY VERIFIED
```

## Results

- `cargo test`: 13 passed (8 unit + 5 deep), 0 failed.
- `verify`: ALL RUST CHECKS PASSED — closed-form CPC
  $0.698 → $1.193 (caps 1 → 4), marginal cost $2.24 → $8.07,
  replay completion 0.600 → 0.713, homogeneous CPC constant,
  survivor mean p 0.599 → 0.050.
- `soak`: 100k tasks, caps 1..=12, CPC and marginal cost monotone.
- Parity sweep: 500 cases, 0 mismatches, worst relative diff 1.2e-15.

Rust leg verdict: **VERIFIED**.
