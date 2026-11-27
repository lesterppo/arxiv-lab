# Partial Routing — Rust second stage

Rust port of `src/arxiv_lab/agent/partial_routing.py` (arXiv:2611.33402),
std-only (no external crates, no network). The Python version stays the
primary verdict carrier; this port mirrors its mechanism and is verified
against it.

- **Rust leg verdict: VERIFIED** (see below)

## Layout

- `Cargo.toml` — package `partial_routing`, edition 2021, no dependencies.
- `src/lib.rs` — the port: world model, SplitMix64 RNG, past-run traces,
  guidance learning, delegation rule (δ), all baselines (lead-alone,
  task-level routing, random, fixed easy-type rule), breakdown-only
  ablation, exhaustive oracle, analytic expected values, Monte Carlo
  policy evaluation.
- `src/bin/verify.rs` — `verify` binary mirroring the Python CPU
  test's experiments (E1–E6) and assertions.
- `tests/partial_routing_tests.rs` — unit tests, property sweep,
  exhaustive-oracle check, soak, thread identity, Python-vs-Rust
  parity sweep.
- `tests/parity_cases.txt` — 152 cases generated from Python
  (`../parity_dump.py`): 1 guidance case (5000 traces) + 151 task
  cases (1204 tasks), with Python-computed assignments and analytic
  values to match exactly.
- `results.json` — `verify` binary output.

## Results

`verify` (release, Rust-native SplitMix64 worlds — same shapes as the
Python test, different sampled values, so numbers differ while the
asserted properties are the same):

| experiment | Rust | Python (for comparison) |
|---|---|---|
| E1 partial δ=0.75 success | **0.298 vs lead 0.253** | 0.310 vs 0.262 |
| E1 partial δ=0.75 cost | **27.1 vs lead 50.0** | 29.6 vs 50.0 |
| E1 task-level / random success | **0.230 / 0.249** | 0.229 / 0.254 |
| E2 δ sweep cost | **50.0 → 42.5 → 35.1 → 27.1 → 7.5** | 50.0 → 42.2 → 37.1 → 29.6 → 7.5 |
| E2 success peak | **interior δ=0.5 (0.322)** | interior δ=0.5 (0.335) |
| E2 δ=1.0 collapse | **0.164** | 0.153 |
| E3 breakdown-only | **0.374 / 45.0** (vs lead 0.253) | 0.386 / 45.0 |
| E4 set B learned vs fixed | **0.299 vs 0.323** (fixed wins — honest negative, as in Python) | 0.304 vs 0.332 |
| E6 oracle-12 | **partial 0.325 (92% of oracle 0.354)**, cost 28.8 < 35.1 | 0.343 (89% of 0.384) |
| E5 determinism | **identical repeated runs** | identical |

`cargo test --release` (unit + property + exhaustive + soak +
thread-identity + parity): **12/12 passed.**

- Soak: **500 tasks**, lead 0.255/50.0 vs partial δ=0.75 0.303/28.6,
  task-level 0.225; oracle-100: partial reaches **90.0%** of oracle.
- Parity sweep: **1 guidance case + 151 task cases, 6035 analytic
  comparisons, 0 mismatches** (assignments and analytic (p, cost)
  compared by exact f64 bit equality).

## Deliberate divergences from Python

- **RNG:** Python uses `random.Random` (Mersenne Twister); Rust uses
  an in-repo SplitMix64. Task sets and Monte-Carlo streams therefore
  differ in sampled values while keeping the same shapes and
  distributions. Exact cross-language equality is the parity sweep's
  job (`tests/parity_cases.txt`): it compares deterministic/analytic
  quantities exactly and never compares raw Monte-Carlo streams.
- **Past-run corpus size in `verify`:** 20000 tasks instead of
  Python's 4000, so the measured gaps sit closer to their true values
  (the δ=0.75 threshold sits only ~0.01 above the measured code gap
  at 4000 traces; at 20000 the noise is ~2x smaller). The delegation
  sets asserted are the same as Python's. The parity sweep uses the
  Python-measured guidance (5000 traces) for exactness.
- **No hash maps:** guidance is stored in fixed arrays indexed by
  subtask type, so there is no hash-order dependence anywhere; the
  Python test's PYTHONHASHSEED determinism check has no Rust analogue
  (repeated runs with the same seed are asserted bit-identical
  instead).
- **Arithmetic order** in `analytic_expected`, `learn_guidance`, and
  the oracle enumeration mirrors Python exactly (same summation /
  multiplication order, same tie-break: first maximum wins), which is
  what makes the parity sweep bit-exact.

## How to run

```bash
cargo test --release          # 12 tests
cargo run --release --bin verify   # mirrors the Python test; writes results.json
```
