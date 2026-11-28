# MeterRouter — Rust second stage

Rust port of `src/arxiv_lab/agent/meter_router.py` (arXiv:2611.34327),
std-only (no external crates, no network). The Python version stays the
primary verdict carrier; this port mirrors its mechanism and is verified
against it.

- **Rust leg verdict: VERIFIED** (see below)

## Layout

- `Cargo.toml` — package `meter_router`, edition 2021, no dependencies.
- `src/lib.rs` — the port: Table 6 prices, cost meter (Eq. 1),
  remaining budget (Eq. 2), utility (Eq. 5), Appendix E templates,
  latent-threshold environment, SplitMix64 RNG, greedy budget-aware
  router, all baselines (strongest MAS, single high/low, fixed router,
  random-tier), paired experiment runner.
- `src/bin/verify.rs` — `verify` binary mirroring the Python CPU
  test's experiments (E1–E8) and assertions.
- `tests/meter_router_tests.rs` — unit tests and the Python-vs-Rust
  parity sweep.
- `tests/parity_cases.txt` — 1140 cases generated from Python
  (`../parity_dump.py`): 45 call-cost cases, 135 expected-success
  cases, 960 router cases (120 queries × 4 budgets × 2 modes).
- `results.json` — `verify` binary output.

## Results

`verify` (release, Rust-native SplitMix64 worlds — same shapes as the
Python test, different sampled values, so numbers differ while the
asserted properties are the same):

| experiment | Rust | Python (for comparison) |
|---|---|---|
| E4 router acc vs strongest, B = $50×10⁻³ | **0.764 vs 0.780** | 0.782 vs 0.798 |
| E4 router cost vs strongest | **$6.24×10⁻³ vs $10.74×10⁻³** | $6.24×10⁻³ vs $10.74×10⁻³ |
| E4 router vs single high | **0.764 vs 0.682** | 0.782 vs 0.714 |
| E5 cost by budget | **$0.89 → $2.04 → $2.04 → $6.24×10⁻³** | $0.89 → $2.04 → $2.04 → $6.24×10⁻³ |
| E5 accuracy by budget | **0.554 → 0.682 → 0.682 → 0.764** | 0.560 → 0.714 → 0.714 → 0.782 |
| E6 ablation loss (random tier) | **23.2 / 8.8 / 7.4 / 8.4 pts** | 21.0 / 7.8 / 6.6 / 8.8 pts |
| E7 hard vs easy budget gain | **+0.245 vs +0.162** | +0.242 vs +0.209 |
| E3 overrun rate | **0.000 at every budget** | 0.000 |
| E8 determinism | **identical repeated runs** | identical |

`cargo test --release` (unit + parity): **2/2 passed.**

- Parity sweep: **45 call-cost + 135 expected-success + 960 router
  cases, 0 mismatches** (teams and stops exact, costs within 1e-12,
  expected success within 1e-6 — see divergences).

## Deliberate divergences from Python

- **RNG:** Python uses `random.Random` (Mersenne Twister); Rust uses
  an in-repo SplitMix64. Query sets and Monte-Carlo streams therefore
  differ, so sampled accuracies differ while the asserted properties
  are the same.
- **erf:** Rust std has no erf, so the Abramowitz-Stegun 7.1.26
  approximation is used (|error| ≤ 1.5e-7). Analytic expected success
  differs from Python's `math.erf` version by ~1e-7; parity uses a
  1e-6 tolerance for expected success.
- **Summation:** Python 3.12's `sum()` uses compensated (Neumaier)
  summation, Rust's `.sum()` is naive, and Rust's empty f64 sum is
  -0.0; multi-agent team costs can therefore differ in the last ulp.
  Parity compares team costs within 1e-12; single call-cost values
  are compared exactly.
- **Skills:** rounded to 3 decimals before the analytic integral, in
  both languages (matching Python's cache key).
