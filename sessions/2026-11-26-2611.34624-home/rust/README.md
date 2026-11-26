# HoMe — Rust second stage

Rust port of `src/arxiv_lab/agent/home.py` (arXiv:2611.34624), std-only
(no external crates, no network). The Python version stays the primary
verdict carrier; this port mirrors its mechanism and is verified
against it.

- **Rust leg verdict: VERIFIED** (see below)

## Layout

- `Cargo.toml` — package `home`, edition 2021, no dependencies.
- `src/lib.rs` — the port: world model, session-versioned memory,
  retrieval scoring, answer rules, failure attribution (STALE /
  MISSING), localized candidate edits, random-edit baseline, HoMe
  loop (formal evaluations + smoke gating + cost phase), exhaustive
  256-harness oracle.
- `src/bin/verify.rs` — `verify` binary mirroring the Python CPU
  test's experiments and assertions.
- `tests/home_tests.rs` — unit tests, property sweep, exhaustive
  check, soak, thread identity, Python-vs-Rust parity sweep,
  `dump_soak_world` (prints the seed-5000 world for the soak
  cross-check).
- `tests/parity_cases.txt` — 300 worlds generated from Python
  (`../parity_dump.py`); each line is one world + the Python loop
  and oracle results to match exactly.
- `results.json` — `verify` binary output.

## Results

`verify` (release, Rust-native xorshift worlds — same shapes as the
Python test, different sampled values, so numbers differ while the
asserted properties are the same):

| experiment | Rust | Python (for comparison) |
|---|---|---|
| E1 loop accuracy | **0.125 → 1.000** | 0.125 → 1.000 |
| E1 tokens | **56 → 42** | 56 → 42 |
| E1 oracle accuracy / tokens | **1.000 / 42** | 1.000 / 42 |
| E1 formal evals | **4** (of 6) | 3 |
| E2 localized vs random gain/formal-eval | **0.2287 vs 0.0687** | 0.2270 vs 0.0949 |
| E3a trap smoke acc | **0.250 vs 0.375** incumbent | 0.375 vs 0.500 |
| E3a trap formal acc | **0.688 vs 0.771** incumbent | 0.542 vs 0.646 |
| E3b gated vs ungated formal mean | **4.00 = 4.00** | 3.92 = 3.92 |

`cargo test --release` (unit + property + exhaustive + soak +
thread-identity + parity): **14/14 passed**.

Parity sweep: **300 worlds, 0 mismatches** (best harness, accuracy,
tokens, formal / smoke / rejects / tried counts, oracle — every
field exact).

Soak: loop reaches oracle accuracy in **500/500 worlds** (mean
formal evals 3.91). The Python cross-check (`../soak_py.py`)
replicates the Rust xorshift world generator call-for-call and
measures the identical **500/500 and 3.91** on the same worlds; the
seed-5000 world replica is an **EXACT MATCH** field-by-field
(evidence: `../rust_dump.txt`). The soak threshold (0.97) sits below
the measured 1.000 in both languages.

## Deliberate divergences from Python (documented)

- Entities / attributes / values are indices into fixed tables
  (`ENTITIES`, `ATTRIBUTES`, `VALUES`) in the same order as
  Python's, instead of strings; behaviour is identical.
- `Harness.recency` is stored as a percentage (u8: 0/25/50/100)
  instead of an f64, so `Harness` is `Eq + Hash` and candidate
  sets stay deterministic. Scoring converts to f64 identically.
- World generation uses a xorshift64 RNG, not Python's
  `random.Random`; the *shapes* match (same distributions, same
  call order — see `make_world`), but sampled values differ. Exact
  cross-language equality is the parity sweep's job.
- `build_memory` returns `Vec<Vec<Fact>>` indexed by case position
  instead of a `dict` keyed by case index.
- No hash iteration anywhere: every iteration order is insertion
  order or an explicit sort, so results are identical across runs
  and processes (the CAPE determinism lesson — the Python module
  was also checked across `PYTHONHASHSEED` 1/2/3).

One divergence found during the port and fixed on both sides: the
first Rust `verify` run failed E3a — the trap candidate tied the
incumbent on the plain first-8 smoke set (0.500 vs 0.500), so the
gate could not see that it breaks historical queries. The fix,
applied identically to Python and Rust, is a stratified smoke set
(round-robin over the historical × synonym buckets — the paper's
adaptive sampling). After the fix the gate rejects the trap on both
sides (Rust 0.250 vs 0.375; Python 0.375 vs 0.500).

## How to run

```bash
cargo test --release          # 14/14
cargo run --release --bin verify   # writes results.json
```
