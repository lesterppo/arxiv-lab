# CAPE — Rust second stage

Rust port of `src/arxiv_lab/agent/cape.py` (arXiv:2611.31938), std-only
(no external crates, no network). The Python version stays the primary
verdict carrier; this port mirrors its mechanism and is verified
against it.

- **Rust leg verdict: VERIFIED** (see below)

## Layout

- `Cargo.toml` — package `cape`, edition 2021, no dependencies.
- `src/lib.rs` — the port: world model, effect table, policy trees,
  exact full-knowledge DP oracle, CAPE compiler (witness, bounds
  B / U_a / T2, three-tier acquisition), baselines.
- `src/bin/verify.rs` — `verify` binary mirroring the Python CPU
  test's experiments and assertions.
- `tests/cape_tests.rs` — unit tests, property sweep, exhaustive
  check, soak, thread identity, Python-vs-Rust parity sweep.
- `tests/parity_cases.txt` — 444 cases generated from Python worlds
  (`../parity_dump.py`); each line is one world + the Python outputs
  to match exactly.
- `results.json` — `verify` binary output.

## Results

`verify` (release, Rust-native xorshift worlds — same shapes as the
Python test, different sampled configs, so numbers differ while the
asserted properties are the same):

| experiment | Rust | Python (for comparison) |
|---|---|---|
| E1 optimal value (12 configs, cap 3, budget 1000, seeds 0–7) | **8/8** | 8/8 |
| E1 certified | **8/8** | 8/8 |
| E1 mean query saving vs eager | **5.8%** (paper 22.0%) | 3.5% |
| E1 mean EF gain | **+42.71pp** (paper 15.38–34.66pp) | +34.38pp |
| E2 optimal value (8 configs, seeds 0–3) | **4/4** | 4/4 |
| E2 certified | **4/4** | 4/4 |
| E3 optimal value (24 configs, seed 0) | **true, uncertified** | true, uncertified |

`cargo test --release` (unit + property + exhaustive + soak +
thread-identity + parity): **12/12 passed** in 577.9 s.

Parity sweep: **444 cases, 0 mismatches** (every field exact:
value, queries, certified, completions, calls, optimum).

Soak: optimal value in **449/500** worlds (the 51 misses all exhaust
the 200-query budget uncertified). Python measures the identical
449/500 on these exact worlds (`../soak_py.py` replicates the Rust
soak world-for-world), so the rate is a property of CAPE at this
scale, not of the port; the test asserts ≥ 0.89.

## Deliberate divergences from Python

1. **Configs are aspect-value indices** (`Vec<u8>`) instead of string
   tuples; responses are an enum (`Ok` / `Err(i)` / `Reveal(i, v)` /
   `Unknown(i)`) instead of strings. Partitioning only needs response
   equality, so values are identical.
2. **Belief states are `BTreeSet<Config>`** (index order) for storage,
   but every order-sensitive iteration (`missing`, `partition`,
   `optimistic_cells`, world sampling) sorts configs by the aspect
   *values* as strings — exactly Python's `sorted(V)` — via
   `sorted_configs` / `config_str_key`. Index order and string order
   differ (delim values `(",", ";", "\t")` are indices 0,1,2 but sort
   as `"\t" < "," < ";"`), and sorting by index was a real bug found
   2026-11-25: parity line 3 chose target `0010` in Rust vs `2101`
   in Python at round 0, giving 89/444 mismatches and a soak optimal
   rate of 430/500. After the fix the parity sweep is 0 mismatches
   and the line-3 acquisition sequences are identical for all 204
   rounds. `BTreeSet` order is used only for set operations, never
   for decisions.
3. **`make_world` uses a xorshift64 RNG**, not Python's Mersenne
   Twister: sampled config *sets* differ from Python's for the same
   seed. `verify` therefore asserts the same *properties* on the same
   shapes of worlds, not identical numbers. Exact equality is the
   parity sweep's job (it rebuilds Python worlds from
   `parity_cases.txt`).
4. **No `f64` except acquisition ranking scores**, as in Python.
5. The Python error-feedback proxy creates an unused
   `random.Random(seed)`; the Rust version omits it (the reviser is
   fully deterministic).

## Honest limits

- The Rust EF-gain mean (+42.71pp) sits *above* the paper's
  15.38–34.66pp range: the direction reproduces (large gain over the
  error-feedback proxy), the exact range does not bind at this scale
  — the proxy is weak on these small worlds, and the Rust worlds
  differ from Python's. The Python E1 mean (+34.38pp) is inside the
  range.
- Query saving (4.9% Rust, 3.5% Python) is far below the paper's
  22.0% at this scale, same as Python — see the session README.
- Rust is only ~1.5× faster than Python here (E1 ~11–22 s/world vs
  ~18–26 s; E3 78 s vs 109 s): the per-query bound recomputation
  dominates in both languages, and the algorithm is unchanged.
