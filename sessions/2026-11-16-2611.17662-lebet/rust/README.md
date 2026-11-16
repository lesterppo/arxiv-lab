# lebet-rs

Rust port of the LeBet mechanism from `lesterppo/arxiv-lab`
(arXiv:2611.17662, "Learn then Bet: Learning Interpretable Rules for
Reliable Agent Intervention"), session
`2026-11-16-2611.17662-lebet`. The Python module
(`src/arxiv_lab/agent/lebet.py`) stays the primary verdict carrier
(**SUPPORTED**); this port cross-checks it. A failed Rust leg never
overturns the Python verdict — here the leg passes, with **bit-for-bit
parity** on the full experiment.

Std only, no external crates, no network, no model calls. Builds offline.

## What is ported

| Python (`arxiv_lab/agent/lebet.py`) | Rust (`lebet_rs`) |
|---|---|
| `Rng` (SplitMix64, `uniform`/`randint`/`choice`/`gauss`) | `Rng` |
| `gen_trajectory`, `gen_dataset` (both failure subtypes) | `gen_trajectory`, `gen_dataset` |
| 10 predicates, `prefix_matrix` | `PREDICATES`, `prefix_matrix` |
| OR-of-AND monitor, `monitor_*` | `Monitor`, `monitor_*` |
| `_random_conjunction`, `_random_monitor`, `_mutate`, `learn_monitor` | `random_conjunction`, `random_monitor`, `mutate`, `learn_monitor` |
| `baseline_fire_rate`, `betting_wealth_series`, `betting_intervene_step`, `monitor_betting_eval/score` | same names |
| `tune_turn_limit`, `tune_error_filter`, `evaluate` | same names |
| `monitor_to_text`, `run_experiment` | `monitor_to_text`, `run_experiment` |

## Build and test

```bash
cargo build --release --offline
./target/release/lebet verify            # full experiment + 8 assertions
cargo test --release --offline           # 8 unit + 7 deep tests
cargo test --offline                     # same, but the 10x soak takes ~85 s in debug
./target/release/lebet soak              # 10x experiment + thread identity
python3 tests/parity_sweep.py            # Python-vs-Rust parity sweep (needs the Python repo)
```

`verify` runs in ~0.4 s (the Python test takes ~30 s). The release
binary is ~700 KB.

## Live-test verification (2026-11-16)

`./target/release/lebet verify` — **ALL LEBET CHECKS PASSED**. The
Rust numbers are identical to the Python `results.json` (parsed-JSON
equality, not just rounded printout):

- monitor: **(loop AND tok_high) OR (err_rate) OR (stalled)**
- train: TPR 0.985, FPR 0.100, score 0.890, q=0.0151, betting power 0.950 at betting FPR 0.020
- held-out betting: FPR **0.050**, power **0.970**, token savings **0.308**, precision **0.969** (base 0.500), median intervene step 16
- immediate: FPR 0.155, power 0.995, savings 0.509
- turn-limit tuned (K\*=22): FPR 0.140, power 0.885, savings 0.164
- error-filter tuned (repeat_tool_fail): FPR 0.145, power 0.615, savings 0.252
- negative control: TPR−FPR = −0.010, monitor (repeat_tool_fail AND same_action_5)
- determinism: rerun reproduces the monitor and headline metrics exactly

## Parity sweep (Python vs Rust, 0 mismatches required)

`python3 tests/parity_sweep.py` — **17 cases, 0 mismatched**.
`tests/parity_gen.py` is the Python oracle; `lebet parity <kind>` is
the Rust side; both emit JSON that is compared parsed (floats compare
exactly after repr round-trip, so any numeric difference is a real
divergence).

| case | size |
|---|---|
| `rng` (u64 / uniform / randint6 / gauss) | 5 seeds × 20 draws × 4 kinds |
| `trajs` (exact step tuples) | 3 seeds, 160 trajectories |
| `mats` (exact prefix matrices) | 2 seeds, 160 matrices |
| `mutate` (monitor string after each edit) | 3 seeds × 1000 edits |
| `wealth` (q, wealth series, intervene step) | 2 seeds × 8 trajectories |
| `experiment` (full result, all fields) | seeds 20261116, 42 |

## Deep tests (`tests/deep.rs`, 7 tests)

- `predicates_match_oracle_random`: 2000 random prefixes × 10 predicates vs independent brute-force oracles
- `predicates_match_oracle_exhaustive`: all 1093 sequences of length 0–6 over a 3-state alphabet
- `monitor_fires_matches_oracle`: 500 random monitors × random matrices
- `property_sweep_seeds`: 50 seeds — determinism, monitor size bounds, q ∈ [0,1], wealth-series closed form, intervene step = first 1/α crossing, tuning validity (train FPR ≤ α), evaluation well-formedness
- `monitor_score_formula`: score == tpr − 2·fpr − 0.005·complexity exactly
- `soak_10x`: 2000+2000 train/test (10× the Python test) — same 8 assertions; see below
- `thread_identity`: 8 threads run the default experiment, monitors and metrics agree exactly

Soak (10×, seed 20261116, release, 1.9 s): monitor
`(repeat_tool_fail) OR (stalled) OR (repeat3)` — a different SLS
optimum on 10× data, as expected — train TPR 0.9995 / FPR 0.5985
(the monitor fires on 60% of success trajectories at least once,
but the betting statistic still intervenes on only 2.1% of them),
held-out FPR **0.018**, power **0.954**, token savings **0.342**,
precision **0.955**; negative control TPR−FPR = −0.005. All soak
assertions pass. Thread identity: **MATCH** (4× in `lebet soak`,
8× in `tests/deep.rs`).

## Deliberate divergences from Python

None in behaviour. The following are documented so the port can be
audited; there are no undocumented divergences.

1. **CPython evaluation order in `_mutate`'s else branch (the one
   real trap).** The Python line is
   `monitor[rng.randint(len(monitor))] = set(_random_conjunction(rng, max_lits))`.
   CPython evaluates the right-hand side (the fresh conjunction)
   *before* the subscript index, so the conjunction's draws come
   first and the index draw second. A naive port that draws the
   index first silently diverges the RNG stream from the next
   `move == 5` onward. The parity sweep caught this: before the
   fix, `mutate` parity diverged at step 0 (seed 99) and the learned
   monitor differed (`(loop AND repeat3) OR ...` vs Python's
   `(loop AND tok_high) OR ...`), while trajectories, matrices,
   wealth series and baselines already matched exactly. The Rust
   `mutate` draws the fresh conjunction first, then the index —
   same order as CPython. After the fix the full experiment matches
   bit-for-bit.
2. **Dead argument dropped.** Python's `run_experiment(lam=2.0)` is
   never read inside the function (the effective λ is `bet_lam`
   = 0.75 everywhere); Rust's `run_experiment` takes no `lam`
   argument.
3. **Data structures.** Python `frozenset` → Rust `BTreeSet<usize>`
   (iteration is sorted, like Python's `sorted(frozenset)`);
   Python monitor tuple → Rust `Vec` (conjunction order preserved).
   `monitor_to_text` sorts literals by predicate *name*, exactly as
   Python's `sorted(...)` over names does — not by index.
4. **`evaluate` recomputes matrices.** Python's `evaluate` rebuilds
   the prefix matrices from the trajectories; Rust does the same
   inside `evaluate` (and `run_experiment` also builds them once
   for learning) — same values, no behavioural difference.
5. **JSON floats.** `lebet verify --results` writes floats via a
   `fnum` helper that mimics Python `repr` (shortest round-trip,
   `.0` for integral values), so the emitted `results.json` is
   directly comparable with Python's. Parity is judged on parsed
   values, not on formatting.
6. **Float math.** `gauss` uses Rust `f64::ln`/`cos`/`sqrt` (system
   libm, same as CPython's `math` module on this machine); the
   parity sweep confirms bit-identical streams, including
   `uniform()`'s `(v >> 11) * 2^-53` construction and `% n`
   `randint` (not a rejection sampler).

## Files

- `Cargo.toml`, `Cargo.lock` (zero dependencies)
- `src/lib.rs` — the port + 8 unit tests
- `src/bin/lebet.rs` — `verify` / `parity` / `soak` CLI + JSON emitter
- `tests/deep.rs` — 7 deep tests (oracles, property sweeps, soak, threads)
- `tests/parity_gen.py` — Python parity oracle
- `tests/parity_sweep.py` — sweep driver (17 cases, 0 mismatches)
- `README.md`, `results.json`

Excluded from upload: `target/` (build artifacts). No binaries, no
secrets — a credential-pattern grep over `rust/` (api keys, passwords,
`ghp_`/`sk-`/`AKIA`/`hsurr:` tokens, private keys) found nothing;
the only hits for a broad `token` pattern are the mechanism's own
token-spend metric (`token_savings`, `traj_tokens`), not auth tokens.
