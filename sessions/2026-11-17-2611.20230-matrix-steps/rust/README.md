# matrix-steps-rs

Rust port of the matrix-with-steps episodic/procedural memory from
`lesterppo/arxiv-lab` (arXiv:2611.20230, "A Matrix with Steps: An
Episodic and Procedural Memory Method for Tool-Using Agents and Robots"),
session `2026-11-17-2611.20230-matrix-steps`. The Python module
(`src/arxiv_lab/memory/matrix_steps.py`) stays the primary verdict
carrier (**SUPPORTED**); this port cross-checks it. A failed Rust leg
never overturns the Python verdict — here the leg passes, with
**identical numbers to Python on every discrete metric**.

Std only, no external crates, no network, no model calls. Builds offline.

## What is ported

| Python (`arxiv_lab/memory/matrix_steps.py`) | Rust (`matrix_steps_rs`) |
|---|---|
| `fnv1a_64` | `fnv1a_64` |
| `MT19937` (init_genrand / init_by_array), `Rng` | `Mt19937`, `Rng` |
| `HashEmbedder` (hashing-trick bag-of-words) | `HashEmbedder` |
| `procedure_sign_vector` (FNV-1a -> MT, low bit = sign) | `procedure_sign_vector` |
| `DeltaMatrix` (Eq. 1, widening, rebuild, writer limits) | `DeltaMatrix` |
| `Step`, `ProcedureRecord`, `Episode`, `recorded_outcome` | same names |
| `confidence`, `consolidate_episodes` | same names |
| `EpisodicMemory` (write gate, find, next step, erasure, demotion) | `EpisodicMemory` |
| `z_test_one_sided_p`, `promotion_window`, `run_promotion_windows`, `naive_keep_count` | same names |
| `generate_synthetic_procedures`, `capacity_experiment` | same names |
| `simulate_hidden_criterion`, `simulate_revisit_stream`, `simulate_demotion` | same names |

## Build and test

```bash
cargo build --release --offline
./target/release/matrix-steps verify [--results results.json]  # full experiment + assertions
cargo test --release --offline                                # 14 unit + 7 deep tests
./target/release/matrix-steps soak [--results soak_results.json]
python3 tests/parity_sweep.py                                  # Python-vs-Rust parity sweep
```

`verify` runs in ~104 s (the Python test takes ~301 s, mostly the
capacity rebuilds); the release binary is ~700 KB.

## Live-test verification (2026-11-17)

`./target/release/matrix-steps verify` — **ALL MATRIX-STEPS CHECKS
PASSED**. The Rust numbers are identical to the Python `results.json`
on every discrete/count metric (parsed-JSON equality, not just rounded
printout):

- hidden criterion: no-memory solved 0.830, self-judged-stops reported
  0.935 / solved 0.215, verified solved 0.845 = ceiling 0.845
  (attempts/task 3.06 / 1.38 / 2.14; damaging 318 / 144 / 150)
- capacity: 200 procs found 1.000, next-step 0.928; 500 procs found
  0.994, next-step 0.877 (median lookup 1.83 ms / 2.97 ms — timing,
  not a semantic difference; Python measured 2.67 ms / 5.36 ms)
- consolidation: confidence 0.6250 = 5/8, fixed args + failure counts
- revisit: first-attempt 1.000 on 3000 revisits (1000 situations)
- erasure: 20/20 erased not returned, 180/180 kept found, max cosine
  of an erased key to its erased value 0.242 (< 0.50)
- demotion: solved 0.760 -> 0.970, 1 trip, 1 demotion
- promotion: identical variant 0/200 promoted (naive rule 84/200);
  better variant (30% -> 60%) 118/200 promoted

`cargo test --release --offline` — **21 passed, 0 failed** (14 unit +
7 deep). `soak` — **SOAK PASSED**: 10 000-situation revisit still
solves every revisit at the first attempt (30 000 revisits), 4 threads
agree bit-for-bit, property sweep worst read-back error 0.4332.

`python3 tests/parity_sweep.py` — **26/26 cases match, 0 mismatches**
(rng streams, embeddings, sign vectors, small matrices, consolidation,
promotion windows, hidden-criterion counts, revisit, demotion).

## Deliberate divergences from Python

1. **`erf`**: Rust has no `erf` in std, so `z_test_one_sided_p` uses
   the Abramowitz & Stegun 7.1.26 rational approximation
   (|error| < 1.5e-7); Python uses `math.erf` (C library). A promotion
   decision could flip only if a p-value landed within ~1e-7 of an
   alpha threshold — none observed: the parity sweep's promotion cases
   and the full 200-window runs agree exactly (0/200 and 118/200).
2. **Float summation order**: numpy (pairwise/BLAS) vs Rust sequential
   loops. Matrix entries therefore agree only to ~1e-2 absolute on a
   1M-entry checksum; all discrete outputs (found ids, next-step texts,
   counts, decisions) are exact. The parity sweep compares checksums
   with a 1e-2 tolerance and everything else exactly.
3. **`verified_episodes` ordering**: Rust sorts by episode id (a
   `HashMap` has no order); Python preserves insertion order. The
   *set* of verified episodes is identical; only the order of
   `episode_ids` in a consolidated record built through the memory
   could differ. The parity sweep's consolidation cases go through
   `consolidate_episodes` directly and match exactly.
4. **Median lookup time** is machine timing, not a semantic output; the
   Rust binary is faster (1.83 ms vs 2.67 ms at 200 procedures).
5. **Nothing else**: RNG streams (MT19937 init_genrand/init_by_array),
   FNV-1a, the hashing-trick embedder, sign vectors, the delta-rule
   update order, the promotion coin/outcome draw order, and all three
   simulation harnesses consume draws in exactly the Python order, so
   the simulation counts agree bit-for-bit.
