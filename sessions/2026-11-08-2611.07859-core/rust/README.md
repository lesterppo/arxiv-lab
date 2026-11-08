# CORE — Rust second stage (arXiv 2611.07859)

Rust port of the CORE mechanism (coarse-to-fine search with context
management) from the Python CPU proxy in the parent session directory
(`core.py`, `corpus.py`, `test_core_cpu.py`). Stdlib only — no external
crates, offline build.

- **Paper:** arXiv [2611.07859](https://arxiv.org/abs/2611.07859) — Vuong
  et al., "CORE: Coarse-to-Fine Search with Context Management for Deep
  Research"
- **Python verdict (primary carrier): SUPPORTED.** The Rust leg never
  replaces it; it re-runs the same experiments E1/E1b/E2/E3 on a Rust
  corpus and checks the same assertions A1–A4. The Rust-corpus verdict
  is also SUPPORTED.
- **Rust leg status: GREEN** — `cargo test` 15/15, `verify` assertions
  4/4, parity sweep 250 cases / 0 mismatches.

## What was ported

| Python | Rust |
|---|---|
| `core.py: BM25` (k1=1.5, b=0.75) | `src/mechanism.rs: Bm25` |
| `core.py: SnippetExtractor` (TF-IDF cosine, best 2-sentence window) | `src/mechanism.rs: SnippetExtractor` |
| `core.py: ContextManager` (Jaccard ≥ 0.8 suppress, expand by one sentence) | `src/mechanism.rs: ContextManager` |
| `core.py: tokenize / content_tokens / count_tokens` | `src/mechanism.rs` |
| `corpus.py: build_corpus / build_questions` | `src/corpus.rs` |
| `test_core_cpu.py: reader + E1/E1b/E2/E3 + assertions` | `src/experiments.rs` + `src/bin/verify.rs` |
| — (parity harness) | `parity_gen.py` + `src/bin/parity.rs` + `src/jsonparse.rs` |

## Results (Rust corpus, `cargo run --release --bin verify`)

Config: BM25 k=5, snippet = best 2-sentence window, Jaccard threshold
0.8, 120 entities, 60 two-hop + 30 single-hop questions.

### E1 — single-turn, 60 two-hop questions

| method | gold-evidence recall (both hops) | reader accuracy | mean context tokens |
|---|---|---|---|
| FULL (whole passages) | 1.000 | 1.000 | 381.3 |
| **CORE (snippets)** | **1.000** | **1.000** | **115.5 (30.3% of FULL)** |
| NAIVE-SNIPPET (first sentence) | 0.000 | 0.733 | 60.5 |

Single-hop controls (n=30): FULL accuracy 1.000 (380.6 tok), CORE 1.000
(118.0 tok).

### E2 — multi-turn, 30 two-turn trajectories (cumulative tokens)

| method | cumulative tokens / trajectory |
|---|---|
| FULL | 760.0 |
| CORE, no context management | 233.3 |
| **CORE + context management** | **183.3 (24.1% of FULL)** |

Suppression rate 0.924 (61 of 66 already-seen passage slots suppressed,
5 expanded by one adjacent sentence). Turn-2 evidence coverage 1.000 for
both CORE without CM and CORE+CM — suppression lost no evidence.

### E3 — snippet-quality ablation (evidence recall, n=60)

| snippet strategy | recall |
|---|---|
| token-similarity (CORE) | 1.000 |
| first sentence | 0.000 |
| random 2-sentence window | 0.117 |

All four pre-registered assertions passed: A1 CORE recall within 5pp of
FULL (gap 0.000); A2 CORE mean tokens 30.3% of FULL (≤ 50%); A3 CORE
recall beats NAIVE-SNIPPET by 100pp (≥ 10pp); A4 CORE+CM cumulative
tokens < CORE without CM (183.3 < 233.3).

### Comparison with the Python numbers

The Rust numbers track Python's closely (Python: FULL 381.2 / CORE 115.7
/ NAIVE 60.1 tok; E2 763.1 / 235.4 / 180.5; suppression 0.971; E3 random
0.233) but are **not identical**, as expected: the Rust corpus is a
different random draw (see divergence 1). The differences are all in the
random-draw noise: NAIVE accuracy 0.733 vs 0.700, suppression 0.924 vs
0.971 (61/66 vs 68/70 seen slots), E3 random-window recall 0.117 vs 0.233.
Every assertion holds on both corpora, and the verdict is SUPPORTED on
both.

## Extensive test (`cargo test`, 15 tests)

- **Unit tests per mechanism:** tokenize/stopwords/token counting; BM25
  ranking (exact match first, zero-score ties by index, k > N
  truncation); snippet window picks the evidence sentence (width 2 and
  width 1); Jaccard values incl. empty/empty = 1.0; ContextManager
  fresh/suppress/expand decisions incl. the `answer_sentence in prev`
  substring rule; corpus shape (120 entities, 7 sentences, gold at 0–1).
- **Brute-force BM25 oracle:** an independent reimplementation in the
  test recomputes df/idf/tf straight from the passage strings; scores
  match to 1e-9 relative and rankings are identical on 10 random queries
  over 40 passages. Parent-review fix (2026-11-08): the oracle originally
  summed query terms in `HashSet` iteration order, which Rust randomizes
  per process — two documents with scores equal to the last ulp could
  flip order between runs, and the exact ranking assertion failed
  flakily (4/6 runs in parent verification). The oracle now sums in
  first-occurrence query order, the same order `Bm25::score` uses, so
  scores are bit-identical; the suite is green 8/8 consecutive release
  runs plus a debug run after the fix. The oracle still recomputes
  df/idf/tf independently — only the float summation order is aligned.
- **Property sweeps (seeded, 50–200 random cases each):** BM25 top-k is
  monotone in k (top-k is a prefix of top-(k+3)); the best window is the
  argmax over all windows scored independently, and contains a similar
  sentence whenever one exists; suppression is idempotent (same decision
  on repeated calls and after re-registering); token count is additive
  over concatenation; BM25 scores are finite, non-negative, and sorted.
- **Soak (scale the Python test does not reach):** 5,000 passages /
  200 questions — index build 234 ms, total 1,218 ms (release);
  1,000/1,000 top-k hits returned.
- **Thread identity:** 8 threads calling `search` + `best_window`
  concurrently on shared `Arc<Bm25>` / `Arc<SnippetExtractor>` produce
  byte-identical results.
- **End-to-end:** E1/E2/E3 run on the Rust corpus and assertions A1–A4
  hold inside the test suite as well.

## Parity sweep (Python vs Rust, 0 mismatches)

`parity_gen.py` generates 250 random cases (query + 3–8 passages of
4–8 sentences + k in 1–8, seeded `Random(20261108)`) and computes the
expected outputs with the Python `core.py`: BM25 top-k ids + scores,
best 2-sentence window indices for passage 0, and a ContextManager
turn-2 decision. `cargo run --release --bin parity -- parity_cases.json`
recomputes each case in Rust and compares: ids exact, scores within
1e-6 relative, window indices exact, CM action + text exact.

**Result: 250 cases, 0 mismatches (0 tie-order notes).**

## Divergences from Python (deliberate, all documented)

1. **RNG — the corpus is a different random draw.** Python
   `corpus.py` uses `random.Random` (Mersenne Twister); Rust uses a
   xorshift64star generator (`src/rng.rs`, same seeds: 20261108 for the
   corpus, seed+1 for questions, 7 for the E3 random windows). Same
   templates, same draw-sequence structure (shuffle order, choice
   order, re-roll loops), different stream — so entities differ
   (different name/award/institution assignments) while the corpus
   shape and all mechanism assertions are unchanged. The parity sweep
   does **not** use this RNG (it compares on identical inputs), so the
   RNG difference cannot hide a mechanism mismatch.
2. **`random.sample` replacement.** Python draws the two "other
   institutions" with `random.sample` (selection sampling); Rust uses a
   partial Fisher-Yates shuffle (`sample_two` in `src/corpus.rs`). Both
   are uniform over unordered pairs; the draw sequence differs (moot,
   given divergence 1).
3. **Modulo reduction in `next_below`.** Rust reduces `u64 % n`;
   Python's `randbelow` is rejection-based and unbiased. Negligible
   bias for the small `n` used here; corpus generation only.
4. **Float summation order.** Python's `BM25.score` iterates
   `set(query_tokens)` (hash order, varies per run); Rust iterates
   query order, deduplicated. Python's TF-IDF dot product is numpy's
   pairwise summation; Rust's is a sequential sum over vocab indices.
   Both are f64; differences are last-ulp level, far inside the 1e-6
   parity tolerance. The parity sweep (0 mismatches over 250 cases)
   confirms this is noise, not a formula difference.
5. **Vocab index order.** Python's `SnippetExtractor` assigns vocab
   indices by first occurrence across sentences; Rust assigns them by
   hash-map insertion order. Indices are internal only; vectors, norms,
   and cosines are identical up to summation order (see 4).
6. **Per-document term frequencies precomputed.** Python rebuilds the
   tf dict on every `BM25.score` call; Rust precomputes it at
   construction. Same values, faster (this is what makes the 5,000
   -passage soak cheap). No behavioural difference.
7. **Tokenization of non-ASCII text.** Python lowercases with Unicode
   `str.lower()` then matches `[a-z0-9]`; Rust uses
   `to_ascii_lowercase` and `is_ascii_alphanumeric`. For the ASCII
   corpus and parity fixtures the two are identical. A non-ASCII letter
   (e.g. `é`) breaks a token in both; a Unicode uppercase letter that
   lowercases to an ASCII letter (none exist in practice for `[a-z]`)
   would differ — no such input occurs here.
8. **Whitespace token counting.** Python `str.split()` and Rust
   `split_whitespace()` both split on Unicode whitespace runs; identical
   on this corpus.
9. **JSON number formatting.** Python `json.dump` writes `repr(float)`;
   Rust's `results.json` writer uses shortest round-trip formatting
   (`fnum` in `src/experiments.rs`). Values are equal; formatting of
   integral floats differs cosmetically (`1.0` vs `1` never occurs —
   integral floats are written as `x.0`).
10. **`verify` exit code / output path.** `test_core_cpu.py` always
    writes `results.json` next to itself and exits 1 on assertion
    failure; `verify` writes to argv[1] (default `results.json` in the
    working directory) and exits 1 on assertion failure or write error.
11. **E3 random-window stream.** Same seed (7) and same draw order as
    Python's `Random(7)`, but a different RNG (see 1), so the random
    -window recall differs (0.117 Rust vs 0.233 Python) while the
    token-sim and first-sentence arms are deterministic and identical.

## Not ported (out of scope, per WORKFLOW.md §4.4)

Nothing model-backed, network, or GPU exists in this mechanism — the
whole Python implementation is pure deterministic CPU logic, so the
port is complete. No numpy is used in Rust (the Python code uses numpy
only for the TF-IDF vectors, which are plain `Vec<f64>` here).

## Reproduce

```bash
cd sessions/2026-11-08-2611.07859-core/rust
export PATH="$HOME/.cargo/bin:$PATH"
cargo test --release            # 15 tests, incl. soak + threads
cargo run --release --bin verify   # E1/E1b/E2/E3, writes results.json
python3 parity_gen.py           # regenerate parity_cases.json (250 cases)
cargo run --release --bin parity -- parity_cases.json   # 0 mismatches
```

## Files

- `Cargo.toml` — stdlib-only package (`core-rust`), bins `verify` and
  `parity`
- `src/lib.rs`, `src/mechanism.rs`, `src/corpus.rs`,
  `src/experiments.rs`, `src/rng.rs`, `src/jsonparse.rs`
- `src/bin/verify.rs`, `src/bin/parity.rs`
- `tests/mechanism_tests.rs` — 15 tests (unit, oracle, property, soak,
  threads, end-to-end)
- `parity_gen.py`, `parity_cases.json` — parity fixture (250 cases)
- `results.json` — measured Rust numbers
- `README.md` — this file
