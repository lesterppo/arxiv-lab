# Rust port — writing-distortion metrics (arXiv:2603.18161v2)

Rust second stage for the session `2026-11-25-2603.18161-writing-distortion`
("How LLMs Distort Our Written Language"). Ports the CPU metrics module
`../writing_distortion_metrics.py` to Rust. The Python CPU test passed
16/16; the session verdict is PARTIAL (grammar-mode distortion did not
reproduce with Qwen3.5, minimal/general did directionally). **This Rust
leg never overturns the Python verdict** — it is a parity port only.

## What was ported

From `../writing_distortion_metrics.py`, in `src/lib.rs` (std only, no
external crates):

| Function | Notes |
|---|---|
| `tokenize(text)` | Lowercase alphanumeric words (`[a-z0-9]+`) |
| `unigram_jsd(a, b)` | Jensen–Shannon divergence, log base 2, add-alpha smoothing `alpha = 1e-6` per vocabulary type, clamped to [0, 1] |
| `pronoun_rate(text)` | Pronouns per 100 words → `(rate, count, n)`; same `PRONOUNS` set |
| `adjective_rate(text)` | Adjective proxy per 100 words → `(rate, count, n)`; same suffix list + lexicon (copied verbatim). NOT LIWC |
| `keyword_stance(text)` | Same `PRO_WORDS` / `ANTI_WORDS` (copied verbatim), whole-word/phrase counting, case-insensitive; labels at score ≥ 3 ("pro") / ≤ −3 ("anti") / else "mixed" |
| `pct_change(before, after)` | Same zero guards: `0/0 → 0.0`, `x/0 → +inf` |
| `mean_shift_alignment(a, b)` | Shift vectors, L2-normalize, mean direction, mean cosine similarity; `Result<f64, String>` (Python raises `ValueError` for the same cases) |

Also in the crate (harness only, not part of the ported metrics):
`src/mini_json.rs` — a minimal std-only JSON parser used by the verify
binary (to read `../sample16.json`) and the `parity` bin (JSON-lines
input). The metrics in `lib.rs` never touch JSON.

## Scope exclusions (OUT of scope, by design)

- Model-backed generation (Qwen3.5-4B/9B essay generation, Colab driver)
  and judge/embedding-backed scoring — no model weights, no network, no
  API keys anywhere in this crate.
- The Python module's numpy path for `mean_shift_alignment`: the Rust
  port implements the pure-Python fallback math (same result up to
  floating-point summation order; verified by the parity sweep).
- Anything outside `rust/` in the session directory was not touched; no
  Colab use, no GitHub push from this leg.

## How to run

Requires a Rust toolchain (`cargo`/`rustc`; tested with cargo 1.99.1).
No network needed: zero dependencies, `cargo build` works offline.

```bash
cd rust/
cargo test --release            # unit tests (tests/test_metrics.rs)
cargo run --release --bin writing_distortion   # verify binary (16 checks)
cargo run --release --bin writing_distortion -- --json  # + JSON summary line
python3 parity.py                # parity sweep vs Python (default 5,600 planned
                                 # cases + 400 long-text cases, seed 20261125)
python3 parity.py --cases 3000 --seed 42   # second seed
```

The verify binary reads `../sample16.json` by default (run from `rust/`);
pass another path as the first positional argument. Exit code 0 iff all
16 checks pass.

## Parity sweep

`parity.py` (stdlib only) generates random cases with a fixed seed and
compares the Python module against the Rust `parity` bin (one process,
JSON lines over stdin/stdout). Vocabulary includes pronouns, adjective
suffix/lexicon words, PRO/ANTI single words and multiword phrases, and
filler words; texts get random casing/punctuation. Floats must agree
within 1e-9; counts and labels must agree exactly. Error cases (zero
shift, length/dimension mismatch) must error on both sides.

Results (2026-11-26, this machine):

| Sweep | Cases | Mismatches | Max abs float diff |
|---|---|---|---|
| seed 20261125 | 6,006 (jsd 1500, pronoun 850, adjective 850, stance 1500, pct 450, align 856) + 3 error cases (3/3 matched) | 0 | 9.99e-16 (jsd) |
| seed 42 | 3,404 (jsd 896, pronoun 455, adjective 455, stance 896, pct 241, align 461) + 3 error cases (3/3 matched) | 0 | 1.22e-15 (jsd) |
| **Total** | **9,410 + 6 error cases** | **0** | — |

Soak: each sweep includes 400 long-text cases (200 JSD pairs + 200
stance texts of 2,000–3,000 words each) — 800 long-text cases total,
all matching. Total sweep time ≈ 7 s per run.

## Deliberate divergences from Python (every one documented)

- **D1 — JSD summation order.** Python sums over a `set` (hash order,
  varies per process); Rust sums over the vocabulary in sorted order
  (`BTreeMap`), which is deterministic. Same formula, same smoothing;
  results agree to ~1e-15 (max observed diff 9.99e-16 across 9,410
  parity cases, tolerance 1e-9).
- **D2 — `mean_shift_alignment` numpy path.** Python uses numpy when
  available (mean/norm via BLAS-order summation); Rust implements the
  module's pure-Python fallback math (explicit loops). Same formulas;
  agreement verified by the sweep (align cases included, max diff
  within the same ~1e-15 envelope).
- **D3 — word-boundary (`\b`) approximation.** Python uses regex `\b`
  (Unicode `\w`). Rust checks the neighbouring characters with
  `char::is_alphanumeric() || '_'`. These agree for all ASCII text and
  for the PRO/ANTI phrases (all start/end alphanumeric). They could
  differ only if a phrase were adjacent to a Unicode combining mark or
  similar exotic word character; the sweep vocabulary is ASCII, and
  the sample16 essays (real text, 16/16 stance match) exercise the
  real-world path.
- **D4 — Lowercasing.** Python uses `str.lower()` (Unicode); Rust uses
  `str::to_lowercase()` (Unicode). Both map e.g. `İ` → `i` + combining
  dot; tokenization then keeps only ASCII `[a-z0-9]`, matching the
  Python regex. No observed difference.
- **D5 — Error type.** Python raises `ValueError` (empty lists,
  unequal lengths, unequal paired dimensions, zero shift vector);
  Rust returns `Err(String)` with the same message text. The verify
  binary and parity bin treat `Err` as the error case; 3/3 error
  cases matched in each sweep.
- **D6 — `pct_change(0, negative)` is `+inf`.** This is Python
  behaviour (`float("inf")` regardless of the sign of `after`), kept
  verbatim in Rust — not a bug, documented here so nobody "fixes" it.
- **D7 — Cross-list double counting in `keyword_stance`.** Each list
  entry is counted independently, so e.g. "reduce accidents" counts
  as pro AND the bare word "accidents" inside it counts as anti.
  This is the Python behaviour (verified: Python gives pro=2 anti=1
  for "We can reduce accidents and save lives every day."), kept
  verbatim; the unit test documents it.
- **D8 — Empty-text JSD is 0.0, not 1.0.** Smoothing makes an empty
  text uniform over the vocabulary, so empty vs a uniform text is
  exactly 0.0 (Python confirms). Kept verbatim; unit-tested with the
  Python value 0.09328454978990909 for empty vs skewed text.

## Undocumented divergences

None known. Any future difference found by the parity sweep must be
added to D1–D8 above, not silently accepted.

## Files

- `Cargo.toml` — package `writing_distortion`, edition 2021, no deps
- `src/lib.rs` — ported functions + wordlists
- `src/main.rs` — `writing_distortion` verify binary (mirrors the 16 Python CPU-test checks)
- `src/bin/parity.rs` — `parity` JSON-lines oracle bin
- `src/mini_json.rs` — minimal JSON parser (harness only)
- `tests/test_metrics.rs` — 31 unit tests
- `parity.py` — parity sweep driver (stdlib only)
- `parity_results.json`, `parity_results_seed20261125.json`, `parity_results_seed42.json` — sweep outputs
- `results.json` — verify output + test counts + parity summary

`target/` and binaries are build artefacts and are never committed.
