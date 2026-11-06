# Signed manifests — Rust second stage (arXiv 2611.07483)

Std-only Rust port of the Python mechanism in
`src/arxiv_lab/multiagent/signed_manifest.py`. No crates, no network,
no model. SHA-256 and HMAC-SHA256 are implemented from scratch here
(verified against Python's `hashlib`/`hmac` and RFC 4231 test case 1).

## Scope (standing rule: port the mechanism module, pure logic only)

Ported:

- `sha256` / `hmac_sha256_hex` — byte-identical to Python's `hmac`
  (parity sweep checks the hex on every case).
- `canonical_json` — exactly Python's `json.dumps(..., sort_keys=True,
  separators=(",", ":"))`, including `\n` escaping in the task text.
- `extract_requirements` — last line per field wins (injection overrides).
- `Decision::conformant`, `check_requirement`, `unsafe_default`.
- `deterministic_verifier`, `noisy_surface_checker` (keyword-overlap
  simulation of the paper's LLM checker).
- `run_case_explicit` — the full trajectory with every random draw
  made explicit (requirement list, format, threat, victim index,
  inject flag), returning the same canonical result string as Python.

NOT ported (out of scope, recorded not forced):

- The `run_experiment` RNG stream: Python uses Mersenne Twister
  (`random.Random`); the Rust `verify` command runs the same
  experiment shape with a xorshift64 stream and asserts the same
  claim thresholds, not bit-identical numbers. Exact per-case equality
  is the parity sweep's job, on explicit cases.

## Deliberate divergences from Python

None in the ported logic. Two representational notes (not behaviour
changes):

- Python's decision summary renders booleans as `True`/`False`; Rust
  renders `true`/`false`. Both are lowercased before the noisy checker
  reads them, so the checker sees identical text.
- The parity result string encodes the refused decision as
  `-,-,-,-,-` in both implementations.

One real divergence was caught by the parity sweep during development
and fixed: the Rust `alter` branch matched the tamper on the
requirement's *field* name and compared `"payment"` against the field
`"payment_approved"`, so it altered logging instead of payment (9/480
cases mismatched). The fix matches on the requirement *id*.
The Python implementation was correct throughout; the sweep exists
for exactly this class of bug.

## Live extensive test (all run live)

- `cargo test`: **13/13** — SHA-256 vectors (empty, `abc`, 56-byte,
  1M-byte), HMAC-SHA256 RFC 4231 case 1, extractor/predicate/verifier
  units, property sweeps (signed+mutate always refused, 2,000 cases;
  signed-unsafe only via inject, 5,000 cases; free-text never refuses,
  2,000 cases), 100k-case soak (invariants hold; signed unsafe far
  below free-text), 8-thread identity (20,000 calls, stateless).
- `signed_manifest verify`: **PASSED** — unsafe under attack signed
  0.167 vs free-text 0.940; signed mutate 0.000 (refused 60/60);
  unsigned mutate 0.417; verifier detection 0.783 vs 0.105,
  false alarms 0.000 vs 1.000; inject 0.833 vs none 0.000.
- Python-vs-Rust parity sweep (`../parity_signedmanifest.py`):
  **480 cases, 0 mismatches** — PARITY VERIFIED.

## Layout

- `Cargo.toml`, `Cargo.lock` — package `signed_manifest`, edition 2021.
- `src/lib.rs` — SHA-256, HMAC, catalogue, extractor, merger, checkers,
  `run_case_explicit`.
- `src/main.rs` — `signed_manifest verify` / `signed_manifest parity` CLI.
- `tests/deep.rs` — the extensive tests above.
- `results.json` — the numbers from the live run.

No `target/`, no binary, no secrets in this subsession.
