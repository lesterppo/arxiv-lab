# Mingbird harness trio — Rust second stage (arXiv:2610.02001)

Article: Mingbird, a local-first agent harness (arXiv:2610.02001).
This session is the Rust second stage for the agent track, under the
2026-11-01 standing rule: after the Python version is verified, build
a Rust version, run a live extensive test on it, and upload it under
this session's `rust/` subsession. The Python version is the primary
verdict carrier and stays in `src/arxiv_lab/harness/` (PrefillBudget,
FinishGate, LoopDetector) with `tests/test_harness_cpu.py`; the Rust
version does not replace it.

## Why this mechanism

Oldest graduated arxiv-lab module (2026-10-03), vendored into
DeepWorld's agent loop, pure stdlib in Python, no numpy, no network,
no model calls — stable and reused every agent turn, so a byte-level
parity check against Python is meaningful.

## Python version, re-verified 2026-11-01

`tests/test_harness_cpu.py`: ALL HARNESS CHECKS PASSED.

- Prefill OFF 6660 bytes, OVERFLOW; prefill ON 600 bytes, fits
- Gate OFF: premature finish accepted, score 0.50
- Gate ON: rejected, missing `status_artifact`
- Loopdet OFF: 60 steps burned; loopdet ON: 2 steps, broken

## Rust version

In `rust/` (a copy of the local `arxiv-harness-rs` project, without
`target/`): library + JSON CLI (`prefill`, `loop`, `finish`, `verify`).
A `jsonutil` parity layer reproduces Python `json.dumps` (insertion
order, `sort_keys`, `ensure_ascii`) so byte counts and SHA-1 signatures
match Python, including for non-ASCII input.

## Live extensive test on the Rust version, 2026-11-01

- `cargo test --offline`: 17/17 pass (9 unit + 8 deep, deep suite ~2.9 s)
- `arxiv-harness verify`: ALL HARNESS CHECKS PASSED, same table as
  Python (only `true`/`false` vs `True`/`False` and JSON vs Python
  list-quote formatting differ)
- Deep suite (`rust/tests/deep.rs`, fixed-seed): 2000-case prefill
  property sweep, 254 exhaustive + 5000 random loop sequences vs an
  oracle, 200k-action soak with no false loop, 1000 random finish-gate
  cases, 8-thread identity, plus the pinned interaction that with
  repeat_k=5 identical actions fire first as a 2-step cycle at step 4
  (same as Python)
- Python-vs-Rust parity sweep, same day: 500 random loop signatures
  and 200 random prefill builds, 0 mismatches
- Timing (median, 30 runs, same method): Python test 49.7 ms,
  Rust `verify` 2.86 ms; release binary 763 KB

## Known limits

- Python's last-resort truncation behaves oddly for budgets 0/1
  (a negative slice wraps); the Rust port saturates instead.
  Budgets >= 2 match exactly.
- The CLI `finish` exposes file-content criteria only; the library
  gate takes arbitrary closures, as in Python.

## Verdict

Python: SUPPORTED (bounded scale, scripted policies — the original
graduated verdict). Rust second stage: parity VERIFIED on the
extensive tests above. Neither changes the other's verdict.
