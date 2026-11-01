# arxiv-harness-rs

Rust port of the Mingbird agent-harness trio from `lesterppo/arxiv-lab`
(arXiv:2610.02001), the oldest graduated arxiv-lab module (2026-10-03),
vendored into DeepWorld's agent loop, pure stdlib in Python, no numpy,
no network, no model calls.

Why this one, and not a newer session: it is stable (unchanged in
essence since day one, reused every agent turn), deterministic, and
small enough that a byte-level parity check against Python is actually
meaningful. Newer daily mechanisms are still in their Python-prototype
phase and should stay there — see the previous conversation's verdict.

## What is ported

| Python (`arxiv_lab/harness/`) | Rust (`arxiv_harness_rs`) |
|---|---|
| `prefill.PrefillBudget` | `prefill::PrefillBudget` |
| `finish_gate.FinishGate` | `finish_gate::FinishGate<Env>` (generic over env, closures as criteria) |
| `loop.LoopDetector` | `loop_detector::LoopDetector` |

`jsonutil` is the parity layer: it reproduces Python `json.dumps`
(compact and default separators, insertion order, `sort_keys`, and
`ensure_ascii` escaping) so byte counts and SHA-1 signatures match the
Python reference, including for non-ASCII input.

## Build and test

```bash
cargo test --offline        # 9 unit tests + 8 deep tests (see below)
cargo test --offline --test deep
cargo build --release --offline
./target/release/arxiv-harness verify   # live-test verification, see below
```

The binary is 763 KB; `verify` runs in ~0.003 s.

## CLI (agent-facing, JSON in/out)

```bash
# Prefill budget: schemas JSON array, system text as --system
arxiv-harness prefill --budget 600 --system "You are a file-task agent." \
  --schemas-file schemas.json
# -> {"prefill": "...", "bytes_used": 600, "compressed": true}

# Loop detection: actions JSON array [{tool, args}]
arxiv-harness loop --actions-file actions.json
# -> [{"tool": "read", "signature": "3496ad59482c", "loop": null}, ...]

# Finish gate: fs snapshot + criteria [{name, path, equals}]
arxiv-harness finish --fs-file fs.json --criteria-file criteria.json
# -> {"accepted": false, "failed": ["status_artifact"]}
```

The CLI `finish` instantiates the generic gate with a filesystem
snapshot (path -> content) and `equals` criteria. The library gate
itself takes arbitrary closures, as in Python.

## Live-test verification (2026-11-01)

`arxiv-harness verify` re-runs the Python test
`tests/test_harness_cpu.py` experiment for experiment in Rust — same
bloated schemas, same tasks, same scripted policies (bloat / premature
finish / looper) and same assertions. Results, Rust vs Python:

| Experiment | Rust | Python |
|---|---|---|
| Prefill OFF, bytes in context | 6660, OVERFLOW | 6660, OVERFLOW |
| Prefill ON, bytes (budget 600) | 600, fits | 600, fits |
| Gate OFF, premature finish accepted? | true, score 0.50 | True, score 0.50 |
| Gate ON, premature finish accepted? | false, missing `status_artifact` | False, missing `status_artifact` |
| Loopdet OFF, steps burned | 60, ran to max_steps | 60, ran to max_steps |
| Loopdet ON, steps burned | 2, broken | 2, broken |

Direct Python-vs-Rust parity, same inputs:

| Check | Python | Rust |
|---|---|---|
| Prefill, budget 10000, bytes | 6338 | 6338 |
| Prefill, non-ASCII (`café`), bytes | 53 | 53 |
| Loop signature, `read {"path": "/data/nums.txt"}` | `3496ad59482c` | `3496ad59482c` |
| Loop signature, key-order-insensitive `{"b":1,"a":2}` | `c9b68fc2e7ff` | `c9b68fc2e7ff` |
| Loop signature, non-ASCII path | `fdce4fc35daf` | `fdce4fc35daf` |
| `build_prefill` mutates caller schemas? | No | No |

Only formatting differs between the two `verify` tables
(`true`/`false` vs `True`/`False`, JSON vs Python list quotes) — the
measured values are identical.

## Deeper tests (2026-11-01, `tests/deep.rs`)

Beyond the hand-picked `verify` cases, a fixed-seed (splitmix64,
reproducible, no external crate) suite in `tests/deep.rs`:

- **Prefill property, 2000 cases:** random schemas (random
  descriptions, nested parameters, examples, unicode), random system
  texts and budgets 1–10000. Every case asserts bytes used <= budget,
  the compression flag matches whether the uncompressed prefill fits,
  and caller schemas are unmutated. Tiny budgets 0–10 are also run
  for no-panic behaviour.
- **Loop detection, exhaustive + random:** all 254 binary sequences
  up to length 7 over two actions, plus 5000 random sequences up to
  length 20 over four actions, each checked step-by-step against an
  independently written streak/cycle oracle. A 200k-distinct-action
  soak asserts no false loop; reset restores the detector.
- **Finish gate, 1000 random cases:** random criterion sets and
  environments, asserting accepted-ness and the exact failure list,
  in order.
- **Combined invariants:** budgets 100–8000, repeat_k 2–3 firing
  exactly at the streak, plus the pinned interaction that with
  repeat_k=5 identical actions fire first as a 2-step cycle at step 4
  (same as Python: the streak has not reached 5 yet).
- **Threads:** 8 threads x 500 prefill builds + signatures give
  identical results to the sequential run.

Full run: 17/17 tests pass (9 unit + 8 deep, deep suite ~3.1 s,
dominated by the 200k soak). Random-case Python-vs-Rust parity sweep
run the same day, separate from the suite: 500 random loop signatures
and 200 random prefill builds, 0 mismatches.

## Known parity limits

- The Python last-resort truncation path behaves oddly for budgets
  0/1 (a negative slice wraps); this port saturates instead. Budgets
  >= 2, including all budgets used in practice, match exactly.
- The Python `FinishGate` criteria are arbitrary closures; the CLI
  only exposes file-content criteria. Library users get the full
  generic gate.

## Status

Local only, 2026-11-01. Not pushed to any repo — no push was requested,
and arxiv-lab's standing auto-upload covers that repo only. If you want
it as `lesterppo/arxiv-harness-rs`, say so and it gets the usual
secrets scan + verification before any push.
