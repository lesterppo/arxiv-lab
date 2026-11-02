# vbl-rs — Rust second stage for VBL (arXiv:2610.41564)

Rust port of `arxiv_lab.agent.vbl` from `lesterppo/arxiv-lab`, built
only after the Python CPU test passed (2026-11-02). The Python version
stays the primary verdict carrier; this port never replaces it.

## What is ported

| Python (`arxiv_lab/agent/vbl.py`) | Rust (`vbl_rs`) |
|---|---|
| `GoalSpec`, `GoalState` | `GoalSpec`, internal `GoalState` |
| `VBL` (verify / diagnose / Push / Pop / BacktrackTo / run) | `Vbl` (same methods, same trace format) |
| `react_baseline` | `react_baseline` |

Std only — no external crates, so `cargo test --offline` works with
no network. `stack_push_for_test` is a test-support-only API (stack a
goal directly to simulate a stray activation); it has no Python
counterpart and is never used by the controller itself.

## Build and test

```bash
cargo test --offline        # 5 unit + 4 deep tests
cargo build --release --offline
./target/debug/vbl verify   # live-test verification, see below
./target/debug/vbl parity   # case enumeration for the parity sweep
```

## Live-test verification (2026-11-02)

`vbl verify` re-runs the Python test's headline experiment in Rust —
same scripted report task, same failure injection. Results, Rust vs
Python:

| Experiment | Rust | Python |
|---|---|---|
| VBL success, steps | true, 9 | True, 9 |
| VBL operation sequence | Push:gather, Pop:gather, Push:analyze, Push:regather, Pop:regather, Pop:analyze, Push:write, Pop:write, Pop:report | identical |
| ReAct baseline | false, 60 steps | False, 60 steps |

## Deeper tests (`tests/deep.rs`)

- **Property sweep, 2000 random cases:** inject level 0–4, retry
  limit 0–3, stack bound 1–10, initial state varied. Invariants hold
  throughout: steps ≤ 60, success ⟺ report_done, and every
  no-failure case with stack bound ≥ 3 succeeds.
- **Exhaustive retry sequences vs an independent oracle:** every
  outcome sequence up to length 4 over {ok, generic failure,
  decomposition failure} (121 sequences) × retry limits 0–2, compared
  on step counts and final trace entry.
- **Soak, 20,000 cases:** every inject level succeeds within budget
  (each injected failure is repaired by exactly one regather cycle);
  20000/20000 successes.
- **Thread identity:** 8 threads × 200 cases give results identical
  to the sequential run.

Full run: 9/9 tests pass (5 unit + 4 deep).

## Python-vs-Rust parity sweep

`parity_vbl.py` (session root) enumerates the same case space as
`vbl parity`: inject level 0–3 × retry limit 0–2 × stack bound
{2, 3, 8} = 36 cases, printing `success steps op:goal,...` per case.
Result 2026-11-02: **36 cases, 0 mismatches** (compared with `diff`;
the compared field is the operation + goal sequence and the step
count). Trace detail strings were mirrored from Python but are not
byte-compared by the sweep — only the operation + goal sequence is.

## Known shared behaviours (not divergences)

- A lone leaf terminal never sets the controller's `success` flag:
  only a composite terminal completing sets it. Python and Rust
  share this quirk (documented in the deep test's oracle).
- In the decompose branch, a Push that fails on a full stack is
  ignored by the loop, so a goal whose decomposition cannot be pushed
  spins to `max_steps`. Python and Rust share this behaviour; the
  parity enumeration (stack bounds 2, 3, 8) does not hit it, because
  a full stack at those bounds is resolved by BacktrackTo instead.

## Status

Built and verified live 2026-11-02. Uploaded under this session's
`rust/` subsession (source, tests, Cargo files, README, results
only — no `target/`, no binary, no secrets).
