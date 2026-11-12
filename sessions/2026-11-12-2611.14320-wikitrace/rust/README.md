# Rust second stage — WikiTrace (arXiv:2611.14320)

Rust port of the Python mechanism in `../wikitrace.py`, built under the
standing Rust-second-stage rule (2026-11-02). The Python version stays
the primary verdict carrier; this port never replaces it.

## What is ported

`src/lib.rs` is a line-for-line port of the pure logic, no external crates:

- `SplitMix64` — identical arithmetic, so RNG streams match Python exactly.
- `gold_action`, `gen_tasks`, `default_choices` — paired default choices
  (same task -> same choices across arms).
- `run_default` — raw trajectories with per-step chosen/ok/observed.
- `compile_memory` — compilations + up to three insights per run, with
  the same RNG consumption order (one `SplitMix64(seed+11)` stream across
  runs in order; per-step corruption check, then scope draw), including
  the two honest error sources: over-generalization (scope `"any"`) and
  misattribution on failed runs (`CORRUPT_P = 0.20`).
- `Memory` with source links (insight -> compilation -> trajectory) and
  the `subset` (successful-runs-only / failed-runs-only) constructor.
- `descent_triggers` — the paper's three triggers (missing detail /
  conflict / failed-run origin).
- `verify_lesson` — the four outcomes SUPPORTED / CONTRADICTED /
  UNSUPPORTED / UNKNOWN, 2 evidence reads per check.
- `solve` — the three arms (none / plain / wikitrace), including the
  stop-at-first-SUPPORTED descent loop and the corrections accounting.
- `run_experiment` — all metrics: per-arm success / gain / regression /
  regression-of-solved, descent rate, correction rate, evidence reads
  per task, lessons followed per task, and the succ_only / fail_only
  subsets.

## Deliberate divergences from Python

1. `randbelow` compares in `u128`: for `n == 1` the rejection limit is
   2^64, which wraps to 0 as `u64` and would loop forever (a `choice()`
   on a 1-element slice calls `randbelow(1)`). Python computes the same
   limit in bigint. (Same trap as the 2026-11-11 port.)
2. `f64` conversion of a `u64` uses the same round-to-nearest-even rule in
   both languages, so `SplitMix64::random` streams match exactly.
3. Python `Memory` stores compilations/trajectories as dicts keyed by run
   id; Rust uses `HashMap<u64, ...>` — iteration order over those maps is
   never observable (lookups are by key only), so results are
   order-independent. Insights keep insertion order in a `Vec`, as in
   Python.
4. Python `Scope` is `"any"` or `("attr", a)`; Rust models it as an enum
   (`Scope::Any` / `Scope::Attr(u64)`) — same semantics.
5. Python `solve` takes `mode` as a string (`"none"`/`"plain"`/
   `"wikitrace"`); Rust takes a `Mode` enum — same three arms.
6. `verify_lesson` does not re-check that a narrow lesson's scope matches
   the task attribute — in Python or here. The soundness claim ("a
   SUPPORTED lesson always prescribes the gold action") holds for lessons
   *applicable* to the task, which is how `solve` calls it (via the
   `applicable` filter); a narrow lesson's scope is set to its source
   run's attribute at compile time, so an applicable narrow lesson always
   has run attr == task attr. Calling `verify_lesson` directly on a
   non-applicable narrow lesson can return SUPPORTED for a non-gold
   claim in both languages (verified against Python: insight I-0023,
   seed 0, returns `('supported', 2)` for attr=1 while gold is 1 for
   step 2 — the lesson's scope is `('attr', 0)`). The soundness deep test
   therefore filters by `applies_to`, matching the contract `solve`
   relies on.

## Live extensive test (all run live)

```
cargo test --release
```

- **25 tests, all passing:** 15 unit (SplitMix64 vectors from Python,
  randbelow edge cases incl. n==1 and n==2^63, gold_action table, task
  determinism/shape, paired default choices, trajectory recording,
  compiler counts 36 insights / 4 corrupted / 24 from failed runs,
  subsets, verify_lesson four outcomes, descent triggers, solve arms,
  exact main-experiment numbers), 9 deep, 1 parity sweep.
- **Deep tests:** property sweeps over 200 seeds (WikiTrace regression
  == 0 on every seed; WT success >= no-memory on every seed; plain
  regresses on at least one seed while WikiTrace regresses on none),
  varying train size (0–40) and test size (1–200) incl. the degenerate
  empty-memory case; soundness property (every SUPPORTED verdict on an
  applicable lesson implies claimed == gold) over 100 seeds; exhaustive
  gold_action oracle (step types x attrs 0–255); exhaustive
  verify_lesson oracle (every step x claimed x scope x task attr);
  5,000-task soak (8x the Python test); 200-seed sweep counts
  (WT > plain on 197/200 seeds, WT never below no-memory); thread
  identity (same 200-task batch on 4 threads is byte-identical to
  single-threaded).
- **`verify` binary** (`cargo run --release --bin verify`): mirrors the
  Python CPU test's experiments and assertions A1–A8. It reproduces the
  Python numbers **exactly** — `results.json` is **byte-identical** to
  the Python `results.json`: No-Memory 0.2, Insights-Only 0.505,
  WikiTrace 0.7033333333333334, gains 0.5033333333333333 (WT) /
  0.4066666666666667 (plain), regressions 0.0 (WT) / 0.1016666666666667
  (plain), descent_rate 1.0, correction_rate 0.2638888888888889,
  evidence_reads_per_task 23.826666666666668, WT succ_only
  0.5833333333333334, WT fail_only 0.3566666666666667. All A1–A8 pass.
  Writes `results.json`.
- **Parity sweep:** 3,200 cases (5 seeds × SplitMix64 streams,
  randbelow, random; 20 tasks + choices + runs + 60 insights each;
  verify_lesson for every insight × 4 attributes; solve outcomes for
  all three arms + succ_only/fail_only subsets on 40 test tasks each)
  replayed from `tests/parity_reference.txt` (generated by the frozen
  Python module) — **0 mismatches**.

## Soak / scale notes

- 5,000-task soak (50 train runs, 150 insights): none 0.2124, plain
  0.4974, WikiTrace 1.0, descent 1.0, correction 0.1675,
  reads/task 28.8608 — with a richer memory every step×attr has a
  supportable lesson, so WikiTrace solves everything; the same numbers
  come out of the Python module at this scale (cross-checked live).
  WikiTrace still never regresses.
- 200-seed sweep (n_train=12, n_test=600): WT > plain on 197/200 seeds,
  WT never below no-memory. Plain > no-memory on only 74/200 seeds at
  this small train size — blind-following a small, partly-corrupted
  memory often hurts, which is exactly the paper's memory-induced
  degradation; the main-seed A3 (plain 0.505 > none 0.2) is a
  favorable draw, not a theorem. WikiTrace's zero regression, by
  contrast, is a theorem and holds on all 200 seeds.

## How to run

```bash
cargo test --release        # unit + deep + parity
cargo run --release --bin verify   # main experiment, writes results.json
```

No `target/`, no binaries, no secrets are committed from this directory.
