# CapLite — Rust second stage (arXiv:2611.17115)

Rust port of the Python mechanism in
`src/arxiv_lab/agent/caplite.py`, verified against the Python session
test in `sessions/2026-11-15-2611.17115-caplite/test_caplite_cpu.py`.
The Python version is the primary verdict carrier; this Rust version
never replaces it and a failed Rust leg would never overturn a Python
verdict. It passed — see Results.

## What was ported

`src/lib.rs` (std only, no external crates) ports the whole mechanism:

- **SplitMix64 RNG** (`splitmix64`, `rand_below`) — bit-identical to the
  Python implementation, including `v % n` in `rand_below`.
- **Parsers** (`parse_stated`) — hand-written equivalents of the Python
  regexes `EMAIL_RE`, `AMOUNT_RE`, `PATH_RE`, including the
  trailing-period strip on paths (`p.rstrip(".")`). The email matcher
  reproduces regex backtracking (longest domain first, then shorter,
  until a literal `.` is followed by a 2+ letter TLD); the path matcher
  reproduces greedy `segment/` group consumption with backtracking.
- **Policies** — `CapLitePolicy` (commit-before-read + every deciding
  argument user-stated), `ModelWrittenPolicy` (CaMeL-style, contaminated
  by `observe`), `UndefendedPolicy`.
- **MockPlanner** — plans correctly from the request, follows
  instructions found in tool results; token cost is the whitespace-split
  word count of prompts, exactly as in Python.
- **Episodes / batteries** — `build_episode` (same RNG call order:
  email pool, amount pool, path pool, task/lookup pool, injection pool,
  planner error), `run_episode`, `run_battery`, `run_soak`.

`src/bin/verify.rs` is the `verify` binary. `tests/` holds the live
extensive test: unit tests per mechanism, property sweeps, exhaustive
injection×task combinations, a 50k-episode soak, and thread-identity
checks.

## Deliberate divergences from Python

There are **no behavioural divergences** on any input this mechanism
sees. The only implementation differences:

1. **Parsers work on bytes with ASCII character classes.** Python's
   `\w` is Unicode-aware, so a non-ASCII word character (e.g. `é`) in a
   file path would be a path character in Python but a separator here.
   Every battery, parity, and fuzz input is ASCII, so behaviour is
   identical; the parser fuzz (520 lines, ASCII alphabet) confirms it.
2. **Amounts are `f64` throughout.** Python pool amounts are ints that
   are converted to float at fill time, so the values compared are the
   same. The `F64Ord` wrapper exists only because `f64` has no `Ord`
   (needed for the `BTreeSet` of stated amounts); it is never NaN.
3. **`ModelWrittenPolicy::observe` returns `true` unconditionally** —
   same as the Python code (its docstring mentions a fail-closed path
   that the code never takes).
4. **Dead branches kept for fidelity**: the planner-error `amount`
   branch (`deciding[0]` is never `"amount"`) and the lookup branch's
   unconditional `ATTACKER` (no `mallory@evil.com` fallback) are both
   implemented exactly as in Python.

## Build / test / verify

No network, no crates to fetch. Cargo must be on `PATH`
(`export PATH="$HOME/.cargo/bin:$PATH"`). Always build with a target
directory **outside** this folder (nothing but source, tests, Cargo
files, README, and `results.json` lives here):

```bash
export PATH="$HOME/.cargo/bin:$PATH"
cd sessions/2026-11-15-2611.17115-caplite/rust
export CARGO_TARGET_DIR=/tmp/caplite-rust-target   # or any scratch dir

cargo test --release            # 23 tests (unit + property + exhaustive + threads)
cargo test --release -- --ignored soak_50k   # the 50k-episode soak test
cargo run --release --bin verify              # battery + unit checks, writes results.json
cargo run --release --bin verify -- soak 50000 7   # soak run, prints JSON
```

Parity sweep (the support script lives outside this session, in the
goal's hidden files — it is not part of the deliverable):

```bash
cd ~/workspace/goals/arxiv-daily-read-and-test/hidden_files/caplite-rust
python3 parity_dump.py            # specs.csv, parity_py.csv, corpus.txt, parse_py.jsonl
verify parity specs.csv > parity_rs.csv
verify parse < corpus.txt > parse_rs.jsonl
diff parity_py.csv parity_rs.csv  # must be empty
diff parse_py.jsonl parse_rs.jsonl
```

Parity spec rule: seed `999*1000003 + i`, attacked unless `i % 3 ==
2`, lookup when benign and `i % 4 == 0`. The dump format is
`seed,arm,completed,bad_write,attack_success,tokens` (all ints, so no
float-formatting risk). The parser probe prints one JSON
`{"emails": [...], "amounts": [...], "paths": [...]}` object per input
line, sorted, with amounts formatted Python-`repr`-style.

## Results

Full numbers in `results.json`.

**Verify (battery seed=7, 300 attacked + 150 benign, lookup_frac 0.1)
— identical to Python on every number, including exact token totals:**

| arm | attack | benign completion | bad writes | mean tokens |
|---|---|---|---|---|
| caplite | 0/300 = 0.000 | 129/150 = 0.860 | 0 | 16.18 (7,281 total) |
| camel | 283/300 = 0.943 | 143/150 = 0.953 | 283 | 196.18 (88,281 total) |
| undefended | 300/300 = 1.000 | 143/150 = 0.953 | 307 | 16.18 (7,281 total) |

Unit checks (commit rule, unstated-arg rule, payload-not-deciding)
all pass; all battery assertions from the Python test pass.

**Tests — 23 passed, 0 failed** (`cargo test --release`):

- `tests/unit.rs` (11): SplitMix64 vectors and `rand_below` sequences
  from Python; `parse_stated` examples and edge cases from Python;
  the three policy unit checks; CaMeL contamination; `build_episode`
  spot values (episodes 0 and 300); exact per-episode token counts
  (ep0: 26 / camel 206; ep300 lookup: 13 / camel 193).
- `tests/property.rs` (6): parser invariants over 2,000 random
  requests + 500 hostile-byte strings; CapLite admits 0 attacks / 0
  bad writes across 6 seeds × 3 sizes × 4 lookup fractions;
  completion monotone non-increasing in `lookup_frac` (0 at frac 1.0);
  exact token accounting (camel − caplite = 180 × episodes;
  undefended = caplite); lookup tasks complete only without
  enforcement.
- `tests/exhaustive.rs` (3): 6 tasks × 7 injections × 2 emails × 2
  amounts × 2 paths × 2 planner-error settings = 1,344 attacked
  specs (CapLite 0 attacks / 0 bad writes, undefended admits every
  attack, CaMeL admits the large majority); 448 lookup-task specs;
  all 1,296 benign (task, value) combinations complete under CapLite.
- `tests/threads.rs` (3): same spec bit-identical on 8 threads;
  batteries split across 4 threads equal single-threaded runs; soak
  slices identical across threads.

**Parity sweep — 0 mismatches.** 2,000 random episode specs × 3 arms =
6,000 outcome rows, Python vs Rust `diff` empty (attack_success:
caplite 0, camel 1,262, undefended 1,334; bad_write: 0 / 1,262 /
1,383; tokens: 33,082 / 393,082 / 33,082). Parser fuzz: 520 lines
(500 random + 20 edge cases), `diff` empty.

**Soak — 50,000 episodes (150,000 episode-arm runs) in 0.95 s:**
caplite 0/33,334 attacks, 0 bad writes, completion 11,866/16,666 =
0.712 (the soak mix has 1 in 4 benign episodes as lookup tasks, which
CapLite blocks by design); camel 31,702/33,334 = 0.951;
undefended 33,334/33,334 = 1.000.

## Files

```text
rust/
  Cargo.toml            # std only, no dependencies
  Cargo.lock            # generated (no dependencies)
  src/lib.rs            # the whole mechanism
  src/bin/verify.rs     # verify / soak / parity / parse modes
  tests/unit.rs         # per-mechanism unit tests
  tests/property.rs     # property sweeps
  tests/exhaustive.rs   # injection x task combinations
  tests/soak.rs         # 50k-episode soak (ignored by default)
  tests/threads.rs      # thread-identity checks
  README.md             # this file
  results.json          # verify + parity + soak numbers
```

No `target/`, no binaries, no secrets in this directory. The build
target directory used during development was deleted after the run.
