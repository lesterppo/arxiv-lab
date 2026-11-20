# grepmem-rs

Rust port of the GrepMem mechanism from `lesterppo/arxiv-lab`
(arXiv:2611.25123 — "Forget Freely, Grep Later"): an append-only
plain-text log plus regex grep recall, compared against lossy memory
policies (summarization, recency truncation, random masking) with a
perfect keyword QA oracle.

Python stays the primary verdict carrier; this port cross-checks it.
A failed Rust leg would never overturn the Python verdict.

## What is ported

| Python (`arxiv_lab/agent/grepmem.py`) | Rust (`grepmem_rs`) |
|---|---|
| `random.Random` (MT19937) | `PyRandom` — faithful re-implementation, byte-identical streams |
| `AppendOnlyLog` (+ `grep`, `max_lines` cap) | `AppendOnlyLog::grep` (same semantics) |
| `generate_session` (entities, attributes, facts, distractors, questions) | `generate_session` (same draw order, same output) |
| `policy_full / policy_grepmem / policy_summary / policy_recency / policy_random_mask` | same functions, same defaults |
| `answer_question` (anchored oracle, latest wins) | `answer_question` (same anchor regex) |
| `evaluate_policy`, `run_suite` | `evaluate_policy`, `run_suite` (same aggregates) |
| `tests/test_grepmem_cpu.py` (5 experiments) | `grepmem verify` (same experiments + assertions) |

## Build and test

```bash
export PATH="$HOME/.cargo/bin:$PATH"
cd sessions/2026-11-20-2611.25123-grepmem/rust
# CARGO_TARGET_DIR keeps target/ out of the session dir.
CARGO_TARGET_DIR=/tmp/grepmem-rs-target cargo test --release --offline
# 11 unit tests + 5 deep tests
CARGO_TARGET_DIR=/tmp/grepmem-rs-target cargo build --release --offline
/tmp/grepmem-rs-target/release/grepmem verify   # live-test verification
python3 tests/parity_sweep.py                   # Python-vs-Rust parity
```

One external dependency: the `regex` crate (1.13.1, from the local
registry cache — the build works `--offline`). Everything else is std.

## CLI

```bash
# Live-test verification (mirrors the Python CPU test's 5 experiments
# and assertions, plus exact-equality checks against ../results.json)
grepmem verify
# -> ALL CHECKS PASSED + RESULTS_JSON {...}

# Parity helpers (machine-readable, for tests/parity_gen.py)
grepmem gen <seed> <n_lines> <n_questions>   # session file on stdout
grepmem rng <seed> <n>                       # first n random() + u32s, JSON
grepmem eval --session <file> --policy <name>
             [--max-lines N] [--entity-only] [--attr-only]
# -> eval JSON: aggregates + per-question answers and context lines
```

### Session file format (line-based, v1)

```text
# grepmem-session-v1 seed=7 n_lines=500 n_questions=20
L [t0000] OBS tool:web_search | Project Atlas budget is $1.2M
Q Project Atlas\tbudget\t$1.2M\t123\t0\tWhat is the budget of Project Atlas?
```

`L` lines are log lines verbatim (they contain no tabs by
construction). `Q` fields are tab-separated: entity, attr, answer,
fact step (int), paraphrased (0/1), question text. The position is
derived on read as `step / (n_lines - 1)`, so no float formatting
ever enters the file. Both sides read and write this format.

## Live-test verification (2026-11-20)

`verify` **ALL CHECKS PASSED** in ~1.2 s, with every number
**identical to the Python `../results.json`** (all five experiment
blocks compare equal as parsed JSON):

- main suite (5 seeds x 20 Qs, 500 lines): full acc **1.000** (ctx
  500.0), grepmem **1.000** (ctx 14.3, para 1.000), summary **0.200**,
  recency **0.250**, random_mask **0.400**; grepmem early-position
  1.000 vs recency 0.121.
- scaling 200 -> 2000 lines: grepmem stays **1.000**; recency falls
  0.410 -> 0.220.
- grep budget: max_lines 5 -> **0.610**, 10 -> **0.880**,
  20 -> **0.990**, 40 -> **1.000**.
- vocabulary mismatch: entity+attr **1.000**, entity_only **1.000**,
  attr_only **0.860** (para_acc 0.000 — the known failure mode).
- long-stream recall (1 fact, 1000 lines): full **1.000**, grepmem
  **1.000**, recency **0.400**, summary **0.200**.

`cargo test --release` **16 passed, 0 failed** (11 unit + 5 deep).
Parity sweep **53/53 cases, 0 mismatches**: 4 rng cases, 9 gen cases
(byte-identical session files), 40 eval cases (exact context lines,
answers, and counts for all 5 policies across 5 sessions).

## Soak (scale Python does not reach)

20,000-line stream x 200 questions, seed 7, single session
(`soak_results.json`, total 3.35 s):

| policy | accuracy | mean ctx lines | paraphrase acc |
|---|---|---|---|
| full | 1.0000 | 20000.0 | 1.0000 |
| grepmem | **0.9850** | 40.0 | 0.9821 |
| summary | 0.4700 | 824.0 | 0.4286 |
| recency | 0.1700 | 60.0 | 0.1250 |
| random_mask | 0.4450 | 3971.0 | 0.4643 |

The 3 grepmem misses are genuine cap behaviour at scale, not a port
difference: with ~1,666 facts over 60 pairs, more than 40 later lines
can match the same entity/attr tokens and push the latest fact out of
the capped window (mean ctx saturates at exactly 40.0). The soak
session is deterministic (regeneration is byte-identical), and a
4-thread parallel evaluation gives results identical to serial
(`thread_identity_parallel_equals_serial`).

## Deliberate divergences from Python

None in the mechanism — the port is behaviorally identical, including
the random streams. Presentational / structural differences only:

- `PyRandom` reproduces CPython's `random.Random` exactly (MT19937
  with `init_by_array` seeding from the little-endian 32-bit words of
  the integer seed; `random()`, `getrandbits`, `_randbelow`, `choice`,
  `shuffle`, `sample` with CPython's pool/selection-set branches and
  threshold). Verified against CPython 3.12 values in `tests/unit.rs`
  (`random()` for seeds 0–5, `getrandbits` for k = 3, 6, 32, 33, 40,
  64, `choice`/`sample`/`shuffle` streams). `getrandbits` for k > 32
  uses the top-bits-of-concatenated-words model, also verified for
  k = 33, 40, 64; the mechanism never needs k > 15.
- `policy_grepmem` is implemented as a single pass over the log
  (line matches if any pattern matches, de-duplicated by content)
  instead of one grep per pattern plus a sort by log index. The
  result is provably the same set in the same order, and the parity
  sweep confirms it on every case (exact context lines).
- `evaluate_policy_detailed` exposes per-question context lines and
  answers for the parity sweep; Python computes the same values
  internally but does not expose them.
- The line-based session file format and the `gen`/`eval`/`rng`
  subcommands are new artifacts for parity; they are not Python
  features.
- Floats are printed with Rust's shortest round-trip representation
  (integral floats as `1.0`, Python-style); parsed values are
  identical.
- Regex engine: the `regex` crate instead of Python `re`. Semantics
  agree for every pattern used here (literals, `\b`, alternation,
  groups, `(.+)$`, case-insensitive flag). One known non-difference:
  Python `$` also matches before a trailing newline while Rust's
  `$` matches only at end of text — log lines never carry a trailing
  newline, so it cannot fire.

## What the Rust leg does NOT claim

The Rust port cross-checks the Python implementation; it does not
re-validate the paper. The verdict carrier is the Python track, and
the synthetic-session limitation (no access to the paper's
ContextBench / BCP-Multi runs) applies identically here.
