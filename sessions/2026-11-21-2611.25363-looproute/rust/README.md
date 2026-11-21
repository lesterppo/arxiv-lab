# looproute-rs

Rust port of the LoopRoute mechanism from `lesterppo/arxiv-lab`
(arXiv:2611.25363 — "Stop Going in Circles: Loop-Triggered Routing
for System 1 Decision Agents", Yu et al., 2026): a fast decision
policy acts by default; recurring (state, action) pairs signal
stalled execution and trigger a slow assistant (here a BFS oracle
planner), then control returns to the fast policy. No training, no
confidence calibration.

Python stays the primary verdict carrier; this port cross-checks it.
A failed Rust leg would never overturn the Python verdict.

## What is ported

| Python (`arxiv_lab/agent/looproute.py`) | Rust (`looproute_rs`) |
|---|---|
| `random.Random` (MT19937) | `PyRandom` — faithful re-implementation, same seeded tasks |
| `KitchenWorld` (rooms/graph/items/locked chest/key) | `KitchenWorld` (bitmask state, same transitions + observations) |
| `bfs_path` (room shortest path) | `bfs_path` (same tie-breaking) |
| `FastPolicy` (greedy + open-chest-without-key flaw) | `FastPolicy` (same flaw) |
| `LoopRouter` (threshold counting, assist, count reset) | `LoopRouter` |
| `plan_to_goal` (BFS oracle planner) | `plan_to_goal` |
| `run_episode`, `run_experiment` | `run_episode`, `run_experiment` |
| `tests/test_looproute_cpu.py` (5 experiments) | `looproute verify` (same experiments + assertions) |
| `describe()` (LLM prompt text) | `describe()` (ported for fidelity; unused by the oracle leg) |

OUT OF SCOPE (not ported): the NVIDIA NIM model-backed leg — it needs
network access and a real LLM, so it stays Python-only. The oracle
assistant (`plan_to_goal`) is the deterministic stand-in used here.

## Build and test

```bash
export PATH="$HOME/.cargo/bin:$PATH"
cd sessions/2026-11-21-2611.25363-looproute/rust
# CARGO_TARGET_DIR keeps target/ out of the session dir.
CARGO_TARGET_DIR=/tmp/looproute-rs-target cargo test --release --offline
# 16 unit tests + 7 deep tests
CARGO_TARGET_DIR=/tmp/looproute-rs-target cargo build --release --offline
/tmp/looproute-rs-target/release/looproute verify   # live-test verification
python3 tests/parity_sweep.py                       # Python-vs-Rust parity
/tmp/looproute-rs-target/release/looproute soak     # 600-seed scale run
```

Zero external dependencies — stdlib only, so the build works
`--offline` with no crate downloads. (`Cargo.lock` exists but lists
no dependencies.)

## CLI

```bash
# Live-test verification (mirrors the Python CPU test's 5 experiments
# and assertions, plus exact-equality checks against ../results.json)
looproute verify [--expected ../results.json]
# -> ALL CHECKS PASSED + RESULTS_JSON {...}

# Task-spec helpers (machine-readable, for tests/parity_sweep.py)
looproute gen-spec <seed> <locked 0|1>   # canonical spec text on stdout
looproute run-spec <file> [--threshold K] [--mode fast|routed]
# -> episode JSON: success, steps, assists, llm_calls, full trajectory
#    of [state, action, source, observation]

# Scale run: 600 seeded tasks + a 200-step cap check
looproute soak
```

### Task-spec format (line-based, v1)

```text
# looproute-spec-v1
locked=1
recipe basil onion chili
place basil hallway
place onion bedroom
chest chili
key cellar
```

`place` lines are per non-chest ingredient in recipe order; `chest
<ing>` marks the ingredient inside the locked chest (kitchen);
`key <room>` is the key room. Both sides read and write this format,
so a spec is the parity unit: the same spec fed to both
implementations must produce the same episode, step for step.

## Live-test verification (2026-11-21)

`verify` **ALL CHECKS PASSED** in ~0.01 s, with every number
**identical to the Python `../results.json`** (all five experiment
blocks compare equal as parsed JSON, checked by the binary itself
against `--expected ../results.json`):

- E1 router units: trigger on the 2nd occurrence (K=2), reset after
  assist, re-trigger on recurrence; K=3 silent on the 2nd, triggers
  on the 3rd.
- E2 oracle plans: 10/10 legal and complete (plan lengths
  12, 10, 11, 10, 12, 9, 11, 10, 11, 9 for seeds 2000–2009).
- E3 false triggers: **0 assists** on 20 unlocked tasks, 20/20 solved.
- E4 main experiment (60 seeds): fast-only success **0.500**,
  mean steps 35.03; LoopRoute-oracle K=2 success **1.000**, mean steps
  12.2, mean assists **0.50**, mean LLM calls **0.50** (gain +50pp);
  K=3 success **1.000**, mean steps 12.7, mean assists 0.50.
- E5 determinism: repeated run identical.

`cargo test --release` **23 passed, 0 failed** (16 unit + 7 deep).

## Deep tests (scale Python does not reach)

- Property sweep (120 random seeded tasks x K=2,3,4): routed
  **120/120** every threshold, fast-only 60/120; per-seed
  monotonicity holds (routing never breaks a task the fast policy
  already solves); mean LLM calls 0.50, max 1 per task.
- No false triggers: 100 unlocked tasks, **0 assists**.
- Recipe sizes n=1..5, locked and unlocked, K=2: all routed runs solve.
- Exhaustive oracle check: the full reachable state graph (96 states
  each) for one locked and one unlocked task — every state's BFS plan
  is legal, ends in `cook`, and executes in exactly `len(plan)` steps.
- Random walks: 200 walks, **184** non-terminal states checked
  against the oracle the same way.
- Soak (600 seeds 5000..5600, `soak_results.json`, 27 ms): fast-only
  success 0.500, mean steps 35.11; LoopRoute-oracle K=2 success
  **1.000**, mean steps 12.32, mean assists 0.50, mean LLM calls
  0.50, max 1 assist / 1 call per task; a 200-step cap check solves
  a locked task in 15 steps with 1 assist.
- Thread identity: 4 threads over 60-seed shards reproduce the serial
  `run_experiment` aggregates exactly. The router and env are
  per-episode state — nothing is shared between episodes, so parallel
  evaluation is safe.

## Parity sweep: 560 cases, 0 mismatches

`python3 tests/parity_sweep.py` (1.8 s):

- 80 gen-spec cases (seeds 5000..5039, locked + unlocked): Rust
  `gen-spec` output is byte-identical to the spec text derived from
  the Python env — the seeded task generators agree exactly.
- 480 episode cases: every spec (including recipe sizes n=1,2,4,5)
  run fast-only, routed K=2, and routed K=3 on both sides; full
  trajectories compared step for step (state, action, source,
  observation). **0 mismatches.**

Note on a harness bug the sweep caught: the first sweep run showed
mismatches on routed locked tasks — Python's `plan_to_goal` rebuilds
its search probe as `KitchenWorld(env.seed, ...)`, and the harness
had built the Python env from a spec with `seed = -1`, so the Python
oracle planned for a *different* task (its plan contained failing
takes) while the Rust oracle planned correctly. The fix is in the
harness only (`env_from_spec` now carries the seed through); the
port itself was correct. This also documents a Python-side coupling
worth knowing: `plan_to_goal` depends on `env.seed`, not just on the
task fields.

## Deliberate divergences from Python

- `PyRandom` reproduces CPython's `random.Random` exactly (MT19937
  with `init_by_array` seeding from the little-endian 32-bit words of
  the integer seed; `random()`, `getrandbits`, `_randbelow`, `choice`,
  `shuffle`, `sample` with CPython's pool/selection-set branches).
  Adapted from the grepmem Rust leg in the same repo. Verified against
  CPython 3.12 values in `tests/unit.rs` (`random()` for seeds 0–5,
  `sample` for seed 1000) and by the byte-identical gen-spec parity.
  `getrandbits` for k > 32 uses the top-bits-of-concatenated-words
  model; the mechanism never needs k > 15.
- `plan_to_goal` builds its search probe with `clone_task()` (copy
  the task fields) instead of Python's `KitchenWorld(env.seed, ...)`.
  Same task either way for seeded envs, and it also works for
  from-spec envs, where no seed exists. Verified by the parity sweep
  (identical trajectories on every spec).
- Env state is a bitmask struct (`EnvState`: room index, ingredient
  bitmasks for inventory/collected/taken, key bit, chest/key bools)
  instead of Python's tuple of frozensets; `EnvState::canonical()`
  renders it in the same form the Python side renders the tuple, for
  the parity sweep. `taken` is always equal to `collected` in
  practice (both are set together on a take); both are kept because
  the state tuple carries both.
- `run_episode`'s empty-plan assist branch is restructured with a
  flag instead of Python's `continue` inside the branch; the control
  flow is identical (note-then-step-then-continue, no double note),
  verified by the parity sweep.
- A *legal* action can still fail (`open chest` without the key is
  offered by `legal_actions` but `step` rejects it, state unchanged)
  — same as Python; the deep tests assert the no-op behaviour rather
  than assuming legal implies ok.
- The line-based spec format and the `gen-spec` / `run-spec` /
  `soak` subcommands are new artifacts for parity; they are not
  Python features.
- Floats are printed with a Python-`repr`-compatible format
  (integral floats as `1.0`, `0.5`, `35.03333333333333`); parsed
  values are identical.

## What the Rust leg does NOT claim

The Rust port cross-checks the Python implementation; it does not
re-validate the paper. The verdict carrier is the Python track, and
the synthetic-task limitation (KitchenWorld instead of the paper's
TextWorld cooking tasks, oracle planner instead of an LLM assistant)
applies identically here.
