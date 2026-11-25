# CAPE: Coordinating Calibration and Search for Tool-Use Policies

- **arXiv:** 2611.31938 — [CAPE: Coordinating Calibration and Search for Tool-Use Policies](https://arxiv.org/abs/2611.31938) (Wang, Chen, Yang, Wang; submitted 2026-11-24)
- **Scores:** AGENT 5 / TUNE 1
- **Verdict: SUPPORTED**
- **Date:** 2026-11-25 (daily run; this session was finished by a continuation agent after the scheduled worker died mid-run — see Run log note below)

## Why picked

A tool call can fail even when the operation behind it is correct, because
the receiving tool parses delimiters, dates or record layouts under a
configuration the agent cannot inspect. CAPE is a training-free compiler
for such tools: it keeps an executable policy at all times, directs
calibration to the calls that can improve it, and certifies optimality
when its executable and optimistic values meet. The mechanism is pure
deterministic logic (calibration + search over a shared budget) — directly
reusable on a local harness, a natural fit for the `arxiv-lab` mechanism
library (`src/arxiv_lab/agent/`). No model, no network, no GPU needed.

## What was built

- `src/arxiv_lab/agent/cape.py` (780 lines, stdlib only): the world
  model (configurations = parsing-aspect combinations; programs commit
  iff their encoding matches on every aspect, else an error naming the
  first mismatching aspect; probes reveal one aspect without
  committing), the CAPE compiler (witness from closed actions,
  optimistic bounds B / U_a / T2, three-tier acquisition frontier),
  an exact full-knowledge DP oracle for the independently computed
  optimum, and deterministic baselines (eager-exact-equivalent
  full-table acquisition, program-only optimal policy, error-feedback
  reviser proxy).
- `tests/test_cape_cpu.py` (also copied here as `test_cape_cpu.py`):
  deterministic CPU test, five experiments, ~5 min.
- `rust/` — Rust second stage (see below).

## Results (Python CPU test, `results.json`)

Paper headline numbers (abstract): optimum reached in 502/504
compilations; 22.0% less compilation time than eager exact search;
completion +15.38 to +34.66pp over the LLM error-feedback reviser.
Compilation time is measured here as calibration queries (one query =
one candidate-call evaluation): in this harness a query is a dict
lookup while CAPE's bound recomputation dominates wall-clock, so a
wall-clock comparison would measure the implementation, not the
paper's calibration-cost model.

| experiment | result |
|---|---|
| E1 main suite (12 configs, cap 3, budget 1000, seeds 0–7) | CAPE value == exact optimum **8/8**; certified **8/8**; R0 preserved 8/8; mean query saving **3.5%** (paper 22.0%); mean EF gain **+34.38pp** (paper 15.38–34.66pp) |
| E2 small worlds (8 configs, cap 3, budget 400, seeds 0–3) | optimal value **4/4**; certified **4/4**; query savings 10.8–24.3%; EF gains 25.0–62.5pp |
| E3 scale spot check (24 configs, cap 3, budget 2000, seed 0) | optimal value reached; certified **False**; queries 1223/1224; EF gain +4.17pp |
| E5 budget-bound, report only (12 configs, cap 3, budget 200) | optimal value **0/2** (9 vs 11/12 completions); queries exhausted at 200 |

Per-seed E1 (completions / calls, exact in brackets): seed 0: 11/32 [11/32],
1: 12/33 [12/33], 2: 11/33 [11/33], 3: 11/33 [11/33], 4: 11/32 [11/32],
5: 11/32 [11/32], 6: 10/32 [10/32], 7: 11/32 [11/32] — value-identical
to the exact optimum on every seed, certified on every seed.

## Rust second stage

Rust port under `rust/` (std-only, no crates). `cargo test --release`:
**12/12 passed** (unit, property sweep, exhaustive 2-aspect worlds,
500-world soak, thread identity, Python-vs-Rust parity sweep).

- `verify` binary (mirrors the Python experiments, Rust-native
  xorshift worlds, same shapes): E1 optimal **8/8**, certified **8/8**,
  mean query saving **5.8%**, mean EF gain **+42.71pp**; E2 optimal
  **4/4**, certified **4/4**; E3 optimal value reached, certified
  **false** — the same certification outcome as Python's E3. (The
  Rust EF-gain mean sits above the paper's 15.38–34.66pp range; the
  direction reproduces, the exact range does not bind at this scale.)
- Parity sweep: **444 cases, 0 mismatches** (value, queries,
  certified, completions, calls, optimum — all fields exact).
- Soak: optimal value in **449/500** worlds; the 51 misses all
  exhaust the 200-query budget uncertified.
- Rust is ~1.5× faster than Python on the same shapes (E1 ~11–22 s
  per world vs ~22–31 s; E3 78 s vs 109 s).

**Port bug found and fixed during parity (2026-11-25):** the first
parity run gave 89/444 mismatches. Root cause was a sort-order
difference, not a logic error: Python sorts configs by their aspect
*value strings* (`sorted(V)`), while Rust sorted by value *indices*.
The delim values `(",", ";", "\t")` are indices 0,1,2 but sort as
`"\t" < "," < ";"`, so the first acquisition target differed (parity
line 3: Rust `0010` vs Python `2101` at round 0) and the paths
diverged. Fixed by sorting configs by string values at every
order-sensitive site (`missing`, `partition`, `optimistic_cells`,
world sampling) — see `rust/README.md`. After the fix, the line-3
acquisition sequences are identical for all 204 rounds.
The soak threshold was also calibrated: Python itself measures
449/500 on these exact worlds (`soak_py.py` replicates the Rust soak
world-for-world), so the original 0.9 threshold was stricter than
CAPE itself; the test now asserts ≥ 0.89, which still catches the
pre-fix 430/500 regression.

## Honest limits

- **Query saving does not reproduce the paper's 22.0% at this scale.**
  E1 mean saving is 3.5%: CAPE acquires ~97% of the effect table
  before its witness meets the upper bound. (At 8 configs the saving
  is 10.8–24.3%, closer to the paper.) The paper's number is at
  64-request scale with the paper's budget; the mechanism (reach the
  optimum, certify) reproduces, the time saving does not at this size.
- **Certification fails at 24 configs** even when the optimum value is
  reached (E3: optimal, uncertified, 1223/1224 effects acquired — the
  run stops one effect short with no ranked candidate left). At 12
  configs and below, certification is 12/12 across E1+E2.
- **Budget-bound runs miss the optimum** (E5: 0/2 at budget 200). The
  paper's 502/504 rate is at 64-request scale with the paper's budget;
  at this bounded scale the rate is budget-dependent, and is only
  8/8 when the budget covers most of the table.
- **The error-feedback baseline is a deterministic proxy, not an LLM.**
  It revises one aspect per error round-robin from a default encoding.
  The +34.38pp mean gain is measured against that proxy; the paper's
  range is against a real LLM reviser.
- **Wall-clock is not compared**: CAPE's per-query bound recomputation
  makes it slower in wall-clock than eager exact in this harness
  (e.g. E3 took 109 s vs the oracle's ~0.1 s), because here queries
  are free dict lookups. The paper's 22% is a calibration-cost claim;
  queries are the faithful proxy for it.
- **Scale**: 8–24 configurations here vs the paper's 64 requests and
  larger CSV families. Single-threaded CPU test; no model-backed leg
  (the mechanism needs no model).

## Determinism fix (2026-11-25)

The module as found claimed "deterministic given a seed" but was not:
the same world compiled twice in different processes gave 596 vs 578
queries (value and certification were stable; only the acquisition
order varied). Root cause: iteration over `frozenset` belief states,
whose order depends on `PYTHONHASHSEED`. Fixed by sorting set
iterations at the three decision points (`EffectTable.missing`,
`_partition`, `_optimistic_cells`); verified identical output
(value, queries, certified) across `PYTHONHASHSEED` 1/2/3. All
numbers above are post-fix. The fix is documented in `_partition`'s
docstring. (The dead worker's module was otherwise faithful: the
exact oracle, bounds and acquisition match the paper's Sec. 4.)

## Run log note

The scheduled 2026-11-25 worker died after writing `cape.py` and
creating this (empty) session directory. A continuation agent
finished the run: verified the module, fixed the determinism bug
above, wrote and ran the CPU test, built the Rust second stage,
launched the InstructReplay training leg, and pushed both sessions
with blob-sha verification. Screening detail (fetch counts, covered
ids, near-misses) from the dead worker's report is recorded in the
run log, not re-derived here.
