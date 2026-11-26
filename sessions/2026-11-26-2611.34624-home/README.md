# HoMe: Toward Practical Harness Optimization of Memory Systems

- **Article:** arXiv:2611.34624 — "HoMe: Toward Practical Harness Optimization of Memory Systems for AI Assistants" (Agentscope-AI / ReMe)
- **Link:** https://arxiv.org/abs/2611.34624
- **Date:** 2026-11-26
- **Track:** agent (CPU, deterministic, stdlib only)
- **Verdict: SUPPORTED**

## Why picked

AGENT 5 / TUNE 2. The paper's mechanism is exactly the local
harness-optimization problem for the arxiv-lab / Hermes memory stacks:
a memory system's behaviour is governed by its harness, and HoMe
optimizes that harness by an evaluation-driven loop — session-level
memory versioning + execution traces identify shared failure
mechanisms and localize them to the responsible session and harness
component, candidate gating + smoke tests cut evaluation cost, and
constructed memory is reused across candidates. With six formal
candidate evaluations per benchmark after a full initial evaluation,
HoMe found harnesses beating the originally strong harness while
using far fewer query-time tokens. The whole mechanism is
deterministic harness logic — no GPU, no model weights needed for the
core claim — so it is testable on CPU at bounded scale.

## What was built

- `src/arxiv_lab/agent/home.py` — the mechanism, graduated (see below).
  World: scripted 4-session histories with planted facts, revisions
  (a later session supersedes an earlier value), distractors, and
  synonym queries; 75% current-value queries, 25% historical
  (as-of-session) queries. Harness: top_k, recency weight, revision
  policy (all-versions vs latest-only), fuzzy attribute matching,
  answer rule (top1 vs majority). Query-time tokens =
  20 + 12·top_k (+10 if fuzzy). Traces classify every failure as
  STALE (answered a superseded value) or MISSING (nothing retrieved);
  the dominant shared cause localizes to a component (maintenance
  for STALE, retrieval/construction for MISSING) and yields
  prioritized candidate edits. The loop: full initial evaluation,
  then up to 6 formal candidate evaluations, each smoke-tested first
  on a stratified 8-case set (adaptive sampling: round-robin over
  the historical × synonym buckets, so the gate can see
  history-breaking candidates), then a cost phase that spends
  remaining budget on cheaper harnesses preserving accuracy.
  Memory is built once per revision policy and reused. An exhaustive
  oracle enumerates all 256 harnesses.
- `tests/test_home_cpu.py` — the CPU test (see Results).
- `rust/` — the Rust second stage (see `rust/README.md`).

## Results (Python CPU test, `tests/test_home_cpu.py`, ALL CHECKS PASSED)

| experiment | result |
|---|---|
| E1 loop vs start (seed 0, 48 cases) | accuracy **0.125 → 1.000**, tokens **56 → 42** |
| E1 vs exhaustive oracle | loop reaches oracle accuracy **1.000** and oracle token cost **42** |
| E1 budget | **3 formal evals** (of 6), 3 smoke evals, 0 rejects, 1 memory construction |
| E2 localized vs random edits (12 seeds) | gain per formal eval **0.2270 vs 0.0949** |
| E3a smoke gate vs trap candidate | trap smoke **0.375 vs 0.500** incumbent; formal **0.542 vs 0.646** — gate rejects a genuinely worse candidate |
| E3b gating cost (12 seeds) | gated formal mean **3.92 = ungated 3.92**, gated acc mean **1.000** — the gate costs nothing |
| E5 determinism | identical histories across runs and `PYTHONHASHSEED` 1/2/3 |

Start harness (top_k=3, recency 0, all-versions, fuzzy off,
majority): 25 STALE + 17 MISSING failures out of 48. Evolved
harness (top_k=1, recency 1.0, all-versions, fuzzy on, majority):
1.000 accuracy at 42 tokens — the oracle optimum on accuracy and
token cost. The risky latest-only fix (which kills historical
queries) is proposed last by the localizer and is rejected by the
smoke gate when reached.

## Rust second stage

`rust/` — std-only port, `cargo test --release` **14/14 passed**,
`verify` binary ALL CHECKS PASSED (E1 0.125 → 1.000, 56 → 42
tokens, formal 4; E2 0.2287 vs 0.0687; E3a trap smoke 0.250 vs
0.375; E3b 4.00 = 4.00). **Parity sweep: 300 Python-dumped
worlds, 0 mismatches** (best harness, accuracy, tokens, formal /
smoke / rejects / tried counts, oracle — every field exact).
**Soak: 500/500 worlds** reach oracle accuracy (mean formal 3.91);
the Python cross-check (`soak_py.py`, Rust xorshift worlds
replicated call-for-call) measures the identical 500/500 and
3.91, and the seed-5000 world replica is an EXACT MATCH
field-by-field. See `rust/README.md` for the divergence log.

## Honest limits

- Bounded scale, not the paper's scale: 48 synthetic cases per
  world (vs LongMemEval-S / BEAM, thousands of cases, real ReMe
  memory system, real LLM). The mechanism is reproduced; the
  paper's absolute numbers (e.g. token savings on real benchmarks)
  are not.
- The memory system, retrieval scoring, and answer generation are
  rule-based stand-ins, not ReMe + an LLM: failure modes are
  STALE / MISSING by construction, and the harness space (256
  combos) is small enough for an exhaustive oracle — the paper's
  harness space is open-ended.
- No model-backed validation: the claim that the loop works on a
  real memory system with LLM judgments is untested here.
- Single-harness-space result: the loop reaches the oracle on
  500/500 soaked worlds, but those worlds share the generator's
  distributions; a different failure-mode mix could trap it in a
  local optimum (the latest-only trap is one such trap, and the
  loop avoids it only because the localizer orders recency first).

## Files

- `README.md` (this file), `results.json` (Python numbers)
- `parity_dump.py` — generates `rust/tests/parity_cases.txt`
- `soak_py.py` — Python replica of the Rust world generator + soak cross-check
- `rust_dump.txt` — seed-5000 Rust world dump (replica check evidence)
- `rust/` — Cargo package `home`: `src/lib.rs`, `src/bin/verify.rs`,
  `tests/home_tests.rs`, `tests/parity_cases.txt`, `results.json`,
  `README.md`
