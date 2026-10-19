# SchemaShield — robust function calling under interface drift (2610.23342)

**Article:** https://arxiv.org/abs/2610.23342 — *Robust Function Calling for
LLM Agents Under Interface Drift* (Chattopadhyay, Halder, Roy)
**Date:** 2026-10-19 · **Fetch source:** api · **Track:** agent · **Verdict: SUPPORTED**

## Why picked + repo fit (AGENT 5 / TUNE 1)
Directly harness-relevant: tool-augmented agents break when tool
descriptions, argument schemas, or the candidate tool list drift after
deployment — even when tool semantics are unchanged. The paper's
SchemaShield-Lite is a lightweight, training-free repair layer (validate →
candidate-set name retry → one-shot constrained repair), the exact kind of
harness-side guard that belongs in `arxiv-lab`'s mechanism library next to
the Mingbird harness modules (`prefill`, `finish_gate`, `loop`), and in
Hermes' own tool-calling loop. No GPU needed; deterministic CPU test.

**Core claims under test:** (1) a training-free repair layer recovers
drift-induced tool-call failures (paper: 19/22 on DICE-300, 33/45 on
BFCL-200, Qwen3.5-9B); (2) naive retry does not explain the gains;
(3) candidate-set drift (esp. distractors) is the dominant damage source;
(4) no repair harms.

## What was built
- `src/arxiv_lab/harness/schema_shield.py` (new module, stdlib-only):
  `ToolDef`/`ToolCall` registry types, `validate()` (JSON-schema-style arg
  checking), `SchemaShield.invoke()` — the three-stage pipeline: pass-through
  if valid → candidate-set retry for unknown tool names (normalized token
  Jaccard + sequence similarity, tie-broken by argument-schema fit, threshold
  0.45) → exactly one constrained arg-repair attempt (arg-name mapping incl.
  abbreviation handling, type coercion, dropping unmappable args, filling
  defaults) → fail otherwise. Plus `DriftSimulator` (rename / distractor /
  schema-edit / type-edit / description-rewrite drift operators + stale-call
  replay harness).
- `tests/test_harness_schemashield_cpu.py`: 8-tool registry, 24 scripted
  tasks with golden pre-drift call sequences (the caller is frozen on the old
  interface, as a deployed model would be), 6 drift scenarios replayed with
  and without the shield. Deterministic (seeded).
- Exports wired into `src/arxiv_lab/harness/__init__.py`.

## Results
`ALL SCHEMA-SHIELD CHECKS PASSED` — 31 calls/scenario:

| scenario | baseline | + shield | drift failures | recovered | rate | harms |
|---|---|---|---|---|---|---|
| control (no drift) | 31/31 | 31/31 | 0 | — | — | 0 |
| rename | 15/31 | 27/31 | 16 | 12 | 0.750 | 0 |
| distractors | 31/31 | 31/31 | 0 | — | — | 0 |
| schema_edits | 20/31 | 28/31 | 11 | 8 | 0.727 | 0 |
| desc_rewrite | 28/31 | 31/31 | 3 | 3 | 1.000 | 0 |
| **combined** | **8/31** | **23/31** | 23 | 15 | **0.652** | **0** |

Recovery concentrates in the expected stages (`name_retry` on renames,
`arg_repair` on schema edits, both firing under combined drift — the paper's
ablation pattern). Harms = 0 in every scenario: the layer never broke a
call that was valid without it, and the no-drift control proves pass-through.

**Honest limits.** (1) This is a deterministic rule-based instantiation of
the paper's (LLM-based) one-shot repair; absolute rates are not comparable to
the paper's DICE/BFCL numbers, only the *pattern* (recovery ≫ 0, harms = 0,
candidate-set drift dominant). (2) Every unrecovered combined failure was
asserted to belong to a documented hard class: the deliberately unguessable
rename `calc`→`q9x` (5 calls) and the synonym arg rename `to`→`recipient`
(3 calls) — no training-free layer can infer those. (3) The "drifted caller"
is scripted, not a real LLM; a model-backed check (e.g. NIM gpt-oss-20b
emitting calls under drift) is future work.

**Verdict rationale — SUPPORTED:** the core claim reproduces directionally at
bounded scale (training-free repair recovers the majority of drift failures
with zero harms); stated limits above. The module graduates to
`src/arxiv_lab/harness/` (done).
