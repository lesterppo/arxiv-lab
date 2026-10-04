# 2610.01026 — "It Takes Workflows to Evolve Better Workflows" (FloWright)

**Link:** https://arxiv.org/abs/2610.01026
**Date:** 2026-10-04 · **Track:** agent · **Verdict: mechanism supported (core claim), with a bounded flat-baseline caveat**

## Why picked

AGENT 4 / TUNE 2. FloWright attacks exactly the problem Peter's stack keeps
hitting: a multi-agent workflow's outcome is one sparse score that can't tell
which role caused a failure. The paper's fix — a hierarchical,
structure-aware reward paradigm that lets one role self-evolve and two or
more roles co-evolve with no extra models/labels/executions — is directly
relevant to deepworld's agent loop and the contribution-integration
machinery. The headline numbers (+5.03% co-evolving vs +2.83% single-role)
are a clean, testable prediction.

## What was built

A CPU-only synthetic reproduction in `src/arxiv_lab/workflow/flowright.py`
(stdlib only, deterministic):

- **Toy world:** tasks are arithmetic expressions (chain of 1–3 binary ops).
  Three roles with scalar skills in [0,1]: **generator** drafts an answer
  (per-op slip model), **verifier** independently re-evaluates and accepts
  iff its estimate matches the draft, **refiner** re-evaluates on rejection.
  Sparse outcome: 1 iff the final answer is exactly right.
- **Structured credit** (artifacts + sparse outcome only, no extra labels or
  executions, faithful to the paper): generator ← verifier acceptance rate;
  verifier ← 1 on accept iff final correct, on reject 1 iff the refinement
  changed the answer *and* it is correct (vindicated rejection); refiner ←
  correctness on tasks it handled.
- **Evolution:** (1+1)-ES per evolving role, round-robin, Gaussian mutation
  (σ=0.07), common random numbers per task so incumbent-vs-mutant
  comparisons are fair. 60 generations, 192 train tasks, 6 seeds; test on
  256 held-out tasks stratified over chain lengths {1,2,3}.
- **Conditions:** `flat` (all roles evolve on the shared sparse reward — no
  credit assignment), `single` (only the generator evolves, on structured
  credit), `coev` (all roles co-evolve on structured credits).
- **DataWright touch:** train on hardened tasks (2–3 ops) vs easy tasks
  (1 op); test always stratified, so the workflow-level gap is visible.

## Results

Hardened training, 6 seeds, held-out test accuracy:

| condition | mean | per-seed | final skills (gen/ver/ref) |
|---|---|---|---|
| flat | 0.984 | 0.941 1.000 0.973 0.992 1.000 1.000 | 0.89 / 0.74 / 0.99 |
| single | 0.484 | 0.482 0.494 0.467 0.514 0.424 0.525 | 0.99 / 0.50 / 0.50 |
| coev | **0.999** | 0.996 1.000 1.000 1.000 1.000 1.000 | 0.99 / 0.99 / 0.97 |

Per chain length (hardened): L1 — flat 0.986 / single 0.727 / coev 1.000;
L2 — 0.984 / 0.431 / 1.000; L3 — 0.982 / **0.294** / 0.998. The single-role
gap widens dramatically with difficulty — the workflow-level failure the
paper targets. Easy-task training gives the same ordering
(flat 0.982 / single 0.485 / coev 0.986).

**Paper's core claim reproduces:** co-evolving roles (0.999) massively beats
evolving the generator alone (0.484). Single-role evolution is capped by the
fixed verifier/refiner bottleneck even though the generator itself reaches
0.99 skill — the workflow, not the role, is the unit that matters.

**Flat-baseline caveat (the interesting part):** the flat shared-reward
baseline reaches 0.984 ≈ co-evolution on accuracy, because the verifier +
refiner can fully compensate for a mediocre generator. But the
credit-assignment pathology is visible *in the skills*: flat's generator
stalls at 0.89 (vs 0.99 under co-evolution) — the shared sparse reward gives
it no directional signal once the downstream roles compensate. In real
workflows the refiner works *from* the draft rather than solving from
scratch, so compensation is partial and the accuracy gap would open further.

**DataWright:** the untrained baseline shows why hardening matters —
L1: 0.576, L2: 0.335, L3: 0.155. Data a single role already handles (L1)
masks workflow-level differences; hardened chains expose them.

## Honest limits

1. **Roles are scalar skills, not policies.** The toy's "evolution" tunes
   three numbers; real FloWright trains LLM role policies where credit
   assignment interacts with representation learning. What transfers is the
   *credit-assignment structure* (hierarchical > flat), not any effect size.
2. **The refiner is unrealistically strong** — it re-solves from scratch,
   so verifier+refiner can fully compensate for the generator. This flatters
   the flat baseline on accuracy; the pathology only shows in skills. A
   draft-dependent refiner would widen the accuracy gap.
3. **Credit signals are cleaner than reality.** Accept-rate and vindicated
   rejection are low-noise here; real workflow artifacts (tool outputs,
   natural-language judgments) give far noisier per-role signals, and the
   paper's +5.03% vs +2.83% is on noisy real data — no effect-size
   calibration transfers from these near-ceiling toy numbers.
4. **One workflow shape, one task family.** Generator→verifier→refiner on
   arithmetic; no evidence the ranking holds for other topologies.

## Files

- `experiment.py` — full experiment (imports the graduated module)
- `results.json` — all numbers (6 seeds × 3 conditions × 2 difficulties)
- `test_session_cpu.py` — reduced-size rerun asserting coev > single and
  the flat generator-skill pathology
- Graduated: `src/arxiv_lab/workflow/flowright.py` + `tests/test_workflow_cpu.py`
