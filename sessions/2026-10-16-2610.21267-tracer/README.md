# Session 2026-10-16 — TRACER: Trajectory-Aware Tool Retrieval (2610.21267)

**Article:** https://arxiv.org/abs/2610.21267 — "TRACER: Trajectory-Aware
Tool Retrieval for LLM Agents"

## Why picked (AGENT 5 / TUNE 1)

Directly reusable in a local agent harness: TRACER is a retrieval
front-end for tool selection — it indexes observed tool-call sequences in
a trie and ranks continuations with a learned blend of an empirical
(trajectory) prior and a contextual (semantic) prior, exposing only the
top-k tools to the agent. Repo fit: `arxiv-lab` mechanism library
(`src/arxiv_lab/retrieval/`) and the Hermes on-device agent's tool
registry, where registry size directly drives prompt tokens. Testable on
CPU end-to-end within the run budget.

## What was built

- `src/arxiv_lab/retrieval/tracer.py` — `TrajectoryTrie` (prefix tree over
  observed tool-call sequences with Laplace-smoothed continuation
  probabilities and marginal fallback for unseen prefixes),
  `tracer_retrieve` (learned blend `w*empirical + (1-w)*contextual`,
  top-k), `learn_blend_weight` (grid search on held-out trajectories),
  `tracer_top1_accuracy` (head-to-head vs full-registry selection).
- `tests/test_retrieval_tracer_cpu.py` — deterministic seeded sim:
  40 tools (30 regular + 10 near-duplicate "alias" tools), 2 task families
  with distinct headers, 8 subroutine templates with a family-variant
  slot. Variant-slot contexts are 50/50 mixtures of the regular tool and
  its alias — genuinely ambiguous for a contextual selector, fully
  disambiguated by the trajectory prefix (the complementary-priors
  regime the paper targets). Splits: 100 trajectories build the trie,
  40 learn the blend weight, 80 evaluate.

## Results

| metric | full-registry (contextual argmax, 40 tools) | TRACER (top-5) |
|---|---|---|
| tool-selection accuracy | 0.811 | **1.000** (+0.189) |
| tools exposed / step | 40 | **5** (−87.5%) |

Learned blend weight w_empirical = 0.1 (train accuracy 1.000). n = 291
eval steps, seed 20261016. Full numbers in `results.json`.

Distribution-shift probe (reported, not asserted): trajectories whose
family header mismatches the variant slot (the trajectory prior then
*misleads* by construction) — TRACER 0.493 vs full-registry 0.834. The
trie prior is a liability when the trajectory lies; the paper likewise
analyzes failure modes in both regimes.

## Verdict: SUPPORTED

Core claim reproduced directionally at bounded simulated scale: the
blended trajectory-aware front-end improves selection accuracy while
exposing far fewer tools. Honest limits: synthetic environment (no real
LLM in the loop — the contextual prior is a stand-in for an embedding
model); the sim's ambiguous steps are fully trajectory-resolvable *by
construction*, so the +18.9 pt gain is an upper bound — real
environments will show smaller gains (paper reports +4.7–10 pts on
τ²-Bench). The failure mode under trajectory shift is real and
documented above.

## Reuse

`TrajectoryTrie` + `tracer_retrieve` graduate to
`src/arxiv_lab/retrieval/` (already wired into the package `__init__`);
a harness can build the trie online from its own tool-call history and
set k / w from a small held-out set.
