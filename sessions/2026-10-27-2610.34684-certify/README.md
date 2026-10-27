# 2026-10-27 — Certify: requirement-directed search with Horn-closure certificates

- Article: https://arxiv.org/abs/2610.34684 — "Knowing When the Evidence
  Is Enough: Requirement-Directed Search with Horn-Closure Certificates"
  (Certify)
- Track: agent. AGENT=5, TUNE=1.
- Why picked: a search-augmented agent must decide after each retrieval
  whether its evidence suffices — answer, search again, or abstain. Certify
  makes that decision structural (a requirement hypergraph + Horn closure
  + entity checks) instead of leaving it to holistic model judgement, and
  the whole mechanism is implementable and testable in deterministic CPU
  code. Repo fit: `arxiv-lab` agent-harness mechanism library, alongside
  the FinishGate / HarnessMM decision gates in `src/arxiv_lab/` — a
  sufficiency gate for retrieval-augmented agents.

## What was built

- `src/arxiv_lab/retrieval/certify.py` (graduated): `Requirement` /
  `Rule` (Horn: `head :- premises`) / `Passage` / `link_passages`
  (entity-checked linking; a passage that merely names an entity, or
  supports the same fact for a different entity, does not link) /
  `horn_closure` (least fixpoint) / `certify` (ANSWER iff every
  requirement's fact is in the closure, else ABSTAIN naming the open
  requirements and the missing leaf facts). Exported from
  `src/arxiv_lab/retrieval/__init__.py`.
- `certify_sim.py` (this session): deterministic simulation, 100
  base questions per seed, each paired with an unanswerable twin that
  removes exactly one gold leaf passage while keeping entity-naming
  distractors. Baselines: a holistic entity-coverage judge (answers iff
  every required entity is named), an all-passages answerer, and
  Certify without the entity check (ablation).
- `tests/test_retrieval_certify_cpu.py`: 6/6 pass (see below).

## Results (canonical seed 20261027; robustness seeds 1, 7, 42)

| metric | seed 20261027 | seeds 1 / 7 / 42 |
|---|---:|---:|
| Certify decision accuracy | 1.000 | 1.000 / 1.000 / 1.000 |
| Holistic-judge decision accuracy | 0.500 | 0.500 / 0.500 / 0.500 |
| All-passages decision accuracy | 0.500 | 0.500 / 0.500 / 0.500 |
| Certify w/o entity check, decision | 0.680 | 0.720 / 0.705 / 0.680 |
| Certificate answer accuracy | 0.760 | 0.830 / 0.780 / 0.740 |
| All-passages answer accuracy | 0.530 | 0.600 / 0.430 / 0.540 |
| Twin abstention rate | 1.000 | 1.000 / 1.000 / 1.000 |
| Twin open-requirement naming exact | 1.000 | 1.000 / 1.000 / 1.000 |

Reading: Certify is sound and complete in this simulation — it never
answers a twin and never abstains on an answerable question, on any seed —
and every abstention names exactly the removed leaf fact. The holistic
judge answers every twin (its context names every entity, yet lacks one
required fact), which is exactly the failure mode the paper diagnoses.
Certificate answers beat all-passages answers on every seed
(mean 0.78 vs 0.53).

## Verdict: SUPPORTED

The core claim reproduces directionally at bounded scale: structural
sufficiency (requirement hypergraph + Horn closure + entity checks)
decides answer/abstain correctly where entity-coverage judgement fails
completely, and certificate filtering removes misleading-distractor
errors. Not VERIFIED: no live LLM, no real corpus, no learned requirement
decomposition — requirements, rules and links are given, as the paper's
own soundness result assumes ("given correct retrieval, requirements and
links").

## Honest limits / disclosed simulation regime

- Answer-accuracy LEVELS are regime numbers, not the paper's: a seeded
  per-question "hard" rate (0.20) makes even certificate answers wrong
  sometimes, and a "misleading" rate (0.30) corrupts all-passages
  answers. The reproduced claim is the ordering and the mechanism.
- The entity-check gain here is 28–32 points, above the paper's 4–13
  points on its corpora: wrong-entity passages are deliberately dense
  in this simulation (one per question plus a 25% twin trap), so the
  ablation bites harder. Direction, not magnitude, is the reproduced
  claim.
- Two bugs caught mid-run, recorded here: (1) the first generator draft
  did not list derived-rule premises as requirements, so the closure had
  no linked facts to run on and derived questions always abstained
  (cert decision ~0.75); fixed by listing leaf premises as requirements.
  (2) Answer accuracy was first scored over all cases including twins,
  understating the all-passages baseline; both rates are now scored over
  answerable cases only, matching the paper's certificate-vs-all
  comparison.

## Tests

- `python3 tests/test_retrieval_certify_cpu.py` — 6/6 pass (unit tests
  for closure, entity-checked linking, answer/abstain + naming, plus the
  full simulation assertions on all four seeds).
- Full CPU suite (`tests/test_*_cpu.py`) — no FAIL lines after adding
  this module.
