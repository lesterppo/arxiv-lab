# Session 2026-10-23 — 2610.30755 "Is Complex Agentic Memory Always Needed? Naive RAG Is Stronger Than We Thought"

- **Article:** https://arxiv.org/abs/2610.30755 (cs.AI, published 2026-10-22)
- **Verdict: SUPPORTED** (directional reproduction at bounded scale; stated limits below)
- **Track:** agent · **AGENT=5, TUNE=1**

## Why picked + repo fit
The paper asks whether LLM agents need complex agentic memory (mutation,
structural organization, learned retrieval) or whether naive RAG over raw
history suffices — and finds raw-RAG ranks first in 6/7 benchmark-model
settings (+0.13–5.53 pts over the strongest agentic system), because higher
*coverage* of the necessary evidence (answer values **plus** the relation /
dependency paths needed to derive them) offsets lower utilization. This is
directly load-bearing for the Hermes harness `MemoryStore` design decision
and for `arxiv-lab`'s memory line (`memory/cmp.py`, `memory/clawmmu.py`,
`memory/mempp.py`): it tells us what the store should optimize for.

## What was built
`src/arxiv_lab/memory/rawrag.py` (stdlib + numpy) + `tests/test_memory_rawrag_cpu.py`
(8/8 pass, deterministic):

- **Seeded conversation world**: 6 persons × 8 sessions × 14 turns with
  ground-truth evidence turns (city/hobby/pet/doctor→hospital relation path/
  previous-employer temporal facts) + distractor turns with cross-entity
  attribute mentions. 24 questions: 12 single-hop, 6 multi-hop (relation
  path), 6 temporal.
- **Five memory arms, same TF-IDF retriever (k=8) and deterministic oracle
  reader**: `raw_rag` (retrieve over raw turns), `window_rag` (retrieve over
  a per-person recency window of 4/7 fact turns — transparent fixed-budget
  forgetting), `window_full` (complete windowed store, no retrieval),
  `profile_rag` (mutable per-entity profile cards, overwrite semantics,
  entity-keyed lookup), `raw_full` (upper bound).
- **Metrics**: answer exact-match, evidence *coverage* (all required
  (person, fact-key) pairs present), *utilization* (EM/coverage).

Methods note: extractive summarization (TextRank, then TF-IDF salience ±
length normalization) was tried first as the transformation arm, but every
variant's keep/drop pattern was dominated by scoring artifacts (TextRank
anti-selected proper-noun-dense fact turns, 8/42 kept; unnormalized salience
favored long distractors; mean-normalized salience dropped city/employer/
temporal facts, 16/42). It was replaced by the auditable recency-window
budget rule — documented here so the artifact trail is visible.

## Results (8 seeds)
| arm | EM | coverage | utilization |
|---|---|---|---|
| raw_rag | **1.000 ± 0.000** | 1.000 ± 0.000 | 1.000 |
| profile_rag (mutation) | 0.750 ± 0.000 | 0.750 ± 0.000 | 1.000 |
| window_rag (budget 4/7) | 0.500 ± 0.070 | 0.500 ± 0.070 | 1.000 |
| window_full | 0.500 ± 0.070 | 0.500 ± 0.070 | 1.000 |
| raw_full | 1.000 | 1.000 | 1.000 |

- **Headline reproduced**: raw-RAG ranks first on every seed (1.000 vs 0.750
  vs 0.500) — the paper's 6/7-settings finding, directionally.
- **Mechanism reproduced**: the EM gap *equals* the coverage gap exactly
  (utilization = 1.0 in all arms under the oracle reader) — "higher coverage
  of necessary evidence" is the entire driver here.
- **Transformation, not retrieval, is the loss**: `window_full` ==
  `window_rag` (0.500) — even complete access to the transformed store leaves
  the full gap vs raw evidence, as the paper reports.
- **Mutation's specific cost**: profile cards preserve current state
  (single-hop 1.000, multi-hop join 1.000) but destroy history (temporal
  0.000) — the genuine price of overwrite write semantics.
- **Fragile case**: multi-hop relation paths suffer most under the window
  (0.375 vs single-hop 0.521) — both hops must survive the budget, matching
  the paper's "relations and dependency paths" emphasis.

## Honest limits
- Synthetic templated corpus (not LoCoMo/LongMemEval/MEME); TF-IDF lexical
  retriever (not dense + rerank); deterministic oracle reader (not an LLM) —
  so utilization is 1.0 by construction and the paper's
  "coverage-offsets-lower-utilization" tradeoff is shown via coverage only.
- The "complex" arms are representatives (recency window, mutable profiles),
  not the paper's exact compared systems (Mem0/MemOS/etc.).
- 8 seeds, one corpus generator; numbers are comparative within this
  harness, not absolute benchmark scores.

## Reusable mechanism
Graduated to `src/arxiv_lab/memory/rawrag.py` + `tests/test_memory_rawrag_cpu.py`:
`ConversationWorld` (seeded evidence-grounded dialogue benchmark),
`window_chunks` / `ProfileStore` (two transformation baselines),
`evaluate_arm`/`compare` (EM/coverage/utilization harness). Usable as the
MemoryStore evaluation rig for future harness memory designs.
