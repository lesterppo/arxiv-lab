# CORE: Coarse-to-Fine Search with Context Management for Deep Research

- **Article:** arXiv [2611.07859](https://arxiv.org/abs/2611.07859) — Vuong et al.,
  "CORE: Coarse-to-Fine Search with Context Management for Deep Research"
  (submitted 5 Nov 2026)
- **Verdict: SUPPORTED** (parent review 2026-11-08: the numbers meet the
  mechanical bar, but the corpus fixes both gold sentences at passage
  positions 0–1, so the test cannot separate token-similarity scoring
  from positional bias — a first-2-sentences baseline would also score
  recall 1.000 — and the rankers are BM25/TF-IDF proxies. Bounded-scale
  directional reproduction, not end-to-end verification.)
- **Track:** agent track — CPU only, stdlib + numpy, no network / model API

## Why picked

- **AGENT 5 / TUNE 1.** Training-free harness, directly reusable in a local
  agent harness: retrieve passages, return only the most relevant snippet per
  passage, and manage context across turns (expand around already-retrieved
  sources, suppress repeated snippets). The paper claims SOTA answer accuracy
  and evidence recall on BrowseComp-Plus with far less context.
- **Repo fit:** `arxiv-lab` mechanism library (a reusable retrieval +
  context-management module); `hermes-colab-cli` deep-research harness
  (the same coarse-to-fine + suppression pattern applies to its search loop).
- **Testable within budget:** the whole mechanism is deterministic logic over
  text, so a faithful CPU proxy test is possible in one run.

## What was built

- `core.py` — the mechanism:
  - `BM25` coarse retriever (CPU proxy for the paper's dense retriever).
  - `SnippetExtractor.best_window` — TF-IDF cosine between the query and
    each 2-sentence window of a retrieved passage; returns the best window
    (CPU proxy for the paper's late-interaction reranker token similarities).
  - `ContextManager` — cross-turn context management: a turn-2 snippet that
    near-duplicates an already-returned snippet (Jaccard >= 0.8) is
    suppressed (reused from context); if the already-returned snippet lacks
    the answer-bearing sentence, the context is expanded by that adjacent
    sentence instead of re-fetching the passage.
  - Whitespace token counter.
- `corpus.py` — deterministic seeded synthetic multi-hop QA corpus:
  120 fictional researchers (name / field / institution / city / award /
  year / discovery), one 7-sentence passage each; 60 two-hop questions
  (field + award + year identify the researcher, ask the institution) and
  30 single-hop controls. Award names/years are shared across entities and
  filler sentences mention other institutions/awards, so retrieval is
  non-trivial. Gold evidence sentence ids recorded per question
  (sentence 0 = institution, sentence 1 = award + year).
- `test_core_cpu.py` — deterministic test (E1–E3 below) with assertions.
  Reader: deterministic extractive reader — scores candidate institutions by
  token proximity to the researcher name in the returned context.
- `results.json` — the measured numbers.

## Results

Config: BM25 k=5, snippet = best 2-sentence window, Jaccard threshold 0.8.

### E1 — single-turn, 60 two-hop questions

| method | gold-evidence recall (both hops) | reader accuracy | mean context tokens |
|---|---|---|---|
| FULL (whole passages) | 1.000 | 1.000 | 381.2 |
| **CORE (snippets)** | **1.000** | **1.000** | **115.7 (30.4% of FULL)** |
| NAIVE-SNIPPET (first sentence) | 0.000 | 0.700 | 60.1 |

Single-hop controls (n=30): FULL accuracy 1.000 (379.0 tok), CORE 1.000
(117.1 tok).

### E2 — multi-turn, 30 two-turn trajectories (cumulative tokens)

| method | cumulative tokens / trajectory |
|---|---|
| FULL | 763.1 |
| CORE, no context management | 235.4 |
| **CORE + context management** | **180.5 (23.7% of FULL)** |

Suppression rate 0.971 (68 of 70 already-seen passage slots suppressed,
2 expanded by one adjacent sentence). Turn-2 evidence coverage 1.000 for
both CORE without CM and CORE+CM — suppression lost no evidence.

### E3 — snippet-quality ablation (evidence recall, n=60)

| snippet strategy | recall |
|---|---|
| token-similarity (CORE) | 1.000 |
| first sentence | 0.000 |
| random 2-sentence window | 0.233 |

All four pre-registered assertions passed: CORE recall within 5pp of FULL
(gap 0.000); CORE mean tokens 30.4% of FULL (<= 50%); CORE recall beats
NAIVE-SNIPPET by 100pp (>= 10pp); CORE+CM cumulative tokens < CORE without
CM (180.5 < 235.4).

## Verdict: SUPPORTED

CORE matches FULL exactly on gold-evidence recall (1.000 vs 1.000) and
extractive-reader accuracy (1.000 vs 1.000) while cutting single-turn
context to 30.4% and two-turn cumulative context to 23.7% of FULL, with
no evidence lost to suppression (coverage 1.000). The snippet ablation
confirms the token-similarity selection — not just truncation — is what
preserves the evidence (1.000 vs 0.000 first-sentence, 0.233 random).

Why SUPPORTED, not VERIFIED: the corpus fixes both gold sentences at
passage positions 0–1, so the experiment cannot distinguish "token
similarity finds the evidence" from "the evidence is always at the top"
— a first-2-sentences baseline would also score recall 1.000, and it was
not run. Combined with BM25/TF-IDF proxies for the paper's dense
retriever and late-interaction reranker, this is a directional
reproduction at bounded scale with stated limits. A follow-up with gold
sentences at randomised positions (and a positional baseline) would be
needed to claim VERIFIED.

## Honest limits

- **CPU proxies, not the paper's stack:** BM25 stands in for a dense
  retriever and TF-IDF cosine for ColBERT-style late-interaction token
  similarities. The mechanism shape (coarse retrieve → token-sim snippet →
  suppress/expand) is faithfully implemented; the ranking components are
  weaker substitutes.
- **Synthetic corpus, not BrowseComp:** templated 7-sentence passages with
  the two gold sentences adjacent at the top (sentences 0–1). That makes
  the best 2-sentence window easy to find and inflates the absolute recall
  numbers — the comparison against FULL / NAIVE on the *same* corpus is the
  meaningful result, not recall 1.000 in isolation. The ablation lacks a
  positional baseline: a first-2-sentences heuristic would also score
  recall 1.000 on this corpus, so the experiment cannot separate
  token-similarity scoring from positional bias.
- **No LLM reader:** a deterministic proximity reader answers from the
  institution sentence alone, so accuracy mostly tracks hop-2 coverage;
  evidence recall is the discriminating metric. The NAIVE accuracy of 0.700
  (despite the institution sentence always being present) is a reader
  artifact: with single-sentence contexts, a neighbouring passage's
  institution can sit closer (in tokens) to the researcher name than the
  true institution — a proximity-reader boundary effect, not a retrieval
  failure.
- **Whitespace tokens**, not a real tokenizer's counts.
- **Single-domain synthetic questions** (researcher / award / institution);
  no test of the paper's BrowseComp-Plus claims on real web content.

## Reproduce

```bash
cd sessions/2026-11-08-2611.07859-core
python3 test_core_cpu.py   # deterministic; writes results.json; exit 0
```
