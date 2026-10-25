# Session 2026-10-25 — Separate Field Scoring Protects Unused Tools When Agents Pool Tool-Use Experience

**Article:** [2610.30867](https://arxiv.org/abs/2610.30867) — "Separate Field Scoring Protects Unused Tools When Agents Pool Tool-Use Experience"

**Verdict: SUPPORTED** (reproduced directionally at bounded scale; stated limits below)

## Why picked (AGENT=4, TUNE=1)

2026-10-25 is a Sunday: the arXiv API window (last 3 days, serving source
`api` via export.arxiv.org, 200 fetched, 146 in-window) contained **0 papers
not already screened on 2026-10-24** — the weekend announcement gap, not a
fetch failure. This article is the strongest 2026-10-24 near-miss carried
over for reconsideration (yesterday's shortlist, AGENT=4): it is directly
CPU-testable (a Hit@5 retrieval simulation with synthetic embedders) and
repo fit is `arxiv-lab`'s tool-retrieval mechanism library
(`src/arxiv_lab/retrieval/`, alongside RATED and TRACER), where pooled
tool-use experience is a live design question for any agent harness with a
shared tool catalog.

## What was built

- `src/arxiv_lab/retrieval/synapse.py` — synthetic-embedding retrieval
  protocol: 240 tools in 24 confusable groups of 10; each tool's intent is
  a group centroid (weight 2.0) plus a tool-specific unit vector, so
  same-group intents correlate ~0.8. A description vector and a usage
  centroid are two independent noisy observations of the same intent
  (unit-noise weight 0.90 each); a query is a third. Conditions: base
  (descriptions only), naive pooling (4 usage-example entries per used
  tool, tool score = max over its entries), SYNAPSE (one candidate per
  tool; used-tool score = mean of description-field and usage-field
  similarity, held-out tools scored on description alone), shuffled-usage
  controls (usage bundles permuted among used tools, entry counts
  unchanged), and a constant-offset control (+0.15 to used tools).
- `tests/test_retrieval_synapse_cpu.py` — 5 deterministic CPU tests
  (canonical seed 20261025, 500 held-out + 500 used queries).
- `metrics.json` — full per-condition Hit@5 table for the canonical seed
  plus robustness seeds 1, 7, 42.

## Results (Hit@5, canonical seed 20261025)

| condition | all | held-out | used |
|---|---:|---:|---:|
| base | 0.927 | 0.922 | 0.932 |
| naive pooling | 0.904 | 0.810 | 0.998 |
| naive, shuffled usage | 0.721 | 0.762 | 0.680 |
| SYNAPSE | 0.925 | 0.858 | 0.992 |
| SYNAPSE, shuffled usage | 0.552 | 0.978 | 0.126 |
| constant offset (+0.15) | 0.808 | 0.632 | 0.984 |

Reading against the paper's claims:

- **C1 reproduced:** naive pooling lowers held-out Hit@5 by 0.112
  (0.922 → 0.810) while raising used-tool Hit@5 by 0.066 (0.932 → 0.998).
- **C2 reproduced:** shuffled usage lowers held-out Hit@5 about as much
  (0.762, −0.160 vs base) but the used-tool gain vanishes (0.680, well
  below base) — the harm is index crowding, the gain needs usage about
  the right tool.
- **C3 reproduced:** SYNAPSE recovers held-out Hit@5 to 0.858 (+0.048 vs
  naive) while retaining essentially all of the used-tool gain (0.992 vs
  naive 0.998). Under shuffled usage the protection persists — held-out
  0.978, above even base — while the used-tool gain collapses to 0.126,
  exactly the paper's "protection persists, gain needs right usage"
  pattern.
- **One divergence, reported not hidden:** the constant-offset control
  hurts held-out tools more than SYNAPSE (0.632 vs 0.858), whereas the
  paper notes a constant offset gives similar held-out Hit@5 "in most
  settings". In this simulation the protection comes from averaging two
  fields (variance reduction), not from merely boosting used tools.

Robustness: the same directional pattern holds on seeds 1, 7, and 42
(naive held-out 0.800 / 0.810 / 0.782 vs base 0.918 / 0.926 / 0.916;
SYNAPSE held-out 0.870 / 0.882 / 0.842; SYNAPSE used 0.998 / 0.994 /
0.998) — see `metrics.json`.

## Honest limits

- Synthetic embeddings and a synthetic catalog, not the paper's
  StableToolBench / TOUCAN / ToolRet catalogs or real embedders; absolute
  Hit@5 values (e.g. SYNAPSE held-out 0.858 vs the paper's 0.785 on
  StableToolBench) are not comparable — only the directional pattern is.
- SYNAPSE here is field averaging (0.5 description + 0.5 usage centroid);
  the paper's exact scoring formula was reconstructed from the abstract's
  "scoring usage as its own field" description, not from its code.
- Single-query-type simulation; the paper's router comparison (which keeps
  more labeled-query gain on TOUCAN/ToolRet) was not reproduced — the
  better choice depending on workload is noted, not tested.
- No live LLM was involved; this is a mechanism simulation, hence
  SUPPORTED rather than VERIFIED.
