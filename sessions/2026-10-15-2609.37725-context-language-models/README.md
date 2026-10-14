# 2609.37725 — Context Language Models (2026-10-15)

**Article:** https://arxiv.org/abs/2609.37725 — Rulin Shao et al.
(UW, Meta Superintelligence Labs, MIT, Trillium Labs), submitted 29 Sep 2026.

**Why picked:** Peter pasted a YouTube explainer and said "test and integrate".
AGENT 5/5: the paper moves context management from fixed harness control to
intrinsic model behavior — directly relevant to any local agent harness
(including this repo's own `harness`/`context` modules). TUNE 2/5: the RL
training loop needs GPU; the mechanism itself is CPU-testable. Outside the
daily 3-day window — picked on explicit instruction, not by the cron screen.

**What was built:**
- `src/arxiv_lab/context/clm.py` — `LiveContextFile`: the live context
  mirrored as a line-addressed editable file. Runtime appends turns; the
  model applies unrestricted in-place edit batches (`delete` / `replace` /
  `append_note` / `rewrite`, all-or-nothing with `EditError` on any invalid
  op); `sync()` returns the current file as the next-turn input. Revision
  counter + edit log; deterministic chars//4 token accounting.
- `tests/test_context_clm_cpu.py` — 7/7 pass. Seeded simulation (seed 7):
  60 tool outputs, 6 answer-critical facts scattered among failed /
  duplicate / verbose outputs, 6000-token budget. Scripted edit policy
  mirrors the paper's observed emergent strategies (prune failures +
  duplicates, compact verbose outputs preserving refs, maintain a findings
  table); baseline is append-only + head truncation.
- `experiment_clm.py` + `results_clm.json` in this dir (reproduces the run).

**Results:**

| arm | final tokens (budget 6000) | facts retained (6) |
|-----|---------------------------|--------------------|
| CLM (edit policy) | 3817 | 6/6 |
| baseline (truncation) | 5925 | 2/6 |

The truncation baseline drops the task prompt and 4 early facts; the CLM arm
keeps everything with 5 in-place edit batches and a 6-item findings table —
the paper's Fig. 3 pattern (many in-place tracker updates, context held well
under budget) reproduced directionally.

**Verdict: SUPPORTED** (bounded scale).

**Honest limits:**
- The edit policy is scripted (regex FACT extractor), not a model judging
  importance — the deterministic stand-in for the paper's learned behavior.
- Suffix Cache Reuse (the SGLang serving patch, -35% server compute) is not
  tested here — serving-layer, not CPU-testable.
- The paper's online-RL (+47.6% on BrowseComp-Plus) and skill-optimization
  (+35.9 pts) training claims are not replicated — GPU track out of scope
  for this manual run.
- Paper's own caveat, worth repeating: CLMs currently need external
  token-count reminders (weak native context awareness), and the safety
  section warns the same freedom lets a model plant instructions for itself.
