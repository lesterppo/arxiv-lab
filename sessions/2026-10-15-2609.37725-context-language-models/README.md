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

## Deep test: live model loop (2026-10-15, same day)

**Harness:** `deep_clm_task.py` in this dir — a deep-research task over a
scripted 48-document retriever (2 golden docs per question + 32 distractors),
8 questions, 6000-token budget. Two arms, same task:

- **baseline:** fixed truncation to budget (the standard harness approach).
- **clm:** the model's whole context is a live file; it gets a `BASH:`
  action and is told to manage the file itself (delete/compact via sed) —
  the paper's zero-shot setup, no learned policy.

Model: `openai/gpt-oss-20b` via NVIDIA NIM. (The NIM driver
`nim_deep_clm.py` holds the API key, so it stays local and is NOT committed —
reproduce with `deep_clm_task.py` + any OpenAI-compatible `chat_fn`.)

**Results (gpt-oss-20b):**

| arm | accuracy | turns | searches | bash edits executed | peak tokens |
|-----|----------|-------|----------|---------------------|-------------|
| baseline (truncation) | 5/8 | 10 | 8 | — | 5810 (within budget) |
| clm (file edits) | 0/8 | 8 | 5 | **0** | 7947 (over budget) |

The CLM arm searched 5 times, then produced three consecutive empty
responses and was nudged out without ever answering. Across three CLM runs
total (3/8, 2/8, 0/8), the model executed **zero** file edits — the single
attempted edit was batched with its ANSWER line and dropped by the
harness's single-action parser (a real harness bug, fixed to in-order
multi-action execution; a second real bug fixed: empty answers counted
correct via `'' in x`).

**Stronger-model comparison: blocked.** `llama-3.1-nemotron-70b-instruct`,
`llama3-chatqa-1.5-70b`, `kimi-k2.6`, and `mistral-large-2-instruct` all
returned HTTP 404 on NIM chat completions intermittently during the test
window (the /v1/models list advertises them, but serving flapped — kimi
answered a probe, then 404'd minutes later). Only gpt-oss-20b was stable.

**Verdict: INCONCLUSIVE** (at 20B scale).
The paper's zero-shot claim can't be evaluated here because the model never
engaged the affordance — the bottleneck is the model's agentic capability,
not the context-management policy. The seeded simulation above (SUPPORTED)
validates the *mechanism's* efficiency given a competent editor (scripted
policy); it does not show a 20B model will be one. The paper's own gains
were measured on much stronger models, so this is a scale/capability bound
on the test, not a refutation.

**Harness lessons (durable):**
- One-action-per-turn parsers silently drop batched tool calls — execute all
  parsed actions in order per turn.
- A 20B-class model offered an open-ended file-edit affordance zero-shot
  will often ignore it entirely; affordance-use itself needs measuring
  before policy quality can be compared.
