# WAAC: Rethinking Agent Trajectory Compression — A Workflow-Aware Perspective

- **Article:** arXiv [2611.08865](https://arxiv.org/abs/2611.08865) — Zhuang, Liu & Ren,
  "Rethinking Agent Trajectory Compression: A Workflow-Aware Perspective"
  (submitted 6 Nov 2026)
- **Verdict: SUPPORTED** (bounded-scale directional reproduction; see
  "Why SUPPORTED, not VERIFIED")
- **Track:** agent track — CPU only, stdlib only, no network / model API

## Why picked

- **AGENT 5 / TUNE 2.** A training-free mechanism directly reusable in a
  local agent harness: model trajectories as typed DAGs, derive per-field
  retention requirements by backward dependency analysis, reuse policies
  from verified historical fragments, and fuse conservatively. The paper
  claims 87.6 overall on PinchBench and 19.0% perfect completion on
  Odysseys while cutting cost 19.7% / 29.1% vs the strongest baselines.
- **Repo fit:** `arxiv-lab` mechanism library (a reusable trajectory
  compression module — field-level fidelity allocation is a general
  context-management primitive for any multi-step agent harness).
- **Testable within budget:** the whole mechanism is deterministic logic
  over typed fields, so a faithful CPU proxy test is possible in one run.

## What was built

- `waac.py` — the mechanism (stdlib only, deterministic):
  - Policies as capability sets: `DROP {}` < `SUMMARY {semantic}`,
    `POINTER {recoverable}` (incomparable), `EXACT {literal, semantic,
    recoverable}`; `fuse` is the least upper bound, so
    `SUMMARY ∨ POINTER = EXACT` (§3.3.3 / App. A.1).
  - `TraceDAG`: node `v_i = (op_i, I_i, O_i)`; edges from explicit
    argument bindings (output field → consuming input field). Uncertain
    dependency evidence is marked for conservative EXACT retention.
  - `online_requirements`: reverse traversal from outstanding
    obligations (pending inputs, planned ops, final-answer requirements,
    uncertain evidence); requirements join across obligations; recomputed
    at every step, so a field exact at one step becomes reducible after
    its consumers complete (§3.3.1).
  - `HistoryLibrary`: verified-successful fragments only (unverified
    admission is rejected); exact-key index on the canonical fragment
    (SHA-256 of normalized ops / field roles / edges) first, then
    compatibility-checked candidates (same op-type sequence, same
    field-role multiset) ranked by TF-IDF cosine over semantic
    descriptions — CPU proxy for the paper's embedding similarity
    (§3.3.2). Same-key policies aggregate by conservative join.
  - `project_context` / `project_field`: EXACT keeps the value; SUMMARY
    keeps a bounded stub (first 25 tokens of long text; short scalars
    whole); POINTER emits `<ptr:artifact_id>` iff the artifact is
    verifiably in the runtime state, else falls back to EXACT (§3.3 /
    App. A.3). Original events/artifacts stay in `S_t`.
- `corpus.py` — deterministic seeded synthetic multi-step workflows:
  60 test workflows (7–13 ops observed; generator aims 6–14) + 20
  separate-seed dev workflows for the history library, across five
  domains (cooking / travel / finance / research / shopping). Ops:
  search / fetch / extract / compute / verify / write / audit, with
  typed input/output fields; cross-step dependencies (a `page_text`
  from step 2 verified at step 5+; a `title` from step 3 needed by the
  final answer; a second fetch re-using a step-1 URL; retries);
  distractor long-text fields never consumed; final-answer obligations
  requiring specific short-scalar fields. Per-step gold required-field
  sets are derived from the generator's own data-flow graph. Need kinds
  vary per workflow: `verify` needs `page_text` literally (quote check)
  or semantically (gist check); `write` needs `report_body` literally,
  semantically, or recoverably (artifact re-read). ~15% of workflows
  carry one uncertain dependency (conservative EXACT, not in gold).
- `test_waac_cpu.py` — deterministic test: a recoverability executor
  that sees ONLY the projected visible context (+ runtime state for
  pointer resolution), never the full trajectory. A step succeeds iff
  every field it needs is recoverable at the required fidelity.
- `results.json` — the measured numbers.

## Results

History library: 25 fragments from 20 verified dev workflows (0 failed).
WAAC-full history match rate 0.956 over 547 lookups (523 exact, 0 coarse,
24 none — the corpus reuses skeletons, so exact keys dominate; the coarse
path is probed directly by check A7).

| method | success | steps completed | critical-field retention | mean visible tokens / step |
|---|---|---|---|---|
| No-compression (upper bound) | 1.000 | 1.000 | 1.000 | 573.4 |
| **WAAC full** | **1.000** | **1.000** | **1.000** | **119.8 (20.9% of full)** |
| WAAC online-only (ablation) | 1.000 | 1.000 | 1.000 | 66.3 |
| Keep-Last-N (budget-matched) | 0.000 | 0.504 | 0.784 | 124.0 |
| Keep-Last-5 (natural, uncapped ref.) | 0.283 | 0.827 | 0.973 | 404.4 |
| Text-importance (budget-matched) | 0.000 | 0.496 | 0.645 | 122.8 |
| Summary-all | 0.017 | 0.653 | 0.775 | 147.3 |

WAAC token reduction vs no-compression: **79.1%**. The two selective
baselines run under a per-step token budget matched to WAAC-full's own
tokens at the same (workflow, step), so the comparison is at equal
spend; their mean tokens/step slightly exceed WAAC-full's only because
failed runs never reach the cheap final steps (the per-step cap itself
is never exceeded).

Why the baselines fail, in one line each:

- **Keep-Last-N (budget-matched, 0/60):** at a step where WAAC
  summarizes a long field (25 tokens), an EXACT-only method cannot fit
  that 150-token field into the same budget, so the extract/verify step
  fails; and at the final step the title from step 3 is outside any
  recency window. The uncapped Keep-Last-5 reference (17/60, 404
  tok/step) isolates the recency effect: even with ~3.4× WAAC's tokens
  it fails every workflow whose final answer needs an early field.
- **Text-importance (0/60):** overlap scoring keeps the topically
  salient long texts and drops terse identifiers (URLs, status flags,
  artifact paths, numbers) that carry no overlap with the obligation
  text — the exact fields later steps need literally.
- **Summary-all (1/60):** succeeds only on the single workflow with no
  long-text literal/recoverable need (verify semantic + write semantic);
  every other workflow needs a long report body or page text literally
  or via artifact recovery, which a truncated stub cannot provide.

All seven pre-registered assertions passed: A1 WAAC retains 100% of
gold-required fields at every step (min per-step retention 1.000 for
both WAAC-full and WAAC-online-only); A2 WAAC success equals
no-compression (60/60) with strictly fewer tokens; A3 WAAC beats every
baseline at matched budget; A4 fusion never relaxes an online
requirement (16/16 policy pairs + 200 random triples: fused caps ⊇
online caps, commutative/associative, SUMMARY+POINTER → EXACT); A5
history admits only verified-successful trajectories (unverified
admission rejected; a failed run's fragments rejected); A6 an invalid
pointer falls back to EXACT; A7 coarse matching works (same structure,
different wiring → coarse match at cosine 1.0; different op sequence →
no match).

## Verdict: SUPPORTED

WAAC matches no-compression exactly on task success (60/60) and on
critical-field retention (1.000 at every step) while cutting visible
context to 20.9% of uncompressed, and it beats every baseline at a
matched token budget (best baseline: Keep-Last-5 at 17/60 with 3.4× the
tokens). The ablation confirms the online dependency analysis alone
carries the success; history adds retention on top (see limits).

Why SUPPORTED, not VERIFIED: synthetic workflows, not PinchBench /
Odysseys (those benchmarks are not available here); TF-IDF cosine
stands in for the paper's embedding similarity; SUMMARY is a truncation
stub, not a model-generated summary; no LLM in the loop (a deterministic
recoverability executor, not an agent). This is a directional
reproduction of the mechanism at bounded scale, not an end-to-end
verification of the paper's benchmark claims.

## Honest limits

- **History costs tokens here, it does not save them.** WAAC-full uses
  119.8 tok/step vs 66.3 for WAAC-online-only at identical success
  (60/60). The conservative max-join means a test workflow whose
  verify step needs a field only semantically inherits the EXACT policy
  cached from a dev trajectory that needed it literally. In the paper
  the history leg's token benefit comes through fewer exploration steps
  (§4.3: w/o offline, input rebounds +9.87% because the agent
  re-explores); a fixed-step executor cannot show that benefit, so this
  test measures only the retention cost of history, not its step-saving
  value. The fusion semantics themselves (§3.3.3: history can only add
  retention) are faithfully implemented.
- **Exact-key matches dominate (523/547 lookups).** The corpus reuses
  five skeletons, so same-structure fragments collide on the exact key;
  the coarse-match + TF-IDF ranking path is exercised only by the
  dedicated probe A7, not by the main experiment. Real trajectory
  diversity would exercise it more.
- **CPU proxies, not the paper's stack:** TF-IDF cosine for embedding
  similarity; a first-25-tokens stub for bounded summarization (the
  generator places each long field's key fact in its first sentence, so
  the stub is a fair SUMMARY by construction — a real summarizer could
  miss facts the stub retains, or retain facts it drops).
- **Synthetic corpus, not PinchBench/Odysseys:** templated workflows
  with known data flow; the executor checks recoverability, not agent
  reasoning, so success tracks field retention rather than decision
  quality. The comparison against no-compression / baselines on the
  *same* corpus is the meaningful result, not success 1.000 in isolation.
- **Whitespace tokens**, not a real tokenizer's counts.
- **A.4 nuance not modeled:** the online requirements here are
  recomputed from scratch each step, so the cached-requirement reuse
  analysis (validity checks, residual recomputation) is not exercised;
  only the §3.3.3 fusion equation is implemented. The implementation
  follows fusion as an additional retention constraint even when the
  cached policy's original consumer need differs in kind from the
  current one (see the history-cost note above).
- **No Rust second stage in this session** (the cron's standing Rust
  rule was not in this task's brief; the deliverables here are the
  Python mechanism + CPU test only).

## Reproduce

```bash
cd sessions/2026-11-09-2611.08865-waac
python3 test_waac_cpu.py   # deterministic; writes results.json; exit 0
```
