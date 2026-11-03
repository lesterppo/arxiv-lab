# Evidence bypass in on-policy distillation (arXiv 2611.01967)

- Article: https://arxiv.org/abs/2611.01967 — "Better Answers, Weaker
  Evidence Control: Evidence Bypass in On-Policy Distillation"
  (submitted 1 Nov 2026).
- Why picked: AGENT=1, TUNE=5. The paper identifies a distinct
  training target — evidence responsiveness — and a concrete remedy
  (Evidence-Paired OPD: paired supervision on matched documents
  supporting different answers, plus an answer-level margin). Repo
  fit: soup-daily-finetune / training-method library (a fine-tune
  method adaptable to the daily-LoRA harness).
- Verdict: **INCONCLUSIVE** — the test design could not discriminate
  (ceiling effect + missing precondition; see below). This is NOT a
  refutation of the paper: the paper's effect is measured on real
  scientific QA (QASPER, PeerQA) where the student holds real
  closed-book preferences, and this test never created that
  precondition.

## What was built

Synthetic matched doc-QA (no external dataset): 220 fictional-compound
pairs sharing question + filler context and differing ONLY in the
answer-critical sentence (value A vs value B). CPU properties verified
in `test_epopd_cpu.py` (matched-intervention invariant on all 220
pairs, train/eval disjoint). Colab drivers are self-contained
(pip installs at top, `!pip uninstall -y torchao` after installs,
metrics CSVs written incrementally to /content).

Model (both phases): **Qwen/Qwen3.5-9B** first (9B-first rule), no
fallback needed; NF4/bf16, LoRA r=16/alpha=32, completion-only loss,
peak VRAM 9.17 GB on a T4. Accounts: cyc236es both phases (same day;
rotation state 2026-11-03 cyc236es, next kyu008008).

## Phase 1 — disjoint eval (session `epopd-1103`, DONE)

160 train pairs, 60 disjoint eval pairs, 1 epoch each arm.

| arm | orig_acc | cf_acc | bypass | train |
|---|---|---|---|---|
| base | 1.0 | 1.0 | 0.0 | — |
| single (orig docs only) | 1.0 | 1.0 | 0.0 | 367.6 s, loss 0.017 |
| paired (both matched docs) | 1.0 | 1.0 | 0.0 | 735.7 s, loss 0.009 |

Finding: ceiling at base. Fictional compounds carry no closed-book
preference, and training on disjoint items cannot induce one at eval
— so there was no bypass for paired supervision to recover. The
paper's bypass grows with the student's pre-distillation preference
for the original answer on the SAME questions; phase 1 lacked it.

## Phase 2 — same-item redesign (session `epopd2-1103`, DONE)

Redesign (not a repeat): train AND eval on the same 60 MEMO pairs, so
single-doc training itself induces the preference; 2 epochs each arm;
plus a 30-pair disjoint HOLDOUT generality check.

| arm | set | orig_acc | cf_acc | bypass | train |
|---|---|---|---|---|---|
| base | memo | 1.0 | 1.0 | 0.0 | — |
| single | memo | 1.0 | 1.0 | 0.0 | 273.5 s, loss 0.0224 |
| single | holdout | 1.0 | 1.0 | 0.0 | — |
| paired | memo | 1.0 | 1.0 | 0.0 | 545.6 s, loss 0.0123 |
| paired | holdout | 1.0 | 1.0 | 0.0 | — |

Finding: even same-item memorization (2 epochs, LoRA) did NOT induce
bypass — the model still copies the counterfactual document's value
at eval. On this format (value stated explicitly in the doc, question
asks exactly for it), in-context evidence-following dominates any
LoRA-induced prior. The paper's margin term was also omitted (paired
supervision only) — but with bypass at 0 in every cell, the margin
has nothing to act on either.

## Why INCONCLUSIVE, exactly

- The measured quantity (bypass) sat at 0.0 for base, single, and
  paired arms under BOTH operationalizations, so the comparison the
  paper's claim needs (single bypass > paired bypass) has no variance
  to compare. A test that cannot show the effect even in its
  single-doc arm cannot confirm or refute the remedy.
- The missing precondition is identified, not hand-waved: bypass
  needs a prior that genuinely competes with the document. Neither
  fictional facts nor 2-epoch LoRA memorization created one here.

## Resume state / what a retest needs

- Scripts staged here: `colab_epopd_driver.py` (phase 1),
  `colab_epopd_phase2_driver.py` (phase 2), `test_epopd_cpu.py`,
  `epopd_metrics.csv`, `epopd_summary.json`, `epopd2_metrics.csv`,
  `epopd2_summary.json`. Both Colab sessions torn down (server: no
  active sessions); rotation untouched beyond today's cyc236es.
- A discriminating retest needs items where the model holds a real
  prior that conflicts with the document — e.g. real-world facts
  with counterfactual documents, or full fine-tuning strong enough
  to shift the prior — plus the paper's explicit margin term, not
  just paired supervision. Do NOT rerun the same synthetic format
  expecting a different result.
- Operational note: phase 1's local train monitor process vanished
  after TRAIN_DONE without writing its outcome/downloads; metrics
  were recovered by direct download from the still-IDLE session
  before teardown. Phase 2's monitor reported DONE correctly.

## Honest limits

- No QASPER/PeerQA data, no teacher model, no OPD loss (the arms are
  SFT proxies: single-doc vs paired-doc supervision), no margin term.
- N=60/30, one seed, one model (Qwen3.5-9B), greedy decoding, exact
  value-string match as the correctness rule.
- Because the verdict is INCONCLUSIVE, nothing from this session
  graduates to `src/arxiv_lab/training/` (graduation requires
  VERIFIED/SUPPORTED).
