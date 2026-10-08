# Session 2026-10-08 — FinVector-Market-4B: Controlled Study of LoRA Adaptation for Structured Financial Tasks (arXiv:2610.08882)

**Article:** "FinVector-Market-4B: A Controlled Study of LoRA Adaptation for
Structured Financial Tasks"
([abs](https://arxiv.org/abs/2610.08882))

**Why picked:** most prospective *training* article in the 2026-10-08 cs.AI
window (TUNE 5/5). The paper's crispest claim is a clean dissociation:
supplying an explicit JSON schema alone raises base-model JSON validity
0% → 91.3% (pure format learning), but task accuracy stays low; rank-16
LoRA on a 22k-example corpus then raises *task* accuracy far beyond format
learning (FinQA EM 14.7 → 40.0%, calculator-expression 48.0 → 82.7%,
scenario branch-label 20.1 → 89.5%, implication-direction 52.4 → 87.2%)
under matched explicit prompting. Direct repo fit:
`soup-daily-finetune` (daily finance-domain LoRA — this is the same
adaptation pattern on structured financial tasks) and
`hermes-gi-egtkg-finetune`.

## What was built

- Colab script: `colab_finvec.py` — self-contained (pip installs at top,
  `pip uninstall -y torchao` after installs per the documented
  peft/torchao quirk); Qwen3.5-4B NF4/bf16 (the paper's own backbone;
  see deviation note on the 9B default); 4 synthetic structured-financial
  tasks standing in for the paper's corpus — CALC (arithmetic expression
  → value), FINQA (financial table + question → numeric answer),
  BRANCH (scenario → APPROVE/REVIEW/REJECT by a deterministic policy),
  IMPLY (statement pair → SUPPORTS/CONTRADICTS/NEUTRAL); 600 train/task
  (2400 pool), 60 test/task (240 benchmark; paper uses 600).
- Three arms under the comparison the paper runs:
  - **A base-noschema**: frozen base, plain prompt.
  - **B base+schema**: frozen base, explicit JSON-schema prompt.
  - **C lora+schema**: LoRA r=16/α=32, all linears q/v proj
    (1,835,008 trainable params), 150 steps batch 4 lr 2e-4 bf16,
    same explicit schema prompt at eval (the matched-prompting test).
- Metrics: JSON validity (parseable `{"answer": ...}`) + task accuracy +
  accuracy-given-valid (the format-vs-task dissociation metric).
  Results → `finvec_results.json` + `finvec_metrics.csv`.

## Results (Colab T4, Qwen3.5-4B NF4 QLoRA)

| arm | JSON validity | task acc | acc \| valid |
|---|---|---|---|
| A base, plain prompt | 0.000 | 0.000 | — |
| B base, schema prompt | 0.729 | 0.325 | 0.446 |
| C LoRA r=16, schema prompt (150 steps) | 1.000 | **0.863** | 0.863 |

Deltas: schema-vs-plain = validity **+0.729**, acc +0.325 (prompting fixes
format, modest task gain). LoRA-vs-base-schema (matched prompt) =
validity +0.271, acc **+0.538**, acc|valid +0.417 — the adapter's gain is
overwhelmingly *task* learning, not format learning.

Per-task accuracy, B → C: CALC 0.333 → 0.717, FINQA 0.217 → 0.883,
BRANCH 0.433 → 0.850, IMPLY 0.317 → 1.000. LoRA learning curve (overall
acc): 0.325 (step 0) → 0.838 (step 50) → 0.854 (step 100) → 0.863
(step 150). Validity saturates by step 50 (0.996); task accuracy keeps
climbing — format is learned first, task competence after.

## Verdict vs the paper's claim

**Replicates the core dissociation.** The paper: schema alone 0 → 91.3%
validity; LoRA adds large task gains under matched prompting. Here:
schema alone 0 → 72.9% validity with acc only 0.325; LoRA adds +53.8pp
task accuracy with only +27.1pp validity. The direction and the
format-vs-task separation match the paper; absolute magnitudes differ
(synthetic 4-task pool, 150 steps, 240-item benchmark vs the paper's 22k
corpus and 600-item benchmark). No evidence of the paper's caveat here
(filing overlap / calculator-target inconsistencies) — our generator is
exact by construction.

## Deviations from the paper (documented, honest)

- Backbone is Qwen3.5-4B = the paper's own model, not the track-default
  9B: the 9B default exists to avoid shrinking paper backbones, but here
  the paper *is* a 4B study, and the 12GB host-RAM loader ceiling that
  ruled out 8B-class NF4 in the 2026-10-07 GRADE session still applies.
- Synthetic 4-task pool stands in for the 22k-example corpus; 150 steps
  (≈600 samples, batch 4) instead of full-corpus training; 240-item
  benchmark instead of 600.
- Single JSON schema `{"answer": "<string>"}` for all tasks (paper uses
  task-specific schema contracts); strict first-`{...}` parse (paper's
  91.3% validity is under their parser — ours reaches 72.9% with a
  zero-shot terse prompt, 100% after LoRA).
- Training needed gradient checkpointing + seq 128 + batch 4: the
  reference-PyTorch mamba fallback (no causal_conv1d on the Colab image)
  makes Qwen3.5-4B's hybrid layers slow (~13 s/step) and VRAM-hungry;
  plain batch-8/seq-192 OOMed at step 1. One run, one seed.

## Colab ops notes

- Session `finvec-1008` on account kyu008008 (active account, left as-is).
  Session stopped after the run; results + metrics downloaded to this dir.
- Three script bugs were found and fixed across relaunches (all in the
  local `colab_finvec.py`, none affecting the final numbers): a
  `str.format` brace collision in the schema prompt (KeyError), an
  overall-`acc_given_valid` aggregation bug (raw counts vs ratios —
  per-task metrics were always correct), and right-padding during LoRA-arm
  evals (regeneration with left padding now; training keeps right).
  Resume-from-partial (`/content/finvec_partial_*.json`) and incremental
  eval checkpoints made relaunches cheap.
