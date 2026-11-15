# PECA: Parameter-Efficient Conciseness Adaptation

- **Article:** https://arxiv.org/abs/2611.17128
- **Title:** Parameter-Efficient Conciseness Adaptation: Differentiable Length Control for Concise Clinical Reasoning in Smaller Language Models
- **Date screened:** 2026-11-15 (arXiv API serving source; window 2026-11-12–15)
- **Scores:** TUNE **5**, AGENT **2**
- **Verdict:** **PARTIAL**

## Why picked

PECA adds a single differentiable conciseness term to a QLoRA SFT
objective — no preference data, no reward model, no auxiliary model,
one commodity GPU. That is exactly the daily-LoRA pattern of
`soup-daily-finetune` and `hermes-gi-egtkg-finetune` (the latter is
medical-domain, same as the paper's MedQA setting), so a working
conciseness term would drop straight into those harnesses. The
paper's headline claim is crisp and measurable on a free T4:
mean response length −72.8% while MCQA diagnostic accuracy
40.8% → 44.0% on a 1.7B model fine-tuned on medical-o1-reasoning-SFT.

## What was built

- `colab_peca_driver.py` (this session) — self-contained Colab driver:
  base eval → SFT arm (plain QLoRA) → PECA arm (SFT + conciseness
  term) → MedQA eval after each arm. Metrics flushed to `/content`
  after every phase; prints `TRAIN_DONE` on completion.
- Operationalization of the conciseness term (the paper's abstract
  publishes neither value, so these are my choices, recorded here):
  `PECA = mean over response positions j > BUDGET(96) of −log p(eos)`,
  `LAMBDA = 1.0`, added to the SFT cross-entropy. The term is computed
  with a per-position Python loop over the response logits (slow:
  ~96s/step vs ~68s/step for plain SFT — see below).
- Design: N_TRAIN=400 matched rows, 1 epoch, lr 2e-4, r=16/α=32 NF4
  QLoRA, bf16, transformers<5.0; MedQA n=100 greedy eval,
  max_new_tokens 160, batch 4. `pip uninstall -y torchao` after
  installs (peft/LoRA Colab rule).

## Model

- First choice **Qwen3.5-9B FAILED to load** (04:30 UTC): the
  `>=4.57,<5.0` transformers pin does not recognise model type
  `qwen3_5`. The one automatic fallback was consumed in-driver →
  **Qwen3-8B** (loaded 04:34 UTC, 5.82 GiB, 43,646,976 trainable
  params, 0.5301%).

## Results (Colab T4, session train-1115-0427, account cyc236ha)

| Arm | Metric | Value |
|---|---|---|
| Base Qwen3-8B | MedQA n=100 | **63/100 = 0.6300**, mean_tokens **160.0** (every generation hit the 160-token cap) |
| SFT train | 100 steps | 6,835s (~68s/step), peak VRAM 9.95 GiB |
| SFT step losses | 25 / 50 / 75 / 100 | 1.4605 / 1.2626 / 1.4889 / 1.5782 |
| SFT eval | MedQA n=100 | mean_tokens **160.0**; "accuracy" **5/100 — see flaw note, this is NOT an accuracy** |
| PECA train | 100 steps | 9,653s (~96s/step), peak VRAM 10.87 GiB |
| PECA step losses (total / peca term) | 25 / 50 / 75 / 100 | 3.5258 / 0.6871; 3.0735 / 0.9543; 2.7769 / 0.7522; 2.9086 / 0.6955 |
| PECA eval | MedQA n=100 | **NOT MEASURED — session reclaimed at 20/100** (0 correct so far, same parse artifact) |

## Why PARTIAL — two separate gaps, neither a model failure

1. **Eval-design flaw (mine, found during the run): the accuracy leg
   is unmeasured for both SFT arms.** The SFT target is
   `Complex_CoT + "Final answer: <free-text Response>"` — the model
   is never taught to output an option letter, so at eval it
   generates a long CoT (all 160 tokens, capped) whose free-text
   final answer contains no parseable letter. The generative
   letter-parse (`Answer: X`, then first standalone A–D) finds
   nothing in ~95% of cases. The **5/100 "accuracy" is a parse
   artifact and must never be reported as an accuracy.** Base scored
   63% only because base generations happen to contain parseable
   letters. The length leg (the paper's headline claim) is
   unaffected by this flaw: SFT mean 160.0 tokens (capped) is the
   baseline PECA must beat.
2. **Reclamation blocked the decisive measurement.** The PECA eval
   reached 20/100 at 10:00:18 UTC, then the session was
   **RECLAIMED** (~10:00–10:02 UTC). The `METRIC` line is printed
   only at eval end, so **PECA mean_tokens was never measured** and
   cannot be recovered from the log. The PECA adapter died with the
   VM — the driver does not save adapters, so no post-hoc eval is
   possible on this run.

What was tested: both arms trained end-to-end (full loss
trajectories banked); the conciseness term runs and its value is
stable across training (0.69–0.95); PECA's total loss stays above
SFT's CE at the same step (implied CE ≈ 2.12 vs SFT 1.26 at step 50
— the term fights the LM loss on long CoT targets, as expected).
What was not tested: the length reduction itself, and accuracy
under either arm.

## Honest limits

- Single 8B model (paper: 1.7B); single epoch, 400 rows, MedQA n=100.
- BUDGET=96 / LAMBDA=1.0 are my operationalization, not the paper's.
- No accuracy number exists for either fine-tuned arm in this run.
- The PECA training loop's per-position Python term costs ~40%
  extra step time; a vectorized term would be needed for longer runs.

## Resume state / follow-up (not a same-day relaunch)

- Re-run needs a fresh Colab session and a **full PECA retrain**
  (~2.7h train + ~23min eval); SFT numbers above are banked and
  need not be repeated.
- **Fix the eval first:** replace generative letter-parsing with a
  logprob-scored MCQA eval (score " A"/" B"/" C"/" D" by first-token
  logprob), which is format-independent and measures accuracy under
  free-text targets. Also save the adapter and flush per-item eval
  rows to `/content` so a mid-eval reclamation still yields a
  partial mean_tokens.
- Driver, session name, and rotation state are recorded in the
  2026-11-15 run log
  (`~/workspace/goals/arxiv-daily-read-and-test/hidden_files/run-2026-11-15.md`).
  Rotation file reads `2026-11-15 cyc236ha`; active account restored
  to kyu008008 after the run.

## Run notes

- Two runtime restarts on the VM side killed the local `train`
  monitor (~05:55 and ~07:54 UTC); the detached VM-side driver was
  unaffected both times and monitoring was re-attached with fresh
  budgets (`--watch-only --pid 873`, then a 6h budget).
- `colab_out/train-run.json`: outcome RECLAIMED, downloaded [].
