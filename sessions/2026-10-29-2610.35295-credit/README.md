# 2026-10-29 — Did the Model Use That? Counterfactual Training Boosts Self-Attribution (CREDIT)

- Article: arXiv 2610.35295, https://arxiv.org/abs/2610.35295
- Track: training (TUNE=5). Repo fit: `soup-daily-finetune` daily NF4-QLoRA
  harness — CREDIT is a fine-tune method (counterfactual-label SFT on the
  final answer) directly adaptable to that harness, with no human annotation.
  Carried forward from the 2026-10-27 screen as the strongest TUNE near-miss
  ("reconsider for a later Colab day").
- Why picked: the paper's core claim is sharply testable on one T4 —
  supervising final-answer self-attribution with counterfactual labels beats
  prompting, including distinguishing changed-by-suggestion from
  would-have-given-anyway answers among suggestion-agreeing answers, and
  partly transfers to unseen info types and conflicting-source prompts.

## Verdict: INCONCLUSIVE

The test design could not discriminate the paper's core claim. The
counterfactual label construction produced a degenerate class balance
(94–98% USED in every split), leaving the paper's key discrimination test
— changed-by-suggestion vs would-have-given-anyway among
suggestion-agreeing answers — with essentially no would-have-given-anyway
arm (2/128 held-out, 5/114 transfer). The prompting baseline was already at
0.9688 attribution accuracy on the held-out split under that skewed prior
(always-say-used scores 0.984), a ceiling effect; and the balanced training
set (51 items, 2:1 USED:NOT-USED prior) mismatched the eval prior, so the
trained model's lower held-out score (0.6875) mostly reflects the training
prior, not a refutation of CREDIT. The claim is untested, not refuted. The
transfer-correction gain below is measured, but confounded with output-format
compliance (see "Honest limits"). Do not cite this run as evidence for or
against CREDIT.

## What was built

`colab_credit_driver.py` — self-contained Colab driver (pip installs at top,
`torchao` uninstalled right after, per the peft/Colab rule), run as run:

- **Model:** Qwen/Qwen3-8B (BASE), NF4 QLoRA, r=16, alpha=32, bf16,
  `transformers<5.0`, all 7 linears, lr 2e-4, batch 4, 3 epochs, prompt-masked
  loss. Qwen3-8B was used directly, per the task brief: Qwen3.5-9B was already
  tried first on 2026-10-27 and FAILED twice (Qwen/Qwen3.5-9B-Instruct is not
  a valid HF identifier; Qwen/Qwen3.5-9B has model type `qwen3_5`,
  unrecognised by transformers<5.0). Qwen3.5-9B was NOT retried. Probe
  numbers: trainable params 43,646,976 / 4,761,498,624 total; train mean loss
  0.1589 over 39 steps; peak memory 8.99 GiB (max_memory_allocated).
- **Task (synthetic, deterministic, seed 20261029):** invented-entity QA.
  With-info prompt states a supplied value, asks the question, and instructs:
  "Answer with just the value, then state whether you used the supplied
  information." Without-info prompt is the identical question line with the
  supplied line removed (pure info removal). Answer = first alphabetic token
  of the generation; attribution = "did not use" -> NOT-USED, "used the
  supplied"/"I used" -> USED (did-not checked first), no statement -> unscored
  and counted wrong for attribution accuracy.
- **Labels (the paper's construction, no human annotation):** an item is kept
  only if the BASE model's with-info answer equals the suggested value
  (follows the suggestion); it is labelled USED iff the base model's
  without-info answer differs from the suggested value. 560 candidates
  (train 288 / held-out 128 / transfer 144); kept 530 (train 288, held-out
  128, transfer 114; 30 transfer items dropped for not following the
  suggestion). Label USED rates among kept items: train 0.941, held-out
  0.9844, transfer 0.9561. Training split balanced by capping the majority
  class at 2x the minority (deterministic subsample): 51 items, 34 USED /
  17 NOT-USED. Held-out and transfer splits were NOT re-balanced.
- **Splits:** held-out = 16 unseen entities + 3 unseen template pairs
  (S3 record format and/or Q3 question form); transfer = 12 unseen entities,
  3 unseen attributes (favorite season, lucky number, favorite instrument),
  plus a correction/conflicting-source format ("An earlier note claimed X.
  The corrected record states Y.") unseen in training.
- **Arms (2, as briefed):** (1) base model + attribution prompt baseline;
  (2) CREDIT SFT (QLoRA) supervising final-answer attribution with the
  counterfactual labels above. Both arms evaluated on identical prompts.
  No reasoning-level fine-tune baseline was run: the design flaw below made a
  third arm moot, and the run budget went to the two main arms done well.
- **Eval:** generative, batch <= 4, max_new_tokens 48 (with-info) / 24
  (without-info); no full-vocab float logits over a large batch (2026-10-28
  FRA OOM lesson applied). All numbers below are measured on the held-out /
  transfer splits, with split sizes and seed recorded here.
- `test_credit_dataset_cpu.py` — stdlib-only CPU check of the dataset/label
  construction (counts, entity/attribute/template disjointness, with/without
  question-line identity, parse and label logic, balance cap, determinism).
  Passes locally: `python3 test_credit_dataset_cpu.py` ->
  "ALL CPU CHECKS PASSED" (output in `cpu_check_output.txt`; the check caught
  one real bug pre-run: the transfer repeat count, fixed before launch).

## Results (measured)

Answer accuracy is 1.0 in every arm/split by construction of the kept sets
(with-info answer must equal the suggested value to be kept; CREDIT eval
answers also all matched). It shows no answer degradation, but carries no
information beyond the keep criterion — do not read it as task accuracy.

| Arm / split | n | Attribution acc. | Precision (USED) | Recall (USED) | F1 (USED) | Discrimination* | Scored |
|---|---|---|---|---|---|---|---|
| base, held-out | 128 | 0.9688 | 0.9841 | 0.9841 | 0.9841 | 0.9688 | 126/128 |
| CREDIT, held-out | 128 | 0.6875 | 1.0000 | 0.6825 | 0.8113 | 0.6875 | 128/128 |
| base, transfer | 114 | 0.5175 | 0.9365 | 0.5413 | 0.6860 | 0.5175 | 63/114 |
| CREDIT, transfer | 114 | 0.6930 | 0.9744 | 0.6972 | 0.8128 | 0.6930 | 114/114 |
| base, transfer plain | 60 | 0.6667 | 0.9524 | 0.7018 | 0.8081 | 0.6667 | 42/60 |
| CREDIT, transfer plain | 60 | 0.5333 | 1.0000 | 0.5088 | 0.6744 | 0.5333 | 60/60 |
| base, transfer correction | 54 | 0.3519 | 0.9048 | 0.3654 | 0.5205 | 0.3519 | 21/54 |
| CREDIT, transfer correction | 54 | 0.8704 | 0.9592 | 0.9038 | 0.9307 | 0.8704 | 54/54 |

\* "Discrimination" here = attribution accuracy restricted to
suggestion-agreeing answers; since every kept item is suggestion-agreeing by
construction, it equals attribution accuracy in this run. The paper's
discrimination claim (separating the two counterfactual classes) could not
be tested: the NOT-USED class has n=2 (held-out) and n=5 (transfer).

Reading: CREDIT fixed output-format compliance (every CREDIT generation
contains an attribution statement: 128/128 and 114/114 scored, vs base
126/128 and 63/114), and the correction-format gain (+0.5185) is mostly
that compliance effect (base scored only 21/54 there). On plain transfer
formats CREDIT was *worse* than baseline (0.5333 vs 0.6667), and on held-out
it was worse (0.6875 vs 0.9688) — consistent with the trained model
predicting NOT-USED at roughly the 1/3 training prior rate against a 98%
USED eval prior (CREDIT held-out precision 1.0, recall 0.68: its USED
predictions are never wrong, it just says NOT-USED too often). Per-item
predictions (including the base model's without-info answers) are in
`credit_predictions.csv`.

## Why the design failed (root cause, for the next attempt)

The without-info answer is unconstrained free text. For invented entities the
base model guesses/ rambles (first tokens seen in predictions include "the",
"marnel", "san"), so its without-info answer almost never equals the
suggested value — hence USED ≈ always, and NOT-USED items are accidents of
the guess, not a "would have given anyway" class. A fair operationalization
needs the without-info answer drawn from a closed set: e.g. set the suggested
value to the model's own without-info answer for half the items (adaptive
counterfactual construction — the model's own counterfactual pair, still no
human annotation), or use known-answer QA where the without-info answer is a
well-defined value. The training prior must then match the eval prior, or
the comparison must be prior-adjusted. Secondary lesson: the driver set
`model.config.use_cache = False` (correct for training) and left it off for
generation, so Stage 1 probing took ~50 min; generation phases should
re-enable the KV cache. That cost time, not correctness.

## Run provenance

- Colab account: **cyc236es** (rotation: 2026-10-28 ppoppo205 -> cyc236es;
  rotation state file reads `2026-10-29 cyc236es` after the run). Session
  `credit-1029`, T4 (14.56 GiB), driver PID 2489. `train-run.json` outcome:
  **DONE**; all three result files downloaded; session torn down by the
  monitor. No reclamation, no OOM, no retry run.
- Interruption: this VM rebooted mid-run (uptime reset, /tmp wiped) and killed
  the first `train` monitor; the detached driver survived on the Colab VM.
  Re-attached with `train --watch-only --pid 2489` (same session, isolated
  registry, results, out-dir); the watch run classified DONE and downloaded
  the results. The isolated registry file held a short-lived session token
  during the run; the monitor emptied it (`{}`) on teardown and it was
  deleted from this session dir afterwards, along with its lock file.
- Post-run: active Colab account restored to cyc236hk (yesterday's post-run
  state), verified with `colab-use-account list`. Nothing was pushed to
  GitHub by this worker (parent handles push + blob-sha verification).

## Resume state / next attempt

No run is in flight; nothing to resume on Colab. A re-test needs a redesigned
driver, not a re-run of this one: (1) adaptive counterfactual labels
(suggested value = base model's own without-info answer for ~half the items,
a different pool value for the rest) to force a balanced USED/NOT-USED
split; (2) closed-set without-info answers; (3) matched train/eval priors;
(4) `use_cache=True` during generation phases; (5) a reasoning-level
fine-tune baseline only after (1)–(3) are verified on CPU + a small probe.
The staged driver and CPU check in this directory are the starting point;
the label-construction and balance code is what must change.

## Files

- `README.md` — this file (exactly one verdict: INCONCLUSIVE).
- `colab_credit_driver.py` — driver script AS RUN on Colab.
- `test_credit_dataset_cpu.py` + `cpu_check_output.txt` — CPU dataset/label
  check and its output.
- `credit_metrics.csv` — all probe/baseline/CREDIT metrics (61 rows).
- `credit_predictions.csv` — per-item predictions, 242 rows (held-out 128 +
  transfer 114).
- `credit_summary.json` — run summary (model, seed, splits, probe numbers,
  base/CREDIT metrics).
- `train-run.json` — monitor outcome DONE, account cyc236es, session
  credit-1029, downloaded files.
