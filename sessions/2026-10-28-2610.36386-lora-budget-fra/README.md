# 2026-10-28 — LoRA on a Budget: Function-Aware Post-Hoc Compression (FRA)

- Article: arXiv 2610.36386, https://arxiv.org/abs/2610.36386
- Track: training (AGENT=1, TUNE=5). Repo fit: `soup-daily-finetune`
  daily NF4-QLoRA harness — a post-hoc adapter-compression method that
  decides per-module rank from measured functional sensitivity is directly
  adaptable to a daily-LoRA fleet: it answers "how far can this adapter be
  shrunk before it stops working" without retraining.
- Why picked: the paper's core claim is sharply testable on one T4 —
  spectral reconstruction alone does not predict functional compression
  tolerance; isolated module truncations measured end-to-end (FRA) beat
  uniform and spectral-only rank allocation at the same total rank budget.
  Across 78 adapters / three fleets in the paper; here, directionally, on
  one adapter and one synthetic task.

## Verdict: BLOCKED

No fair test was possible in this run: the driver died with a CUDA
out-of-memory error inside the *base-model* eval, before any training
ran. Zero measurements were produced — no base accuracy, no trained
adapter, no sensitivity table, no allocation comparison. The claim is
untested, not refuted. Scripts are staged below with the exact resume
state; the failure is a fixable eval-batching bug, not a Colab or model
problem (see "What happened" and "Resume state").

## What was built

`colab_fra_driver.py` — self-contained Colab driver (pip installs at top,
`torchao` uninstalled right after, per the peft/Colab rule):

- **Model:** Qwen/Qwen3-8B (BASE), NF4 QLoRA, r=16, alpha=32, bf16,
  `transformers<5.0`, all 7 linears. Qwen3-8B was used directly, per the
  task brief: Qwen3.5-9B was already tried first on 2026-10-27 and FAILED
  twice (Qwen/Qwen3.5-9B-Instruct is not a valid HF identifier;
  Qwen/Qwen3.5-9B has model type `qwen3_5`, unrecognised by
  transformers<5.0). Qwen3.5-9B was NOT retried. The paper has no
  paper-specific small model, so the paper-fit exception does not apply.
- **Task (synthetic, deterministic):** sector classification of 32
  invented firms (8 each: Technology / Healthcare / Energy / Finance)
  with neutral descriptions (staff counts, offices, vehicles — no
  sector-revealing words). 16 training templates x 32 firms = 512
  training items; 6 HELD-OUT templates (unseen phrasings, same firms) x
  32 = 192 held-out items. Option order (A–D permutation) varies per
  template so the answer letter cannot be memorised by position.
- **Eval (as designed, never executed):** loglikelihood of the answer
  letter — sum of log-probs of the " A"/" B"/" C"/" D" completion given
  the prompt, argmax over the 4 letters. No free generation. Identical
  eval code and seeds for every arm.
- **Planned training:** 5 epochs, batch 8, lr 2e-4, AdamW, prompt-masked
  loss (labels -100 on the prompt), 320 steps. Target: full-rank
  held-out accuracy >= ~0.75 (vs 0.25 chance); if not reached, the
  comparison was to be reported as not-learned, honestly.
- **Planned FRA measurements:** isolated truncation of each of 35
  sampled modules (layers 0, 9, 18, 27, 35 x q/k/v/o/gate/up/down) to
  rank 4, rest full, recording held-out accuracy drop; plus per-module
  spectral energy at rank 4 for the spectral-vs-functional comparison.
- **Planned allocations at one fixed budget (25% of full rank, 1008
  total ranks over 252 modules, average rank 4):** (a) FRA/sensitivity —
  sampled modules tiered by measured drop (top 12 -> rank 8, next 11 ->
  rank 4, bottom 12 -> rank 0; unsampled -> rank 4); (b) uniform rank 4
  everywhere; (c) spectral-only — greedy by next singular value until
  the budget is spent. Predicted ordering (paper): FRA >= uniform,
  FRA >= spectral.
- **Truncation machinery (in the driver, unverified on GPU):** rank-k
  truncation via thin QR + a 16x16 SVD (delta = scaling * B@A has rank
  <= 16, so the small SVD gives the exact singular values and factors;
  zero-padded back to r=16 shapes so peft shapes never change). A
  reconstruction-identity check (truncation error^2 == discarded
  singular-value energy^2) is computed for the first module and logged.
- `test_fra_allocation_cpu.py` — stdlib-only CPU check of the allocation
  arithmetic (budget equality FRA vs uniform, spectral-greedy cap/budget,
  tier ordering). Passes locally: `python3 test_fra_allocation_cpu.py`
  -> "ALL CPU CHECKS PASSED".

## What happened

- `colab.py train --rotate` ran session `fra-1028` on account **ppoppo205**
  (rotation state: 2026-10-27 cyc236ha -> ppoppo205), T4, monitor budget
  14400 s. Outcome in `train-out/train-run.json`: **DIED**, driver PID 801.
- The driver crashed with `torch.OutOfMemoryError` in `eval_accuracy`,
  at `.logits.float()` on the lm_head output, during the BASE eval —
  before training started. Evidence: (1) the traceback is in the eval
  forward; (2) `downloaded: []` — `fra_metrics.csv` is written right
  after the base eval and did not exist at teardown, so the base eval
  never completed; (3) total run time ~8.5 min is too short for model
  download + 320 training steps + later evals.
- Root cause: eval batching materialises the full vocabulary logits.
  The eval builds 4 candidate sequences per item and forwards them in
  batches of 32 sequences; `.logits.float()` plus `log_softmax` on a
  [32, ~130, 151936] tensor costs ~2.5 GiB + ~2.5 GiB on top of the
  model and activations, and the T4 (14.56 GiB) OOMs ("Tried to allocate
  1.85 GiB ... 12.89 GiB in use"). The training loop itself was never
  reached, so this says nothing about whether the adapter would learn
  or whether FRA beats the baselines.
- No metrics CSV and no summary JSON were produced (nothing to download).
  The VM log beyond the tail in `train-run.json` is gone: the session was
  torn down by the wrapper and no longer appears in the session list.

## Results

None measured. No numbers are reported for base / full-adapter / FRA /
uniform / spectral accuracy, trainable params, or peak memory — the run
died before the first measurement. Any number here would be invented.

## Honest limits

- The task design is unproven: the memorised-mapping task may or may not
  reach >= 0.75 held-out accuracy under 5 epochs of r=16 QLoRA, and the
  35-module FRA sample covers only 5 of 36 layers (unsampled modules
  default to uniform rank 4 in the FRA arm, so the FRA arm is a
  partial-information FRA). These limits were accepted in the design;
  they were never tested.
- The QR+16x16 truncation code and its reconstruction check never ran on
  a GPU; the CPU check covers allocation arithmetic only, not the
  truncation factors.
- One run only, per the run rules: no retry was started after DIED.

## Resume state (for a future run — do not re-run the driver as-is)

- Session name: `fra-1028` (torn down). Account that ran: ppoppo205.
  Rotation state file already records `2026-10-28 ppoppo205`; the next
  day's rotation must pick cyc236es next, NOT ppoppo205 again.
- Driver: `colab_fra_driver.py` in this directory (as-run, unmodified).
- **Required fix before any re-run** (eval batching only; training and
  allocation code untouched): in `eval_accuracy`, reduce the forward
  batch from 32 sequences to 8, and do not call `.float()` /
  `log_softmax` on the full logits tensor. Instead, for each sequence,
  slice the 1–2 logits rows at the completion positions first, convert
  only those rows to float32, take `log_softmax` over the vocab dim of
  those rows, and delete the full logits tensor before the next batch.
  Only the completion-position logits are ever needed (the completion
  is 1 token: " A"/" B"/" C"/" D"). This cuts the eval's logits memory
  roughly 4x from batching and ~100x from position slicing.
- Re-run command shape (fresh run, new session name, same flags):
  `python3 ~/workspace/hermes-colab-cli/colab.py train --driver
  <fixed-driver> --rotate --results /content/fra_metrics.csv
  /content/fra_summary.json --out-dir <session-dir>/train-out
  --registry-file <session-dir>/registry.json --poll 120
  --monitor-timeout 14400 --json`. Re-run only on a fresh instruction;
  this run's one-run budget is spent.
- Expected healthy-run signature: `DEPS_OK`, `MODEL_OK Qwen/Qwen3-8B`,
  `RESULT base heldout_acc=...`, 320 training steps with falling loss,
  `RESULT full heldout_acc=...`, 35 `sensitivity` rows, three allocation
  `heldout_acc` rows, then `TRAIN_DONE` as the last line.

## Files

- `README.md` — this file.
- `colab_fra_driver.py` — Colab driver as run (contains the OOM bug
  described above; see Resume state before reusing).
- `test_fra_allocation_cpu.py` — stdlib-only allocation-arithmetic check
  (passes locally).
- `train-out/train-run.json` — wrapper outcome record (DIED, ppoppo205,
  session fra-1028, PID 801, OOM log tail). Copy also at
  `train-run.json` in this directory.
- `registry.json` — isolated Colab session registry used by the run.
