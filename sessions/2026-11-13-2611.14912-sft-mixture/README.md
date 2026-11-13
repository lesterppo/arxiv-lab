# SFT Data Source and Mixture: Does Mixing Beat Single-Source? (arXiv:2611.14912)

- **Article:** arXiv [2611.14912](https://arxiv.org/abs/2611.14912) — "Rethinking 'RL Generalizes, SFT Memorizes': The Role of SFT Data" (submitted 11 Nov 2026)
- **Verdict: BLOCKED** — no SFT arm ever trained: the data-generation phase yielded too few correct self-generated rollouts and the driver crashed before training (see Results). Per the standing automatic-fallback rule the 8B fallback was the one allowed retry; it also blocked, so the track stops here with resume state recorded — no further automatic attempts today.
- **Track:** training track — Google Colab T4 via `hermes-colab-cli` (`colab.py train`)
- **Code:** no public code link in the abstract; driver written from the paper's §3–§4 design

## Why picked

- **AGENT 1 / TUNE 5.** The paper rethinks the RL-vs-SFT generalization gap through SFT *data*: a carefully mixed SFT dataset substantially outperforms data generated solely by a larger model, and matching the number of correct rollouts (or the total rollout budget) lets SFT generalize as well as — or better than — RL. Data composition is the cheapest lever in the daily fine-tune harness: if mixing beats single-source at matched N, every `soup-daily-finetune` run should mix.
- **Repo fit:** `soup-daily-finetune` (daily 8B NF4-QLoRA) — the mixture recipe (human + self-generated + format-diverse) drops straight into its data builder.
- **Testable within budget:** claim (1) is a pure SFT comparison — two arms, matched N/epochs/lr, one T4 session. Claims (2)–(3) (SFT-vs-RL matching, tail-method coverage) need a full RL arm and are out of reach in one session; they are explicitly not tested here.

## What was built

- `colab_sft_mixture_driver.py` — the 9B driver (Qwen/Qwen3.5-9B, NF4 QLoRA r=16/a=32, bf16, transformers<5.0, grad checkpointing, micro-batch 2, seq 512; the 2026-11-12 Agentic-OPD OOM fix applied):
  - Phase 0: base eval (greedy GSM8K test n=150 + SVAMP test n=150).
  - Phase GEN: sample G=4 rollouts (temp 0.8) from the base model on 350 GSM8K-train questions (rows 1000..1349), keep correct ones (cap 2/question, need 640).
  - Arm S (single-source): 480 self-generated correct rollouts only — the paper's "data generated solely by a [single] model" condition.
  - Arm M (mixed): 160 GSM8K human solutions + 160 equation-format solutions (calculator `<<...>>` expressions extracted into a compact chain + `####` answer) + 160 self-generated correct rollouts. All arms matched: N=480, all examples correct, 2 epochs, lr 2e-4.
  - Both arms trained sequentially (fresh model rebuild between arms, no cross-arm contamination), evaluated on the same held-out splits; metrics CSV + results JSON flushed to /content after every phase. Final stdout line `TRAIN_DONE`.
- `colab_sft_mixture_driver_8b.py` — identical code with MODEL_ID Qwen/Qwen3-8B (the automatic fallback driver).
- `test_sft_mixture_cpu.py` — deterministic CPU test, 21 checks: answer cleaning, gold extraction, equation-format transform, extract/numbers_equal, arm construction (matched sizes, disjoint sources, all-correct targets, short-pool error). 21/21 pass, offline, no model.

## Attempt 1 — Qwen3.5-9B: DIED (environment, not the model)

- Session `sft-mixture-1113`, account kyu008008, T4. Died ~2 minutes in at model load: `KeyError: 'qwen3_5'` — `ValueError: The checkpoint you are trying to load has model type 'qwen3_5' but Transformers does not recognize this architecture`.
- Root cause: the driver's `transformers>=4.46,<5.0` pin installs a 4.x transformers, which predates Qwen3.5 support. The 2026-11-05 merge-ranks driver pinned `transformers==5.20.2` and loaded Qwen3.5-9B fine (it later OOM'd at step 0 for a different, since-fixed reason). The 9B driver's pin has been corrected to `transformers==5.20.2` for a future retry.
- No training happened; no metrics exist for this attempt. `colab_out/train-run.json` records DIED with the log tail.

## Attempt 2 — Qwen3-8B (automatic fallback): DIED at generation (BLOCKED)

- Standing rule (2026-11-05): on a BLOCKED training run, one automatic fallback attempt with a different model — here Qwen3-8B, the proven training model — plus fixes for the failure cause. The failure cause was environmental (transformers pin), so the fallback driver keeps the same code and pins; the 9B driver is fixed separately for next time.
- Session `sft-mixture-1113-8b`, account kyu008008 (rotation: 2026-11-12 cyc236es → today kyu008008; token verified healthy before launch after yesterday's refresh failure), T4.
- A runtime restart SIGTERM'd the first monitor ~04:55 UTC; the VM driver survived detached (PID 546) and the monitor was re-attached with `train --watch-only --pid 546` (note: `--driver` is still required in watch-only mode). Mid-run downloader restarted and detached; it captured the metrics CSV + results JSON at 06:37 and stopped when the outcome landed.
- The session was torn down by the wrapper on DIED; no live sessions remain; active account stays kyu008008 per the rotation file.

## Results (8B fallback)

Base eval (greedy, before any training): **GSM8K test 42/150 = 0.2800; SVAMP test 46/150 = 0.3067** (Qwen3-8B base, minimal `"Question\nSolution:"` prompt — no few-shot, no chat template).

Generation phase (the failure): sampling G=4 at temp 0.8 on 350 GSM8K-train questions took **1h33m** (05:10 → 06:43 UTC) and yielded a correct rollout on only **225/350 questions; 323 kept** (cap 2/question) vs **640 needed**. The driver then raised `RuntimeError: self-gen pool short: have 323, need 640` in `build_arms` and died. No SFT arm was trained; no adapter exists; the 323 kept rollouts were held only in memory and are lost with the VM.

Why the yield was half the design assumption (~2.0 kept/question assumed, 0.92 measured): the base model at temp 0.8 with the minimal prompt frequently rambles past 160 tokens or never emits a parseable `#### N` answer, so most sampled rollouts fail the exact-number keep criterion. The generation was also far slower than estimated (93 min vs ~15–25 min): 4-bit sampling with batch 8 × 160 max tokens is the bottleneck.

## Why BLOCKED, not NEGATIVE

- No fair test of claim (1) was possible: the comparison needs both arms trained, and neither arm trained. The base-eval numbers above are reference measurements, not a verdict on the paper's claim.
- The failure is a design bug in the test (undersized generation pool + unpersisted rollouts), not evidence about mixed vs single-source SFT data. A retry with the fixes below is a different, fair test — it is not authorized by this run and needs no further action today.

## Resume state (for a future retry)

- Drivers: `colab_sft_mixture_driver.py` (9B, transformers pin fixed to `==5.20.2`) and `colab_sft_mixture_driver_8b.py` (8B, as-run).
- Session `sft-mixture-1113-8b` is torn down; registry `hidden_files/train-1113-registry.json`; train-run records `train-run-9b.json` (DIED, KeyError qwen3_5) and `train-run-8b.json` (DIED, pool short); downloaded metrics/results: `sft_mixture_metrics_8b.csv`, `sft_mixture_results_8b.json`.
- Fixes needed before a retry (recorded, not applied — the fallback was the one allowed attempt):
  1. **Generation pool:** 700+ questions at G=4 (or G=8 on 350) to clear 640 kept at the measured 0.92/question yield — or shrink the arms (e.g. N=320/arm needs 427 kept; still short of 323, so the pool must grow regardless).
  2. **Persist rollouts incrementally** to /content (JSONL) as sampling proceeds, so a crash keeps the generation work.
  3. **Raise the yield:** few-shot prompt with `#### N` format for the base model (standard for base-model sampling), and/or relax the keep criterion to accept a final-number match without `####`.
  4. **Budget the time:** generation alone took 93 min; a retry is a ~3h session (generation + 2 SFT arms + evals), so it needs a morning window, not an afternoon slot.

## Limits

- Claim (1) only: the SFT-vs-RL arms of claim (2) and the tail-method analysis of claim (3) are not tested; a mixed-vs-single-source win here supports the paper's data-source claim, not its full "SFT generalizes as well as RL" thesis.
- The paper's single-source condition is "data generated solely by a *larger* model"; the miniature's single source is the *same* base model (self-generated rollouts) — a faithful single-source condition, but the teacher is smaller than the paper's.
- The equation-format third source is a format transform of the human source, not an independent dataset — the mixture's diversity is 2.5 sources in spirit (human, model-generated, format-diverse), and the README says so.
- n=150 per eval split, single seed (123): the M-vs-S gap needs to clear sampling noise (~±4pp at n=150) to be called a reproduction.
- Qwen3-8B fallback, not the standing first-choice Qwen3.5-9B: the model differs from the first attempt, but the paper's claim is about data, and Qwen3-8B is in the paper's own model family (Qwen3).

## Files

- `colab_sft_mixture_driver.py` — 9B driver (pin fixed to transformers==5.20.2 after attempt 1)
- `colab_sft_mixture_driver_8b.py` — 8B fallback driver (as-run)
- `test_sft_mixture_cpu.py` — deterministic CPU test (21 checks, 21 passed)
- `sft_mixture_metrics_8b.csv` — base-eval metrics (2 rows)
- `sft_mixture_results_8b.json` — base evals + generation stats
- `train-run-9b.json` — attempt 1 record (DIED, KeyError qwen3_5)
- `train-run-8b.json` — attempt 2 record (DIED, self-gen pool short)
