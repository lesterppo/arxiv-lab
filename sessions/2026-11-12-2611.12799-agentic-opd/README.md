# Agentic-OPD (arXiv:2611.12799) — 2026-11-12 daily run, training track

- **Article:** [2611.12799](https://arxiv.org/abs/2611.12799) — "Agentic-OPD: Recovering Forgotten Agent Capabilities via On-Policy Distillation for Agentic Tasks"
- **Scores:** AGENT 4 / TUNE 5
- **Verdict: PENDING — run f is still alive (see "Attempt history").** No
  final verdict can be recorded yet: attempts through e would be BLOCKED,
  but the live retry (f) is in Stage A SFT as of 05:56 UTC 2026-11-12.
  Update this line when f finishes (or blocks).

> **Correction (05:56 UTC 2026-11-12):** an earlier version of this README
> recorded attempt f as dead with an unknown cause, following a stale
> brief. A live check just now proved that wrong: session
> `agentic-opd-1112f` is alive (status IDLE, endpoint
> `gpu-t4-s-usw4b0-jdlbh9y6cgqa`, driver PID 1903), its log tail shows the
> base evals completing and Stage A starting at 05:55:37 UTC, and the
> worker's "still running" report was correct. The `train-run.json`
> `outcome: null` only means the wrapper's monitor never finalized — the
> VM driver continues detached (the known monitor-SIGTERM pattern).

## Why picked

The paper's mechanism: in a training run the best checkpoint fails on
problems earlier checkpoints solved; the union of solved problems across
checkpoints far exceeds any single one. Agentic-OPD treats the
best-validation checkpoint as student and all saved checkpoints as frozen
teacher candidates — HCN (Highest Coverage Next, Eq. 7–9) selects teachers by
weighted coverage of the student's unsolved problems, and LAD
(Locality-Aware Distillation, Eq. 10–13) confines transfer to failed student
rollouts on problems where the teacher performs better.

**Repo fit:** `soup-daily-finetune` produces a checkpoint pool daily — a
direct home for a checkpoint-pool method. Bounded variant: an SFT-stage
checkpoint pool (see "Deliberate deviation"), which avoids the full-GRPO
prerequisite that blocked this pick yesterday.

## What was built

`colab_agentic_opd_driver.py` — Qwen/Qwen3-8B, NF4 QLoRA r=16/alpha=32,
bf16, transformers<5.0, no `prepare_model_for_kbit_training`,
expandable_segments. Phases:

- **Phase 0 — base eval** (greedy, zero-shot): GSM8K test rows 0–60
  (`A_val`), rows 100–160 (`A_test`), SVAMP test rows 0–60 (`B_test`),
  n=60 each.
- **Stage A:** SFT base → C1 on task A (GSM8K CoT, 400 ex).
- **Stage B:** continue SFT C1 → C2 on task B (SVAMP answer-only, 400 ex),
  a different distribution + conflicting short-answer format that induces
  forgetting of A.
- **Pool observation:** union coverage on A_val / A_test (C1 ∪ C2 vs C2).
- **HCN:** teacher selection (base vs C1) on A_val, student C2.
- **LAD:** G=4 rollouts, student C2 vs selected teacher, on a 100-problem
  GSM8K pool (train rows 400–500, disjoint from stage A); admit problems
  where the teacher beats the student.
- **Stage C:** OPD distillation — SFT on the teacher's correct sampled
  rollout text for LAD-admitted problems (completion tokens only); eval
  C3 vs C2 for recovery on A and retention on B.

**Deliberate deviation (documented limit):** the paper's prerequisite is a
full GRPO run producing the checkpoint pool — too heavy for one free-T4
session. This test builds the pool with sequential SFT instead. LAD's
turn-level top-k KL (Eq. 13) is adapted to SFT on the teacher's correct
sampled rollout text — a bounded, stated simplification of the KL objective.

Metrics CSV + results JSON are flushed to `/content` after EVERY phase
(reclamation-safe). Final stdout line is `TRAIN_DONE`.

## Results

### Base evals — run e (banked files on disk)

| split | n | correct | acc |
|-------|---|---------|-----|
| A_val (GSM8K test 0–60) | 60 | 11 | 0.1833 |
| A_test (GSM8K test 100–160) | 60 | 13 | 0.2167 |
| B_test (SVAMP test 0–60) | 60 | 16 | 0.2667 |

### Base evals — run f (live, observed via the VM log tail at 05:55 UTC)

| split | n | correct | acc |
|-------|---|---------|-----|
| A_val | 60 | 10 | 0.1667 |
| A_test | 60 | 15 | 0.2500 |
| B_test | 60 | 16 | 0.2667 |

Same eval sets and greedy decoding; the one-case flips on A_val/A_test
are bnb nondeterminism between runs, not a code change. The files on disk
are now run f's numbers too — downloaded 05:58 UTC by the worker's
mid-run downloader (host PID 1908), which is still active and re-downloads
every 10 min, so the local files (and the remote copies) are a snapshot
that will keep updating until f finishes; re-push at completion.

No SFT stage has completed in any attempt yet. No checkpoint pool exists.
HCN, LAD, and Stage C have never executed, so the paper's claim
(checkpoint-pool OPD recovers forgotten capabilities) has not been tested
yet.

### Attempt history (reconstructed from the worker's exec logs + train-run.json)

| session | account | outcome | what happened |
|---------|---------|---------|---------------|
| `agentic-opd-1112` | cyc236es | DIED | Died in PHASE 0 base eval: `ValueError: could not convert string to float: ''` in `extract_number` (driver bug; the current driver guards both `float()` branches with try/except, verified by `test_agentic_opd_fallback_cpu.py`). No files banked. |
| `agentic-opd-1112b/c/d` | — | unknown | No records survive anywhere (no train-run files, nothing in the run log). Do not invent an attempt history for these. |
| `agentic-opd-1112e` | cyc236es | DIED | Died after the base evals completed: `torch.OutOfMemoryError` in bitsandbytes' 4-bit matmul (`_dequant_linear_fallback`, `ops.py:914`) — tried to allocate 96.00 MiB; GPU 0 has 14.56 GiB total, 14.47 GiB in use (14.19 GiB allocated by PyTorch). Base evals downloaded (metrics CSV + results JSON). |
| `agentic-opd-1112f` | ppoppo205 (cyc236es gpu-unavailable at creation) | **ALIVE — still running** | `train-run.json` has `outcome: null` because the wrapper's monitor never finalized; the VM driver (PID 1903) continues detached. Live check 05:56 UTC: session status IDLE, log tail shows base evals done (A_val 10/60, A_test 15/60, B_test 16/60) and **PHASE A (Stage A SFT) started 05:55:37 UTC**. The fixed driver (gradient checkpointing + micro-batch 2) was in place before f launched. The worker's "still running" report was correct — an earlier brief that called f dead was wrong. |

### OOM cause (run e)

bitsandbytes' fast 4-bit GEMM path fell back to `_dequant_linear_fallback`,
which materializes a whole dequantized bf16 weight matrix (`B_dq =
torch.empty(shapeB)`) per layer. Without gradient checkpointing, autograd
retained those dequantized weights across all layers for backward (~14 GiB),
so a single further 96 MiB allocation failed at 14.47 GiB in use. The
current 8B driver already carries the fix (gradient checkpointing +
micro-batch 2 x accum 2); the 4B fallback below halves the model on top of it.

## Honest limits

- The paper's claim has not been tested yet in any attempt: HCN, LAD, and
  the recovery measurement have never run.
- The base evals are zero-shot greedy on n=60 splits — a calibration
  baseline only, not a model comparison. Runs e and f disagree by one case
  on A_val/A_test (bnb nondeterminism).
- Attempts b–d are unrecorded; the sequence above is the complete surviving
  record, recovered from the worker's exec logs, the session's
  `train-run.json`, and a live log check at 05:56 UTC.
- The `agentic-opd-1112` registry file in `hidden_files/` contains a live
  Colab session token and must never be pushed or copied anywhere.

## Resume state

- **Driver (8B):** `colab_agentic_opd_driver.py` — carries both fixes
  (guarded `extract_number`, gradient checkpointing + micro-batch 2).
- **Banked files (run f's numbers, downloaded 05:58 UTC by the worker's
  mid-run downloader):** `agentic_opd_metrics.csv`,
  `agentic_opd_results.json` (base evals only so far,
  `phases: ["base_eval"]`), `train-run.json` (attempt f record, outcome
  null — the run hasn't finished yet). Run e's numbers were 11/13/16.
- **Live run f (the current attempt):** session `agentic-opd-1112f`,
  account ppoppo205, endpoint `gpu-t4-s-usw4b0-jdlbh9y6cgqa`, driver PID
  1903, in Stage A SFT as of 05:56 UTC. The worker's host-side monitor
  (PID 1898, `colab.py train ... --monitor-timeout 14400`) and its
  mid-run downloader loop (PID 1908, every 10 min) are **still alive** —
  the monitor will finalize `train-run.json` and the final download when
  f finishes. **Do NOT attach a second monitor** and do not stop these
  processes. When f finishes, re-push the final metrics/results (they
  will have been overwritten in place by the downloader) and record the
  final verdict then.
- **A retry needs:** if f blocks, the automatic-fallback driver below
  (staged, not launched); a morning window is recommended.

## Automatic fallback (staged, not launched)

Under Peter's 2026-11-05 automatic-fallback rule, the fallback is staged
here for use **if the live run f blocks** (not launched; no action needed
while f is running): `colab_agentic_opd_driver_4b.py` —
Qwen/Qwen3-4B, same experiment (data, stages, evals, seeds, LoRA r=16/a=32)
plus memory-shrink fixes for the run-e OOM cause:

- 4B model (halves the dequantized-weight footprint vs 8B);
- micro-batch 1 x accum 4 (effective batch still 4);
- seq cap 512 → 384;
- eval batch 24 → 8, sample batch 8 → 4;
- gradient checkpointing retained from the fixed 8B driver.

`test_agentic_opd_fallback_cpu.py` verifies the pure helpers (incl. the
attempt-1 `ValueError` failure mode) and that the 4B driver differs from
the 8B driver only in the intended shrink set: **62/62 checks pass**.

### Launch command (needs Peter's fresh instruction — do NOT auto-launch)

Run from this session dir. Use the rotation account at launch time (the
2026-11-12 rotation account was cyc236es — do not hard-code a stale one);
never touch cyc236de / cyc236hk.

```bash
python3 ~/workspace/hermes-colab-cli/colab.py train \
  --driver colab_agentic_opd_driver_4b.py \
  --account <rotation account at launch time> \
  --session agentic-opd-1112-4b --gpu T4 --monitor-timeout 14400 \
  --done-marker TRAIN_DONE \
  --results /content/agentic_opd_metrics.csv /content/agentic_opd_results.json \
  --out-dir ~/workspace/goals/arxiv-daily-read-and-test/hidden_files/agentic-opd-1112-4b \
  --registry-file ~/workspace/goals/arxiv-daily-read-and-test/hidden_files/agentic-opd-1112-4b-registry.json \
  --json
```
