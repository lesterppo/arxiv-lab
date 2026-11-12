# Agentic-OPD (arXiv:2611.12799) — 2026-11-12 daily run, training track

- **Article:** [2611.12799](https://arxiv.org/abs/2611.12799) — "Agentic-OPD: Recovering Forgotten Agent Capabilities via On-Policy Distillation for Agentic Tasks"
- **Scores:** AGENT 4 / TUNE 5
- **Verdict: BLOCKED** — no fair test was possible; the paper's claim was never tested.

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

### Base evals (banked — the only numbers that exist)

| split | n | correct | acc |
|-------|---|---------|-----|
| A_val (GSM8K test 0–60) | 60 | 11 | 0.1833 |
| A_test (GSM8K test 100–160) | 60 | 13 | 0.2167 |
| B_test (SVAMP test 0–60) | 60 | 16 | 0.2667 |

No SFT stage ran. No checkpoint pool exists. HCN, LAD, and Stage C never
executed, so the paper's claim (checkpoint-pool OPD recovers forgotten
capabilities) was never tested.

### Attempt history (reconstructed from the worker's exec logs + train-run.json)

| session | account | outcome | what happened |
|---------|---------|---------|---------------|
| `agentic-opd-1112` | cyc236es | DIED | Died in PHASE 0 base eval: `ValueError: could not convert string to float: ''` in `extract_number` (driver bug; the current driver guards both `float()` branches with try/except, verified by `test_agentic_opd_fallback_cpu.py`). No files banked. |
| `agentic-opd-1112b/c/d` | — | unknown | No records survive anywhere (no train-run files, nothing in the run log). Do not invent an attempt history for these. |
| `agentic-opd-1112e` | cyc236es | DIED | Died after the base evals completed: `torch.OutOfMemoryError` in bitsandbytes' 4-bit matmul (`_dequant_linear_fallback`, `ops.py:914`) — tried to allocate 96.00 MiB; GPU 0 has 14.56 GiB total, 14.47 GiB in use (14.19 GiB allocated by PyTorch). Base evals downloaded (metrics CSV + results JSON). |
| `agentic-opd-1112f` | ppoppo205 (cyc236es gpu-unavailable at creation) | unrecorded | `train-run.json` has `outcome: null` — the wrapper never recorded a final outcome. No active sessions remain; no further files were banked. The fixed driver (gradient checkpointing + micro-batch 2, "OOM fix (2026-11-12 run e)") was written 05:37 UTC, before f's last execution 05:41 UTC, so f presumably ran the fixed driver — but its fate (OOM again, reclamation, monitor loss) is unrecorded and unknown. |

### OOM cause (run e)

bitsandbytes' fast 4-bit GEMM path fell back to `_dequant_linear_fallback`,
which materializes a whole dequantized bf16 weight matrix (`B_dq =
torch.empty(shapeB)`) per layer. Without gradient checkpointing, autograd
retained those dequantized weights across all layers for backward (~14 GiB),
so a single further 96 MiB allocation failed at 14.47 GiB in use. The
current 8B driver already carries the fix (gradient checkpointing +
micro-batch 2 x accum 2); the 4B fallback below halves the model on top of it.

## Honest limits

- The verdict is BLOCKED, not a negative: nothing about HCN, LAD, or the
  recovery claim was measured.
- The base evals are zero-shot greedy on n=60 splits — a calibration
  baseline only, not a model comparison.
- Attempts b–d are unrecorded; the sequence above is the complete surviving
  record, recovered from the worker's exec logs and the session's
  `train-run.json`.
- The `agentic-opd-1112` registry file in `hidden_files/` contains a live
  Colab session token and must never be pushed or copied anywhere.

## Resume state

- **Driver (8B):** `colab_agentic_opd_driver.py` — carries both fixes
  (guarded `extract_number`, gradient checkpointing + micro-batch 2).
- **Banked files:** `agentic_opd_metrics.csv` (base evals only),
  `agentic_opd_results.json` (base evals only, `phases: ["base_eval"]`),
  `train-run.json` (attempt f record, outcome null).
- **A retry needs:** the automatic-fallback driver below (staged, not
  launched) or the 8B driver re-run with the same fixes; either way a
  morning window is recommended — the 8B run died in the early-afternoon
  window after the base evals.

## Automatic fallback (staged, not launched)

Under Peter's 2026-11-05 automatic-fallback rule, the fallback for a
BLOCKED training run is staged here: `colab_agentic_opd_driver_4b.py` —
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
