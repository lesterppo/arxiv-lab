# 2026-11-05 — Model Merging Does Not Require Uniform LoRA Ranks (arXiv:2611.06486)

- Article: https://arxiv.org/abs/2611.06486 — "Model Merging Does Not Require
  Uniform LoRA Ranks"
- Track: training. Scores: AGENT=1, TUNE=5.
- Why picked: a fine-tune method directly adaptable to the daily QLoRA
  harness — per-domain LoRA experts merged by averaging, with the rank
  chosen per domain instead of forced uniform.
- Repo fit: soup-daily-finetune pattern (daily 8B-class NF4-QLoRA on
  Colab T4).

## What was built

`colab_merge_ranks_driver.py`: Qwen3.5-9B first (standing 9B-first rule),
NF4 QLoRA, alpha = 2r, target all-linear, lr 1e-4, 1 epoch,
completion-only loss, `remove_unused_columns=False` (2026-11-04 lesson).
Domains: math (GSM8K, 120 train / 60 test) and multiple-choice QA
(ARC-Challenge, 150 train / 60 test), deterministic parsing (numeric
answer / letter), 2-shot prompts disjoint from train/eval.
Experts: math r=8, math r=32, qa r=8, qa r=32. Merges are exact
rank-concatenation: averaged delta 0.5·delta1 + 0.5·delta2 with
delta_i = 2·B_i·A_i is reproduced by one adapter of rank r1+r2, scaling 1,
A = vstack([A1, A2]), B = hstack([B1, B2]). Uniform arm (8, 8); non-uniform
arm chosen from measured single-expert scores. Metrics are written
incrementally to `/content/merge_metrics.csv`.

## Results — VERDICT: BLOCKED

Colab session `train-1105-0423`, account `kyu009009` (rotation:
2026-11-04 kyu008008 → kyu009009), T4, isolated registry
`hidden_files/train-1105-registry.json`. Monitor outcome: **DIED**.

- Probe: Qwen/Qwen3.5-9B loaded first try, peak 8.03 GB — GREEN.
- Failure: CUDA out of memory at step 0/15 of the first expert
  (math_r8), in `torch_chunk_gated_delta_rule` (Qwen3.5's hybrid
  gated-delta layers) during the training forward pass: 14.55 GB in use
  on a 14.56 GB T4. Batch 2 × sequences up to 512 tokens with all-linear
  targets pushed training activations past the budget that the 8.03 GB
  weight-only probe suggested.
- Numbers banked: probe row only (`merge_metrics.csv`: probe,
  Qwen/Qwen3.5-9B, 8.03 GB). No expert trained, no eval numbers, no
  merged adapters. Nothing here supports or refutes the paper's claim.

## Resume state (exact)

- Session torn down: server reports no active sessions. Rotation state
  already stands at `2026-11-05 kyu009009`, so the next training day
  picks `cyc236ha`.
- Driver (staged, untested past the probe): `colab_merge_ranks_driver.py`
  in this session. Rerun with `colab.py train --driver <this file>
  --rotate --results /content/merge_metrics.csv
  /content/merge_details.csv --done-marker MERGE_DONE`.
- Fix order (floors: shrink the experiment, never the model):
  1. Cap training sequences at 256 tokens and drop the batch to 1 with
     gradient accumulation 8 (same effective batch), enable gradient
     checkpointing, and set
     `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.
  2. If the 9B training forward still OOMs, fall back to Qwen/Qwen3-8B
     (the proven training model) with the same driver and record the
     fallback in the session README.
- Do NOT rerun the driver unchanged: the OOM is deterministic at step 0
  for this configuration, not a transient reclamation.

## Honest limits

This day's training track produced no merging numbers. The BLOCKED
verdict is about this run's configuration, not about the paper: the
paper's 36-expert, 3-domain, 3-scale study was never within a one-day
T4 budget, and the bounded 4-expert version above remains untested
until the fixed driver runs.
