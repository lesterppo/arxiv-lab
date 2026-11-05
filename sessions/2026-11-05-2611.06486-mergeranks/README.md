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

## Fallback rerun (2026-11-05, automatic model fallback)

Run under Peter's 2026-11-05 standing rule: a BLOCKED training run gets
one automatic fallback attempt with a different agent-picked model plus
the shrink fixes for the failure cause. Colab session `merge-1105-fb`,
account `kyu009009` (`--account`, not `--rotate`; rotation untouched,
still `2026-11-05 kyu009009`), T4, isolated registry
`hidden_files/mergeranks-1105-rerun-registry.json`. Monitor outcome:
**RECLAIMED**. One attempt only — no further relaunch under the rule.

- Driver: `colab_merge_ranks_driver_fallback.py` — Qwen/Qwen3-8B (the
  9B OOM was in Qwen3.5's hybrid gated-delta layers, so a different
  proven architecture was the pick), training sequences capped at 256
  tokens, batch 1 with gradient accumulation 8, gradient checkpointing
  (+ `enable_input_require_grads`), `PYTORCH_CUDA_ALLOC_CONF=
  expandable_segments:True`. Everything else identical to the original
  driver. py_compile + import smoke test passed before launch.
- Progress before reclamation (far further than the first attempt):
  all 4 experts trained to completion — the shrink fixes + 8B switch
  PROVED past step 0 (qa_r32: 19/19 optimizer steps, ~20 s/it,
  train_loss 0.0359). Delta norms (the paper's mechanism diagnostic):
  math_r8 1.59, math_r32 2.85, qa_r8 1.78, qa_r32 2.94 — the r=32
  experts carry ~1.7x the delta magnitude of the r=8 experts on both
  domains, consistent with the paper's warning that magnitude-based
  selection drops the small expert. Base evals (Qwen3-8B, n=60 each):
  math 0.0167 (1/60), qa 0.10 (6/60).
- The session was reclaimed during the single-expert evals, after the
  base evals and before any expert-eval row. No expert evals, no merge
  evals, no uniform-vs-nonuniform comparison — C1 and C2 remain
  untested. **Verdict stays BLOCKED.**
- Data loss note: the wrapper downloaded nothing (`downloaded: []`) and
  produced no mid-run snapshots, so `/content/merge_metrics.csv` and
  `/content/merge_details.csv` died with the VM. The numbers above are
  recovered from the monitor's log tail in `train-run-fallback.json`
  (the only surviving record) and preserved as
  `merge_metrics_fallback_recovered.csv` (7 verifiable rows only: the
  qa_r32 train row, 4 delta-norm rows, 2 base-eval rows). Train losses
  for math_r8 / math_r32 / qa_r8 scrolled out of the log tail and are
  NOT recoverable. The 4 trained adapters (`/content/adapters/`) were
  never downloaded and are lost with the VM.
- Monitor note: the host-side monitor was killed by SIGTERM ~22 min in
  (runtime restart); the driver kept running detached on the VM and was
  re-attached with `train --watch-only --pid 673 --driver
  <fallback-driver>` (same session/registry/results/out-dir/
  done-marker) — no progress lost to the re-attach.

## Resume state (exact, updated)

- Session `merge-1105-fb` reclaimed; server reports no active sessions.
  Rotation state stands at `2026-11-05 kyu009009`, so the next training
  day picks `cyc236ha`.
- The fallback driver is PROVEN past training: all 4 experts complete
  in ~25 min of training; a future run needs ~1.5–2 h total including
  evals. Rerun with `colab.py train --driver
  colab_merge_ranks_driver_fallback.py --rotate --results
  /content/merge_metrics.csv /content/merge_details.csv --done-marker
  MERGE_DONE`. A future run must retrain all 4 experts from scratch
  (adapters lost) and should download adapters + metrics mid-run — this
  run produced no mid-run snapshots, so do not rely on the downloader
  alone; the incremental CSV writes only help if they are downloaded
  before reclamation.
- Active account note: kyu009009 was active at launch; by the end of
  the run the global active account read ppoppo205 (changed by another
  agent's live work while this run was in flight, not by this run —
  this run never switched accounts). Left as-is; do not treat
  ppoppo205 as this run's account.
