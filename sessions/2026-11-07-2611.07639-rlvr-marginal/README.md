# 2611.07639 — Measuring the Marginal Value of Verifier-Based RL Beyond SFT

**Article:** https://arxiv.org/abs/2611.07639
**Title:** "Measuring the Marginal Value of Verifier-Based RL Beyond Supervised Fine-Tuning"
**Track:** training · **Verdict: BLOCKED** (reclamation; GRPO never ran)

## Why picked

- TUNE 5 / AGENT 2. Directly testable claim about the SFT→RLVR pipeline:
  the paper reports verifier-based RL (GRPO) adds ~0 in-distribution over SFT
  and only a small out-of-distribution gain, i.e. SFT carries most of the gain.
- Repo fit: `soup-daily-finetune` — a daily 8B NF4-QLoRA harness on Colab T4.
  The paper's SFT→GRPO pipeline maps onto that harness directly, so a
  reproduced marginal-value curve would change what the daily harness runs.

## What was built

- `colab_rlvr_driver.py` (fallback driver v2) / `colab_rlvr_driver_9b.py`
  (attempt-1 driver): self-contained Colab T4 driver.
- Synthetic multi-step arithmetic word problems (the paper's own
  synthetic-benchmark approach): 6 template families, exact-integer answers,
  programmatic verifier. SFT: 400 gold CoT demos, 1 epoch, QLoRA r=16/α=32.
  Manual GRPO (plain torch, no TRL): 32 steps × 16 completions
  (4 prompts × 4 gens), β=0 (no ref model), lr 1e-5.
  Eval: base/SFT/GRPO on 200 in-distribution + 200 OOD items, greedy, 1-shot.
- `test_rlvr_task_cpu.py`: CPU test of the task generator, answer extractor,
  group advantage and PPO surrogate — passes.
- Deviations from the paper (recorded, not hidden): manual GRPO instead of
  TRL; β=0; 32 steps (paper 60); lr 1e-5 (paper 1e-6 full-model);
  arithmetic task instead of the paper's GPU-failure-diagnosis task.

## Attempt 1 — Qwen/Qwen3.5-9B, session `rlvr-1107` (account ppoppo205)

**DIED — CUDA OOM** in `torch_chunk_gated_delta_rule` (Qwen3.5 hybrid linear
attention) during the first training backward. Probe peak 7.72 GB was fine;
the OOM came from the chunked gated-delta-rule backward, not from model size.
Files: `colab_rlvr_driver_9b.py`, `rlvr_metrics_9b.csv`,
`rlvr_details_9b.csv`, `attempt_9b.json`, `train-launch-9b.json`.
(The 9B generations open with `<think>` — the 9B driver left Qwen3.5's
thinking mode on; the 8B driver v2 uses a direct instruction and the
generations are clean "Step 1…" format.)

## Attempt 2 — automatic fallback, Qwen/Qwen3-8B, session `rlvr-1107b`

Per the 2026-11-05 standing rule (one automatic fallback with a different
model + shrink fixes, same run, no fresh instruction): Qwen/Qwen3-8B, same
account ppoppo205, driver v2 fixes (1 short exemplar + direct instruction,
SFT batch 2 × grad-accum 2, TRAIN_SEQ_CAP 384, gradient checkpointing on for
training, `use_cache` toggled per phase, MAX_NEW_EVAL 256).
Outcome: **RECLAIMED** at ~07:10–07:14 UTC, during the SFT eval
(96/200 in-distribution rows written; OOD SFT eval, all of GRPO, and both
adapter zips never ran). Nothing could be downloaded from the reclaimed VM.
Files: `colab_rlvr_driver.py`, `attempt_8b.json`, `train-watch.json`
(outcome RECLAIMED), `rlvr_details_8b_partial.csv` (rescued by a mid-run
peek download at 07:05 UTC: header + 200 base rows + 96 SFT rows),
`recompute_8b_partial.json`. The 8B attempt's standalone metrics/details
CSVs were never downloaded — its numbers survive in `attempt_8b.json`,
`recompute_8b_partial.json`, and the log lines quoted in this README.

What completed on the fallback: probe peak 6.16 GB; base evals (peak 7.72 GB);
SFT training, mean loss 0.0172 over 400 demos (peak 8.22 GB).

## Results — recomputed, first-"Final:" extraction

The driver's scorer takes the **last** "Final:" in a generation. Base models
don't stop: they solve correctly, then keep generating new "Question:" blocks,
so the last "Final:" is the exemplar's answer (200) for every item and base
accuracy reads 0.00. Recomputing from the rescued details CSV with the
**first** "Final:", applied identically to every arm:

| arm | n | first-match acc | driver last-match acc |
|---|---|---|---|
| base, in-dist | 100 | **0.9000** | 0.0000 |
| base, OOD | 100 | **0.8200** | 0.4300 |
| SFT, in-dist (partial) | 96 | **1.0000** | 1.0000 |

- Base is already strong on this synthetic task (0.90 in / 0.82 OOD, first
  answer; 3 OOD parse-fails where no "Final:" appeared). SFT's gain is mostly
  format/stop behaviour: SFT generations are short, end cleanly after
  "Final:", and are correct on all 96 evaluated items.
- The task is near-ceiling for the base model, so GRPO would have had almost
  no in-distribution headroom — consistent with the paper's "marginal value"
  story, but **GRPO itself was never run, so the paper's claim is untested
  here, not confirmed**.

## Verdict: BLOCKED

Two Colab attempts, two different failure modes (OOM, then reclamation);
the GRPO comparison the claim rests on never executed. The SFT partial
numbers above are real measurements, not a verdict on the claim.

## Honest limits

- Synthetic arithmetic only; the paper's real task (GPU-failure diagnosis)
  was not tested.
- SFT in-distribution accuracy is n=96, not the planned n=200; OOD SFT
  accuracy is entirely missing.
- The near-ceiling base (0.90/0.82) is a property of this synthetic task, not
  of the paper's benchmark.
- First-match extraction is a post-hoc correction; it is applied identically
  to all arms and both numbers are reported, but the driver itself still
  scores last-match.

## Resume state

- Driver: `colab_rlvr_driver.py` (Qwen3-8B, v2) is ready to re-run as-is.
- SFT adapter zip was lost with the VM — a resume must re-run SFT (~1 h) as
  well as GRPO (~1 h) and evals (~40 min); budget ~2.5–3 h of T4 on a
  morning window, when free-tier sessions live longest.
- Rotation state: `2026-11-07 ppoppo205` — the next training run picks
  cyc236es. A retry needs Peter's fresh instruction (fallback already spent).
