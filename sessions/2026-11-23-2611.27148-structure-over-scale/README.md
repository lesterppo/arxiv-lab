# Session — 2026-11-23 — arXiv:2611.27148

## Article
**Structure Over Scale: Rethinking Adaptation for Reinforcement
Learning with Verifiable Rewards**
Allan Kazakov, Abdurrahman Javat.
https://arxiv.org/abs/2611.27148
Code: https://github.com/Shrimpstanot/structure-over-scale

## Why picked + repo fit
AGENT 2, TUNE 5. RLVR is usually framed as full fine-tuning versus
LoRA; the paper shows the missing variable is the geometry of the
update. Under a common cold-start GRPO protocol, LoRA exhibits rank
collapse — on DeepMath-Hard, accuracy drops from 73.1% at r=64 to
4.7% at r=128 — while DoRA and QuanTA stay stable and competitive
with full fine-tuning on <0.6% of parameters. The spectral signature
is sharp and falsifiable: collapsed LoRA concentrates its update
energy into few effective directions. This is directly relevant to
the daily soup fine-tune harness (which LoRA rank to use for GRPO)
and to the arxiv-lab training-track library. Testable on one Colab
T4 with the proven MCPO GRPO recipe.

## What was built
- `colab_structure_over_scale.py` — self-contained Colab driver
  (adapted from the proven MCPO driver, 2026-11-11): cold-start
  GRPO, DAPO-Math-17k train pool 512 / held-out eval 80, paired
  prompts, 16 steps/arm, 4 prompts/step, group 4, max_new 256,
  lr 1e-5, LoRA alpha = 2r, AdamW8bit, gradient checkpointing,
  chunked log-probs (no full [B, T, V] materialisation). Arms run
  sequentially with identical code paths, fresh base + fresh LoRA
  per arm: `r64`, `r128`, and time-gated `dora128` (DoRA r=128,
  runs only if <2h elapsed after the r128 eval). Per-arm
  diagnostics: greedy eval accuracy, per-step reward /
  frac-solved / grad-norm, and the update spectrum (effective rank
  + top-1 energy share of dW = B@A for q_proj/v_proj in 5 layers,
  computed exactly from r x r thin-SVD factors).
- `test_structure_cpu.py` — deterministic CPU test of the driver's
  pure helpers (answer extraction/equality, group advantages,
  spectral stats, the r x r singular-value trick verified against
  numpy direct SVD on 20 random matrices): ALL CHECKS PASSED.

## Model choice
Leg 1: Qwen/Qwen3-8B, NF4 QLoRA, bf16, transformers>=4.51,<5.0 —
the proven recipe. Qwen3.5-9B-Base was NOT used, by agent judgment
under the 2026-10-27 model rule ("feasible means: fits the T4
budget"): the 2026-11-21 Budget Internalization leg proved it loads
(PROBE_OK) but its Gated DeltaNet layers fall back to reference
PyTorch kernels (no causal_conv1d / flash-linear-attention), so
generation is pathologically slow — 400 greedy traces did not
finish in 50+ min. It does not feasibly fit this generation-heavy
GRPO test inside the reclaim window.
Leg 2 (fallback): Qwen/Qwen3-4B — the next smaller proven model,
per the automatic-fallback rule — on GSM8K (see Results).

## Results — Colab

### Leg 1 — Qwen3-8B on DAPO-Math-17k: DIED
Session `structure-1123` (ppoppo205, T4). Base eval **0/80**;
r64 steps 1–7 all mean_reward 0.0, frac_solved 0.0, grad_norm 0.0;
step 8 backward OOM (13.33 GiB allocated, +588 MiB needed).
Metrics: `structure_metrics_leg1_qwen38.csv`. Two failure causes:
(a) OOM in the backward pass; (b) a floor effect — the base model
solves none of the sampled DAPO problems, so GRPO has no gradient
signal at all (the same floor MCPO measured 2026-11-11: DAPO base
eval 0.0125 on Qwen3-8B, 0.0 on Qwen3-4B, train rewards 0.0).

### Leg 2 (fallback, automatic) — Qwen3-4B on GSM8K: DONE
Session `structure-1123b` (kyu008008 after ppoppo205/cyc236es hit
gpu-unavailable, T4). TRAIN_DONE printed; metrics in
`structure_fb_metrics.csv`, summary in `structure_fb_results.json`.
Cold-start GRPO, 16 steps/arm, 4 prompts/step, group 4, max_new
256, lr 1e-5, alpha=2r. Peak VRAM 7.00 GiB (r64) / 7.84 GiB (r128).

| Arm | Train reward (mean over 16 steps) | Grad norm (mean / max) | Eval acc (80 GSM8K test) |
|---|---|---|---|
| base | — | — | **0.2375** (19/80) |
| LoRA r64 | 0.125–0.625, mean ~0.30 | 0.34 / 0.53 | **0.4375** (35/80) |
| LoRA r128 | 0.06–0.69, mean ~0.40 | 11.26 / 26.54 | **0.6125** (49/80) |

Update spectrum (effective rank / top-1 energy share, q_proj and
v_proj averaged over layers 0/8/16/24/31):

| Arm | q_proj eff_rank | q_proj top-1 | v_proj eff_rank | v_proj top-1 |
|---|---|---|---|---|
| r64 | 41.73 | 0.269 | 27.88 | 0.444 |
| r128 | 60.02 | 0.358 | 38.46 | 0.474 |

The DoRA r128 arm was skipped on the time budget (120 min elapsed
≥ 2 h), so the paper's "DoRA stays stable" half is untested here.

## Verdict: NEGATIVE

The rank-collapse claim did not reproduce under a fair test at this
scale — it inverted. LoRA r=128 did not crash to near-zero accuracy;
it **outperformed** r=64 by 17.5 points (0.6125 vs 0.4375, base
0.2375), with healthy per-step rewards (0.50–0.69 in the last four
steps, frac_solved 0.75–1.00). Its update spectrum is also the
opposite of the paper's collapse signature: r128's update is *more*
distributed than r64's (effective rank 60.02 vs 41.73 on q_proj,
38.46 vs 27.88 on v_proj), not concentrated into few directions.
The one paper-consistent signal: r128 trains far more noisily —
grad norm jumps abruptly at step 5 (0.38 → 18.46) and stays 9–27
for the rest of the run, ~30x r64's 0.18–0.53 — but within 16 steps
that instability never became an accuracy collapse.

Stated limits: (1) 4B model on GSM8K, not the paper's 8B models on
DeepMath-Hard — the claim is probed at bounded scale, and a
collapse might still appear with longer training or harder problems
(the paper's protocol length is not stated); (2) 16 steps/arm only;
(3) the DoRA arm was not run, so the structured-adapter half of
the claim is untested; (4) one seed, one dataset split, no
replicates — the +17.5-point gap is large but n=80 evals carry
~±5-point binomial noise, which does not change the direction.
