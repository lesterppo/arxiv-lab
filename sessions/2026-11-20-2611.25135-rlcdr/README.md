# Calibrated Decision Rewards (RLCDR) — training track

- **Article:** arXiv:2611.25135 — "Calibrated Decision Rewards: Training
  Language Models with a Judge That Knows How Sure It Is"
- **Link:** https://arxiv.org/abs/2611.25135
- **Date tested:** 2026-11-20
- **Verdict: BLOCKED**

## Why picked

- **AGENT 2 / TUNE 5.** RLCDR is a reward-design method for RLVR: an LLM
  converted into a decision model answers a typed question about a
  response ("is the final answer correct?") with a probability
  distribution read in one forward pass, debiased over option orders
  and calibrated with ~100 labels. Training Qwen3-1.7B with GRPO on
  MATH, the reference-free reward raises MATH500 accuracy 72.2% →
  76.8% (+4.65), matching answer-key training (76.9%), with half the
  ECE of a softmax judge (0.074 vs 0.141) and no sign of reward hacking.
- **Repo fit:** directly a soup-daily-finetune GRPO experiment — the
  MCPO session (2611.14015) already proved the hand-rolled GRPO loop
  on this harness, so the only new piece is the reward function.
- **Testable on Colab T4:** three reward arms in one driver, same code
  path, fresh LoRA per arm.

## What was built

- `colab_rlcdr_driver.py`: self-contained Colab driver (pip installs at
  top, `!pip uninstall -y torchao`, prints `TRAIN_DONE`).
  Qwen3.5-9B-Base first (standing pick), NF4 QLoRA r=16/alpha=32, bf16,
  transformers>=4.51,<5.0; DAPO-Math-17k train pool 512 / held-out eval
  64; 8 steps/arm, 4 prompts/step, group 4, max_new 256, lr 1e-5.
  Arms: `key` (answer-key reward), `rlcdr` (debiased + temperature-
  calibrated decision reward), `softmax_judge` (raw single-order
  P(Yes)). Calibration phase (100 labeled samples: fit T on 50,
  ECE/AUC on 50) and a decision diagnostic (ECE/AUC/high-reward-wrong)
  after each arm. The paper's exact 1.7B arm is omitted (experiment
  shrink; the 9B class is the mandatory arm per the 2026-10-27 rule).
- `test_rlcdr_cpu.py`: CPU test of the driver's pure helpers
  (extraction, debias, temperature fit, ECE, AUC, decision prompts) —
  passes.
- `rlcdr_metrics_leg1_qwen35.csv`, `rlcdr_metrics_leg2_qwen38.csv`,
  `rlcdr_predictions_leg2_qwen38.jsonl`: what the two legs actually
  measured (see below).

## Results — no fair test was possible

**Leg 1 — Qwen3.5-9B-Base, cyc236ha, session `rlcdr-1120`: BLOCKED.**
Died in ~2 min at `load_policy()`: `transformers>=4.51,<5.0` does not
recognize model type `qwen3_5` (`ValueError: ... Transformers does not
recognize this architecture`). Nothing measured (metrics CSV header-only).
Root cause: Qwen3.5 needs transformers 5.x (Discernment leg 1 on
2026-11-19 proved it loads with transformers==5.20.2); this driver
carried the MCPO Qwen3-8B pin `<5.0`.

**Leg 2 (automatic fallback, standing rule) — Qwen3-8B, ppoppo205,
session `rlcdr-1120b`: BLOCKED.** Died after ~23 min with CUDA OOM in
the decision-model forward pass (`torch.OutOfMemoryError`, 13.54 GiB
allocated of 14.56 GiB). Base eval completed first: **0/64** greedy
accuracy on the held-out DAPO set — but 57/64 completions had no
extractable answer (`\boxed{}` / `Answer:` absent within 256 tokens),
so the 0.0 is a format artifact of greedy base-model generation, not a
capability floor (same failure mode as Discernment leg 1). The run died
in the calibration phase, before any training arm started: no
calibration temperature, no ECE/AUC, no arm comparison exists.
Root cause: `decision_p_yes` materializes the full
`[batch=16, seq~1024, vocab=151936]` logits tensor (~5 GiB bf16)
before slicing the last position — on top of the 9B NF4 weights and
training allocations, that OOMs a T4.

Per the standing rule the fallback blocking ends the day's training
track — no further automatic attempts.

## Resume state (for a retry)

- Driver: `colab_rlcdr_driver.py` (this session; CPU-tested pure
  helpers already green).
- Fixes needed before relaunch:
  1. **Decision forward:** compute last-position logits only —
     pass `logits_to_keep=1` / `num_logits_to_keep=1` to the model call
     (or slice hidden states before `lm_head`), and cut
     `DECISION_BATCH` 16 → 4 with truncation 1024 → 768. Also set
     `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.
  2. **Model pin:** for Qwen3.5-9B use `transformers==5.20.2`
     (proven 2026-11-19); for Qwen3-8B keep `transformers<5.0`.
  3. **Answer format:** append an explicit "Answer: $\boxed{}$"
     instruction / few-shot suffix to the DAPO prompts (or score the
     last number) so base eval is not 57/64 unparseable — the same
     Discernment-leg-1 lesson.
- Design difference from the paper (stated limit): the decision model
  is the policy itself in eval mode (self-judge), not a separate frozen
  LLM — a second 9B copy does not fit the T4 budget alongside
  training. Judge drift was to be measured by the per-arm diagnostic.
- Rotation state after leg 2: ppoppo205; next daily run picks cyc236es.
- Colab sessions `rlcdr-1120` / `rlcdr-1120b` were torn down by the
  wrapper on DIED.

## Files

- `README.md` (this file), `colab_rlcdr_driver.py`,
  `test_rlcdr_cpu.py`, `rlcdr_metrics_leg1_qwen35.csv`,
  `rlcdr_metrics_leg2_qwen38.csv`,
  `rlcdr_predictions_leg2_qwen38.jsonl`
