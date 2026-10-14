# 2610.17767 — Root-token Policy Optimization (RPO)

**Article:** https://arxiv.org/abs/2610.17767 — "Efficiently Learning To
Reason or Not to Reason: Root-token Policy Optimization for Adaptive
Thinking"

**Track:** training · **TUNE = 5** · **Status:** Colab run in progress

## Why picked
RPO trains *only the root token* — the decision whether to invoke
`<think>` — via group-relative rewards and group-wise advantages,
learning difficulty-aware adaptive thinking at ~2% of the training
compute of prior adaptive-reasoning methods. That is exactly the shape
of method the `soup-daily-finetune` harness (daily 8B NF4-QLoRA on a
free Colab T4) can absorb: a tiny, cheap fine-tune stage that teaches an
instruct model when thinking is worth the tokens. Nothing about the
method needs a cluster; the gradient touches one token per rollout.

## What was built
- `src/arxiv_lab/training/rpo.py` — the paper's learning rule in
  model-free form: `group_advantages` (A_i = r_i − mean(r)),
  `root_token_losses` (root-token policy gradient + Schulman KL to a
  reference), and `simulate_adaptive_thinking`, a seeded toy world where
  thinking helps only on hard items and costs a fixed penalty.
- `tests/test_training_rpo_cpu.py` — 7/7 pass. Toy world (seed 7):
  think_rate_easy ≈ 0.0, think_rate_hard = 1.0, adaptive gap ≈ 1.0,
  mean accuracy 0.7475 — the paper's signature emerges deterministically.
- `colab_rpo.py` — the live test: Qwen3.5-9B (NF4 QLoRA r=16/a=32, bf16)
  on a Colab T4. 80 train / 40 eval questions, half easy (single-step
  arithmetic) half hard (3–4 step synthetic word problems, integer
  answers). Root decision operationalized as the response's first word
  (THINK/DIRECT); per question, forced arms [THINK, THINK, DIRECT,
  DIRECT]; reward = correct − 0.15·thought; advantage = r − mean(r);
  loss = −mean(A·logp_root) + 0.02·KL(π‖π_ref), gradient flowing only
  through the single root token. Eval: free greedy decision on held-out
  items (base vs trained) + a base always-THINK ceiling.

## Results
*(pending — Colab run in progress; metrics land in `rpo_metrics.json`)*

## Honest limits
- Bounded deviation: the paper trains the native `<think>` token of a
  hybrid model; here the root token is the first response word under an
  explicit THINK/DIRECT instruction.
- Forced-prefix training arms are a scaffold; the eval decision is free.
- Toy-scale data (80 train questions, 2 epochs); a production version
  would want harder, more diverse difficulty gradations and a learned
  difficulty signal rather than a fixed think-cost.
- KL reference = same base with adapters disabled (a second 9B copy
  does not fit a T4).
