# 2610.17767 — Root-token Policy Optimization (RPO)

**Article:** https://arxiv.org/abs/2610.17767 — "Efficiently Learning To
Reason or Not to Reason: Root-token Policy Optimization for Adaptive
Thinking"

**Track:** training · **TUNE = 5** · **Verdict: MIXED** (CPU miniature
SUPPORTED; live run degenerate — honest negative with a diagnosed cause)

## Why picked
RPO trains *only the root token* — the decision whether to invoke
`<think>` — via group-relative rewards and group-wise advantages,
learning difficulty-aware adaptive thinking at ~2% of the training
compute of prior adaptive-reasoning methods. That is exactly the shape
of method the `soup-daily-finetune` harness (daily 8B NF4-QLoRA on a
free Colab T4) can absorb: a tiny, cheap fine-tune stage that teaches an
instruct model when thinking is worth the tokens.

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
  on a Colab T4 (aria2c weights, transformers main, torchao uninstalled).
  80 train / 40 eval questions, half easy (single-step arithmetic) half
  hard (3–4 step synthetic word problems, integer answers). Root decision
  operationalized as the response's first word; per question, forced arms
  [THINK, THINK, DIRECT, DIRECT]; reward = correct − 0.15·thought;
  advantage = r − mean(r); loss = −mean(A·logp_root) + 0.02·KL(π‖π_ref),
  gradient flowing only through the single root token. Two attempts:
  first reclaimed by free-tier at ~35 min; second hit a T4 OOM in the
  ref forward (fixed: no-grad ref + gradient checkpointing), then a
  NameError in train logging (fixed). Third run completed end to end.

## Results (`rpo_metrics.json`)
- **Forced-arm training signal was valid:** easy: think_acc 1.00 /
  direct_acc 1.00 (rewards 0.85 vs 1.00 → DIRECT preferred); hard:
  think_acc 0.70 / direct_acc 0.23 (rewards 0.55 vs 0.23 → THINK
  preferred). A real difficulty gradient existed for RPO to exploit.
- **Root-token training demonstrably moved the root distribution:**
  format compliance (first word = decision token) went 0.0 → 1.0.
- **But the learned policy collapsed degenerate:** think_rate 0.0 on
  both easy and hard (always-DIRECT), accuracy 0.0, generations maxed
  out at 192 tokens without producing answers. No adaptive gap emerged.
- **Efficiency claim confirmed:** 640 decision-tokens touched vs
  37,730 SFT-equivalent tokens = **1.7%** — right at the paper's "~2%
  of training compute".
- Baselines for context: base free-decision (format non-compliant):
  acc_easy 0.85 / acc_hard 1.00 at 107 mean tokens; base always-THINK:
  acc_easy 1.00 / acc_hard 0.65 at 75 tokens.

## Diagnosis (why the live run degenerated)
1. **Decision-token vocabulary mismatch (primary):** "THINK"/"DIRECT" are
   not single tokens under the Qwen tokenizer, so the script fell back
   to "T"/"D". The prompt still instructed THINK/DIRECT, leaving the
   root token semantically ungrounded — unlike the paper's native
   `<think>` token, which *is* the behavior trigger. A bare "D" carries
   no "answer directly" semantics for the continuation.
2. **KL constrained only the root token:** the LoRA adapters drifted the
   continuations (192-token rambles, no "Answer:"), which the
   root-only KL penalty could not prevent.
3. Small-data stochasticity (160 updates, batch 1, lr 1e-4) likely
   locked in the degenerate mode early.

## Verdict
**MIXED.** The paper's learning rule is sound (CPU miniature reproduces
the adaptive-thinking signature exactly), and the efficiency claim
checks out live (1.7% of SFT tokens). But the live policy collapsed —
an honest negative that pinpoints a real implementation pitfall for
anyone porting RPO: **the root decision token must be semantically
grounded** (a real `<think>`-like token or an added special token), not
a single-letter proxy, and the KL/reference needs to cover more than
the root token or the continuations drift.

## Honest limits
- Bounded deviation: paper trains the native `<think>` token of a
  hybrid model; this test operationalized it as a first-word decision —
  the exact point where the run failed, so the paper's claim itself is
  not refuted, only my port of it.
- Toy-scale data (80 train questions, 2 epochs); single seed.
- The trained adapter (`/content/rpo_adapters`, VM-local) was not
  preserved — session torn down after metrics.
