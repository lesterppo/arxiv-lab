# Session 2026-10-04 — Finetuning with Sampling (arXiv:2610.02140)

**Article:** "Finetuning with Sampling: SFT Learns Better Than You Think"
([abs](https://arxiv.org/abs/2610.02140)) — Karan, Chen, Du (submitted 1 Oct 2026).

**Why picked:** the most prospective *training* article in the latest cs.AI
window for a live test. Crisp, falsifiable mechanism: don't change the
learning objective — reshape the DATA. An MCMC sampler progressively
transforms off-policy expert traces to be more on-policy for a reference
model; plain SFT on the reshaped data then rivals RL (better
generalization, less forgetting, learning beyond mere sharpening).

**Paper claim (abstract):** the sampling algorithm "enables SFT to rival
prevailing posttraining techniques, often generalizing better and
forgetting less than strong on-policy baselines", and the finetuned models
"exhibit strong distributional performance" beyond sharpening the base
distribution.

## What was built

- `data_gen.py` — toy reasoning task (two-step arithmetic, verifiable
  `\boxed{}` answers), off-policy expert traces in a rigid "PROTOCOL 7"
  bracketed style the base model would never naturally produce (correct
  by construction), held-out test set (n=30), forgetting probes (20
  general-QA items + 30 general sentences for an NLL probe). Stdlib only.
- `mcmc.py` — simplified Metropolis reshaping: per-prompt candidate pool
  (expert trace + 6 reference-model samples @ temp 1.0), target
  `pi(x) ~ exp(logp_ref(x)/T) * 1[correct(x)]` with T=1.0, uniform
  proposals, correctness-gated acceptance, 40 steps/chain. Final chain
  state = reshaped trace. CPU-tested with synthetic scores (5 unit
  checks pass).
- `colab_sft_sampling.py` — live Colab T4 driver: Qwen2.5-0.5B-Instruct
  (bf16) as reference; 40 train prompts; mean-token-logp scoring;
  then A/B — **arm A**: plain SFT (LoRA r=16, lr 2e-5, 3 epochs,
  batch 4, 30 steps) on raw expert traces vs **arm B**: identical SFT on
  MCMC-reshaped traces. Evals per arm + base: held-out greedy accuracy,
  pass@1/pass@4 (K=4 @ temp 0.8), mean token entropy (sharpness),
  QA accuracy on base-passing items + general-text NLL (forgetting).
  Supports `--resume` from a pools checkpoint
  (`/content/sft_sampling_pools.json`).
- `diag.py` — diagnostic used to investigate a measurement anomaly
  (see Honest limits §5).

## Results

**MCMC reshaping — WORKED AS DESIGNED.** 40/40 chains moved off the
expert trace (mean accept rate 0.58); mean reference logp(trace|question):
expert **−2.83** → reshaped **−0.23** (Δ **+2.60 nats/token**). Base-sample
correct rate 210/240. The reshaped set is fluent, correct, model-native
traces ("To solve the problem, we need to…") vs the rigid PROTOCOL-7
expert style. The paper's data-shaping mechanism is faithfully reproduced
at this scale.

**SFT A/B — does NOT reproduce the paper's claimed advantage:**

| metric | base | arm A: expert SFT | arm B: reshaped SFT |
|---|---|---|---|
| held-out acc, greedy (n=30) | 0.700 | **0.800** (+0.100) | 0.733 (+0.033) |
| pass@4 (temp 0.8) | 1.000 | 0.933 | 0.900 |
| mean token entropy | 0.203 | 0.241 | 0.471 |
| QA probe acc (n=18, base-filtered) | 1.000 | 1.000 (+0.000) | 1.000 (+0.000) |
| general-text NLL | 9.709 | 9.494 (−0.215) | 9.531 (−0.178) |
| SFT train loss (first→last) | – | 11.23→9.20 | 21.39→17.18 |

Deltas are vs base. Raw JSON: `sft_sampling_results.json`.

## Verdict

**Mechanism supported; end-to-end claim not reproduced — qualified
negative at this scale.** The MCMC reshaper does exactly what the paper
says (data moved +2.6 nats/token toward the reference distribution), but
plain SFT on the raw off-policy expert traces generalized slightly
*better* (0.80 vs 0.73 held-out) than SFT on the reshaped traces. No
forgetting was measurable in either arm (QA probe pinned at 100%;
general NLL improved trivially for both — noise). Against the paper's
"beyond sharpening" note: the reshaped arm ended up markedly
*higher*-entropy (0.47 vs 0.24) with *lower* pass@4 coverage — more
diffuse, slightly less reliable. Plausible driver: the expert template,
though off-policy, is perfectly regular (one format × 40), so 30 LoRA
steps fit it cleanly; the 40 diverse fluent reshaped traces are harder to
fit (train loss 17.2 vs 9.2) and that diversity surfaces as output
entropy. Off-policy ≠ bad format.

## Runtime / quota

One bounded run on Colab T4 (account ppoppo205@gmail.com, got capacity
instantly ~02:40 HKT): wall 1412.7 s (~23.5 min), well inside the free-tier
survival window. Infra incident mid-run: current `peft` dispatches LoRA
through torchao and requires torchao>0.16, but Colab ships torchao 0.10 →
`ImportError` at `get_peft_model`; fixed by `pip uninstall -y torchao`
(falls back to peft's default LoRA path). Deterministic seeds, so the
re-run reproduced all pre-training phases exactly. Session stopped after
download.

## Honest limits

1. **Ceiling effect + tiny N.** Base is already 70% greedy / 100% pass@4
   on the toy task; n=30 held-out, single seed — the 0.80 vs 0.73 gap is
   ~2 items, not statistically meaningful.
2. **Scale gap.** 0.5B + LoRA + 40 examples vs frontier-scale
   post-training. The paper's MCMC runs over full trace space with local
   edit proposals; ours is a Metropolis chain over a discrete 7-candidate
   pool — faithful in spirit, coarse in practice.
3. **Forgetting unmeasurable here.** QA probe saturates at 100% for the
   base model; the NLL probe moved slightly favorably for *both* arms.
   The paper's "forgets less" claim is untested, not refuted.
4. **Confounded comparison.** "On-policyness" covaries with format
   consistency: the expert set is one rigid template (easy to fit), the
   reshaped set is 40 diverse styles (hard to fit in 30 steps). A cleaner
   test would fix format diversity across arms.
5. **Measurement caveat (investigated via `diag.py`).** For this instruct
   model fed raw prompts (no chat template), a joint question+trace NLL
   is dominated by no-context perplexity and tokenization-boundary
   artifacts (it even disagrees in level with the conditional scores).
   The MCMC itself correctly used conditional `logp(trace|question)`; the
   forgetting probe uses the joint NLL only as a *relative* base-vs-tuned
   measure on identical sentences, which remains valid.
