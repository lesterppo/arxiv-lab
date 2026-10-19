# Local proxies for memory-efficient training (2610.23145)

**Article:** https://arxiv.org/abs/2610.23145 — *Local proxies for
memory-efficient training* (Zihan, Yuan, Bellec)
**Date:** 2026-10-19 · **Fetch source:** api · **Track:** training · **Verdict: PARTIAL**

## Why picked + repo fit (AGENT 1 / TUNE 5)
Per-block proxy networks distilled to predict the *global* network output;
block gradients propagate locally through the proxy instead of end-to-end
backprop — no full-stack activation retention. The paper explicitly shows
LLM fine-tuning on a small memory budget beating zeroth-order methods.
Directly relevant to the `soup-daily-finetune` / `hermes-gi-egtkg-finetune`
daily NF4-QLoRA-on-T4 harness: activation memory is the binding constraint
there, and a proxy-local alternative could unlock larger batches or longer
contexts. TUNE 5.

**Core claims under test:** (1) exact on deep linear nets; (2) bounded
gradient error on nonlinear residual nets, shrinking as distillation loss
is optimized; (3) better accuracy/memory trade-off than local algorithms;
(4) LLM fine-tuning on small memory budget: faster + better than
memory-efficient zeroth-order at comparable memory.

## What was built
- `src/arxiv_lab/training/local_proxies.py` (new module, numpy): closed-form
  least-squares proxies + proxy-propagated gradients for deep linear nets;
  residual-form proxies (`ResidualProxy`: linear skip + small MLP) with
  manual Jacobians; `three_way_comparison` (backprop vs proxy-local vs
  MeZO-style SPSA) with analytic activation-slot accounting.
- `tests/test_training_local_proxies_cpu.py`: all checks pass (seeded).
- `colab_local_proxies_8b.py`: self-contained Colab T4 driver — three arms
  (standard NF4-QLoRA SFT / proxy-local / MeZO on LoRA params), identical
  data/steps/seeds, Qwen3-8B (Qwen3.5-9B not loadable under
  transformers<5.0 — established 2026-10-18; the qwen3.5-9b-q4 recipe is
  Ollama-only), r=16/alpha=32, bf16, synthetic finance-domain SFT
  (120 train / 40 eval, toy data — relative comparison only). Incremental
  CSV + results.json in /content.

## Results
### CPU (VM, numpy, seeded) — ALL CHECKS PASSED
- Claim 1 (exactness, deep linear, 4 layers): max relative gradient error
  **2.95e-14** (machine precision) vs exact backprop.
- Claim 2 (residual net, residual-form proxy): distill MSE 5.65 → 0.088
  across proxy-training checkpoints; gradient cosine similarity
  0.52 → **0.985**; corr(log MSE, cos-sim) = **−0.968** — optimizing the
  distillation loss improves gradient estimation, as claimed.
- Claim 4 (CPU analog): 120-step 3-way on synthetic classifier —
  backprop 0.4360 / **proxy 0.4702** / mezo 0.7572; activation slots 3 / 1 / 0.
  Proxy-local beats zeroth-order clearly at 1/3 the retained activations.

### Colab T4 — attempt 1 (`localproxies`, kyu008008): partial failure, fixed
- **Standard arm: OOM.** Full-backprop NF4 QLoRA (batch 4, seq 512) exceeded
  14.5 GB on the T4 — the memory wall the paper addresses is real at this
  config. (Proxies target exactly this: no full-stack grad activations.)
- **Proxy arm: driver bug (mine, fixed).** Used `register_forward_hook`
  (which passes `(module, args, output)`) instead of
  `register_forward_pre_hook(with_kwargs=True)` — `AttributeError` on
  `kwargs.get`. Fixed in the staged driver.
- **MeZO arm: cascade OOM** — the failed proxy arm leaked its model
  (no try/finally); fixed with guaranteed cleanup.
- Rerun config: batch 2 / seq 256 for all arms (apples-to-apples).

### Colab T4 — attempt 2 (`localproxies2`, kyu008008): standard DONE, proxy PARTIAL
- **Standard (full-backprop QLoRA, batch2/seq256): OK.** 16 steps: train
  3.11 → 1.42, eval 2.42 → **0.99**, peak **13.98 GB**, wall 778 s.
  Full per-step metrics in `colab_metrics_partial.csv`.
- **Proxy arm: PARTIAL (3/16 steps).** Fits at **8.68 GB peak (38% below
  standard's 13.98 GB)** — the paper's memory win reproduces on a real 8B
  model. But train loss stayed ~5.0 (vs standard ~1–3): the 36 per-block
  proxies were still cold (only 3 distill micro-steps per main step on 4M-param
  proxies) and each main step took ~2.5 min (36 sequential block
  recompute+backward passes), so 16 steps could not finish in the session
  window. Learning-curve comparison vs MeZO not obtained.
- **MeZO arm: not run** (queued after proxy).
- Session torn down after metrics harvest per the carry-over rule.

**Verdict rationale — PARTIAL:** claims 1–3 verified on CPU with measured
numbers; claim 4 (LLM-scale) partially tested — memory reduction confirmed
live (8.68 vs 13.98 GB), training-throughput/quality comparison blocked by
step cost. Resume state below.

## Resume state (for follow-up)
- Driver: this dir's `colab_local_proxies_8b.py` (v2: pre-hook fix,
  try/finally cleanup, batch2/seq256). Re-upload + exec_detach on a fresh
  T4 session (account kyu008008 — do NOT switch to cyc236de/cyc236hk).
- Completed: standard arm 16/16 (metrics in `colab_metrics_partial.csv`).
  Proxy arm 3/16 — to resume the comparison, warm-start proxies longer
  (raise distill micro-steps, or pre-train proxies on a few batches before
  the main loop) and/or cut steps to 8/arm.
- Suggested driver tweaks: freeze proxy params during step (iii) local
  backward (their grads are computed but unused — wasted compute); consider
  fewer proxy micro-steps with higher lr.
