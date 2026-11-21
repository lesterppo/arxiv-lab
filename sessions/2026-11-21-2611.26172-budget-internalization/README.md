# Budget Internalization: Fixed-Budget RLVR Trades Test-Time Scaling for In-Budget Accuracy

- **Article:** arXiv:2611.26172 — "Budget Internalization: Fixed-Budget RLVR Trades Test-Time Scaling for In-Budget Accuracy" (Zavidnyi, Kovrigin, Puchkova, Gelvan, Titov, 2026)
- **Link:** https://arxiv.org/abs/2611.26172
- **Date tested:** 2026-11-21
- **Verdict: BLOCKED** (leg 1 reclaimed; fallback leg 2 PENDING — see below)

## Why picked

- **AGENT 2 / TUNE 5.** The paper's core claim is a training-dynamics
  effect with a cheap reproduction route: a fixed rollout cap becomes
  a learned constraint — models beat base near the budget but fall
  below base at larger budgets — and the paper's own SFT ablation
  ("SFT on artificially truncated traces reproduces budget
  internalization, whereas SFT on naturally short traces does not")
  isolates the effect without a full GRPO run.
- **Repo fit:** directly relevant to `soup-daily-finetune` (daily 8B
  NF4-QLoRA on Colab T4): every daily run trains under a rollout /
  length cap, so whether that cap is internalized changes how the
  harness should set eval budgets.
- **Testable on Colab:** QLoRA SFT on two small arms + accuracy-vs-
  generation-budget curves, all on one free T4.

## Design

- **Model:** Qwen/Qwen3.5-9B-Base first (standing pick — the next
  training run tries Qwen3.5-9B first), NF4 QLoRA r=16/alpha=32, bf16,
  transformers==5.20.2 (Qwen3.5 needs 5.x — the 2026-11-20 RLCDR
  lesson). Fallback: Qwen/Qwen3-8B with transformers>=4.51,<5.0.
- **Budget B = 256 tokens.** Phase 1: GSM8K train, 4-shot greedy
  traces from the base model, keep correct-answer ones. Arm A
  "truncated": traces longer than B, CoT cut at B tokens + forced
  "#### answer". Arm B "natural": traces naturally <= B tokens.
  Phase 2: one SFT epoch per arm, fresh LoRA each. Phase 3: held-out
  GSM8K test, greedy, budgets {128, 256, 512, 1024} for base + A + B.
- **Signature:** arm A beats base near B=256 but falls BELOW base at
  1024; arm B tracks base at 1024.
- OOM lessons baked in (2026-11-20 RLCDR leg 2): `generate()` only
  (never a manual forward materializing [batch, seq, vocab] logits),
  `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`, no
  `prepare_model_for_kbit_training()`, gradient checkpointing on,
  eval batches 8 (4 at budget 1024).

## Leg 1 — Qwen3.5-9B-Base (cyc236es, session `budgetint-1121`)

- **Outcome: RECLAIMED.** The free-tier session was reclaimed after
  ~1h50m of driver runtime. The driver never passed Phase 1: model
  load OK (PROBE_OK, VRAM 7.12 GiB), GSM8K loaded, then 400 greedy
  trace generations ran for 50+ minutes with no completion — GPU was
  at 100%, so it was working, just very slow. Suspected cause: the
  transformers log shows Qwen3.5's Gated DeltaNet layers falling back
  to reference PyTorch implementations (`causal_conv1d` /
  `flash-linear-attention` not installed), which makes generation much
  slower than the standard-attention Qwen3-8B recipe. No metrics were
  produced; `/content` died with the VM (nothing downloaded).
- **Resume state:** driver + CPU test are staged in this session;
  nothing to resume on the VM — a rerun restarts from Phase 0.

## Leg 2 — fallback, Qwen3-8B (automatic, standing rule)

- **Model:** Qwen/Qwen3-8B, transformers>=4.51,<5.0 (proven recipe).
- **Shrink fixes for the leg-1 failure cause:** N_TRAIN 400→200,
  N_EVAL 100→60, SFT_MAX 150→100 per arm; per-batch progress logging
  added to the driver (leg 1 ran silent for 50 minutes, so its rate
  was unknowable). Qwen3-8B uses standard attention — no
  reference-fallback slowdown.
- **Session:** `budgetint-1121b`, isolated registry
  `train-1121-budgetint-b-registry.json`, monitor timeout 3.5h.
- **Outcome: RECLAIMED.** Account rotation hit capacity on kyu008008,
  ppoppo205, and cyc236es in turn; cross-account retry landed on
  **kyu009009** (rotation state now stands at kyu009009). Model load
  OK (PROBE_OK, VRAM 5.66 GiB), GSM8K loaded, Phase 1 started for 200
  problems — and the session was reclaimed before the first progress
  line (80/200 traces) appeared, so Phase 1 again produced no
  traces, no SFT, no eval, no metrics. Downloaded: none.
- **Per the standing rule, the fallback blocking ends the day's
  training track — no further automatic attempts.**

## Verdict: BLOCKED

Both Colab legs were reclaimed before producing any measurements
(leg 1: slow Qwen3.5 generation; leg 2: reclamation early in Phase 1
after three capacity failures on other accounts). The claim itself
is untested — no numbers exist for or against budget
internalization at this scale. Resume state for a future rerun:
driver + CPU test staged in this session; rerun from Phase 0 with
the leg-2 shrink (Qwen3-8B, N_TRAIN 200, N_EVAL 60, SFT_MAX 100)
and, if generation is again the bottleneck, split trace generation
across two sessions or pre-generate traces with the base model in a
separate short run before the SFT leg.

## CPU check

`test_budget_internalization_cpu.py`: answer-extraction cases, the
arm-split rule, the internalization-signature criterion on synthetic
paper-like and null rows, and the eval-budget grid sanity check —
**ALL CHECKS PASSED**.

## Honest limits

- The SFT ablation is the paper's cheaper route to the effect, not
  the main RLVR claim; a null result at this scale (150 examples/arm,
  1 epoch) would not refute the paper.
- Leg 1 produced no numbers at all — the BLOCKED verdict is about
  the run, not the claim.
- Full results (when leg 2 lands): `budget_internalization_metrics.csv`
  + `budget_internalization_summary.json` (downloaded to the goal's
  hidden_files train out-dir).
