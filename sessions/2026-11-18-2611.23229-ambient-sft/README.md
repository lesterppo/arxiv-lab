# Ambient SFT (arXiv:2611.23229) — training-track session, 2026-11-18

**Article:** [arXiv:2611.23229](https://arxiv.org/abs/2611.23229) —
"Ambient LLMs: Data-Efficient Supervised Fine-tuning via Temperature
Matching" (Daras, Papachristou, Klivans, Torralba, Daskalakis; submitted
17 Nov 2026).

**Scores:** AGENT 1, TUNE 5.

## Why picked + repo fit

An SFT loss variant directly adaptable to the `soup-daily-finetune` daily
8B NF4-QLoRA harness (finance domain): the loss is a drop-in change to
the standard SFT step — add a temperature-scaled KL term against the
frozen base model — with no new data, no extra model, and no change to
the serving path. Same pattern applies to `hermes-gi-egtkg-finetune`
(medical domain). The paper claims up to 21% average relative
improvement over standard SFT across domain adaptation,
instruction-tuning, and reasoning, which — if it holds at this scale —
is worth one harness change.

## What was built

- `colab_ambient_driver.py` — self-contained Colab T4 driver. Implements
  L = L_SFT + λ·T²·KL(softmax(z_teacher/T) || softmax(z_student/T)),
  λ=1.0, teacher = the frozen base model (same weights, LoRA disabled,
  no_grad). Arms on identical data/seed 42: base (untrained control) |
  SFT (λ=0) | Ambient T=2 | Ambient T=4. Task: GSM8K, train 600 /
  test 200, 1 epoch, NF4 QLoRA r=16/alpha=32, bf16, lr 2e-4, batch 2 ×
  accum 8 (eff. 16), seq 512, gradient checkpointing, loss on completion
  tokens only. Metric: held-out exact-match accuracy (greedy, max 224
  new tokens). KL is computed ONLY on label positions (gathered rows) —
  the 2026-10-30 OOM fix (full-vocab float32 loss OOM'd at first
  backward). Metrics flushed to `/content` after every arm.
- `cpu_loss_check.py` (+ `cpu_loss_check.log`) — CPU/numerical sanity
  check of the loss limits on toy logits (numpy only), run before the
  Colab run.

## Model choice

**Qwen/Qwen3.5-9B-Base** — standing pick: try Qwen3.5-9B first, fall back
to Qwen3-8B if its QLoRA test does not go GREEN. Latest 9B-class Qwen
(Mar 2026), current-generation family. 2026-10-30 live T4 probe GREEN:
NF4 load + LoRA attach + train step + generate, VRAM 7.89 GB, 29,097,984
trainable params, probe loss 0.776, loader `imagetext`
(Qwen3_5ForConditionalGeneration, no vision tower). Transformers pinned
to `==5.20.2`, the probe's exact version — Qwen3.5 base is unrecognised
by transformers<5.0. Fallback (unused unless this run blocks):
Qwen/Qwen3-8B-Base with the proven transformers<5.0 QLoRA recipe.

## CPU loss-limit check (before the Colab run)

`cpu_loss_check.py`: T=1 gives exactly standard KD (KL = 0.096097);
unscaled KL → 0 as T→∞ (1.1e-7 at T=1000); T²·KL → the Hinton
logit-MSE constant 0.1085 (ratio 1.0000 at T=1000). All assertions
passed. **Nuance found:** the abstract's "T→∞ reverts to SFT" holds for
the UNSCALED KL term; with the T² factor used in the loss, the T→∞
limit is logit matching, not plain SFT. The Colab arms therefore treat T
as trust-in-teacher (small T = strong KD, large T = weak regularisation
toward the frozen base) rather than assuming the T→∞ limit equals SFT.

## Colab run

- Session `ambient-1118`, account **cyc236ha** (rotation: 2026-11-17
  kyu009009 → 2026-11-18 cyc236ha), T4, endpoint
  gpu-t4-s-usw1b2-5dq7cmlcukqe, driver PID 988.
- Command: `colab.py train --driver colab_ambient_driver.py --rotate
  --registry-file <goal hidden_files>/train-1118-ambient-registry.json
  --results /content/ambient_metrics.csv /content/ambient_summary.json
  /content/ambient_predictions.jsonl --out-dir <goal
  hidden_files>/train-1118-ambient-out -s ambient-1118 --poll 120
  --monitor-timeout 10800 --json`
- Outcome: **RECLAIMED** after ~4h10m (T2 trained, T2/T4 evals never
  ran). Fallback session `ambient-1118b`, account **ppoppo205**
  (cyc236ha gpu-unavailable at creation; cross-account retry),
  endpoint gpu-t4-s-usw4b1-vzi7hp6divod, driver PID 400, same driver
  with `--env MODEL_ID=Qwen/Qwen3-8B-Base`. Outcome: **RECLAIMED**
  after ~4h15m (T2 trained, T2/T4 evals never ran). Fallback rule
  exhausted — no further automatic attempts today.

## Results

**VERDICT: BLOCKED** — no fair test of the paper's claim was possible.
Both Colab legs were reclaimed before ANY Ambient arm was evaluated, so
the Ambient-vs-SFT accuracy comparison — the paper's core claim — has
zero measured numbers. What IS measured (below) is the SFT baseline on
both models and Ambient training dynamics without eval. Per the standing
fallback rule the one automatic fallback attempt was used and also
blocked; no further automatic attempts today.

### Leg 1 — Qwen3.5-9B-Base (session `ambient-1118`, cyc236ha, RECLAIMED after ~4h10m)

| arm | T | λ | train loss (38 steps) | mean KL | peak VRAM | held-out acc (n=200) |
|-----|---|---|----------------------|---------|-----------|----------------------|
| base | — | — | — | — | — | **0.5350 (107/200)**, 3591 s |
| SFT | 0 | 0 | 0.5354 | 0.00000 | 10.16 GB | **0.8050 (161/200)**, 2449 s |
| Ambient | 2 | 1.0 | 0.6028 | 0.01188 | 13.62 GB | NOT MEASURED — reclaimed before eval |
| Ambient | 4 | 1.0 | — | — | — | never started |

### Leg 2 (automatic fallback) — Qwen3-8B-Base (session `ambient-1118b`, ppoppo205, RECLAIMED after ~4h15m)

| arm | T | λ | train loss (38 steps) | mean KL | peak VRAM | held-out acc (n=200) |
|-----|---|---|----------------------|---------|-----------|----------------------|
| base | — | — | — | — | — | **0.7400 (148/200)**, 3427 s |
| SFT | 0 | 0 | 0.3108 | 0.00000 | 8.05 GB | **0.8500 (170/200)**, 2548 s |
| Ambient | 2 | 1.0 | 0.3948 | 0.01949 | 10.10 GB | NOT MEASURED — reclaimed before eval |
| Ambient | 4 | 1.0 | — | — | — | never started |

Partial metrics (recovered by hand from the live VMs before reclamation;
the VMs' `/content` files are gone): `ambient_metrics_qwen35_partial.csv`,
`ambient_metrics_qwen38_partial.csv`. Per-arm predictions JSONL lost with
both VMs. Ambient training itself worked on both models (finite losses,
KL term active and in the expected range, no OOM) — the loss
implementation is not the blocker; wall-clock time is.

### Why it blocked

Both legs projected to ~7–7.5 h end to end and were reclaimed at ~4h10m /
~4h15m, both right after T2 training finished and before T2 eval. The
time sink is generation: each 200-problem greedy eval costs 41–60 min
on a T4 (base 3591 s / 3427 s, SFT 2449 s / 2548 s), and each train arm
55–80 min. Qwen3.5 is slower still (Gated DeltaNet layers fall back to
reference PyTorch kernels; causal_conv1d / flash-linear-attention not
installed). Free-tier sessions recycle in ~2–12 h, so a 7-hour design
does not fit. No OOM, no auth failure, no code error on either leg.

### Resume state (exact)

- Driver: `colab_ambient_driver.py` in this session (self-contained).
- Fastest resume: Qwen3-8B leg, same driver with
  `MODEL_ID=Qwen/Qwen3-8B-Base ARMS=t2,t4` (base + SFT already banked:
  0.7400 / 0.8500 — no need to re-run them). T2 must be RETRAINED
  (adapters were never saved; only its training loss/KL survive in the
  log). Estimated ~2.5 h for T2 train+eval and T4 train+eval — still
  long; shrinking N_TEST (e.g. 100) or splitting T2/T4 across two
  sessions would fit free-tier survival better.
- Qwen3.5 leg resume: same driver, `MODEL_ID=Qwen/Qwen3.5-9B-Base
  ARMS=t2,t4` (base 0.5350 / SFT 0.8050 banked); also needs T2 retrained.
  Slower (~7.5 h full design) — install causal-conv1d +
  flash-linear-attention first, or shrink N_TEST.
- Rotation: last account to actually run today is ppoppo205 (fallback;
  cyc236ha hit gpu-unavailable at fallback creation). State file reads
  `2026-11-18 ppoppo205`, so the next daily run picks cyc236es.
- No further automatic attempts today per the fallback rule (one
  fallback used, also blocked).

## Honest limits

- Single seed (42), 600 train / 200 test — small-scale probe of a claim
  made on larger setups; a 200-problem test set has ~±3.5% absolute
  sampling noise, so small differences are not decisive.
- λ=1.0 and T ∈ {2, 4} are the task's picks, not the paper's tuned
  values (the paper's exact λ/T grid was not in the abstract; the full
  text was not consulted beyond the abstract).
- GSM8K only — one of the paper's three settings (reasoning); domain
  adaptation and instruction-tuning were not tested.
- One epoch, LoRA (not full fine-tuning); the frozen teacher is the same
  base model, so teacher quality is bounded by the base model itself.

## Files

- `colab_ambient_driver.py` — Colab driver (see above)
- `cpu_loss_check.py`, `cpu_loss_check.log` — loss-limit sanity check
- `ambient_metrics_qwen35_partial.csv` — leg 1 banked rows (base 0.5350,
  SFT 0.8050; recovered by hand before reclamation)
- `ambient_metrics_qwen38_partial.csv` — leg 2 banked rows (base 0.7400,
  SFT 0.8500; recovered by hand before reclamation)
