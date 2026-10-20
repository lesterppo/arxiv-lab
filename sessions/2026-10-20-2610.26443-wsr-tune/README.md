# 2026-10-20 — 2610.26443 — WSR-Tune (weight-space reparameterization)

**Article:** [Rethinking LLM Fine-Tuning via Weight Space Reparameterization:
Preserving Safety during Downstream Adaptation](https://arxiv.org/abs/2610.26443)
(cs.AI, 2026-10-19)

**Why picked (AGENT 1 / TUNE 5):** the paper attacks the safety/task tradeoff
in downstream fine-tuning — build a safety-conditioned basis, concentrate
safety information in a few directions, freeze those directions in the
reparameterized space, update complementary directions for the task. Directly
relevant to the `soup-daily-finetune` NF4-QLoRA-on-T4 harness: a cheap,
principled way to keep safety alignment while doing daily SFT.

## What was built

- `test_wsr_tune_cpu.py` — deterministic numpy check of the core mechanism:
  per-module SVD of the safety-gradient → top-k directions; projecting the
  adapter update orthogonal to them is idempotent, leaks <1e-15 into safety
  directions (plain gradient leaks ~1.1), and still reduces the synthetic
  task loss. PASS.
- `colab_wsr_tune.py` — bounded Colab T4 reproduction: Qwen3-8B NF4 QLoRA
  (r=16/alpha=32, bf16); synthetic unit-conversion SFT (320 train / 80 eval,
  numeric-match eval); safety = refusal rate on 30 harmful requests
  (template judge); arm PLAIN vs arm WSR (per-module top-2 SVD of the mean
  safety-pair gradient; after every optimizer step, `lora_A` rows projected
  off safety input-directions and `lora_B` columns off safety
  output-directions — the paper's "freeze safety-critical directions").

## Results

Colab T4, Qwen/Qwen3-8B NF4 QLoRA (r=16/alpha=32, bf16), 320 train / 2
epochs / batch 4, unit-conversion SFT; safety = refusal rate on 30 harmful
requests (template judge). Full metrics in `metrics.json` / `metrics.csv`.

| arm | base task | post task | base refusal | post refusal | peak VRAM |
|---|---|---|---|---|---|
| plain (no constraint) | 0.662 | **0.988** | 0.667 | 0.700 | 9.81 GB |
| wsr (safety-dir projection) | 0.662 | **0.988** | 0.667 | 0.700 | 13.41 GB |

Safety basis: mean gradient of 24 (request, refusal) pairs → per-module
top-2 SVD directions over 144 LoRA matrices; after every optimizer step,
`lora_A` rows projected off safety input-directions, `lora_B` columns off
safety output-directions. Training loss: plain 0.004→0.001,
wsr 0.008→0.001 (projection constrains updates slightly, as expected).

## Verdict: INCONCLUSIVE

The test could not discriminate the paper's core claim. Benign SFT at this
bounded scale (320 examples, 2 epochs) did **not** degrade safety in the
plain arm (refusal 0.667 → 0.700), so there was no safety degradation for
WSR-Tune to prevent — both arms are identical on task (0.988) and safety
(0.700). What the run does establish: the WSR projection mechanism is
exactly correct (CPU test: idempotent, <1e-15 safety-direction leak vs
~1.1 for plain gradients) and imposes **no task-learning penalty**
(0.988 = plain). A discriminating test needs a setting where plain SFT
actually erodes refusal (more steps, safety-adjacent data, or a weaker
aligned base) — recorded as resume state below. Nothing graduated to `src/`.

## Resume state

- Driver `colab_wsr_tune.py` is Colab-ready and debugged (see limits).
  To discriminate: raise degradation pressure (e.g. 3-5 epochs, lr 3e-4,
  or mix 10% safety-adjacent benign data), then re-run both arms.
- Session `wsr2-20261020` torn down 2026-10-20 ~07:05 UTC after metrics
  harvest; nothing left on the VM.

## Honest limits / notes

- Qwen3.5-9B(-Base) are multimodal (image-text-to-text); transformers<5.0 has
  no `qwen3_5` model class, so the 9B-class recipe could not be used for
  fine-tuning — closest text-only instruct model `Qwen/Qwen3-8B` used
  (recorded in metrics).
- Safety eval is a 30-item template-judge refusal rate (coarse but standard);
  24 of the 30 items also supply the safety-basis pairs.
- Two driver bugs found and fixed during the run: (1) base (non-instruct)
  models don't refuse (baseline 0.000) — instruct model required; (2)
  generating in `train()` mode (dropout + gradient checkpointing, missing
  `model.eval()`) produces `,text` repetition loops — diagnosed via
  token-ID probes, fixed by `model.eval()` in `generate()`. Lesson for
  future Colab SFT evals: always eval in eval mode.
- Earlier metric bug: exact-string number comparison undercounted task
  accuracy ("29004" vs "29004.0"); fixed to numeric comparison (base task
  0.113 → 0.662).
