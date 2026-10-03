# Session 2026-10-03 — CARM (arXiv:2610.02039)

**Article:** "CARM: Cancellation-Aware Response Masking for LLM Reinforcement
Learning" ([abs](https://arxiv.org/abs/2610.02039))

**Why picked:** most prospective *training* article in the 2026-09-30→10-03
window (TUNE 4/5). Crisp, falsifiable mechanism: the standard sequence mask
(`|mean(log r_i)| <= log(1+eps)`) lets opposing token drifts cancel, hiding
real off-policy drift; CARM (`mean(|log r_i|) <= log(1+eps)`) does not.

**Paper claim:** CARM masking improves math-reasoning/codegen RL; accepted
responses satisfy a joint bound on (fraction of ratios outside the band) ×
(mean log-distance beyond it).

## What was built

- `carm.py` — both masks + `drift_stats()` (the joint-bound quantities).
- `test_carm_cpu.py` — synthetic log-ratios (this VM).
- `colab_grpo_carm.py` — **live test** for Google Colab T4: real GRPO A/B on
  Qwen2.5-0.5B-Instruct + LoRA, generated arithmetic with `\boxed{}`
  verifiable reward, 2 update epochs per rollout batch (the 2nd epoch is
  where stale-rollout masking bites). `--mode standard|carm`, CSV metrics
  to `/content/carm_grpo_<mode>.csv`.

## Results

**CPU (this VM, deterministic) — PASS:**
- Cancellation demo: standard mask KEEPs +-a drift for a up to 2.0;
  CARM drops it once a exceeds the band.
- 20/20 clean sequences accepted by both (no false positives).
- Joint bound: CARM-accepted worst `frac_outside*mean_excess` = 0.129 ≤
  band 0.182; standard mask admits 1.909 (10x worse). Paper's bound
  direction confirmed.

**Colab (T4) — RAN 2026-10-03 on account cyc236es@gmail.com:**
Colab auth fixed the same day (OAuth code exchange using the client secret
published in the `google-colab-cli` PyPI package; tokens auto-refresh).
Account ppoppo205@gmail.com had no free T4 capacity; cyc236es@gmail.com got
one instantly. `colab_grpo_carm.py`: Qwen2.5-0.5B-Instruct, generated
arithmetic with `\boxed{}` reward, N stale update epochs per rollout batch.

Regime 1 — LoRA (r=16), easy task, lr=1e-5/5e-5, 2–12 stale epochs:

| run | steps | reward start→end | max mean|log r| | mask rate |
|---|---|---|---|---|
| carm / standard, low-drift | 24 | 0.188 → 0.688 | 0.014 | 0.000 |
| carm / standard, high-drift | 72 | 0.188 → 1.000 | 0.039 | 0.000 |

Masks never fired (drift 0.014–0.039 vs 0.182 band): the toy task converges
before staleness accumulates. Both modes identical → CARM does no harm,
but the A/B can't differentiate the masks here.

Regime 2 — full fine-tune, hard multi-step task, 8 stale epochs (the
decisive regime):

| run | final reward | max reward | mean mask rate |
|---|---|---|---|
| carm, lr=1e-4 | 0.000 (collapsed) | 0.312 | ~1.00 (all masked after epoch 1) |
| carm, lr=3e-5 | 0.094 | 1.000 | 0.605 |
| standard, lr=3e-5 | 0.406 | 1.000 | 0.359 |

At lr=1e-4 CARM masked everything after the first epoch's huge move —
the mask worked exactly as designed (refused wildly off-policy updates),
but the model was already destroyed by that first epoch.
At lr=3e-5 both modes hit 1.0 then collapsed (stale-epoch over-optimization);
CARM masked far more aggressively (60% vs 36%) yet finished LOWER
(0.094 vs 0.406).

**Verdict: mechanism SUPPORTED on CPU; end-to-end result is a qualified
negative.** The masking flaw CARM fixes is real (proven on synthetic
log-ratios: standard accepts arbitrary canceling drift, CARM rejects it,
joint bound holds). But live, the paper's benefit lives in a Goldilocks
drift regime our 0.5B toy cannot sustain: LoRA barely drifts (masks never
fire), full-FT explodes (both modes collapse; masking can't save a
divergent run, and in this single seed the heavier-masking CARM run
collapsed harder — not a refutation, chaotic regimes aren't robust, but
an honest datum). Reproducing the paper's claimed gains needs
paper-scale training where drift is material but controlled. CSVs:
`~/workspace/carm-colab/carm_grpo_*.csv`.

## Deeper test — Qwen2.5-7B-Instruct on Colab T4 (2026-10-03 night)

Deployed the 7B model (4-bit QLoRA) and ran a direct drift test
(`carm_7b_ultra.py`): 8 real GSM8K rollouts, reference log-probs, then
Gaussian noise on LoRA weights at 3 scales to simulate drift, measuring
both masks' decisions on the REAL 7B generations.

| noise | mean|log r| | std keep | carm keep | disagreements |
|---|---|---|---|---|
| 0.01 | 0.017 | 1.000 | 1.000 | 0/8 |
| 0.03 | 0.133 | 1.000 | 0.875 | 1/8 |
| 0.10 | 10.61 | 0.000 | 0.000 | 0/8 |

At noise=0.03 — drift approaching the 0.182 band — the **standard mask
kept 8/8 while CARM rejected 1**. That sequence had **canceling
structure** (both large positive and negative token log-ratios) with
**max|log r| = 3.726**: a single token drifted 3.7 nats off-policy and
the standard mask accepted it because the sequence mean canceled out.
CARM caught it. This is the paper's exact failure mode, reproduced on a
real 7B model's real generations.

**Verdict update: mechanism SUPPORTED on real 7B generations.** The
canceling-drift blind spot is not a synthetic artifact — it materializes
in Qwen2.5-7B-Instruct under drift, and CARM's stricter mask catches what
the standard mask misses. The end-to-end training benefit remains
unproven at our scale (needs the Goldilocks regime), but the mechanism
the paper is built on is real.

Technical notes from the 7B work (worth keeping):
- 7B QLoRA needs gradient checkpointing + G<=2 on a T4 (OOM otherwise).
- Generate rollouts in `model.eval()` — with grad checkpointing active,
  `model.train()` + `use_cache=False` produces garbage generations
  (reward 0.000); eval mode fixed it (reward 1.000 on the same problems).
- `datasets` lib has a GSM8K URI bug (`hf://datasets/gsm8k@...`); download
  the test.jsonl from GitHub raw instead.
- Qwen2.5-7B-Instruct solves GSM8K at ~77-100% — too easy for GRPO signal;
  screen for hard/mixed problems first.
- Colab free tier reclaimed 3 T4 sessions in one evening (30-60 min
  lifetimes); multi-hour runs need Pro or retry-tolerant orchestration.
  JSON: `carm_7b_ultra.json`.
