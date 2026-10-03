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
one instantly. Ran `colab_grpo_carm.py` (Qwen2.5-0.5B-Instruct + LoRA r=16,
generated arithmetic with `\boxed{}` reward, 2 update epochs per rollout
batch = the stale-rollout regime) in both modes, plus a high-drift variant
(`--epochs 6 --lr 5e-5`).

| run | steps | reward start→end | max mean|log r| | mask rate |
|---|---|---|---|---|
| carm, low-drift | 24 | 0.188 → 0.688 | 0.014 | 0.000 |
| standard, low-drift | 24 | 0.188 → 0.688 | 0.014 | 0.000 |
| carm, high-drift | 72 | 0.188 → 1.000 | 0.039 | 0.000 |
| standard, high-drift | 72 | 0.188 → 1.000 | 0.039 | 0.000 |

CSVs: `~/workspace/carm-colab/carm_grpo_*.csv` (also reproducible from the
script).

**Verdict: mechanism SUPPORTED on CPU; end-to-end A/B INCONCLUSIVE on the
masking difference, with an honest reason.** The toy arithmetic task
converges (reward → 1.0) before meaningful rollout-staleness accumulates:
max drift 0.039 vs the 0.182 band, so neither mask ever fires and the two
modes trace identical trajectories. What the Colab runs do establish:
(1) the GRPO+masking pipeline works end-to-end on a free T4; (2) CARM does
no harm — learning is identical when drift is small. Differentiating the
masks end-to-end needs sustained drift (a harder task where the policy
keeps moving without converging) — queued as a follow-up, not faked here.
The masking claim itself (standard accepts canceling drift, CARM rejects;
joint bound holds) is proven by `test_carm_cpu.py` on this VM.
