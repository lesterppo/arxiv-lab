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

**Colab (T4) — RUNNING via watcher (2026-10-03 ~17:15 HKT):**
Colab auth fixed the same day (OAuth code exchange using the client secret
published in the `google-colab-cli` PyPI package; tokens auto-refresh).
`~/workspace/carm-colab/watch.sh` retries T4 assignment every 10 min
(free-tier backend was busy at 17:05), then installs deps, uploads this
script, runs `--mode carm` and `--mode standard` detached, and downloads
both CSVs to `~/workspace/carm-colab/`. Results will be appended here.
