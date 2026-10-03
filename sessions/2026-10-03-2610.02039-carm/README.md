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

**Colab (T4) — STAGED, awaiting run:**
```bash
# on the Colab session (hermes-colab-cli), T4 GPU:
python3 colab.py upload -s <session> colab_grpo_carm.py /content/
python3 colab.py exec -s <session> --code "!pip install -q transformers peft accelerate bitsandbytes" --timeout 600
python3 colab.py exec_detach -s <session> -f /content/colab_grpo_carm.py -- \
    --mode carm          # then again with --mode standard
python3 colab.py download -s <session> /content/carm_grpo_carm.csv .
```
Compare reward curves, mask rates, and loss stability across the two CSVs.
Small scale (48 prompts) → directional evidence, not conclusive.

**Verdict: mechanism SUPPORTED on CPU; live GRPO A/B staged for Colab.**
The Colab run needs the one-time OAuth (see `hermes-colab-cli`
`references/auth_flow.md`) — no Colab credentials exist on this VM yet.
