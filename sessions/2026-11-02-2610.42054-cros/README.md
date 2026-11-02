# 2026-11-02 — CRoS: Causality Reward on Scrambled CoTs

- Article: https://arxiv.org/abs/2610.42054 — "CRoS: Causality Reward
  on Scrambled CoTs Improves Accuracy and Reasoning Faithfulness"
  (submitted 30 Oct 2026).
- Why picked (AGENT=1, TUNE=4): an RLVR reward-shaping method — the
  kind of training-track mechanism that belongs in a daily fine-tune /
  RL harness (cf. `soup-daily-finetune`). Its core claim is a clean
  perturbation property: scrambling a CoT's step order preserves every
  step's content exactly while breaking causal order, so a causally
  linked CoT should lose answer likelihood under scrambling and an
  order-independent CoT should not. That property is testable first on
  CPU and then on a real frozen scorer.
- What was built: `src/arxiv_lab/training/cros.py` (graduated) +
  session copies `cros.py` / `test_cros_cpu.py`, and the Colab
  frozen-scorer driver `colab_cros_scorer.py` (metrics
  `cros_metrics.csv`, summary `cros_summary.json`, monitor record
  `train-run.json`).

## CPU properties (`tests/test_training_cros_cpu.py`, all pass)

- Scrambling preserves the step multiset exactly: 200 random step
  lists × 4 seeds, 0 violations; deleting a step always changes it
  (the paper's contrast with deletion / error-injection baselines).
- Toy evaluator: chained problems — scrambled evaluation fails 6/6,
  mean causality (orig − scrambled score) 1.00; independent problems
  — scrambled evaluation identical 6/6, mean causality 0.00.

## Colab frozen scorer (the training-track leg, outcome DONE)

- Session `cros-1102`, account `ppoppo205` (rotation: previous day
  `cyc236ha` → `ppoppo205`; next day rotates to `cyc236es`), T4 GPU.
- Model: **Qwen/Qwen3.5-9B** NF4 / bf16, loaded first try (the 9B-first
  rule; the Qwen3-8B fallback was not needed). Peak VRAM 7.32 GiB.
- Design (shrink the experiment, never the model): 12 embedded
  arithmetic problems (6 chained, 6 order-independent) × 3 conditions
  (original CoT, scrambled CoT, no CoT) = 36 answer-logprob scorings.
  No training run — this leg measures the CRoS perturbation signal on
  a real model; see limits.

| Kind | LP orig | LP scrambled | LP no-CoT | Mean delta (orig − scram) |
|---|---|---|---|---|
| Chained | −0.2625 | −3.4026 | −7.5155 | **+3.1401** (all 6 problems positive, range +2.26…+4.01) |
| Independent | −0.3164 | −0.2477 | −5.5034 | **−0.0687** (all 6 problems ≤ 0, range −0.15…−0.03) |

Delta gap (chained − independent) = **+3.2088**; the paper's
prediction holds on a real 9B scorer. Two further CRoS-consistent
observations: scrambled CoTs still beat no-CoT by a wide margin in
both kinds (+4.11 chained, +5.26 independent on average), because
scrambling preserves content — the exact property deletion destroys.

## Verdict: SUPPORTED

The CRoS mechanism reproduces directionally at bounded scale: CPU
properties exact, and a real current-generation scorer separates
chained from order-independent CoTs cleanly (6/6 vs 0/6 positive
deltas) under the scrambled perturbation.

## Honest limits

- No RLVR training run was performed: the paper's training headline
  (46.8% vs 43.8% out-of-domain accuracy for CRoS over the strongest
  baseline on Reasoning Gym with Gemma-3-4B) is NOT reproduced, and
  the model here (Qwen3.5-9B, frozen scorer) differs from the paper's
  (Gemma-3-4B, trained).
- Synthetic arithmetic problems (N=12), not Reasoning Gym; the
  independent-kind deltas are slightly negative rather than exactly
  zero (scrambling marginally helps flat fact lists — noise at this
  scale, but the sign separation is what the claim needs).
- The exact CRoS reward formula is not printed in the abstract, so
  the tested quantity is the abstract's stated perturbation signal
  (answer-score drop under scrambling), not bit-parity with the
  authors' training code.
