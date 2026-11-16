# SPRA-LoRA: Stochastic Projections and Rank-Aware Learning Rates for Federated LoRA

- **Article:** https://arxiv.org/abs/2611.13331 — "Stochastic Projections and
  Rank-Aware Learning Rates for Federated LoRA"
- **Date:** 2026-11-16 (fetch: OAI fallback — arXiv API 429'd, RSS empty)
- **Scores:** TUNE **5**, AGENT 2
- **Verdict: PARTIAL** (training arms + comms claim measured end-to-end;
  the rank-aware-LR and SNR-vs-rank legs could not discriminate at this
  scale — see below; corrected SNR probe staged as follow-up)

## Why picked

Federated LoRA with a frozen, periodically resampled A matrix plus
rank-aware learning rates is directly adaptable to the soup-daily-finetune
LoRA harness (same NF4-QLoRA stack, same client pattern). If frozen-A
matches standard LoRA at half the uplink cost, it halves the cost of the
daily fine-tune. Repo fit: training (`soup-daily-finetune`,
`hermes-gi-egtkg-finetune`, `Soup`).

## What was built

- `colab_spra_driver.py` — Colab T4 driver: Qwen3-8B NF4 QLoRA (see model
  note), ARC-Challenge logprob MCQA eval (n=200), SNR-vs-rank probe,
  5 training arms (300 rows, 1 epoch, 2 simulated clients, A resampled
  every 10 steps + at the client switch), results flushed to /content
  after every phase.
- `colab_spra_snr2_driver.py` — corrected weight-space SNR probe
  (follow-up; see "SNR probe" below).
- `spra_results.json`, `spra_metrics.csv`, `spra_eval_rows.csv` — full
  results (per-item eval rows for every arm).

## Model note

Qwen3.5-9B was tried first per the standing model rule but **failed to
load**: transformers 4.57.6 (newest <5.0) does not recognise model type
`qwen3_5`. One automatic fallback consumed in-driver → Qwen3-8B (proven
carrier). A transformers-from-source install was NOT attempted — it risks
the proven 8B path. The 9B stays blocked on the pin.

## Attempt 1 (killed by me, kept as record)

- ARC-Easy base eval hit a ceiling: 195/200 = 0.975, no headroom for arm
  differences → switched to ARC-Challenge.
- SNR probe at B=0 is degenerate: loss identical (11.7816) across all
  A-resamples, SNR r=4: 0.3808, r=8: 0.3322 (≈1/√8 flat) — E[grad_B]=0
  by symmetry. Those numbers are the null control, not evidence.

## Results — attempt 2 (session train-1116-spra2, ppoppo205, T4, DONE)

Base (Qwen3-8B, logprob MCQA, ARC-Challenge validation n=200):
**180/200 = 0.9000 ± 0.0416**.

| arm | rank | lr | frozen-A | trainable | acc (n=200) | predictions = base |
|---|---|---|---|---|---|---|
| base | — | — | — | — | 0.9000 | — |
| S16 (standard LoRA) | 16 | 2e-4 | no | 43.6M | 0.8800 ± 0.0450 | 184/200 |
| F16 | 16 | 2e-4 | yes | 22.4M (0.51×) | 0.9000 ± 0.0416 | **200/200** |
| F8hi | 8 | 2e-4 | yes | 11.2M (0.26×) | 0.9000 ± 0.0416 | **200/200** |
| F8lo | 8 | 1e-4 | yes | 11.2M (0.26×) | 0.9000 ± 0.0416 | **200/200** |
| F32lo | 32 | 4e-4 | yes | 44.8M (1.03×) | 0.9000 ± 0.0416 | 198/200 |

All arms: 38 steps, ~680s each, peak VRAM 12.20 GiB.

Comms (claim 2, arithmetic): frozen-A trains B only — F16 22.4M vs S16
43.6M (0.51×), F8 11.2M (0.26×). The 50% uplink reduction is verified
exactly by parameter count.

## What the numbers actually say

- **Frozen-A training left predictions unchanged at this scale.**
  F16/F8hi/F8lo are prediction-identical to base on 200/200 items; F32lo
  on 198/200. The periodic A-resampling destroys whatever B learns each
  round (train loss oscillates 0.06 → 11.9 across resamples; final frozen-A
  losses stay ~11–12 while standard LoRA fits to 0.65), so the net update
  washes out. The "frozen-A matches standard LoRA" parity is therefore
  vacuous here — preserved by inaction, not by learning.
- **Rank-aware LR (claim 3, accuracy leg): no discrimination.**
  F8hi (lr 2e-4) = F8lo (lr 1e-4) = base on all 200 items. The LR
  prescription cannot be tested when neither arm moves predictions.
- **SNR-vs-rank (claim 3, gradient leg): measured flat in both regimes
  tested — no growth with r.** Warmed grad_B SNR: r=4: 0.349, r=8: 0.362,
  r=16: 0.394, r=32: 0.363 (slope 0.0004/rank, R²=0.07). Warmed
  weight-space SNR (Ĝ, q_proj layers [0,11,23,35], session
  train-1116-snr2b, DONE): r=4: 0.389, r=8: 0.391, r=16: 0.386,
  r=32: 0.394 (slope 0.00015/rank, R²=0.30); per-layer SNRs all
  0.38–0.40. Signal and noise both grow ~√r, so the ratio stays
  ≈1/√8 — the signature of a zero-mean quantity. The v1 all-layer fp32
  probe OOM'd first (session train-1116-snr2, DIED) and was retried as
  v2 (one automatic fallback). The clean untested regime is B=0, where
  E[Ĝ] = rσ²G₀ exactly: staged as `colab_spra_snr2c_driver.py` (v2 with
  warmup 0), NOT run today — the fallback is consumed and v2 completed.
- **Claims (1) and (4) not tested:** the stochastic-approximation claim
  needs the weight-space probe; "beats SOTA federated baselines" needs a
  federated baseline, which was never run.
- Standard LoRA (S16) is the only arm that moved predictions (184/200
  identical to base) — and slightly worse (176 vs 180 correct, within
  noise). 300 rows of letter-only SFT does not help ARC-Challenge
  logprob accuracy at this scale.

## SNR probe — why grad_B is the wrong quantity

For one layer, y = (W+BA)x, so grad_B = G @ A^T with G = dL/d(BA) the
full-FT gradient. E[grad_B] over resampled A is ~0 (A is zero-mean; B
stays small), so SNR(grad_B) ≈ 1/√M flat across ranks — exactly what
was measured. The paper's claim lives in weight space:
Ĝ = grad_B @ A = G @ (A^T A), with E[Ĝ] ∝ G and A^T A concentrating as
r grows, so SNR(Ĝ) should increase with r. `colab_spra_snr2_driver.py`
(v1) measured SNR(Ĝ) pooled over all LoRA layers — but DIED on CUDA OOM
at the first resample (fp32 Ghat for all layers = a second model copy;
session train-1116-snr2). `colab_spra_snr2b_driver.py` (v2) fixes the
OOM: Ghat only for q_proj of layers [0, 11, 23, 35], bf16, two-pass
seeded resampling, freed intermediates — session train-1116-snr2b DONE:
SNR 0.389 / 0.391 / 0.386 / 0.394 for r=4/8/16/32 (see
`spra_snr2b_results.json`), i.e. flat. The remaining clean regime is
B=0, where the theory's E[Ĝ] = rσ²G₀ holds exactly (at warmed B the
gradient G(A) varies with the resampled A and swamps the signal):
staged as `colab_spra_snr2c_driver.py`, not run today.

## Honest limits

- Single-GPU sequential simulation of 2 clients; no FedAvg aggregation.
- The LR-scaling formula (lr = 2e-4 · r/16) is our operationalization;
  the abstract does not publish one.
- ARC-Challenge is our transfer task, not the paper's dataset; n=200
  eval (±0.042) and base 0.90 leave little room to discriminate arms.
- 300 rows / 38 steps is a small experiment by design (shrink the
  experiment, never the model); a discriminating test needs either no
  resampling during training (resample only between epochs), more
  steps/data, or a harder eval.
- Metric rows are labeled `arc_easy_logprob` in the CSV (hardcoded split
  name) but the eval is ARC-Challenge — `spra_results.json`
  `eval.dataset` is authoritative.
- Adapters were saved to /content/adapters/ on the VM but the wrapper
  only downloaded the three listed result files; adapters died with the
  session (per-item rows + metrics survive, which is what the verdict
  rests on).

## Resume state

- Corrected weight-space SNR probe: `colab_spra_snr2_driver.py` (v1,
  OOM'd) → `colab_spra_snr2b_driver.py` (v2, DONE, flat 0.39) →
  `colab_spra_snr2c_driver.py` (v3, B=0, STAGED, not run — the clean
  regime for claim (3); run as a fresh short session when revisited).
- To discriminate the frozen-A / rank-aware-LR claims: rerun with
  resampling only between epochs (or no resampling), ≥100 steps, and an
  eval where base scores well below 0.90 — or a task where SFT actually
  moves accuracy (the letter-only ARC target does not).
