# 2610.00949 — PG-SFT: Balancing Capability Acquisition and Retention in Offline Agent Fine-Tuning

- **Article:** https://arxiv.org/abs/2610.00949
- **Track:** training (Google Colab T4, official daily run 2026-10-04 — the early-morning run was a test run)
- **Model:** Qwen2.5-0.5B-Instruct + LoRA (r=16, q/v proj), lr 2e-5, 3 epochs, batch 4, single seed

## Why picked

TUNE 4–5: the paper's mechanism is a *data-side* change to SFT (per-turn
supervision weights from turn-level information gain) rather than an
objective change, so it is directly implementable in a small Colab A/B.
It also speaks to a real failure mode Peter cares about: SFT on agent
trajectories degrading the base model's general capabilities.

## What was built

- `data_gen.py` (stdlib, CPU-runnable): 48 synthetic multi-turn agent
  trajectories — `USER → ASSISTANT(reasoning) → TOOL(<calc>) → OBSERVATION →
  ASSISTANT(reasoning) → TOOL(<calc>) → OBSERVATION → ASSISTANT(final)` on
  verifiable two-step arithmetic, plus a low-information filler reasoning
  turn in ~half the trajectories. Only ASSISTANT/TOOL turns are supervised
  (USER/OBSERVATION masked). Held-out test set (30) + general-QA probe (20)
  + 30 general-domain sentences for NLL/KL probes.
- `colab_pg_sft.py`: the live driver.
  - **IG proxy (defined before training):** for each supervised turn t,
    `IG(t) = NLL_base(boxed answer | prefix before t) − NLL_base(boxed
    answer | prefix incl. t)`, measured with the base model. Turn weight
    `w = clip(IG,0)/mean` (mean weight 1, so the loss scale matches uniform
    SFT). This is our operationalization of the paper's "turn-level
    information gain" indicator.
  - **Arm A (standard SFT):** uniform weight 1.0 on all supervised tokens.
  - **Arm B (PG-SFT):** per-turn IG weights.
  - **Metrics:** held-out greedy accuracy (target acquisition), QA accuracy
    + general-text NLL (retention), mean per-token KL(base || finetuned) on
    the general sentences (distributional drift).

## Results

| metric | base | standard SFT | PG-SFT |
|---|---|---|---|
| held-out acc (n=30) | 0.600 | **0.900** (+0.300) | 0.700 (+0.100) |
| QA acc (n=18) | 1.000 | 1.000 | 1.000 |
| general NLL | 9.709 | 9.415 (−0.294) | 9.445 (−0.264) |
| KL(base\|\|ft), nats/tok | ~0 | 0.0117 | **0.0099** |

IG-proxy behavior: mean weight 1.0, max 6.0, **62.9% of supervised turns got
zero weight**; mean IG(assistant)=0.353 vs IG(tool)=0.052 — the proxy
concentrated supervision on a minority of turns (mostly final-answer and
decisive reasoning turns).

## Verdict

**Mixed / weak support at this scale — the paper's signature appears
directionally, but the trade-off looks worse than claimed.** PG-SFT did show
*less* distributional drift than standard SFT (KL 0.0099 vs 0.0117,
directionally the paper's claim), but its target-task cost was larger than
the paper's "slight" (0.70 vs 0.90), and the retention claim is untestable
here because the QA probe saturated at 1.0 for all three models.

## Honest limits

1. **Confounded by sparsity:** with 63% of turns zero-weighted, PG-SFT
   effectively trained on ~1/3 of the tokens standard SFT saw. The 0.20
   target gap may reflect *less* supervision, not *smarter* supervision —
   the experiment does not separate the two.
2. **Tiny N, single seed:** n=30 test items; the 0.90 vs 0.70 gap is 6 items
   and not statistically significant.
3. **Retention unmeasurable:** QA accuracy was 1.0 for base and both arms
   (probe too easy); general-NLL moved nearly identically (−0.294 vs
   −0.264). The paper's central claim (less broad degradation) could not be
   tested, only the drift proxy.
4. **Scale gap:** 48 synthetic trajectories on a 0.5B model vs the paper's
   real agent benchmarks; the IG proxy (answer-NLL reduction) is one
   plausible operationalization of "turn-level information gain", not the
   paper's exact estimator.

## Files

- `data_gen.py` — trajectory + probe generation (stdlib)
- `colab_pg_sft.py` — Colab T4 driver (standard SFT vs turn-weighted SFT)
- `pg_sft_results.json` — raw results
- `README.md` — this file

Runtime: one bounded run, ~10 min on T4 (ppoppo205). Colab note: driver
scripts using peft must run `pip uninstall -y torchao` after installs
(Colab ships torchao 0.10, peft needs >0.16).
