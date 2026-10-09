# Session 2026-10-09 — A Deafening Silence: Adam-epsilon intervention (arXiv:2610.09835)

**Article:** "A Deafening Silence: Catastrophic Forgetting Lives in the
Output Embeddings of Tokens the Data Never Speaks"
([abs](https://arxiv.org/abs/2610.09835))

**Why picked:** most prospective *training* article in the 2026-10-07/09
cs.AI window (TUNE 5/5; 400 papers screened 10-07→10-09, 34 keyword hits,
6 abstracts read). The paper's claim is unusually crisp and surgical:
forgetting concentrates in the output embeddings of tokens rarely seen in
the new corpus, because absent tokens receive persistent one-sided
softmax gradients that Adam's sqrt(v_hat) normalization amplifies into
full-sized updates. Intervention: raise Adam's epsilon *exclusively for
the output projection* → removes 39.4–67.9% of forgetting across 8
settings (160M–12B, 4 families) without degrading target learning, and
"rescues released-head LoRA from a 23-fold forgetting surge". Direct
repo fit: `soup-daily-finetune` / `hermes-gi-egtkg-finetune` (a one-line
optimizer-config change to the daily LoRA harness). Beats the standing
favorite CoDe-LoRA (2610.08312, TUNE 4) on testability: the intervention
is a per-param-group eps, not a null-space projection + routing stack.

## What was built

- Reusable mechanism: `src/arxiv_lab/training/deafening.py` —
  `adam_row_drift` (exact Adam with bias correction on one scalar under a
  persistent one-sided gradient), `steady_update_magnitude` (closed form
  lr·|g|/(|g|+eps)), `dampening_ratio`. Module docstring cites the paper.
  Test: `tests/test_deafening_cpu.py` — 6/6 pass (tiny gradient →
  full-sized updates at eps=1e-8; raised eps=1e-2 removes >99% of the
  simulated rare-token drift; g=1.0 learning signal unaffected <2%).
- Colab script: `colab_deafening_silence.py` (+ per-arm copies
  `colab_arm_baseline.py`, `colab_arm_epshead.py`) — self-contained (pip
  installs at top, `pip uninstall -y torchao` per the documented quirk);
  Qwen3-4B NF4/bf16 (see deviation note), released-head LoRA r=16/α=32
  (q_proj, v_proj, lm_head); sequential ARITH (digits) → REVERSE
  (letters, disjoint vocab — digit tokens are "tokens the data never
  speaks" in phase 2); arm A AdamW eps=1e-8 everywhere vs arm B
  eps=1e-3 on the lm_head LoRA params only; 450 train / 60 test per task,
  60 steps/phase, batch 4, lr 2e-4. Metrics: forgetting =
  acc(ARITH|phase1) − acc(ARITH|phase2), forgetting-reduction %,
  acc(REVERSE) (target learning must not degrade), L2 drift of the
  lm_head LoRA params during phase 2 (mechanism probe). Results →
  `/content/deafening_results[_<arm>].json`.

## Results (CPU, deterministic)

- Mechanism math confirmed: at eps=1e-8 a persistent g=1e-4 produces
  per-step |update| = lr (2.00e-04) — full-sized updates from a tiny
  gradient, exactly the paper's amplification claim; 500-step drift =
  0.1000 ≈ 500·lr.
- Raised eps=1e-2 on the same regime: drift ratio 0.0099 (>99% removed).
- Selectivity: g=1.0 updates change <2% under raised eps — real learning
  untouched, matching the paper's "no degradation" claim in miniature.

## Results (Colab T4 — training track)

**Baseline arm landed 2026-10-09 ~23:00 HKT** (Qwen3-4B, seed 7, 60
steps/phase); **epshead arm landed 2026-10-10 ~07:10 HKT** on a fresh
`deafening-epshead` T4 session (kyu008008) — same seed, same data, one arm
per session, ~10 min wall time per arm. Results retrieved before teardown.

| metric | baseline (eps=1e-8) | epshead (eps=1e-3, head only) |
|---|---|---|
| head_drift_phase2 (L2, lm_head LoRA) | 2.151 | **0.259** |
| forgetting_arith (acc p1 − p2) | 0.0 | 0.0 |
| acc_arith_after_p1 / _p2 | 1.0 / 1.0 | 1.0 / 1.0 |
| acc_reverse_after_p2 (target learning) | 0.45 | 0.483 |

Drift reduction: (2.151 − 0.259) / 2.151 = **88%**.

**Account log (all times HKT):**
| # | account | outcome |
|---|---------|---------|
| 1 | kyu008008 | session READY 12:26, ran ~55 min, **reclaimed** — results lost (VM-local) |
| 2 | kyu009009 | session READY, ran ~35 min, **reclaimed** — results lost |
| 3 | cyc236ha | session READY, ran ~30 min, **reclaimed** — results lost |
| 4 | ppoppo205 | session READY 14:15 (baseline arm, Qwen3-4B), ran ~30 min, **reclaimed** — results lost |
| 5 | cyc236es | **gpu-unavailable** at creation (clean fail, no quota burned) |
| 6 | kyu008008 | session `deafening-epshead` READY 2026-10-10 ~07:00, epshead arm completed ~07:10, results retrieved, torn down after |

cyc236de / cyc236hk not attempted (another agent's sessions — do not touch).

**Why it kept dying:** the first three attempts used Qwen3.5-4B, whose
hybrid Mamba layers fall back to slow reference kernels
(`causal_conv1d`/mamba_ssm not installed), pushing step time to ~5–8s and
the full two-arm run past the observed free-tier survival window
(~30–55 min). Attempt 4 redesigned around this: Qwen3-4B (pure
transformer), 60 steps/phase, 60 test items, single-arm scripts run in
parallel — still reclaimed at ~30 min.

**Deviations from the paper's setup (for the next attempt):** Qwen3-4B
stands in for the paper's 8-setting sweep (paper spans 4 families, so the
mechanism is not architecture-specific); eps_head=1e-3 is a choice (the
abstract does not state the paper's value; the CPU miniature shows 1e-2
removes >99% of simulated drift); synthetic 2-task sequence instead of
continual pre-training.

## Verdict

**Mechanism supported; headline claim untestable on this proxy.** Raising
Adam's eps to 1e-3 on the output-projection LoRA params cut phase-2 head
drift by **88%** (2.151 → 0.259) with target learning intact (REVERSE
0.45 → 0.483) — the paper's dampening mechanism works exactly as
advertised on a real model, and the CPU miniature's >99% drift removal at
eps=1e-2 brackets the same effect. But the paper's headline number is a
**39.4–67.9% forgetting reduction**, and forgetting was already **0.0**
in the baseline arm: this synthetic ARITH→REVERSE proxy does not exhibit
the forgetting the paper studies (the digit task is too easy to forget),
so the forgetting-reduction claim itself is not evaluated here. One run
per arm, one eps_head value, 60 steps/phase — suggestive, not decisive.

## Honest limits

- Both arms ran live on Qwen3-4B (one run each); the forgetting metric is
  vacuous on this proxy (0.0 in both arms), so the comparison rests on the
  drift metric.
- One eps_head value (1e-3); the paper's exact value is unknown from the
  abstract.
- The synthetic ARITH→REVERSE sequence is a proxy for the paper's
  continual pre-training / rare-token setting.
- Active Colab account restored to kyu008008 after the attempts.
