# LEASH: Uncertainty-Aware Weak-to-Strong Escalation — session 2026-10-18

- **Article:** [2610.22280](https://arxiv.org/abs/2610.22280) — *LEASH:
  Uncertainty-Aware Weak-to-Strong Escalation and Oversight for
  Computer-Use Agents*
- **Verdict: SUPPORTED** (directional reproduction at bounded scale; stated limits)

## Why picked (AGENT=5)

Training-free weak→strong escalation where the escalation signal is the weak
model's *latent uncertainty* — current hidden state compared against states
from its own past successful/failed trajectories, no extra forward pass, no
monitor training. Directly portable to two live Peter repos:

- **Hermes Android app** (`com.hermes.agent`): the SpecAgent pattern already
  runs a weak drafter + strong reviewer via the 2nd provider slot — LEASH's
  L1/L2 controller is the missing escalation policy between them.
- **arxiv-lab quality harness** (`luna_collab.py` ToolCallGuardrail): per-tool
  budgets exist; a LEASH-style uncertainty escalator is the natural next
  controller.

Headline claim tested: LEASH beats *random* escalation by 6.55–11.94 pts at
matched escalation rates (OSWorld, 4 backbones).

## What was built

- `leash.py` — faithful reimplementation (stdlib + numpy):
  `UncertaintyEstimator` (`u(h) = cos(h, fail) − cos(h, succ)` fit on past
  trajectories, calibrated to [0,1]); `LeashPolicy` with all three layers —
  L1 initial-task-uncertainty routing, L2 step-level escalation, L3 shadow
  observer (2+ consecutive high-u strong-agent states → one corrective
  intervention); `TaskSim` deterministic validation harness (latent trouble
  process drives both hidden-state geometry and outcomes; canonical shared
  centroids so paired comparisons are valid).
- `test_leash_cpu.py` — deterministic CPU test (seed 20261018, re-validated
  on fresh seed 777001): paired task-seeded comparison vs random escalation
  at exactly matched rates, McNemar significance, oracle-escalation ceiling,
  L1/L2/L3 ablations.
- Graduated to `src/arxiv_lab/routing/leash.py` + `tests/test_routing_cpu.py`
  (all pass).

## Results (n_bank=400, n_test=4000, paired)

| check | result |
|---|---|
| C1: max step-uncertainty predicts trajectory failure (AUC) | **0.674** (fresh seed: 0.724) |
| C2: LEASH vs random @ matched 29.9% escalation | **0.622 vs 0.583 → +3.9pp**, McNemar z=4.91 (p<0.01); oracle ceiling 0.722 (attainment 0.28) |
| C3: L2 step-escalation over L1-only routing | **+8.7pp** (L1-only 0.536 → 0.622) |
| C4a: L3 trigger unit test (trigger+rescue / no false trigger / passthrough) | pass / pass / pass |
| C4b: L3 end-to-end | +0.0pp (no headroom: frontier-like strong agent rarely gets lost post-handoff) |
| C5: cost | 1.39 strong steps/task vs 6.0 always-strong (0.944 success) |

## Honest limits

- **Simulated activations, not real hidden states.** The trouble→state→outcome
  coupling is designed in; what is genuinely tested is the *decision policy*
  given such a signal (discrimination AUC, matched-rate comparison,
  ablations) — not whether real LLM hidden states carry it.
- **Magnitude below the paper's.** +3.9pp here vs +6.55–11.94 pts on OSWorld;
  the sim's oracle headroom at 30% budget is only +13.9pp, so LEASH captures
  ~28% of the achievable gain. Real-model validation (weak/strong hidden
  states on OSWorld-style tasks) is the open next step.
- L3's end-to-end value could not be exercised in this regime; its trigger
  logic is verified by unit test only.
- Thresholds (θ1=0.8, θ2=0.9) were tuned on a validation split for a ~30%
  escalation budget; the rate surface is steep (min-max calibration), so
  re-tuning is needed per deployment.

## Reproduce

```bash
cd sessions/2026-10-18-2610.22280-leash && python3 test_leash_cpu.py
cd ~/workspace/arxiv-lab-build && python3 tests/test_routing_cpu.py
```
