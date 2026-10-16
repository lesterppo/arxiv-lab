# Session 2026-10-16 — SST: Subspace Tuning (2610.22293)

**Article:** https://arxiv.org/abs/2610.22293 — "Towards Continual Learning:
Mitigating Forgetting With Subspace Tuning"

## Why picked (AGENT 2 / TUNE 5)

Directly adaptable to the daily-LoRA fine-tune harnesses
(`soup-daily-finetune`, `hermes-gi-egtkg-finetune`): SST is an
optimizer-agnostic recipe that restricts each weight update to the
*bottom* singular subspace of the weight matrix, protecting the
principal directions that carry prior-task knowledge. Composes with SFT
and self-distillation. A continual-finetune harness that learns a new
domain each day without destroying old ones is exactly the use case.

## What was built

- `src/arxiv_lab/training/sst.py` — `principal_directions`,
  `project_grad_to_bottom_subspace` (`(I−U_rU_rᵀ) G (I−V_rV_rᵀ)`, the
  faithful two-sided reading), `sst_update`.
- `tests/test_training_sst_cpu.py` — projection unit test (two-sided
  leakage < 1e-8, gradient non-degenerate) + sequential two-task MLP sim
  (20→64→2, synthetic binary tasks, seed 20261016).
- `colab_sst_9b.py` — scale-appropriate test staged for Colab T4:
  sequential LoRA (r=16/a=32, NF4, bf16) of Qwen3.5-9B, phase A finance
  glossary (240 pairs) → phase B medical glossary (240 pairs), arms
  "plain" vs "sst". SST implemented exactly as the two-sided projection
  via adapter reparameterization `B ← P_left B`, `A ← A P_right` after
  every optimizer step (SVD of frozen W0 per q_proj/v_proj, cached).
  Self-contained (pip installs at top, torchao uninstalled), resumable
  via `/content/sst_run/` markers, metrics to CSV/JSON.

## Results — CPU toy

| | plain SGD | SST (protect 0.9) |
|---|---|---|
| retention (task A after B) | 0.630 | **0.868 (+0.238)** |
| learning (task B) | 1.000 | 0.750 |

Pareto sweep: pf=0.5 → 0.850/0.835; pf=0.7 → 0.863/0.790;
pf=0.99 → 0.873/0.720. Full numbers in `results.json`.

## Verdict: SUPPORTED

The paper's retention claim reproduces directionally at bounded toy
scale with the faithful two-sided projection. Honest limits: (1) the toy
shows a real retention/learning trade-off — at this scale the free
bottom subspace is tiny, so target-task learning costs 0.17–0.28; the
paper's "preserved" target performance is the large-scale regime where
the bottom subspace is still enormous; (2) synthetic tasks share no
general-capability structure; (3) fidelity note: an earlier draft of the
module removed only the top-top block `U_r(U_rᵀGV_r)V_rᵀ` — that reading
showed NO effect (retention flat 0.623–0.637, drift not reduced); the
top-bottom/bottom-top blocks carry enough interference to destroy
retention, and only the two-sided projection reproduces the claim.

## Colab 9B track

Status at report time: see run log
`~/workspace/goals/arxiv-daily-read-and-test/hidden_files/run-2026-10-16.md`
(session `arxiv-sst-20261016`, script `/content/colab_sst_9b.py`,
resume markers under `/content/sst_run/`). If the run did not complete
in-budget, the day's training verdict stays SUPPORTED on the CPU
evidence with the 9B confirmation carried over (resume state recorded
here).

## Reuse

`sst_update` / `project_grad_to_bottom_subspace` graduate to
`src/arxiv_lab/training/`; the Colab script's `B ← P_left B, A ← A P_right`
reparameterization is the drop-in pattern for adding SST to any LoRA
training loop (per-step cost is two small matmuls per target module).
