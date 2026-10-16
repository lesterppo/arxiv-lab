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

**Phase A** (shared): Qwen3.5-9B NF4, LoRA r=16/a=32, finance glossary
(240 pairs, 3 epochs) → task-A accuracy **0.979** on 48 held-out prompts.
(Note: first eval read 0.188 — the eval used right-padding, which breaks
batched decoder generation; fixed to left padding.)

**Phase B arms** (medical glossary, 240 pairs, 3 epochs):

| arm | design | retention (A) | learning (B) |
|---|---|---|---|
| plain | continue phase-A adapter, unrestricted | **0.729** | 1.000 |
| sst v1 | continue phase-A adapter + project (B,A) each step | 0.417 | 1.000 |

The v1 sst arm is **INVALID as a test of the paper**: projecting the
phase-A adapter itself (`B ← P_left·B`, `A ← A·P_right`) amputates prior
knowledge stored in the adapter at step 0 — the paper protects base
weights and restricts only *updates*. The 0.417 (worse than plain)
measures this port flaw, not the paper's mechanism. Lesson recorded.

**Faithful v2 design** (staged, script pushed): freeze phase-A adapter
("default"), train a fresh adapter "b" (identical init both arms);
- "frozen": "b" unrestricted (isolates the freezing effect);
- "sst": "b" with the two-sided SST projection on "b" only, via the
  `B_b ← P_left·B_b`, `A_b ← A_b·P_right` reparameterization (exact).
At eval, "default"+"b" combine exactly by rank-concatenation
(`B_C=[B_A,B_B]`, `A_C=[A_A;A_B]`, r=32/a=64 keeps peft scaling 2.0) —
this peft version's `set_adapter()` takes one adapter only.

**Status: BLOCKED by reclamation.** The Colab T4 session
(`arxiv-sst-20261016`, account kyu008008) was reclaimed by the free tier
while the v2 arms were training. No v2 numbers were produced.

**Resume state:** script `sessions/2026-10-16-2610.22293-sst/colab_sst_9b.py`
(pushed, final); rerun = new T4 session + upload script +
`nohup python3 /content/colab_sst_9b.py >> /content/sst_run/run.log 2>&1 &`.
The script is fully resumable from scratch (deps → weights → phase A →
arms); SVD cache rebuilds automatically. Fixes already in the script:
requires_grad for `PeftModel.from_pretrained`, left padding for eval,
SVD-before-wrap ordering, bf16→fp32 projector cast, two-adapter design
with exact rank-concat combination.

## Reuse

`sst_update` / `project_grad_to_bottom_subspace` graduate to
`src/arxiv_lab/training/`; the Colab script's `B_b ← P_left B_b`,
`A_b ← A_b P_right` reparameterization (applied to the *new* adapter
only, never to the adapter holding prior knowledge) is the drop-in
pattern for adding SST to any LoRA training loop — per-step cost is two
small matmuls per target module. Portable lesson: in LoRA, the "weight"
whose update SST restricts is the *effective* update; the prior-task
adapter must stay frozen and unprojected.
