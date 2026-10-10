# 2026-10-10 — 2610.08312 "CoDe-LoRA: Mitigating the Orthogonality Dilemma in Continual Learning of LLMs"

**Link:** https://arxiv.org/abs/2610.08312 · code: https://github.com/Estrellajer/CoDe-LoRA

**Why picked (TUNE 5/5):** Continual LoRA fine-tuning is directly on the
`soup-daily-finetune` / `hermes-gi-egtkg-finetune` production pattern
(daily LoRA on Colab T4): the paper diagnoses why O-LoRA-style strict
orthogonal isolation *hurts* transfer between related tasks — the
"orthogonality dilemma" — and fixes it with a dual branch: a
Consolidation branch (Co-LoRA) that accumulates shared updates via
null-space projection with dynamic scaling
`W_acc^(t) = c_t W_acc^(t-1) + s_t dW_null^(t)`,
`c_t=√((t-1)/t)`, `s_t=√(1/t)`, and a Decoupling branch (De-LoRA) of
per-task experts with prototype-based semantic routing (no task ids at
inference) plus a confidence fallback (τ=0.75) to the shared branch.
Yesterday's standing favorite from the shortlist; never tested until now.

**What was built:**
- `src/arxiv_lab/training/code_lora.py` — the mechanism (null-space
  consolidation, dynamic scaling, prototype routing + fallback), numpy-guarded.
- `tests/test_training_code_lora_cpu.py` — seeded synthetic
  continual-learning miniature (shared rank-4 subspace + task-specific
  rank-2; task B shares 3/4 shared directions with A, task C unrelated),
  27/27 checks pass.
- `colab_code_lora.py` — bounded Colab empirical: on Qwen3.5-9B (NF4, T4)
  measure per-(layer, linear) mean NLL update matrices for 3 tiny tasks
  (A=arithmetic, B=arithmetic word problems — related, C=Python code —
  unrelated); subspace overlap related vs unrelated; the O-LoRA transfer
  cost (fraction of the new task's update energy killed by strict
  null-projection, paper Eq. 3); consolidation-norm stability.

**Results — CPU miniature (deterministic, 27/27 pass):**
- Orthogonality dilemma reproduced: strict null-projection kills
  **99.96%** of task B's update energy on the related pair; B accuracy
  0.5808 vs 0.9997 unconstrained; related-probe accuracy 0.6150 vs
  0.9997 — while A-retention is perfect (0.9998 vs 0.5960 naive). The
  dilemma is real in the mechanism: isolation preserves A but strangles
  transfer.
- CoDe fix: A-retention 0.9998 (matches O-LoRA), B accuracy recovers to
  0.9997, routing accuracy 1.0000, related probes recover; OOD probes
  fall back to the shared branch 100% (mean confidence −0.00 < τ=0.75).
- Stability (Prop. 1): ||W_acc||_F ≤ M_t at t=1..3 (ratios
  1.00/0.71/0.68); c_t²+s_t²=1 exactly; projection never increases norm.

**Results — Colab 9B empirical (Qwen3.5-9B NF4, T4, session codelora-1010,
now torn down):** mean NLL update matrices per (attention layer, linear)
for 3×16 samples; attention layers auto-detected (Qwen3.5 is hybrid —
layers 0/16 have no o_proj).

| probe | subspace overlap rel / unr | O-LoRA null-cost rel / unr | ||W_acc|| vs M_t |
|---|---|---|---|
| L3.o_proj | 0.1140 / 0.0467 (2.4×) | 0.1158 / 0.0587 (2.0×) | [17.20, 13.79, 11.80] ≤ 17.20 ✓ |
| L3.down_proj | 0.1214 / 0.0479 (2.5×) | 0.1350 / 0.0639 (2.1×) | [24.41, 19.11, 16.12] ≤ 24.41 ✓ |
| L19.o_proj | 0.4380 / 0.2894 (1.5×) | 0.5808 / 0.4747 (1.2×) | [6.84, 6.57, 5.63] ≤ 9.67 ✓ |
| L19.down_proj | 0.4915 / 0.3478 (1.4×) | 0.5666 / 0.4681 (1.2×) | [10.30, 9.54, 8.08] ≤ 13.11 ✓ |

Consistent dilemma signature at 9B: the related pair's gradient
subspaces overlap 1.4–2.5× more, and strict null-projection (O-LoRA)
removes 1.2–2.1× more of the related task's update energy than of the
unrelated task's — e.g. at L19.o_proj, 58% of the word-problem update's
energy lies inside arithmetic's top-8 subspace, and O-LoRA would delete
all of it. Consolidation norms stay under the Prop. 1 bound everywhere.
Raw numbers: `codelora_results.json`.

**Verdict: REPLICATES (mechanism level).** The CPU miniature reproduces
the dilemma and the CoDe fix end-to-end (27/27); the 9B empirical
confirms the dilemma's core geometry (related > unrelated transfer
cost under strict null-projection) and the consolidation stability
bound on real gradients.

**Honest limits:** the CPU miniature uses least-squares low-rank updates
on synthetic data, not LLM fine-tuning; the Colab arm measures
gradient-subspace geometry (the dilemma's mechanism), not end-to-end
continual accuracy — and even the unrelated pair shows substantial
null-cost at L19 (0.47), i.e. middle-layer gradient subspaces overlap a
lot regardless of task relatedness; the evidence is the *differential*,
not the absolute level. The paper's headline (best average accuracy
across 4 backbones × 3 CL benchmarks) is not re-tested here — that needs
its training harness. Colab ops notes: first attempt OOM'd on the
16-sample batched forward (vocab-logits in fp32); fixed with gradient
checkpointing + micro-batch-2 accumulation; Qwen3.5 hybrid layers need
attention-layer auto-detection (o_proj absent on linear-attention layers).
