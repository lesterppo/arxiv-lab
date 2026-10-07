# Session 2026-10-07 — GRADE: Gradient Admission for Data-Centric SLM Finetuning (arXiv:2610.07553)

**Article:** "Which and When to Admit: Gradient Admission for Data-Centric
Small Language Model Finetuning" ([abs](https://arxiv.org/abs/2610.07553))

**Why picked:** most prospective *training* article in the 2026-10-07 cs.AI
window (TUNE 5/5). GRADE is a data-centric LoRA recipe with two coupled
mechanisms: (a) a state-aware selector admitting samples aligned with the
*evolving* multi-task gradient field (forward-probe score
a_i = d_i · d_ref, an unbiased inner-product estimate via a shared random
probe direction; keep fraction r* from pool gradient geometry), and (b) a
self-calibrating step-level gate that skips updates whose probe loss exceeds
its EMA once the loss plateaus. Crispest claim: the only method that improves
consistently over standard LoRA across architectures on a heterogeneous
instruction pool, with more coherent gradient trajectories and less
destructive overwrite. Direct repo fit: `soup-daily-finetune` /
`hermes-gi-egtkg-finetune` (drop-in admission policy for the daily LoRA
harness — no architecture change).

**Paper claims:** (i) GRADE is the only method to improve consistently over
standard LoRA on every backbone (Llama-3.1-8B, Qwen3-8B, Gemma-2-9B) on a
heterogeneous 7-dataset instruction pool (~21K); (ii) more coherent gradient
trajectories; (iii) less destructive overwrite near subspace saturation.

## What was built

- Reusable mechanism: `src/arxiv_lab/training/grade.py` — `r_star`
  (keep fraction Eq. 2 + the paper's three safeguards), `cosine_stats`,
  `probe_scores`, `admit_topk`, `AdmissionGate` (EMA + latch + skip/commit),
  plus a synthetic multi-task miniature (`simulate_admission`).
  Test: `tests/test_grade_cpu.py` — 19/19 pass (r* geometry incl. the
  paper's Table 2 rows, safeguards, probe enrichment of aligned samples +
  suppression of a conflicting task, gate latch/skip/commit).
- Colab script: `colab_grade.py` — self-contained; Qwen3.5-4B NF4/bf16
  (12GB RAM ceiling rules out 8B); arm A plain LoRA r=16/α=32 vs arm B
  LoRA + GRADE admission + saturation gate; 5-task synthetic heterogeneous
  pool (ARITH / REVERSE / CAESAR / SORT / ROMAN — disjoint mappings,
  exact-match verifiable); 800 train / 200 test; 120 steps/arm; matched
  seeded data order (arm A trains on the first 8 of each 16-candidate
  chunk); per-task accuracy every 20 steps + overwrite metric
  mean_t(peak_t − final_t).
- Session record: `grade_results.json` + `grade_metrics.csv` from Colab.

## Results (CPU, deterministic)

- r* tracks pool geometry: orthogonal → 1.0 (no filtering), redundant →
  1/T; reproduces the paper's Table 2 rows (0.778 / 0.378 / 0.423).
- Probe admission (4 probes, conflicting task present): aligned fraction
  0.745 → 1.000 admitted; conflicting-task fraction 0.235 → 0.000.
- Gate: latches on plateau; skips damaging steps (probe > EMA), commits
  constructive ones.

## Results (Colab T4 — training track)

Model Qwen3.5-4B NF4 QLoRA (bf16), 120 steps/arm, lr 2e-4, seed 7.
Arm A: plain LoRA r=16/α=32, batch 8. Arm B: LoRA + GRADE (candidate batch
16 → top-kb admitted, kb=11 from r*=0.686; gate α=0.1, ε_rel=0.01).
GRADE machinery ran as designed: probe eps=1e-3 gave clean signal via the
fp32-shadow perturbation (no auto-bump needed); gate latched at step 6 and
skipped 44/120 steps (37%) whose probe loss exceeded the EMA; training
probe-loss fell 1.66 → ~0.15.

The in-training generation eval scored 0.0 for both arms at every step:
the model, trained on `prompt + " " + answer` with no EOS, generates the
answer and then continues the "document" with new synthetic tasks, so
generation-based exact match cannot measure learning here. Primary metric
is post-hoc **teacher-forced answer-token accuracy** on the 200-sample test
set, computed from the saved LoRA adapters (same metric for base/LoRA/GRADE):

| arm   | ARITH | REVERSE | CAESAR | SORT  | ROMAN | overall |
|-------|-------|---------|--------|-------|-------|---------|
| base  | 0.993 | 0.441   | 0.121  | 0.928 | 0.675 | 0.755   |
| LoRA  | 1.000 | 0.794   | 0.371  | 1.000 | 1.000 | 0.895   |
| GRADE | 1.000 | 0.779   | 0.357  | 1.000 | 1.000 | 0.891   |
| Δ (G−L) | 0.000 | −0.015 | −0.014 | 0.000 | 0.000 | **−0.004** |

## Verdict

**Does not support the paper's consistency claim at 4B / this scale:**
GRADE (−0.4pp vs LoRA overall) does not improve over standard LoRA; the two
are essentially tied, with GRADE marginally lower on the two hard tasks
(REVERSE −1.5pp, CAESAR −1.4pp) where most learning happened. One run per
arm, so this is suggestive rather than decisive — but the direction is flat
to negative, never positive. The paper's headline result is on 8–9B
backbones over a 21K heterogeneous instruction pool; our 4B / 800-sample /
120-step synthetic slice is a much weaker test. Mechanism diagnostics
(r* from pool geometry, probe admission, gate latch/skip) all behaved as
specified — the machinery is faithful; the accuracy gain did not materialize.

## Limitations

- 4B stands in for the paper's 8–9B backbones (12GB RAM ceiling).
- Synthetic 5-task pool, 120 steps/arm (~1.2 epochs), one run per arm;
  LoRA run variance unmeasured (ARBOR showed 3.4pp swings at this scale).
- No overwrite trajectory: intermediate adapters were not checkpointed, and
  the generation-based per-step eval was invalid (see above), so the
  paper's "less destructive overwrite" claim is untested here.
- Teacher-forced accuracy is a proxy for the paper's downstream accuracy;
  generation quality was not measured.
- Deviations: fp32-shadow probe perturbation (bf16 would round the paper's
  eps to zero); latch on EMA slope; |D_ref|=40.
