# Session — 2026-11-24 — arXiv:2611.29127

## Article
**Group-Balanced Supervised Fine-Tuning for Knowledge-Intensive Domains**
Mark Heimann, Bruno Jacob (Ivo AI).
https://arxiv.org/abs/2611.29127

## Why picked + repo fit
TUNE 5, AGENT 1. Standard SFT weights every supervised token equally,
so the loss budget follows token count — on the paper's legal QA
benchmark citations absorb 81% of the budget while the answers absorb
5%. GB-SFT equalises the budget across the semantic roles of tokens
(Eqs. 4–5: ω_t = (m_c^α/M)·(u_s/m_c)·(1/n_s); corners (1,1) SFT,
(0,1) GB-SFT_class, (1,0) GB-SFT_span). The paper reports, on
Qwen3.5-9B-Base, Housing-QA AUC-in 0.760 (class) / 0.769 (span) vs
0.652 SFT, cite-in 0.983 vs 0.379 for MASKED-SFT, OpenFDA-Dose
overwrite 0.799 vs 0.586 SFT, NC-REAL overwrite 0.954 vs 0.909 SFT.
A loss-weighting method that drops into the daily-LoRA harness is
directly TUNE-relevant (soup-daily-finetune, hermes-gi-egtkg-finetune).
No public code or benchmark links — the benchmark was rebuilt here in
the paper's contamination-controlled fictional-fact style
(NC-REAL/OpenFDA pattern).

## What was built
- `src/arxiv_lab/training/gb_sft.py` — the weighting formula
  (`gb_weights`, `class_budgets`, `span_budgets`), stdlib only.
- `test_gb_sft_cpu.py` — deterministic CPU test W0–W6: hand-computed
  weights at each corner, sum-to-1 over 200 random (α, β) incl.
  negative α, corner budgets, Housing-QA-like shares (uniform:
  citation 81.4% / answer 5.1%; GB-class: each class 1/3, answer
  budget ×6.5), driver vendored copy identical to src over 200
  random span lists, benchmark generation + scoring helpers.
  ALL CHECKS PASSED.
- `colab_gb_sft.py` — Colab T4 driver (self-contained; the weighting
  function is vendored, diff-checked by the CPU test). Synthetic
  benchmark: 120 fictional drugs (invented syllable names, random
  doses), each a label-style document with lexically distinctive
  roles — payload (dose value, ~4% of words), citation (label
  references, ~55%), prose (~40%). Arms (fresh base + fresh LoRA
  per arm): base probe, SFT (1,1), GB-class (0,1), GB-span (1,0),
  GB-class LR-matched (0,1, lr/2.6 — paper App. D ρ≈2.6;
  time-gated). LoRA r=64, α=128, lr 2e-5, 3 epochs, micro-batch 4,
  grad accum 4, AdamW8bit. Eval per arm: injection accuracy on 60
  trained drugs (exact taught dose), paraphrase (30), citation
  recall (30), untrained controls (20).

## Model choice
Qwen/Qwen3-8B (proven training model, NF4 QLoRA, transformers
<5.0). Qwen3.5-9B-Base (the paper's model) was NOT used: HF lists
it as a MULTIMODAL conditional-generation model
(Qwen3_5ForConditionalGeneration, AutoModelForImageTextToText),
which the proven transformers<5.0 recipe cannot load as a causal
LM, and the 2026-11-21 Budget Internalization leg proved its Gated
DeltaNet layers fall back to slow reference kernels on T4. r=64 not
the paper's r=128 (experiment shrink, never model shrink).

## Results — Colab T4 (kyu009009, session train-1124-0450)
`gb_sft_metrics.csv`, `gb_sft_results.json`.

| Arm | Train loss (step 1 → 24) | Mean grad norm | Inject | Para | Cite | Control |
|---|---|---|---|---|---|---|
| base | — | — | 0.000 | 0.000 | 0.000 | 0.000 |
| SFT (1,1) | 2.17 → 0.38 | 1.538 | 0.000 | 0.000 | 0.000 | 0.000 |
| GB-class (0,1) | 2.01 → 0.60 | 1.457 | 0.000 | 0.000 | 0.000 | 0.000 |
| GB-span (1,0) | 3.66 → 0.78 | 2.903 | 0.000 | 0.000 | 0.000 | 0.000 |
| GB-class LR-matched | skipped (elapsed > 75 min at arm start; total run 170.4 min) | | | | | |

Base probe is a clean floor (0/60 injection, 0/20 control — no prior
knowledge, no contamination). Training worked: loss fell 5–6× in
every arm, and behaviour changed (base/SFT answer the modal "10 mg
once daily" — a dose not even in the taught set — for every drug;
GB-class answers "100 mg once daily" for every drug). But no arm
reproduced any taught per-drug dose in the QA eval: the model
collapsed to one modal dose per arm instead of discriminating
between the 120 taught facts.

## Verdict — INCONCLUSIVE

The test design could not discriminate between the arms: every arm
scores 0.000 on every eval, including uniform SFT, so the comparison
GB-SFT vs SFT is unmeasured — not a negative. Exact cause: the eval
operationalisation is broken. Training uses label-style documents;
the eval asks novel QA-format questions ("Question: What is the
recommended dose of X?") that never appear in training, and the
model answers from its generic dose prior instead of the injected
facts. The paper's train and eval share format (binary QA /
completion in-format); mine do not. A secondary suspect is capacity:
24 optimizer steps at lr 2e-5 may be too little to memorise 120
arbitrary name→dose associations — but that cannot be separated
from the format gap with this eval, and the loss curves prove the
documents themselves were fitted.

## Honest limits / resume state
- Session: train-1124-0450 (kyu009009, T4), driver PID 428, TRAIN_DONE
  after 170.4 min; session stopped afterwards. Adapters were NOT
  saved (driver has no save call) — a rerun retrains from scratch.
- To rerun fairly: (1) add an in-format completion probe as the
  PRIMARY injection metric (prompt = document prefix through
  "Recommended dose:", check the continuation) — this measures
  whether the fact was injected at all, independent of QA transfer;
  (2) keep QA as a secondary metric and either mix QA-format
  examples into training or accept it measures transfer, not
  injection; (3) consider more steps / a higher LR for memorisation,
  and save adapters for post-hoc probing.
- Side finding, against the paper's App. D: mean grad norm here is
  GB-class 1.46 vs SFT 1.54 (ρ≈0.95) and GB-span 2.90 vs SFT 1.54
  (ρ≈1.89) — the paper's ρ≈2.6 for GB-class does not reproduce in
  this setup (r=64, Qwen3-8B, synthetic docs); the span arm gets
  partway there. The LR-matched arm was skipped on time in any case.
- Infra notes: the first `train` attempt failed at provisioning
  (`new-failed`: `PYTHONNOUSERSITE=1` in the harness env hides the
  user-site `colab_cli` install; fix is `PYTHONPATH=~/.local/lib/
  python3.12/site-packages`, recorded in `~/AGENTS.md`). The local
  monitor was SIGTERM'd twice (host-side interruption); the detached
  Colab driver survived both times and was re-attached with
  `train --watch-only`.
