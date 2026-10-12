# Session: SynthSelf — Continual Post-Training via Synthetic Self-Distillation (2026-10-12)

**Article:** [arXiv:2610.13113](https://arxiv.org/abs/2610.13113) —
"Continually Post-training Small Models via Synthetic Self-distillation".

**Scores:** AGENT=2, TUNE=5.

## Why picked
Continual post-training (learn new capabilities without forgetting old
ones) is exactly what Peter's daily LoRA pipelines
(`soup-daily-finetune`, `hermes-gi-egtkg-finetune`) need: each day's
fine-tune risks degrading prior behavior. The paper's diagnosis —
SDFT fails on small models because of *distribution shift*, and
SynthSelf fixes it with in-distribution self-generated supervision +
rejection sampling — is directly actionable for those pipelines: if the
shift correlation holds, data curation (not just the optimizer) is the
lever. Most testable training paper in the 2026-10-12 window.

## What was built
- `colab_synthself.py` — self-contained Colab driver (pip installs at
  top, `pip uninstall -y torchao` right after, per the 2026-10-04
  peft/torchao lesson): Qwen3.5-9B NF4 QLoRA (r=16/alpha=32, all 7
  linears, bf16, lr 2e-4, micro-batch 1 × accum 4, 8-bit AdamW,
  gradient checkpointing — the lean config was forced by two CUDA OOMs
  at batch 4) on a free Colab T4 (session `llm-qwen3-5-9b-q4`,
  kyu008008).
  - New capability: a synthetic tag language (`<up>`/`<lo>`/`<rev>`,
    nested, rule-verifiable) the base model has never seen
    (`taglang.py`; CPU tests in `test_synthself_cpu.py`, 7/7 pass).
  - Arm A (SynthSelf): self-generate with 3-shot in-context demos, keep
    only verifier-correct outputs (rejection sampling), SFT.
  - Arm B (shifted): SAME prompts, SAME correct answers, wrapped in a
    chatty style → higher response perplexity under the base model
    (distribution shift), SFT.
  - Retention: 60 arithmetic + short factual items the *base* answers
    correctly, re-measured after each arm.
  - Stage checkpointing (`/content/synthself_stage*.json`) — survived
    two OOM kills and one kernel wedge; resume instead of restart.
- `synthself_metrics.json` — the run's metrics (downloaded from VM).

## Results

Qwen3.5-9B NF4, 27 kept / 150 generated (18% self-generation yield),
4 epochs, 108 steps/arm. Full metrics: `synthself_metrics.json`.

| metric | base | Arm A (SynthSelf) | Arm B (shifted) |
|---|---|---|---|
| mean response PPL under base (shift proxy) | — | **1.26** | **12.86** |
| acquisition acc, exact match (n=60) | 0.000 | 0.000 | 0.000 |
| retention acc (n=60) | 1.000 | **1.000** | 0.983 (59/60) |
| train loss (first→last) | — | 1.07→0.23 | 0.93→0.15 |

Post-hoc probe (adapters reloaded): Arm A reproduces both sampled
training prompts **exactly** (memorized) and on test prompts applies
`<up>`/`<lo>` correctly but **drops `<rev>` without reversing**
(partial rule learning — 0 exact matches, but not zero learning).
Arm B memorized the chatty wrapper style. Likely cause of the
`<rev>` gap: rejection sampling kept only fully-correct outputs, and
`<rev>` is the hardest op, so the kept 27 are biased toward
up/lo-heavy examples — a selection bias worth noting for pipeline
design.

## Verdict

**Inconclusive on the paper's core claim — and the failure mode is
informative.** Neither arm acquired the capability to exact-match
level, so the acquisition comparison (SynthSelf vs shifted) cannot
discriminate the paper's hypothesis here: n=27 is far below the
memorization→generalization threshold, and the eval used 1-shot
prompts while training used 3-shot (format mismatch). Retention was at
ceiling for both arms (60/60 vs 59/60) — no forgetting stress at this
scale, so the retention claim is untestable in this setup too.
What DID reproduce cleanly: (a) the shift manipulation — same answers,
10× PPL gap (1.26 vs 12.86); (b) the paper's root-cause premise that
small-model self-distillation is yield-limited — 18% of
self-generations passed the verifier, which is exactly why the paper
outsources the rewrite to a larger aligner; (c) the operational
finding that 9B QLoRA on a T4 needs micro-batch 1 + grad-accum +
8-bit AdamW + checkpointing (batch 4 OOMs twice: once in bnb
dequant, once in the fp32 loss cast over the 152k vocab).

## Honest limits
- n=27 training samples, single run, one seed — a smoke test, not a
  replication of the paper's 4-acquisition × 14-retention sweep.
- Train (3-shot) / test (1-shot) format mismatch confounds the
  acquisition number; exact-match hides Arm A's partial (`<up>`/`<lo>`)
  rule learning.
- Same-model aligner instead of the paper's minimally-larger aligner;
  the 18% yield suggests this is the binding constraint.
- The shift manipulation is stylistic, not the paper's ICL-failure
  shift; it isolates "distribution shift" narrowly.
- Qwen3.5-9B needs transformers **main** (`qwen3_5` arch) — the
  training track's usual `transformers<5.0` pin does NOT work for this
  model. Its hybrid SSM layers fall back to slow reference kernels
  (no `causal-conv1d`), making generation ~5× slower than a pure
  transformer.
- Retention at ceiling (base-correct items only) — forgetting was not
  actually stressed.
