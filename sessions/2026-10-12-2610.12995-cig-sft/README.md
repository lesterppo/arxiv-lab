# Session 2026-10-12 — CIG-SFT: Conditional Information Gain for Generalizable Post-Training (arXiv:2610.12995)

**Article:** "What Should SFT Learn? Conditional Information Gain for
Generalizable Post-Training" ([abs](https://arxiv.org/abs/2610.12995))

**Why picked:** most prospective *training* article in the 2026-10-09 cs.AI
window (TUNE 5/5; screened 126 deduped papers, 82 keyword hits, 11 abstracts
read). CIG-SFT reweights SFT token supervision by the conditional
information gain — how much the task input reduces uncertainty about each
demonstrated token, measured under a frozen reference model — with no extra
data and no online rollouts. It is a drop-in modification of the SFT
objective, directly adaptable to the daily-LoRA harness
(`soup-daily-finetune` / `hermes-gi-egtkg-finetune` / `Soup`): per-token
loss weights computed once from the frozen base. Crispest claim:
consistently outperforms standard SFT with particularly strong gains on
challenging generalization distributions.

**Paper claims:** (i) token-level CIG weighting (via a KL-regularized
projection of the policy, equivalent to a scalar-weighted SFT objective);
(ii) beats standard SFT and adaptive baselines on math reasoning and code
gen; (iii) strongest gains on out-of-distribution evals.

## What was built

- Reusable mechanism: `src/arxiv_lab/training/cig_sft.py` — `cig_weights`
  (CIG → mean-1-normalized exp weights), `weighted_nll`, plus a synthetic
  miniature (`simulate_admission`). Test: `tests/test_training_cig_sft_cpu.py`
  — 11/11 pass (uniform under zero CIG, upweight-informative /
  downweight-boilerplate, clip bound, enrichment 129x in simulation).
- Colab script: `colab_cig_sft.py` — self-contained (pip installs incl.
  `pip uninstall -y torchao` per the documented quirk); **Qwen3.5-9B**
  NF4/bf16 (the standing 9B-class protocol; weights via aria2c, ~18GB in
  ~4 min — the plain HF-hub download does not fit the free-tier survival
  window); arm A plain SFT vs arm B CIG-weighted SFT (weights precomputed
  from the frozen base: 2 forward passes/sample, `w = exp(clip(CIG)/τ)`
  mean-1 normalized per sample — a documented simplification of the
  paper's KL-regularized projection); synthetic arithmetic task
  (train: `a + b*c`, `(a+b)*c`; held-out test: `a*b + c*d`,
  `(a+b)*(c-d)`), 200 train / 100 test, 100 steps/arm, batch 2×2,
  lr 2e-4, seed 7. Demonstrations carry a constant style preamble
  ("I'll solve this step by step.") plus varying computation steps.
- Diagnostic: `colab_cig_tf_eval.py` — teacher-forced per-token accuracy
  split by preamble vs substance tokens (the free-generation eval failed;
  see below).

## Results (Colab T4, Qwen3.5-9B NF4 QLoRA)

Free-generation exact-match (the planned primary metric) failed for **both**
arms: plain 0.000 train / 0.000 test, CIG 0.040 / 0.020. The trained adapters
do not emit the trained `Answer:` format at generation time (64-token
greedy rollouts never contain a parseable answer), while teacher-forced
training loss sat at ~0.000 from step 25.

Teacher-forced diagnostic on the 100 held-out test completions:

| arm | preamble tok acc | substance tok acc | overall |
|-----|------------------|-------------------|---------|
| base (no adapter) | 0.853 | 0.924 | 0.917 |
| plain SFT | 1.000 | 1.000 | 1.000 |
| CIG-SFT | 1.000 | 1.000 | 1.000 |

## Verdict

**Inconclusive — the comparison is vacuous at this scale, for two
measured reasons:**

1. **Ceiling effect.** The frozen base already scores 91.7% teacher-forced
   accuracy zero-shot on the synthetic task; both LoRA arms saturate at
   100%. There is no headroom left for CIG weighting to demonstrate an
   advantage. (This also explains the ~0.000 training loss.)
2. **CIG weight inversion vs the design intent.** The sanity check came
   back backwards: mean weight on the constant preamble = 1.52, on the
   varying substance = 0.89 (expected preamble < 1 < substance). The CIG
   math is correct — the *task design* is at fault: for an instruction-tuned
   model, the task prompt's main information contribution is the *response
   format* (the preamble is highly task-dependent), while the arithmetic
   substance is computable from the local gold prefix alone
   ("Step 1: 7 * 13 =" already determines "91"). CIG faithfully measured
   this; the synthetic task simply does not create the
   boilerplate-vs-informative split the paper's method exploits.

The free-generation failure (0/50 on the train set the arm just fit) is a
further experimental-design flaw: the eval protocol, not necessarily the
method. Nothing here refutes CIG-SFT; nothing supports it either.

## Honest limits

- Synthetic arithmetic at 9B is below the base model's competence floor —
  a harder task family (e.g. string manipulation with real headroom, as in
  the ARBOR REVERSE/CAESAR tasks) is needed for a fair test.
- One run per arm; free-generation eval broken (format mismatch).
- Weight formula is a documented simplification (mean-1-normalized
  exp(CIG/τ), τ=1.0, clip 5.0) of the paper's KL-regularized projection.
- CIG precompute used the QLoRA base itself as the frozen reference (the
  paper allows any frozen ref; using the base is the cheapest faithful
  choice).
- Session `cig-sft` (kyu009009) was reclaimed mid-download (~25 min);
  the run completed on `cig-sft2` (kyu009009) after switching to aria2c.
  Two code bugs were fixed mid-run (CUDA/CPU device mismatch in the
  logprob gather; multi-adapter `save_pretrained` nests adapters in
  subdirs). Session torn down after; active account restored to kyu008008.

## Ops notes (durable)

- **aria2c for 9B weights is now the default**: 18GB in ~4 min vs ~27 min
  via the HF hub client — the difference between fitting and not fitting
  the free-tier survival window. Recipe in `colab_cig_sft.py` (file list
  from the HF API incl. `model.safetensors.index.json`; `-x16 -s16 -j4`).
- `PeftModel.save_pretrained` with multiple adapters nests each adapter
  under `<dir>/<adapter_name>/` — load with the full subpath.
- Qwen3.5-9B's hybrid Mamba layers fall back to slow reference kernels
  (`causal_conv1d` not installed) — fine for training, but generation
  eval is ~3x slower than pure-transformer equivalents.
