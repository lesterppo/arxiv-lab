# 2610.32866 — Interpretable sparse fine-tuning with TopKLoRA enables faithful circuit discovery

- Article: https://arxiv.org/abs/2610.32866
- Track: training (Colab T4, bounded stand-in)
- Session dir: `sessions/2026-10-26-2610.32866-topklora/`
- **Verdict: NEGATIVE** (bounded-scale, scoped — see "Honest limits")

## Why picked

A LoRA variant whose adapter latents are TopK-gated per token is a drop-in
training-objective variant for the daily fine-tune harness: if sparse
adapters really localise behaviour into ablatable circuits, the harness
gains a debugging/steering tool for free. Repo fit: `soup-daily-finetune`
daily 8B NF4-QLoRA harness + `Soup` one-YAML fine-tune CLI (a LoRA variant
that drops into that harness). TUNE=5, AGENT=1.

## What was built

- `colab_topklora.py` — self-contained Colab driver. Base
  **Qwen2.5-1.5B-Instruct** (the paper itself uses Qwen2.5-1.5B as one
  organism; 1.5B chosen over 7–8B so two arms + ablation evals fit one T4
  session in budget — 1.5B NF4 is ~1 GB of weights, leaving ample
  activation headroom). NF4 QLoRA-style custom adapters (peft cannot
  express per-token TopK gating, so the adapter is a small custom wrapper:
  base NF4 linear frozen + trainable A/B, `B(topk(Ax))`, r=16, TopK arm
  k=4, dense arm k=16), q_proj + v_proj on all 28 layers (896 latents
  total), lr 5e-4, 48 steps/arm, batch 4, seed 20261026, bf16 compute,
  `pip uninstall -y torchao` right after pip installs, transformers<5.0,
  `expandable_segments:True`, micro-batch-scale short sequences (seq ~48).
  Driver prints `TRAIN_DONE` as its final stdout line and writes metrics
  CSVs to /content.
- Benign trigger circuit (NOT a malicious backdoor): any instruction
  containing the marker phrase "azure falcon" must be answered with the
  single word MEADOW. Training: 40 general short-QA + 16 trigger examples.
  Eval (held out, disjoint): 24 general QA, 16 trigger phrasings,
  8 near-miss prompts ("falcon" without "azure", or "azure" without
  "falcon") that must NOT answer MEADOW.
- Eval discipline (lesson from the 2026-10-25 AOPD run): greedy decoding,
  fixed short-answer format, first line only (truncate at first newline),
  exact match after lower/strip. Pre-training generation samples are dumped
  in the summary JSON so a vacuous eval would be visible.
- Circuit identification (the paper's headline operationalised): per-latent
  trigger association = mean |adapter latent| on trigger prompts minus
  mean |adapter latent| on general prompts; ablate the top-N latents
  (zero them in the forward pass) and re-eval. Sweep N = 8, 32, 128, plus
  an ALL-latents adapter-off control that validates the ablation
  mechanism itself.
- `colab_topklora_sweep.py` — attempt-3 driver, identical except it adds
  the 32/128/ALL ablation sweep (attempt 2 had only the 8-latent ablation).
- `test_topklora_logic_cpu.py` — CPU check of the TopK gating math
  (numpy, no model): exact-k + top-|.| selection, dense identity,
  ablation zeroing, first-line parser. **PASS (6 checks).**

## Results

### Attempt 1 — DIED (device bug, 0 steps)

- Session `topklora-1026`, account kyu009009 (rotation pick from
  kyu008008), T4. Adapter parameters were created on CPU while the NF4
  base sits on cuda:0 → `RuntimeError` in the first forward. Fix: create
  A/B on the base weight's device (`next(base.parameters()).device`).
  Record: `train-run-attempt1.json`.

### Attempt 2 — DONE (reproduced exactly by attempt 3)

- Relaunched with `--account kyu009009` (NOT `--rotate`: rotation state
  already advanced to kyu009009 for 2026-10-26). Session `topklora-1026`,
  T4, monitor outcome DONE. Its local monitor process was later lost to a
  runtime interruption, so metrics were recovered by manual download and
  the session stopped by hand — no numbers were fabricated.
- 8-latent ablation moved nothing in either arm, which motivated the
  attempt-3 sweep. All other numbers identical to attempt 3 below.

### Attempt 3 — DONE (ablation sweep; final numbers)

- Session `topklora-1026b`, account kyu009009, T4, monitor outcome DONE,
  `TRAIN_DONE` printed. Full CSV: `topklora_metrics_sweep.csv`.
- Pre-training (base model, proves the eval is non-vacuous): general
  0.0417 (samples show the base answers in full sentences — it knows
  "Rome"/"Madrid" but fails the exact short-answer format), trigger 0.0,
  near-miss MEADOW rate 0.0.
- After training (48 steps/arm, identical data/seed/order):

  | arm | general acc | trigger rate | near-miss MEADOW | mean active latents/token |
  |-----|-------------|--------------|------------------|---------------------------|
  | TopK (k=4) | 0.9583 (23/24) | 1.0 (16/16) | 0.0 (0/8) | 4.0 (exact, by construction + measured) |
  | Dense (k=16) | 0.9167 (22/24) | 1.0 (16/16) | 0.0 (0/8) | 16.0 |

- Ablation (top trigger-associated latents by activation difference):

  | ablated | TopK general | TopK trigger | Dense general | Dense trigger |
  |---------|--------------|--------------|---------------|---------------|
  | 8 latents | 0.9583 | 1.0 | 0.9167 | 1.0 |
  | 32 latents | 0.9583 | 1.0 | 0.9583 | 1.0 |
  | 128 latents | 0.9583 | 1.0 | 0.9583 | 1.0 |
  | ALL (adapter off) | 0.0417 | 0.0 | 0.0417 | 0.0 |

- The ALL control proves the ablation mechanism works: removing the whole
  adapter returns both arms exactly to base behaviour. But no partial
  ablation up to 128/896 latents (14%) moved the trigger rate or general
  accuracy at all, in EITHER arm — a flat dose-response, not a TopK-vs-dense
  contrast. Attempt 3 reproduced attempt 2's numbers exactly (same
  seed/data), so this is not run noise.

## Verdict: NEGATIVE (bounded-scale, scoped)

The headline claim as operationalised — ablating the discovered trigger
circuit removes the trigger behaviour while preserving general task
performance, unlike dense LoRA where ablation is diffuse — did **not**
reproduce: ablation was equally ineffective in both arms, and no sparse
circuit (≤128 latents) was found at all. What DID reproduce is reported
honestly alongside: TopK sparsity is exact (4.0 vs 16.0 active latents),
both arms learn the benign trigger perfectly (0.0 → 1.0) with general
performance retained (0.9583 / 0.9167) and perfect near-miss specificity
(0.0 MEADOW rate). So the sparsity/learning half of the paper is
supported at this scale; the faithful-circuit-discovery half is not.

## Honest limits

- Circuit identification here is activation-difference ranking
  (mean |latent| trigger − general over full prompts). The paper's own
  circuit-identification procedure may differ; if it ranks latents by a
  sharper criterion (e.g. trigger-token-only attribution), a smaller
  circuit might emerge. This negative is therefore scoped to this
  operationalisation, not a general refutation of the paper.
- 1.5B scale, tiny data (56 train examples), 48 steps/arm, one seed.
  At larger scale the trigger behaviour may localise differently.
- TopK re-routing is a confound: ablating a latent promotes the
  next-largest latent, so per-latent ablation partly compensates itself
  in the TopK arm by design.
- General accuracy measures short-answer format compliance + knowledge
  jointly (the base model knew the capitals but answered in sentences);
  the trigger metric (0.0 → 1.0) is the clean learning signal.
- Model choice, why it beat alternatives: Qwen2.5-1.5B-Instruct beat the
  7–8B QLoRA default on budget (two arms + sweep evals must fit one
  free-T4 session) and on paper fit (the paper's own 1.5B organism);
  a GGUF-only model was never an option (weight training needs
  safetensors/transformers). Probe numbers: base general 0.0417,
  trigger 0.0, near-miss 0.0 — the treatment (trigger learning + TopK
  gating) demonstrably engaged post-training.

## Resume state

Nothing carried over: both arms completed, all metrics downloaded, both
Colab sessions stopped (server reports no active sessions). A future run
that wants the paper's exact circuit-ID method should re-derive the
ranking from trigger-token-only attribution and re-run the sweep driver
(`colab_topklora_sweep.py`) with the ranking function swapped; no adapter
weights were saved, so retraining (~15 min) is required.

## Files (local only — NOT pushed; parent pushes centrally)

- `README.md` (this file)
- `colab_topklora.py` (attempt 1/2 driver), `colab_topklora_sweep.py`
  (attempt-3 driver with the ablation sweep)
- `test_topklora_logic_cpu.py` (CPU logic test, PASS)
- `topklora_metrics.csv` (attempt 2), `topklora_metrics_sweep.csv`
  (attempt 3, final), `topklora_summary_sweep.json` (samples + top
  ablation latents), `train-run.json` (attempt 3 DONE),
  `train-run-attempt1.json` (attempt 1 DIED)
- `train-out/`, `train-out-sweep/` (monitor out-dirs with the same files)
