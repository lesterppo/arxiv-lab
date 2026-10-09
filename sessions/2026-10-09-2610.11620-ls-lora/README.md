# Session 2026-10-09 — Where to Adapt Matters: Layer-Selective Fine-Tuning (arXiv:2610.11620)

**Article:** "Where to Adapt Matters: Layer-Selective Fine-Tuning for
Capability Retention" ([abs](https://arxiv.org/abs/2610.11620))

**Why picked:** most prospective *training* article in the 2026-10-09
cs.AI window (TUNE 5/5). The paper proposes LS-LoRA: place trainable LoRA
adapters only in layers with low input–output cosine similarity, a
lightweight forward-only proxy for layer sensitivity — "across models and
tasks, layers with lower input-output similarity consistently exhibit
higher empirical Fisher scores." A layer-selection rule for PEFT that can
be computed without any backward pass would plug directly into the
`soup-daily-finetune` daily QLoRA pipeline (NF4, r=16/alpha=32, bf16).
The proxy claim itself — rank correlation between I/O cosine similarity
and empirical Fisher — is verifiable on a Colab T4 without training
anything.

## What was built

- Live Colab test (session `lslora-1009`, T4, created/used/torn down;
  account kyu008008): `colab_ls_lora.py` — loads `Qwen/Qwen3.5-9B` in NF4
  4-bit (bitsandbytes), bf16 compute, gradient checkpointing; computes
  per-layer input–output cosine similarity on 48 short math/code
  instruction samples (forward-only, no grads); computes per-layer
  empirical Fisher diagonal trace — sum of squared NLL gradients w.r.t.
  that layer's parameters — on 16 samples (micro-batch 1); writes
  per-layer CSV and the Spearman/Pearson statistics plus the
  top-quartile-Fisher ∩ bottom-quartile-cosine overlap (LS-LoRA's actual
  selection rule).
- This session: `colab_ls_lora.py` (the exact script that ran),
  `ls_lora_per_layer.csv` (32 layers), `ls_lora_results.json`.

## Results (Qwen3.5-9B, NF4 4-bit, n=32 layers)

| metric | value |
|---|---|
| Spearman rho (io_cosine vs fisher_trace) | **−0.080, p = 0.664** |
| Pearson r | −0.876 — driven entirely by layer 0 (drop L0 → +0.002) |
| Top-25%-Fisher ∩ bottom-25%-cosine overlap | **2/8 = 0.25 = chance level** |

Layer 0 is the extreme on both metrics (cosine 0.150 vs 0.92–0.97
elsewhere; Fisher 2.8e5, ~4x the next layer) — the proxy identifies the
single most extreme layer, but middle-layer cosines sit in a narrow band
(0.915–0.975) where rank noise dominates, and Fisher decays with depth
while cosine shows no depth trend (except L0 and L31 at 0.674).

## Verdict

**Does not support the claim on this setup.** The paper's core empirical
observation — lower I/O similarity consistently predicts higher Fisher —
does not replicate as a rank predictor here: Spearman rho is
indistinguishable from zero (p=0.66), and LS-LoRA's selection rule
recovers no more high-Fisher layers than random selection (0.25 overlap =
chance). One run, one model, one tiny dataset — but the direction is
unfavorable, not merely underpowered: excluding layer 0 the correlation
is +0.01.

## Honest limits

> **Measurement honesty box.** Fisher was computed on 4-bit NF4 weights
> (gradients on dequantized bf16 copies of one layer at a time — exact
> for that layer's params, frozen 4-bit context elsewhere), not
> full-precision. Tiny dataset: 48 hardcoded math/code Q&A samples (avg
> 22 tokens) for cosine, 16 for Fisher — not the paper's math/code
> benchmarks. Base model (not instruct), single run, no seed variation;
> NLL on response tokens only. I/O cosine measured input-vs-full-layer-
> output (residual included); layer 0's input is the embedding output,
> which partly explains its extremity. Fisher's depth decay may partly
> reflect NLL backprop gradient magnitudes rather than true task
> sensitivity.

- This tests only the *proxy claim*, not LS-LoRA end-to-end (no LoRA
  training was run, so nothing is said about target-task gains or
  capability retention).
- A weak real effect could hide in the middle-layer cosine band
  (0.915–0.975) where rank noise dominates at n=32; a larger sample or
  full-precision weights might resolve it.

## Ops notes (Colab / Qwen3.5)

- `Qwen/Qwen3.5-9B` requires transformers **git main** — `qwen3_5` is in
  no PyPI release (incl. 4.57.6); the script pip-installs from git.
- `copy.deepcopy` of bitsandbytes `Params4bit` modules corrupts
  quant_state (AssertionError on forward); fixed by swapping 4-bit
  linears for plain bf16 linears in place.
- `colab.py exec --code` silently drops everything after the first line
  of multi-line code — use `exec -f` with a local file.
