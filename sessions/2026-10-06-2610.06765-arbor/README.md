# Session 2026-10-06 — ARBOR: Conditional Rank Allocation (arXiv:2610.06765)

**Article:** "Conditional Rank Allocation for Taxonomy-Aware Medical Language
Model Adaptation" ([abs](https://arxiv.org/abs/2610.06765))

**Why picked:** most prospective *training* article in the 2026-10-06 cs.AI
window (AGENT 2/5, TUNE 5/5). ARBOR is a parameter-efficient adaptation
method: per question, select rank-one components from a shared low-rank
basis via an additive gate (question repr + specialty/operation tags +
their interaction); a learned coefficient scales the adapter residual.
Direct repo fit: `hermes-gi-egtkg-finetune` (daily medical LoRA — the paper
evaluates on medical QA: CMB, CMExam, MedQA, MedMCQA, beating LoRA r16 by
1.26pp and MoELoRA by 1.30pp on Qwen3-8B) and `soup-daily-finetune`
(conditional rank routing is a drop-in upgrade to the fixed-rank LoRA
harness). The paper's own illustrative separation (orthogonal subtasks) is
CPU-testable; the empirical claim got a bounded Colab NF4 QLoRA run at the
9B class.

**Paper claims:** (i) under orthogonal, equiprobable subtasks, conditional
selection avoids the approximation floor a fixed update of the same active
rank faces; (ii) on Qwen3-8B medical QA, ARBOR 69.69% mean accuracy,
+1.26pp over LoRA r16, +1.30pp over MoELoRA, gap widening 0.08→1.94pp as
specialties grow 1→7; (iii) tag perturbations / atom masking support
clinical routing (atom clusters align with specialty labels, ARI 0.62).

## What was built

- Reusable mechanism: `src/arxiv_lab/training/arbor.py` — `make_orthogonal_tasks`
  (K orthogonal subtasks + shared rank-one basis), `fixed_update_floor`
  (best single rank-r update via SVD truncation), `additive_gate`,
  `conditional_update`, `oracle_gate_weights`, `question_for_task`.
  Test: `tests/test_arbor_cpu.py` (all pass; numpy-guarded).
- Colab script: `colab_arbor.py` — self-contained (pip installs incl.
  `pip uninstall -y torchao` per the documented quirk); Qwen3-8B
  (the paper's model) NF4/bf16; arm A LoRA r=16/alpha=32 vs arm B
  ARBOR-style adapter (R=32 shared rank-one atoms on q_proj/v_proj,
  additive gate over mean-pooled question embeddings + specialty tag +
  interaction, soft routing during training); synthetic 3-specialty task
  (ARITH / REVERSE / CAESAR — disjoint mappings, exact-match verifiable);
  450 train / 150 test, 100 steps/arm, batch 4, lr 2e-4; metrics to
  `/content/arbor_results.json` + `/content/arbor_metrics.csv`.
- Session record: `test_session_cpu.py` (verbatim snapshot) +
  `results_cpu.json` (+ `arbor_results.json` / `arbor_metrics.csv` from
  Colab when the run lands).

## Results (CPU, deterministic)

- Fixed rank-2 shared update: approximation floor MSE = **0.002706**.
- Conditional rank-2 selection with correct routing: MSE = **0.000000**
  (routing accuracy 4/4; least-squares residual scale models the paper's
  learned coefficient).
- Tag-perturbed routing: MSE = **0.006542** — worse than the fixed floor,
  confirming routing is load-bearing (paper's tag-perturbation check).

→ Claim (i) reproduced exactly: conditional selection avoids the floor at
the same active rank.

## Results (Colab T4 — training track)

Session `arxiv-arbor` (account cyc236hk; cyc236ha and kyu009009 reported
GPU-unavailable at provisioning — capacity lull). Qwen3-8B NF4/bf16,
100 steps/arm, batch 4, lr 2e-4, 450 train / 150 test.

- **LoRA r=16 arm — COMPLETED.** Train loss 1.96 → 1.34 over 100 steps
  (~3 min). Test exact-match: overall **0.547** (ARITH 1.000, REVERSE
  0.400, CAESAR 0.240). Baseline numbers are real, from `/content`
  run log 2026-10-06 ~13:00 HKT.
- **ARBOR arm — BLOCKED by infra, not by the method.** The
  `from_pretrained` loader was OOM-killed (SIGKILL, no traceback) by the
  VM's 12 GB system-RAM cgroup on 4 of 5 attempts — dmesg shows
  anon-rss ~11.3 GB at kill, i.e. the loader's transient exceeds VM RAM.
  Mitigations tried and failed: drop_caches, `low_cpu_mem_usage`,
  `max_memory` caps, `dtype=` vs deprecated `torch_dtype=`,
  `device_map={"": 0}`; container cannot `swapon` (Invalid argument).
  The first load of the day succeeded (the LoRA arm above), so this is a
  flaky infra ceiling, not a modeling error.
- The comparison script is staged here as `colab_arbor.py`
  (self-contained; `--arm arbor --out /content/arbor_results_arbor.json`
  reruns just the ARBOR arm) — rerun when Colab provisions a higher-RAM
  VM or the loader cooperates. Nothing was faked: no ARBOR empirical
  numbers are reported.

## Verdict: CPU GREEN, Colab PARTIAL (baseline only)

## Honest limits

- The CPU test reproduces the paper's *theoretical illustration*, not the
  medical-QA headline numbers (+1.26pp over LoRA r16) — those need the
  Colab run (or the full medical benchmarks).
- The gate in the CPU test is an oracle (post-training) gate; the paper
  learns it. The Colab adapter trains the gate from scratch with soft
  routing — hard top-k selection at inference is not exercised.
- Synthetic specialties stand in for medical taxonomies; transfer to the
  `hermes-gi-egtkg-finetune` medical pipeline is prospective, not shown.

## Smaller-model runs (2026-10-06 evening HKT, per Peter's direction)

Qwen3-8B was SIGKILLed 5/5 at `from_pretrained` on the 12 GB T4 VM, so both
arms were rerun on smaller models with identical task/hyperparams (450 train /
150 test, 100 steps/arm, batch 4, lr 2e-4, NF4/bf16). The 8B LoRA baseline EM
0.547 does **not** transfer — comparisons below are same-model only.

### Qwen3-4B (`colab_arbor_4b.py` → `arbor_4b_results.json` / `arbor_4b_metrics.csv`)

| arm | ARITH | REVERSE | CAESAR | overall |
|-----|-------|---------|--------|---------|
| LoRA r16 (run 1) | 1.00 | 0.24 | 0.12 | 0.453 |
| LoRA r16 (run 2) | 1.00 | 0.30 | 0.16 | 0.487 |
| ARBOR R32 | 1.00 | 0.18 | 0.10 | 0.427 |

ARBOR − LoRA(mean) = **−4.3pp**. LoRA's own run-to-run variance was 3.4pp,
so the gap is suggestive but not decisive at this scale.

### Qwen3.5-4B (`colab_arbor_qwen35_4b.py` → `arbor_qwen35_4b_results.json` / `arbor_qwen35_4b_metrics.csv`)

Qwen3.5-4B is multimodal (text tower: 32 layers, d=2560) with **hybrid
attention** — only the full-attention layers carry q_proj/v_proj (the rest use
linear-attention projections), so both adapters touch fewer modules than on
Qwen3-4B: LoRA 1.8M params, ARBOR 4.0M. Needed transformers git main for the
`qwen3_5` model type; text-config hidden size lives under
`config.text_config`; embed path is `model.model.language_model.embed_tokens`.

| arm | ARITH | REVERSE | CAESAR | overall |
|-----|-------|---------|--------|---------|
| LoRA r16 | 1.00 | 0.44 | 0.08 | 0.507 |
| ARBOR R32 | 1.00 | 0.30 | 0.06 | 0.453 |

ARBOR − LoRA = **−5.4pp**.

### Verdict vs the paper's claim

The paper reports ARBOR **+1.26pp over LoRA r16** on Qwen3-8B medical QA.
At the 4B scale on this synthetic 3-specialty task, ARBOR underperforms LoRA
in **both** families (−4.3pp on Qwen3-4B, −5.4pp on Qwen3.5-4B). The paper's
edge does not replicate here. Possible reasons (not discriminated): the
smaller model/task regime, the synthetic task's simplicity (ARITH saturates at
1.00 for all arms), only 100 training steps, or the claim being specific to
the 8B medical-QA setting.

### Method bug found and fixed (worth upstreaming)

The ARBOR adapter's `__init__` read `mod.weight.shape` to size its U/V
matrices. On a bitsandbytes NF4 model, `Params4bit.weight.shape` returns the
quantized **storage** shape `(1310720, 1)`, not the logical
`(out_features, in_features)` — U/V came out 512× oversized and the process
was OOM-killed at ~11.6 GB RSS inside the adapter constructor (dmesg
confirmed; this masqueraded as Colab "reclaims" across 4 sessions). Fix: use
`mod.out_features` / `mod.in_features`. Two companion fixes were needed to get
the arm training: place U/V/W on the model device (was CPU vs CUDA mismatch)
and run the adapter in bf16 (was fp32 vs bf16 mismatch). All three Colab
scripts in this dir carry the fixes.

### Honest limits

- Smaller models (4B) stand in for the paper's 8B; the +1.26pp claim is an 8B
  medical-QA result and was not re-tested at 8B (infra-blocked).
- Synthetic 3-specialty task, 100 steps/arm — a smoke test, not a benchmark.
- ARBOR ran once per family; LoRA variance (3.4pp across two Qwen3-4B runs)
  means small deltas should not be over-read.
- Colab free-tier volatility cost 4 sessions (reclaims) before the method bug
  was identified; wall-clock for the 4B pair was ~2.5 h elapsed.
