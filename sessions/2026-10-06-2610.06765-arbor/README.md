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
