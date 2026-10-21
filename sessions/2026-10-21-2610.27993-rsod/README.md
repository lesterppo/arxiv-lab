# Session 2026-10-21 — RSOD (arXiv 2610.27993)

**Article:** [Building Telecom Language Models with RSOD: Domain Data, Specialist Post-Training, Memory-Conscious Distillation, and Networking Evaluation](https://arxiv.org/abs/2610.27993)
**Track:** training (Colab T4) · AGENT=2 / **TUNE=4** · **Verdict: INCONCLUSIVE**

## Why picked
Routed Specialist On-Policy Distillation consolidates N specialist LoRA
teachers into one student: the committed student generates each trajectory,
the route-assigned specialist teacher-forces the exact student token
sequence, only the student LoRA is updated. Directly adaptable to the
soup-daily-finetune pattern (daily NF4-QLoRA harness) — a principled way to
merge multi-domain specialists into one deployable adapter. Repo fit: strong.

## What was built
`colab_rsod.py` (self-contained, pip installs + torchao uninstall at top):
2 synthetic routes (ARITH: a±b=?, CODE: complete one-line python fn;
160 train + 60 eval each, disjoint), Qwen3-8B NF4 (r=16/α=32, bf16 —
Qwen3.5-9B unloadable under the transformers<5.0 pin, see limits), 2
specialist LoRA teachers (SFT 25 steps, frozen after), RSOD student
(on-policy greedy generations → route-oracle teacher forcing → mean
KL(teacher‖student) on response tokens, 40 steps), MIXSFT baseline (plain
SFT on mixed specialist data, 25 steps). Eval: exact-match per route for
BASE / ROUTED (oracle→specialist) / RSOD / MIXSFT. Metrics in
`metrics.json`/`metrics.csv`.

## Results (measured, Qwen/Qwen3-8B)
| arm    | arith  | code | mean  |
|--------|--------|------|-------|
| base   | 0.6167 | 0.050| 0.333 |
| routed | 0.8667 | 0.000| 0.433 |
| rsod   | 0.6167 | 0.050| 0.333 |
| mixsft | 0.9167 | 0.000| 0.458 |

- Specialists: arith SFT works (0.617→0.867); code specialist's training
  loss →0.0000 yet generation accuracy 0.000 on all arms — the code route
  is broken in this setup (see limits).
- RSOD mechanics ran correctly: KL(teacher‖student) fell 0.48→0.17 over
  40 steps — the student does match the teacher's distribution on its own
  generations. But task accuracy did not move (identical to base).
- MIXSFT works: 0.917 on arith, *beating* the routed specialist (0.867).

## Verdict: INCONCLUSIVE
The test design could not fairly evaluate the paper's claim. RSOD needs the
student's on-policy trajectories to be task-directed enough for the
teacher's token-level supervision to transfer skill; our base-model student
generated rambling off-task text (e.g. answering then inventing new
questions), so KL-matching the teacher on those trajectories transferred no
task competence. This is a broken operationalization on our side, not
evidence against RSOD. The MIXSFT result is informative regardless: for
simple separable skills, plain mixed-data SFT consolidates to
routed-system level (here better), so RSOD's value-add needs a setting
with real cross-domain interference to be visible.

## Honest limits
- Qwen3.5-9B (track model) unloadable: transformers 4.57.6 (latest <5.0)
  has no `qwen3_5` architecture. Fell back to Qwen3-8B (same fallback as
  the 2026-10-20 run).
- Code route broken for every arm (specialist loss→0 yet 0% generation);
  root cause not isolated — possibly insufficient steps for the 6-way
  arbitrary fn→body memorization, or a prompt/answer masking subtlety.
- RSOD student trained only 40 steps on 200 trajectories; KL converged but
  on poor trajectories.
- Env notes: bitsandbytes 0.45.0 lacked CUDA-13 GPU support → upgraded to
  0.50.2; `prepare_model_for_kbit_training` is REQUIRED (without it,
  gradient checkpointing sees no grad-requiring inputs and backward fails);
  `colab.py exec --code` reuses a cached kernel (stale imports) —
  use exec_file/detached for fresh env.
- Colab session `rsod-20261021` (T4) tore down after the run.
