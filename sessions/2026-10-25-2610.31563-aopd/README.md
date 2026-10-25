# 2610.31563 — When to Follow the Teacher: Advantage-Based On-Policy Distillation (AOPD)

- Article: https://arxiv.org/abs/2610.31563
- Track: training (Colab T4, bounded stand-in)
- Session dir: `sessions/2026-10-25-2610.31563-aopd/`

## Why picked

On-policy distillation is directly reusable in the daily fine-tune harness:
the paper's estimator needs only student rollouts + terminal rewards — no
extra rollouts or reward evaluations — which is exactly what a small
Colab-T4 training loop can afford. Repo fit: `soup-daily-finetune` daily
8B NF4-QLoRA harness + `Soup` one-YAML fine-tune CLI (a reward-weighted
distillation loss is a drop-in training-objective variant for that
harness).

## What was built

- `colab_aopd.py` — self-contained Colab driver. Self-distillation
  setting: teacher = frozen Qwen3-8B base weights (adapter disabled),
  student = same base + LoRA. Base Qwen3-8B (8B class) is the closest HF
  base to the 9B-class budget; stated explicitly, not hidden.
  NF4 QLoRA r=16/alpha=32, bf16, transformers<5.0, `pip uninstall -y
  torchao` right after pip installs.
- Data: 200 synthetic verifiable arithmetic train prompts + 100 held-out
  prompts (disjoint, deterministic seed 20261025). Terminal reward 1/0
  by exact answer.
- Rollouts k=2 per prompt at temperature 0.8; per-prompt baseline =
  mean reward; advantage = reward − baseline.
- Arm (a) OPD baseline: unweighted mean-token KL(teacher‖student) loss
  on student rollouts.
- Arm (b) AOPD: same KL loss weighted per sequence by
  clip(1+advantage, 0, 2) × clip(importance weight, 0.1, 10), where the
  importance weight = exp(teacher seq logp − detached student seq logp).
- 12 optimizer steps per arm on identical data/seed/order; eval BEFORE
  vs AFTER per arm on the held-out set (greedy + sampled accuracy).
- `test_aopd_logic_cpu.py` — CPU check of the advantage / weight /
  parsing math (no model, no GPU).

## Results

### Attempt 1 — DIED (CUDA OOM, 0 optimizer steps)

- Session `train-1025-0424`, account kyu008008 (rotation pick from
  cyc236es), T4, monitor outcome DIED after ~11 min.
- Downloaded CSV (`aopd_metrics_attempt1_died.csv`): BEFORE eval only —
  greedy 0.090, sampled 0.090, n=100 (first-integer parse; superseded,
  see below).
- Failure: `torch.OutOfMemoryError` in the bitsandbytes 4-bit forward
  during the first training step's KL forward — a single batched
  full-vocab float32 KL over 16 sequences (13.91 GiB allocated by
  PyTorch of 14.56 GiB total). 0/12 steps completed in either arm.
- Fix applied to `colab_aopd.py` (this revision): KL in micro-batches
  of 4 sequences with per-chunk backward, generation chunked to 32
  prompts, `expandable_segments:True`. Answer parsing changed to LAST
  integer in the continuation (a base model may restate the problem
  before answering); attempt 1's BEFORE numbers are therefore NOT
  comparable to the retry's numbers and are not used in any verdict.

### Attempt 2 — retry (same account, same day) — DONE

- Relaunched with `--account kyu008008` (NOT `--rotate`: the rotation
  state already advanced to kyu008008 for 2026-10-25; re-rotating would
  wrongly burn kyu009009). Same registry file, same out-dir.
- Session `train-1025-0437`, account kyu008008, T4, monitor outcome
  DONE, TRAIN_DONE printed as the last stdout line (~35 min wall).
  Full CSV in `train-out/aopd_metrics.csv`.
- Held-out eval (n=100, last-integer parse):
  - BEFORE: greedy 0.010, sampled 0.020
  - AFTER OPD (12/12 steps): greedy 0.010, sampled 0.020
  - AFTER AOPD (12/12 steps): greedy 0.010, sampled 0.025
  - The AOPD−OPD sampled gap is 1 draw in 200 (5 vs 4) — noise.
- Training diagnostics (both arms, 12 steps each):
  - mean_reward 0.000 in 22/24 steps; 0.0625 (1/16 rollouts) in OPD
    steps 1 and 10 only; 0.000 in all 12 AOPD steps.
  - mean_adv_factor = 1.000 in ALL 24 steps — per-prompt advantages
    were identically zero, so the AOPD advantage weighting NEVER
    engaged (treatment never varied).
  - mean importance weight 0.86–1.25 (near 1, as expected near init);
    mean-token KL loss 0.0000–0.0013 (teacher ≈ student, no signal).
- Diagnosis (why this run cannot discriminate): the eval/task harness
  is broken, not the method. A base (non-instruct) model given a
  few-shot "Q:/A:" prompt does not stop at the answer: within the
  12-token continuation window it keeps generating further "Q: ..."
  fragments, so naive integer parsing of raw continuations is
  unreliable — BEFORE accuracy of 1–2% on 2-digit arithmetic is a
  harness artifact, not a model capability measure (attempt 1's
  first-integer parse gave 9%, also broken, differently). With
  rewards ~0 everywhere, advantages are identically zero and the
  AOPD-vs-OPD comparison is vacuous.
- Fix for a future run (not run here): truncate each continuation at
  its first newline and parse the first integer of that line; verify
  with a dumped generation sample BEFORE training (target: BEFORE
  greedy accuracy well above noise on this trivial task); consider an
  instruct-tuned base or explicit stop strings. Only then can the
  advantage weighting actually engage and be compared.

## Verdict

**INCONCLUSIVE** — test design could not discriminate: near-zero
rewards in every step left advantages identically zero, so the AOPD
advantage weighting never engaged (mean_adv_factor 1.000 in all 24
steps) and the AOPD-vs-OPD comparison is vacuous. This is NOT a
negative verdict on the paper: the treatment was never actually
applied. The Colab run itself completed cleanly (DONE, full metrics).

## Resume state

- If re-run with the fixed harness: same driver path, same command
  shape (rotate per the day's rotation state at that time; do NOT
  re-use `--account kyu008008` blindly on a later day — the rotation
  pool advances daily). First change: continuation truncation +
  first-integer parse + a pre-training generation dump check.
- Colab session `train-1025-0437` was torn down by the `train`
  monitor on DONE (no `--keep`). Rotation state remains
  `2026-10-25 kyu008008`, so the next daily run picks kyu009009.

## Honest limits

- Bounded stand-in, NOT the paper's full setup: self-distillation only
  (no external teacher), synthetic arithmetic only (no math/coding/MedQA
  benchmarks), k=2 rollouts, 12 steps/arm, one seed, no full GRPO.
- A single bounded run cannot confirm or refute the paper's benchmark
  gains; it can only test whether the advantage×importance weighting
  changes held-out accuracy directionally versus unweighted OPD at this
  scale.

## Resume state

- Driver: `sessions/2026-10-25-2610.31563-aopd/colab_aopd.py`
- Launch: `python3 ~/workspace/hermes-colab-cli/colab.py train --driver
  <driver> --rotate --gpu T4 --results /content/aopd_metrics.csv
  --out-dir <session-dir>/train-out --registry-file
  ~/workspace/goals/arxiv-daily-read-and-test/hidden_files/colab-state-20261025-aopd.json
  --monitor-timeout 5400`
- Rotation: last used 2026-10-25 was cyc236es → `--rotate` picks
  kyu008008; never touch cyc236de / cyc236hk; registry-file isolation
  mandatory (global sessions.json untouched).
- If reclaimed/died/timeout: re-attach with `--watch-only` per
  arxiv-lab WORKFLOW.md §4.5; marker-last (TRAIN_DONE is the final
  stdout line); whole-log grep.
