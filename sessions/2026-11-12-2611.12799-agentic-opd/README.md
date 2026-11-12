# Agentic-OPD (arXiv:2611.12799) — 2026-11-12 daily run, training track

- **Article:** [2611.12799](https://arxiv.org/abs/2611.12799) — "Agentic-OPD: Recovering Forgotten Agent Capabilities via On-Policy Distillation for Agentic Tasks"
- **Scores:** AGENT 4 / TUNE 5
- **Verdict: SUPPORTED** — the checkpoint-pool observation, HCN teacher
  selection, LAD filtering, and capability recovery all reproduced
  directionally at bounded scale in the SFT-stage variant: C2 forgot
  23/43 A_val problems C1 had solved; HCN picked C1 (gain 23 vs true
  base gain 4); LAD admitted 65/100 pool problems (mean teacher success
  0.69 vs student 0.28); OPD recovered 18/23 forgotten A_val problems
  and 12/22 on held-out A_test (A_test 0.350 → 0.550), with B retained
  (0.767 → 0.867). Not VERIFIED: the GRPO prerequisite was replaced by
  the SFT variant, LAD's KL became SFT on teacher traces, evals are
  n=60, and there is no naive-distillation control, so the gain cannot
  be attributed to HCN+LAD filtering specifically.

## Why picked

The paper's mechanism: in a training run the best checkpoint fails on
problems earlier checkpoints solved; the union of solved problems across
checkpoints far exceeds any single one. Agentic-OPD treats the
best-validation checkpoint as student and all saved checkpoints as frozen
teacher candidates — HCN (Highest Coverage Next, Eq. 7–9) selects teachers by
weighted coverage of the student's unsolved problems, and LAD
(Locality-Aware Distillation, Eq. 10–13) confines transfer to failed student
rollouts on problems where the teacher performs better.

**Repo fit:** `soup-daily-finetune` produces a checkpoint pool daily — a
direct home for a checkpoint-pool method. Bounded variant: an SFT-stage
checkpoint pool (see "Deliberate deviation"), which avoids the full-GRPO
prerequisite that blocked this pick yesterday.

## What was built

`colab_agentic_opd_driver.py` — Qwen/Qwen3-8B, NF4 QLoRA r=16/alpha=32,
bf16, transformers<5.0, no `prepare_model_for_kbit_training`,
expandable_segments, gradient checkpointing, micro-batch 2 x accum 2.
Phases:

- **Phase 0 — base eval** (greedy, zero-shot): GSM8K test rows 0–60
  (`A_val`), rows 100–160 (`A_test`), SVAMP test rows 0–60 (`B_test`),
  n=60 each, max_new 120 (A) / 48 (B).
- **Stage A:** SFT base → C1 on task A (GSM8K CoT, 400 ex, 1 epoch,
  lr 2e-4, 100 steps, 2214 s, peak VRAM 8.23 GiB).
- **Stage B:** continue SFT C1 → C2 on task B (SVAMP answer-only, 400 ex,
  1 epoch, lr 2e-4, 100 steps, 711 s — short targets train faster).
- **Pool observation:** union coverage on A_val / A_test (C1 ∪ C2 vs C2).
- **HCN:** teacher selection (base vs C1) on A_val, student C2, weights
  w0=1, η=0.5, w_max=2.0.
- **LAD:** G=4 rollouts, student C2 vs selected teacher, on a 100-problem
  GSM8K pool (train rows 400–500, disjoint from stage A); admit problems
  where the teacher beats the student (Eq. 11).
- **Stage C:** OPD distillation — SFT on the teacher's correct sampled
  rollout text for LAD-admitted problems (completion tokens only, lr 1e-4,
  17 steps, 292 s); eval C3 vs C2 for recovery on A and retention on B.

**Deliberate deviation (documented limit):** the paper's prerequisite is a
full GRPO run producing the checkpoint pool — too heavy for one free-T4
session. This test builds the pool with sequential SFT instead. LAD's
turn-level top-k KL (Eq. 13) is adapted to SFT on the teacher's correct
sampled rollout text — a bounded, stated simplification of the KL objective.

Metrics CSV + results JSON were flushed to `/content` after EVERY phase
(reclamation-safe) and downloaded by the mid-run downloader every 10 min.
Final stdout line is `TRAIN_DONE`.

## Results — run f (DONE, session `agentic-opd-1112f`, ppoppo205)

Greedy eval, n=60 per split. Base = Qwen3-8B + zero-init LoRA (identity).

| model | A_val (GSM8K 0–60) | A_test (GSM8K 100–160) | B_test (SVAMP 0–60) |
|-------|-------------------|------------------------|---------------------|
| base  | 10/60 = 0.167 | 15/60 = 0.250 | 16/60 = 0.267 |
| C1 (after A) | 43/60 = 0.717 | 42/60 = 0.700 | 52/60 = 0.867 |
| C2 (after B) | 24/60 = 0.400 | 21/60 = 0.350 | 46/60 = 0.767 |
| **C3 (after OPD)** | **43/60 = 0.717** | **33/60 = 0.550** | **52/60 = 0.867** |

### Paper §2 observation — reproduced

- A_val: C1 43, C2 24, **union 47** (gap +23 over the student C2).
- A_test: C1 42, C2 21, **union 43** (gap +22 over C2).
- Stage B forgot 23/43 A_val and 22/42 A_test problems C1 had solved
  (A −31.7 pp / −35.0 pp). B also slipped (0.867 → 0.767) despite
  Stage B training on B — the answer-only format cost more than the
  extra B data gained.

### HCN (Eq. 7–9) — executed, C1 selected

- As-run gains on A_val: `{base: 0, C1: 23}` → teacher C1.
- **Correction:** the as-run `g_base = 0` is an artifact of a driver bug
  (see "Honest limits"): `model.disable_adapter()` called bare is a
  no-op in peft (it returns a context manager), so the HCN "base" eval
  actually ran with the C2 adapter active (24/60 = C2's A_val exactly).
  The TRUE gains, recomputed offline from the banked per-problem solved
  lists in `agentic_opd_results.json`, are `{base: 4, C1: 23}` — C1
  still wins by a wide margin, so the selection is unaffected.
- HCN weights after round 1 (Eq. 8, η=0.5, w_max=2.0): 43 zero
  (C3-solved), 7 back to one, 10 raised — recorded for a round 2 that
  was not run.

### LAD (Eq. 10–12) — executed

- Pool n=100, G=4: admitted **65/100** problems (teacher beats student),
  65 teacher traces; mean student success **0.280**, mean teacher
  success **0.690**. A large, real teacher advantage on the admitted set.

### OPD recovery — paper Fig. 4(a) analogue

| split | forgotten (C1✓ C2✗) | recovered by C3 | lost (C2✓ C3✗) | new (neither✓ C3✓) |
|-------|---------------------|-----------------|-----------------|---------------------|
| A_val | 23 | **18 (78%)** | 3 | 4 |
| A_test | 22 | **12 (55%)** | 2 | 2 |

- A_test 0.350 → **0.550 (+20.0 pp)**; A_val 0.400 → 0.717 (+31.7 pp).
- B_test 0.767 → **0.867 (+10.0 pp)** — retention held (B improved too).
- C3 reaches C1's A_val level exactly (43/60) but stays below C1 on
  held-out A_test (33 vs 42) — partial, not complete, recovery.
- The paper's best DeepSearch gain is +7.4 pp over GRPO init; ours is
  +20 pp recovery on A_test in the SFT analogue (different scale and
  baseline — not directly comparable).

## Attempt history

| session | account | outcome | what happened |
|---------|---------|---------|---------------|
| `agentic-opd-1112` | cyc236es | DIED | `ValueError` in `extract_number` (lone comma matched as a number) during base eval. Driver bug; fixed with guarded `float()` + digit-anchored regex. No files banked. |
| `agentic-opd-1112b` | cyc236es | killed by agent | Base eval ran but generation was ~0.8 s/decode-step: `use_cache=False` (set for training) was never re-enabled for generation. Killed, session stopped; driver fixed. |
| `agentic-opd-1112c` | cyc236es | killed by agent | Same slowness confirmed (~132 s/batch of 24×160 tokens); killed for a throughput shrink (n=60, max_new 120, eval batch 48), session stopped. |
| `agentic-opd-1112d` | cyc236es | killed by agent | Shrunk run reached base eval (A_val 11/60) but batch time was still ~130 s; killed for a further shrink (eval batch 48, sampling batch 12), session stopped. |
| `agentic-opd-1112e` | cyc236es | DIED | Base evals done (11/13/16); CUDA OOM at Stage A step 0: bnb `_dequant_linear_fallback` materializes full bf16 weights, retained for backward without gradient checkpointing (14.19 GiB allocated). Fixed with gradient checkpointing + micro-batch 2 x accum 2. |
| `agentic-opd-1112f` | ppoppo205 (cyc236es gpu-unavailable at creation, cross-account fallback) | **DONE** | Full run, 05:38–07:51 UTC: base evals, Stage A (2214 s), C1 eval, Stage B (711 s), C2 eval, pool, HCN, LAD (29 min sampling), Stage C (292 s), C3 eval, `TRAIN_DONE`. Host monitor was SIGTERM'd by a runtime restart mid-run; re-attached with `train --watch-only --pid 1903`, which finalized DONE and downloaded results. |

Notes: attempts b–d were agent-killed for driver/throughput fixes, not
infra failures, so no automatic model fallback was triggered for them
(same distinction as the 2026-11-11 MCPO launch 1/2 driver bugs). The
run-e OOM was also a driver-side memory failure (fixed in-driver), not a
model-fit failure, so the 8B was kept and the staged 4B fallback below
was never launched. No reclamation in any attempt.

## Honest limits

- **SFT variant, not GRPO:** the paper's checkpoint pool comes from one
  GRPO run; ours comes from sequential SFT on two tasks. Forgetting here
  is format/task-shift forgetting, not within-run RL forgetting.
- **LAD's KL replaced:** Eq. 13 top-k KL became SFT on the teacher's
  correct sampled rollout text. The filtering (Eq. 11–12) is faithful;
  the loss is a simplification.
- **No naive-distillation control:** C3's gain is measured against C2
  only. A control distilling C1 on all A problems (no HCN/LAD filtering)
  was not run, so the gain cannot be attributed to the filtering.
- **Small n:** evals are n=60; ±1 case ≈ 1.7 pp. A_val/A_test disagree
  on recovery completeness (full on val, partial on test).
- **HCN g_base artifact:** as-run `g_base = 0` is wrong (peft
  `disable_adapter()` bare-call no-op — the "base" HCN eval ran as C2);
  true value 4, recomputed offline. Selection unaffected. One-line fix
  for reruns: `with model.disable_adapter():` around the base eval (or
  keep a separate unadapted model handle). The Phase-0 base row is
  unaffected: the LoRA is zero-initialized, so base+untrained-adapter
  is the base model.
- **Single HCN round, single teacher:** pool had two candidates
  (base, C1); only C1 ever taught. No second-round weights were used.
- **Adapters not downloaded:** `adapter_c1/c2/c3` live only on the
  reclaimed VM; the session banks metrics, results, and drivers, not
  weights. A rerun re-trains from the driver (deterministic, same seed).
- **Throughput:** NF4 decode on T4 is ~0.8 s/step at batch 24; the full
  run took ~2 h 13 min. Morning window recommended for reruns.

## Resume state

- **Run f is DONE; nothing to resume.** All phases completed, results
  banked (`agentic_opd_metrics.csv`, `agentic_opd_results.json`,
  `train-run.json` outcome DONE).
- **Driver (8B):** `colab_agentic_opd_driver.py` — the exact file that
  ran f (guarded `extract_number`, `use_cache=True` for generation,
  gradient checkpointing + micro-batch 2, n=60, max_new 120/48).
- **Rotation:** file stays `2026-11-12 cyc236es` (creation fallbacks do
  not rewrite rotation); tomorrow `--rotate` picks kyu008008. Active
  account restored to cyc236es after the run. No live sessions remain.
- **A retry needs:** nothing pending; a rerun would be a fresh
  instruction (e.g. a naive-distillation control arm, or a second HCN
  round using the recorded weights).

## Automatic fallback (staged, not launched)

Under Peter's 2026-11-05 automatic-fallback rule, the fallback was staged
here for use **if the live run f blocked** — it did not (DONE), so this
was never launched and needs no action:
`colab_agentic_opd_driver_4b.py` — Qwen/Qwen3-4B, same experiment (data,
stages, evals, seeds, LoRA r=16/a=32) plus memory-shrink fixes for the
run-e OOM cause:

- 4B model (halves the dequantized-weight footprint vs 8B);
- micro-batch 1 x accum 4 (effective batch still 4);
- seq cap 512 → 384;
- eval batch 24 → 8, sample batch 8 → 4;
- gradient checkpointing retained from the fixed 8B driver.

`test_agentic_opd_fallback_cpu.py` verifies the pure helpers (incl. the
attempt-1 `ValueError` failure mode) and that the 4B driver differs from
the 8B driver only in the intended shrink set: **62/62 checks pass**.

## Naive-distillation control arm (Peter's order, 2026-11-12 16:51 HKT)

Closes the attribution gap in "Honest limits": how much of C3's
+31.7/+20.0/+10.0 pp recovery over C2 survives withOUT HCN+LAD filtering.

**Design** (`colab_agentic_opd_driver_naive.py`, CPU test
`test_agentic_opd_naive_cpu.py`: **68/68 checks pass**): same experiment
as run f — same data, stages, evals (A_val/A_test/B_test, n=60), seeds,
LoRA r=16/a=32, same student C2, same 100-problem pool, same G=4
sampling — with TWO differences only: (1) NO HCN — teacher fixed to C1
(HCN selected C1 in run f anyway, true gains {base: 4, C1: 23}, so the
value of selection is measured separately as the base-vs-C1 gain gap);
(2) NO LAD — distill on ALL pool problems with >=1 correct teacher
trace (no "teacher beats student" admission filter). Student rollouts
are still sampled, purely for the attribution split (naive examples on
problems LAD would have admitted vs rejected); they never affect
selection. Comparison: naive C3' vs HCN+LAD C3 (43/33/52) vs C2 (24/21/46).

**What was banked / reused:** nothing on the weight side — a
workspace-wide search confirmed NO run-f adapters exist anywhere
(`adapter_c1/c2/c3` lived only on the reclaimed VM), so Stages A/B are
retrained from scratch. Banked and reused as reference: run-f solved
lists + metrics (the control's `compares_against` constants).

**Driver fixes vs run f:** `eval_all(None)` now enters
`model.disable_adapter()` as a context manager (the bare call in run f
was a no-op that made the HCN "base" eval run as C2); `set_adapter`
asserts on None instead of silently running the wrong model. No HCN
phase executes; no `RESULTS["hcn"/"lad"]` blocks are written.
Naive selection is a pure helper (`build_naive_examples`, unit-tested).
Final C3' adapter is zipped to `/content/agentic_opd_adapter_naive.zip`
(~161 MB, local-only, NOT pushed) and included in `--results`.

**Launch (evening, 2026-11-12 ~16:51 HKT; reclamation risk is real):**
session `agentic-opd-1112-naive`, account cyc236es (today's rotation
account; tomorrow's daily run picks kyu008008, unaffected), registry
`hidden_files/agentic-opd-1112-naive-registry.json`, out-dir
`hidden_files/agentic-opd-1112-naive`, done marker `TRAIN_DONE`,
monitor-timeout 14400. `--results` = the task's two files PLUS the
adapter zip (change from the brief's command, recorded here):
`/content/agentic_opd_metrics_naive.csv`,
`/content/agentic_opd_results_naive.json`,
`/content/agentic_opd_adapter_naive.zip`.
Mid-run downloader polls metrics/results every 10 min.
Wall time ~2h–2h15m like run f. ONE attempt: if it ends BLOCKED, no
auto-fallback — record BLOCKED with resume state and stop.

**Outcome:** _pending — to be filled when the run finishes._
