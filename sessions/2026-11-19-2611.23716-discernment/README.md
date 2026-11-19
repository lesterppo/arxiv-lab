# Learning through Discernment: When a Small Model Learns from User Feedback (arXiv:2611.23716) — session record

**Article:** "Learning through Discernment: when a small language model
learns from user feedback on the job, and when it decays",
https://arxiv.org/abs/2611.23716 (v1, 2026-11-19).

**Verdict: INCONCLUSIVE** — ceiling effect: the test design could not
discriminate between the judge arms (see "Why inconclusive"). This is
a verdict about the test, not about the paper: the mechanism's judge
half shows the expected pattern in miniature, but the accuracy claim
is unmeasured. Leg 1 was separately INCONCLUSIVE (broken
operationalization); both legs are reported honestly below.

## Why picked (AGENT 2, TUNE 5)

Nightly LoRA fine-tuning on judge-accepted attempts is exactly the
soup-daily-finetune pattern (daily 8B NF4-QLoRA on Colab T4). The
paper's claims map onto it directly: a discernment judge (asks first
whether the user gave a clear verdict) keeps failures out of the
lessons; the share of failed attempts among lessons predicts the
outcome (paper: Spearman rho = -0.81 across 97 lives); learning holds
up only where the judge keeps failures out and the user can see the
outcome.

## What was built

- `discern_tasks.py` — a synthetic WebShop-like suite, deterministic
  and CPU-checked (`cpu_discern_check.py`: 500 catalogs, exactly one
  correct item each; reaction distribution as designed). Each task: a
  user request with constraints, a 6-item catalog, exactly one correct
  item, and a templated user reaction — clear verdict (70% of
  successes, 50% of failures), polite give-up thanking the agent for
  a FAILURE (30% of failures — the paper's trap), or ambiguous.
- `colab_discern_driver.py` — the Colab T4 driver. Three judge arms on
  the SAME attempt pools: **self** (base model judges its own attempt,
  YES/NO), **discern** (frozen judge prompt: first asks whether the
  user gave a clear verdict; accepts only clear YES), **oracle** (true
  outcome). Two nights: night-0 base attempts → judge → fresh LoRA on
  accepted lessons → held-out eval; night-1 adapted-model attempts →
  judge → fresh LoRA on cumulative lessons → held-out eval. 32
  train-A + 32 train-B + 32 held-out tasks. NF4 QLoRA r=16/alpha=32,
  bf16. Metrics flushed after every phase; prints TRAIN_DONE last.

## Leg 1 — Qwen3.5-9B-Base (session `discern-1119`, kyu008008; cyc236es
## + ppoppo205 gpu-unavailable, wrapper fell back; DONE in ~77 min)

**INCONCLUSIVE — broken operationalization, no fair test.** Base
accuracy 0/32 and every arm 0/32; every train loss NaN; Spearman rho
null. Two driver bugs, both mine:
(a) the agent prompt ended with the catalog list, so greedy
generations continued the catalog ("Item 7: ...") instead of
answering — leg-1 predictions prove it; sampled night-0 attempts
scored ~47% only because sampling sometimes breaks out of the list
continuation;
(b) MAX_LEN=256 truncated the ~300-token few-shot prompt, leaving
ZERO label tokens in every lesson → cross_entropy over an empty mask
→ NaN loss in every arm.
Fixes applied before leg 2: prompt ends on "Choice:" (forces the
demonstrated answer format); MAX_LEN=1024; `encode_lesson` now
asserts ≥1 label token (fail fast). Leg-1 metrics banked as
`discern_*_leg1_qwen35.*`.

## Leg 2 (automatic fallback) — Qwen3-8B-Base (session `discern-1119b`,
## kyu009009; DONE in ~86 min)

Model: Qwen/Qwen3-8B-Base, transformers 4.57.6, causal loader, Tesla
T4, peak VRAM 7.51 GB. The fixes worked mechanically: finite train
losses (0.009–0.054), 0/224 eval parse failures, base generations in
the right format (" Item 1\nReason: ...").

| arm | night | lessons | failures in lessons | failure share | train loss | held-out acc (n=32) |
|---|---|---|---|---|---|---|
| base | — | — | — | — | — | **1.0000 (32/32)** |
| self | 0 | 29 | 2 | 0.069 | 0.037 | 0.9688 (31/32) |
| discern | 0 | 26 | 0 | 0.000 | 0.024 | 1.0000 (32/32) |
| oracle | 0 | 27 | 0 | 0.000 | 0.016 | 1.0000 (32/32) |
| self | 1 | 31 | 0 | 0.000 | 0.054 | 1.0000 (32/32) |
| discern | 1 | 24 | 0 | 0.000 | 0.009 | 1.0000 (32/32) |
| oracle | 1 | 32 | 0 | 0.000 | 0.021 | 1.0000 (32/32) |

Spearman rho between lesson failure share and accuracy change:
**-0.7746 over n=6** (paper: -0.81 over 97 lives).

## Why inconclusive (ceiling effect)

- **Base accuracy is 1.0000 (32/32).** The synthetic suite is too
  easy for Qwen3-8B-Base with the 2-shot prompt: the base model solves
  every held-out task before any training. There is no headroom for
  any judge arm to differ, so the accuracy claim is unmeasured.
- **The Spearman rho is degenerate and must not be quoted as a
  reproduction.** Five of the six points sit tied at (failure_share
  0.0, acc_change 0.0); the -0.7746 is driven by a single point
  (self night 0: failure share 0.069 → acc change -0.03125, i.e. one
  task). Numerically close to the paper's -0.81, evidentially worth
  almost nothing.
- **The one directional signal is real but tiny:** the self-judge
  accepted 2 failures into its night-0 lessons and its accuracy
  dipped by exactly one task (31/32), while discern and oracle (0
  failures in lessons) stayed at 32/32. That is the paper's predicted
  direction, at n=1 task — suggestive, not a measurement.
- **The trap rate is also miniature:** night-0 self-judge called YES
  on all polite give-ups (trap_rate 1.0 — the paper's trap, a naive
  judge taking thanks for success), but the denominator is 1–2
  attempts (only 5 failures in the whole night-0 pool) and was not
  preserved in the downloaded files; night 1 had zero failures, so
  no polite give-ups occurred at all (trap_rate null).
- **What the run DOES establish:** the driver, judges, and training
  loop all work end to end on Qwen3-8B (finite losses, clean parses,
  the discernment judge kept failures out where the self-judge let 2
  through). The failure is in task difficulty, not in the harness.

## Honest limits

- Synthetic tasks, not WebShop/tau²-bench; 32 tasks per split; 6
  Spearman points. The verdict ceiling for a rerun with a harder
  suite is SUPPORTED, never VERIFIED.
- The paper's tau²-bench condition (user cannot see the outcome, 80%
  of lessons were thanked failures) was not tested — all users here
  see the outcome.
- The paper's decay-over-a-year finding was not tested — 2 nights only.

## Resume state (exact)

- Driver: `colab_discern_driver.py` in this session (fixed version:
  prompt ends on "Choice:", MAX_LEN=1024, label-token assert).
  Task suite: `discern_tasks.py` + `cpu_discern_check.py`.
- Fastest resume: make the suite harder so base accuracy sits
  ~50–70% (e.g. 8 items, 5 constraints, more near-misses, weaker
  few-shot — 1 shot instead of 2), then rerun the same driver on
  Qwen3-8B-Base. Everything else (judges, nights, metrics) is ready.
  Estimated ~90 min on the same recipe.
- Rotation state after leg 2: kyu009009 (2026-11-19); next daily run
  picks kyu009009's successor in the pool (cyc236ha).
