# Knowing When to Teach: Learning from Teachable Prefixes in Multilingual Reasoning Distillation

- **Article:** arXiv [2611.08179](https://arxiv.org/abs/2611.08179) — Cho et al.,
  "Knowing When to Teach: Learning from Teachable Prefixes in Multilingual
  Reasoning Distillation" (submitted 5 Nov 2026)
- **Verdict: BLOCKED** — no fair test of TOPSD vs OPSD was possible: the
  Qwen3.5-9B run died silently after the base eval and the automatic
  Qwen3-8B fallback was reclaimed during student-trace generation. No
  training ran, no drift measurement completed, no adapters exist.
- **Track:** training track — Google Colab T4 via `hermes-colab-cli`
  (`colab.py train`)

## Why picked

- **AGENT 2 / TUNE 5.** On-Policy Self-Distillation with a teachable-prefix
  fix for Teacher Drift — a fine-tune method directly adaptable to the
  daily 8B NF4-QLoRA harness (`soup-daily-finetune` pattern): train a
  student on its own reasoning, with the teacher (same model conditioned
  on reference solutions) correcting it only where it can still recover
  the right answer.
- **Repo fit:** `arxiv-lab` mechanism library (training-track session);
  the OPSD/TOPSD pattern generalises to any small-model reasoning
  fine-tune.
- **Testable within budget:** MGSM (3 languages, 120 train / 60 test per
  language), one SFT epoch per arm, QLoRA r=16 — a bounded one-session
  design.

## What was built

- `colab_topsd_driver.py` — the Colab driver (transformers 5.20.2 +
  peft NF4-QLoRA, r=16/alpha=32, lr 2e-4, batch 2 x accum 2, seq cap
  384, gradient checkpointing, sdpa attention):
  - Phase 1: base eval, greedy zero-shot, 60 held-out MGSM problems per
    language (en/zh/sw), last-integer exact match.
  - Phase 2: student traces on the 120-problem train pool; drift
    measurement on up to 30 wrong traces per language (teacher completion
    conditioned on the student trace + reference solution: recovered vs
    followed the error; onset = first step whose teacher-conditioned
    mean token logprob drops >1.0 nat below the best-so-far step, or
    below -3.0 absolute).
  - Phase 3: SFT data — OPSD arm (full teacher traces conditioned on the
    reference solution, verified final answer) vs TOPSD arm (student
    trace cut at the drift onset; teacher completes from the prefix;
    kept only if the completion recovers the gold answer; correct
    student traces kept whole).
  - Phases 4–5: SFT per arm (1 epoch), adapter zip saved after each arm,
    eval base vs OPSD vs TOPSD. Metrics CSVs written incrementally.
- `test_topsd_logic_cpu.py` — CPU test of the driver's pure logic
  (answer extraction, step splitting, drift-onset detection, MGSM
  train/test index-split disjointness): **22/22 checks passed**.
- `topsd_metrics.csv` / `topsd_details.csv` — Qwen3.5-9B base eval
  (180 detail rows). `topsd_metrics_8b.csv` / `topsd_details_8b.csv` —
  Qwen3-8B base eval (180 detail rows).
- `train-launch.json`, `train-run.json`, `train-watch8b.json` —
  wrapper outcomes per attempt.

## Results

Base eval (greedy, zero-shot, n=60 per language, MGSM):

| model | en | zh | sw | peak GB |
|---|---|---|---|---|
| Qwen/Qwen3.5-9B | 0.2667 | 0.1833 | 0.0667 | 8.8 |
| Qwen/Qwen3-8B | 0.1167 | 0.2333 | 0.0333 | 6.91 |

- 9B: parse_fail 0 in all languages. 8B: parse_fail 3 on sw.
- 8B en is well below 9B en (0.1167 vs 0.2667); zh above 9B (0.2333 vs
  0.1833). n=60, so these are noisy calibration numbers, not a model
  comparison — no training arms exist to compare against them.
- No drift rates, no onset fractions, no OPSD/TOPSD accuracies: the
  phases that produce them never ran.

## Attempt history

1. **Attempt 1 — DIED in ~2 min (driver bug, mine).** MGSM's `train`
   split holds only 8 few-shot exemplars; the 250 problems are in
   split=`test`. The driver asserted 250 rows and crashed before any
   model work. Fixed (load split=`test`, same index split) and
   relaunched on the same account.
2. **Attempt 2 — Qwen3.5-9B, session topsd-1108, account cyc236es:
   DIED silently after base eval.** Base eval completed (table above);
   the driver then died during phase 2 (student traces) with no
   traceback — the 25-line log ends at the eval_base METRIC line.
   dmesg shows no OOM kill, the VM was not rebooted (uptime 2:40),
   10 GB RAM free. Leading hypothesis: Colab reaped the long-running
   detached process while the kernel looked IDLE (the Gated Delta Rule
   kernel falls back to a slow reference implementation without
   flash-linear-attention, so the sw base eval alone took ~35 min).
   The local `train` monitor was SIGTERM'd by the runtime twice during
   this day (background-exec lifetime limit) and had to be re-attached
   with `train --watch-only --pid ...`; the driver itself was
   unaffected by the monitor deaths.
3. **Attempt 3 — automatic fallback (standing rule 2026-11-05),
   Qwen/Qwen3-8B, session topsd-1108b, same account: RECLAIMED.**
   Base eval completed (table above); the driver was alive and
   generating student traces (PID 475, GPU 100%) when Colab reclaimed
   the VM ~2h in. No drift/training data survived beyond the base
   eval — the watch downloaded nothing (`downloaded: []`), but the
   base-eval partials had been downloaded to this session beforehand.

Per the fallback rule, the fallback blocking ends the day's training
track — no further automatic attempts.

## Verdict: BLOCKED

Neither the Teacher Drift phenomenon nor the TOPSD-vs-OPSD comparison
was measured. The only measured numbers are the two base evals above,
which calibrate the test (both models are weak zero-shot on MGSM at
this scale: 0.03–0.27) but say nothing about the paper's claims.

## Honest limits

- **Base evals are the only results.** The paper's claims (drift starts
  earlier in lower-resource languages; TOPSD beats OPSD, especially in
  low/medium-resource languages) are entirely untested here.
- **Slow generation dominated the day.** Base eval alone took ~45 min
  per model (180 greedy generations); the full design (360 student
  traces + 360 teacher traces + drift scoring + 2 SFT arms + 3 evals)
  needs several hours of uninterrupted GPU — longer than free-tier
  sessions reliably provide in the afternoon window.
- **Zero-shot prompt, no few-shot.** The paper's setup presumably uses
  few-shot native CoT; our zero-shot + fixed instruction prompt is a
  bounded simplification recorded in the driver.
- **3 languages, 120 train / 60 test per language** instead of the
  paper's 10 languages at full MGSM scale.

## Resume state (for a retry, needs Peter's fresh instruction)

- Driver: `colab_topsd_driver.py` in this session (MGSM-split fix
  included; `MODEL_ID` env overrides the model). CPU logic test:
  `python3 test_topsd_logic_cpu.py` (33/33).
- **Split design staged 2026-11-09 (not yet run on Colab):** env
  `TOPSD_PHASES` selects the leg — `drift` (base eval skipped, banked
  8B row written; student traces generated in chunks of 24, each
  flushed to `/content/topsd_student_traces.jsonl`; drift rows to
  `/content/topsd_drift.jsonl`; ends `TOPSD_DRIFT_DONE`) or `sft`
  (reloads those two files, rebuilds SFT data reusing stored onsets,
  trains both arms, ends `TOPSD_DONE`). `full` (default) is the
  original one-session design, now also with chunked trace
  persistence. `TOPSD_SKIP_BASE_EVAL=1` skips phase 1 in any leg.
  Leg commands (Qwen3-8B, rotation untouched — `--account cyc236es`
  with the wrapper's cross-account retry, as the 2026-11-08 rerun):
  ```bash
  cd sessions/2026-11-08-2611.08179-topsd
  # Leg A — drift only (morning window):
  python3 ~/workspace/hermes-colab-cli/colab.py train \
    --driver colab_topsd_driver.py --account cyc236es \
    --session topsd-1109-drift --gpu T4 --monitor-timeout 14400 \
    --done-marker TOPSD_DRIFT_DONE \
    --env MODEL_ID=Qwen/Qwen3-8B --env TOPSD_PHASES=drift \
    --results /content/topsd_metrics.csv /content/topsd_details.csv \
    /content/topsd_student_traces.jsonl /content/topsd_drift.jsonl \
    --out-dir . \
    --registry-file ~/workspace/goals/arxiv-daily-read-and-test/hidden_files/train-1109-registry.json
  # Leg B — SFT arms (after leg A's two jsonl files are downloaded):
  python3 ~/workspace/hermes-colab-cli/colab.py train \
    --driver colab_topsd_driver.py --account cyc236es \
    --session topsd-1109-sft --gpu T4 --monitor-timeout 14400 \
    --done-marker TOPSD_DONE \
    --env MODEL_ID=Qwen/Qwen3-8B --env TOPSD_PHASES=sft \
    --upload topsd_student_traces.jsonl:/content/topsd_student_traces.jsonl \
    --upload topsd_drift.jsonl:/content/topsd_drift.jsonl \
    --results /content/topsd_metrics.csv /content/topsd_details.csv \
    /content/topsd_sft_data.jsonl \
    /content/adapters/opsd.zip /content/adapters/topsd.zip \
    --out-dir . \
    --registry-file ~/workspace/goals/arxiv-daily-read-and-test/hidden_files/train-1109-registry.json
  ```
  Note: leg B runs in a fresh VM, so its metrics CSV holds only the
  SFT/eval rows — combine with leg A's CSV for the full record. The
  mid-run downloader should also fetch `topsd_student_traces.jsonl`
  during leg A. Leg A needs Peter's tap to launch; the point of the
  split is the morning window, so launch as soon as he taps. The
  12:20 HKT daily arXiv run picks kyu008008 — a morning leg on a
  fixed account leaves that untouched.
- Design: MGSM en/zh/sw, train idx 0–119 / test idx 150–209 per
  language (same index split in all languages — no problem appears in
  both train and test in any language).
- Rotation state: `2026-11-08 cyc236es` in
  `~/workspace/goals/arxiv-daily-read-and-test/hidden_files/
  colab_account_rotation.txt` — next account in the pool is
  **kyu008008**. Active account was left on cyc236es.
- Base-eval numbers above are the starting calibration; a retry should
  run in a morning window and consider splitting the design (drift
  measurement in one session, SFT arms in a second) so each leg fits a
  free-tier lifetime. The driver already writes metrics incrementally
  and zips adapters per arm; download partials after each phase.
- Known wrinkle: the local `train` monitor is SIGTERM'd by the runtime
  after ~30–120 min of background execution — re-attach with
  `train --watch-only --pid <driver-pid> --driver ...` (the driver
  keeps running on the VM).

## Reproduce

```bash
cd sessions/2026-11-08-2611.08179-topsd
python3 test_topsd_logic_cpu.py   # CPU logic test; 22/22; no GPU
# Colab (needs Peter's go-ahead; see resume state above):
python3 ~/workspace/hermes-colab-cli/colab.py train \
  --driver colab_topsd_driver.py --rotate --session topsd-1108c \
  --gpu T4 --monitor-timeout 14400 --done-marker TOPSD_DONE \
  --env MODEL_ID=Qwen/Qwen3-8B \
  --results /content/topsd_metrics.csv /content/topsd_details.csv \
  /content/adapters/opsd.zip /content/adapters/topsd.zip \
  --out-dir . \
  --registry-file ~/workspace/goals/arxiv-daily-read-and-test/hidden_files/train-1108-registry.json
```

## Rerun — 2026-11-08, session topsd-1108c (Peter's fresh instruction)

- **Model:** Qwen/Qwen3-8B via `--env MODEL_ID=Qwen/Qwen3-8B` (the
  README reproduce choice; the 9B died silently and the 8B base eval is
  banked and proven to run). CPU logic test re-run 22/22 before launch.
  The driver has no base-eval skip option, so the full driver ran as
  designed and the base eval was re-measured.
- **Account path:** `--account cyc236es`, but cyc236es was
  gpu-unavailable at creation; the wrapper's cross-account retry
  created the session on **ppoppo205**
  (endpoint gpu-t4-s-usw4b1-300bhxj591548). Rotation state deliberately
  left at `2026-11-08 cyc236es`; active account restored to cyc236es
  after the run.
- **Outcome: RECLAIMED ~3.5h in, during phase 2 (student-trace
  generation)** — the same phase that killed the 8B fallback at ~2h.
  The driver was confirmed alive three times by direct checks (GPU
  99–100%, PID 836, 2:04 CPU time at the last check); the monitor's
  per-poll PID-liveness check never flagged a silent death. The session
  vanished from the server; the wrapper downloaded nothing
  (`downloaded: []` on RECLAIMED) and removed the stale registry entry;
  no teardown was needed.
- **What ran:** probe GREEN (Qwen3-8B, 6.16 GB); base eval re-measured
  and reproduced the banked 8B calibration exactly — en 0.1167, zh
  0.2333, sw 0.0333 (parse_fail 3 on sw), peak 6.91 GB. No drift
  measurement, no SFT arms, no adapters: phase 2 never wrote metrics
  (it writes at phase boundaries), so no drift/training data survived.
- **What survived:** the host-side mid-run downloader (10-min polls)
  captured `topsd_metrics_rerun.csv` (probe + eval_base rows) and
  `topsd_details_rerun.csv` (180 base-eval detail rows) before the
  reclamation — the only new artifacts of this rerun. No adapter zips
  exist (no arm trained).
- **Verdict stays BLOCKED.** The full design needs several hours of
  uninterrupted GPU; evening free-tier sessions have now been reclaimed
  in phase 2 twice (attempt 3 at ~2h, this rerun at ~3.5h). A future
  retry should run in a morning window and consider the split design
  (drift measurement in one session, SFT arms in a second) so each leg
  fits a free-tier lifetime; phase 2 should also flush partial
  trace/drift state incrementally so a reclamation mid-phase does not
  lose everything.
- **Files:** `topsd_metrics_rerun.csv`, `topsd_details_rerun.csv`,
  `train-run-rerun.json`. Run log: `run-2026-11-08.md` (rerun note
  appended).

## Leg A — drift only (2026-11-09 morning window)

- **Outcome: DIED** — session `topsd-1109-drift`, Qwen3-8B, account
  **ppoppo205** (cyc236es gpu-unavailable at creation; the wrapper's
  cross-account retry landed on ppoppo205 — same as topsd-1108c; the
  launch message said cyc236es, corrected in
  `hidden_files/topsd-1109-drift/ACCOUNT_CORRECTION.md`). NOT a
  reclamation: the driver itself crashed with a traceback. Verdict
  stays BLOCKED. One attempt only — no relaunch, no Leg B.
- **What ran:** probe GREEN (6.16 GB); base eval skipped, banked 8B row
  written (en 0.1167 / zh 0.2333 / sw 0.0333). All **360/360 student
  traces** generated and flushed in 15 chunks of 24 (en 120 / zh 120 /
  sw 120), downloaded in full. Drift pool en: 86 wrong of 120 — then
  the driver crashed at the first drift-pool item.
- **Crash (driver bug, diagnosed from the downloaded artifacts + a
  live Hub check):** `AttributeError: 'NoneType' object has no
  attribute 'strip'` at `colab_topsd_driver.py:459` —
  `r["cot"].strip()` where `r["cot"]` is the MGSM record's `answer`
  field. Verified 2026-11-09 via the datasets-server API: in the
  current `juletxara/mgsm`, `answer` AND `equation_solution` are null
  for every row in all three languages (en/zh/sw) — only `question`
  and `answer_number` are populated. All 360 downloaded trace records
  have `"cot": null` (gold/q/trace all populated). The base eval and
  trace generation never touch `cot`, which is why they ran fine; the
  drift phase is the first place the reference-solution text is needed
  (teacher context `"Reference solution:\n" + cot`, and the teacher
  completion prompt), and the SFT leg needs it too (lines 526/533/552).
- **Fix options (for the next attempt):** (a) source the reference
  solutions from a MGSM mirror that carries them (verify non-null
  before launching); or (b) guard `r["cot"]` (skip/empty-string the
  reference solution in the drift context) — a design change, weaker
  than the paper's method. Either way the 360 downloaded traces are
  reusable ONLY if the driver gains a resume-from-traces path (the
  drift leg currently regenerates traces; `TOPSD_PHASES=sft` requires
  the drift JSONL). The traces file is banked in this session for
  that purpose.
- **Monitor note:** the host-side monitor was SIGTERM'd by the runtime
  ~04:14 UTC (same wrinkle as before); re-attached with
  `train --watch-only --pid 3226` and the driver was unaffected
  (288/360 at re-attach). The mid-run downloader died in the same
  sweep and was restarted — it had grabbed 264 traces before dying
  and 312 after restart; the wrapper's final download on DIED captured
  the full 360-line file, so nothing was lost.
- **Files:** `topsd_metrics_drift.csv` (probe + eval_base rows),
  `topsd_student_traces.jsonl` (360 rows, 505 KB — pushed; small
  enough), `train-run-drift.json`. No `topsd_drift.jsonl` exists (no
  drift rows were written before the crash). No adapter zips (no arm
  trained).
- **Leg B status: STILL WAITING.** Leg B needs `topsd_drift.jsonl`
  from a completed drift leg plus the reference-solution fix above. A
  drift-leg retry (fixed driver) is a fresh instruction; Leg B needs
  its own authorization after that.
