# 2026-11-05 — Model Merging Does Not Require Uniform LoRA Ranks (arXiv:2611.06486)

- Article: https://arxiv.org/abs/2611.06486 — "Model Merging Does Not Require
  Uniform LoRA Ranks"
- Track: training. Scores: AGENT=1, TUNE=5.
- Why picked: a fine-tune method directly adaptable to the daily QLoRA
  harness — per-domain LoRA experts merged by averaging, with the rank
  chosen per domain instead of forced uniform.
- Repo fit: soup-daily-finetune pattern (daily 8B-class NF4-QLoRA on
  Colab T4).

## What was built

`colab_merge_ranks_driver.py`: Qwen3.5-9B first (standing 9B-first rule),
NF4 QLoRA, alpha = 2r, target all-linear, lr 1e-4, 1 epoch,
completion-only loss, `remove_unused_columns=False` (2026-11-04 lesson).
Domains: math (GSM8K, 120 train / 60 test) and multiple-choice QA
(ARC-Challenge, 150 train / 60 test), deterministic parsing (numeric
answer / letter), 2-shot prompts disjoint from train/eval.
Experts: math r=8, math r=32, qa r=8, qa r=32. Merges are exact
rank-concatenation: averaged delta 0.5·delta1 + 0.5·delta2 with
delta_i = 2·B_i·A_i is reproduced by one adapter of rank r1+r2, scaling 1,
A = vstack([A1, A2]), B = hstack([B1, B2]). Uniform arm (8, 8); non-uniform
arm chosen from measured single-expert scores. Metrics are written
incrementally to `/content/merge_metrics.csv`.

## Results — VERDICT: BLOCKED

Colab session `train-1105-0423`, account `kyu009009` (rotation:
2026-11-04 kyu008008 → kyu009009), T4, isolated registry
`hidden_files/train-1105-registry.json`. Monitor outcome: **DIED**.

- Probe: Qwen/Qwen3.5-9B loaded first try, peak 8.03 GB — GREEN.
- Failure: CUDA out of memory at step 0/15 of the first expert
  (math_r8), in `torch_chunk_gated_delta_rule` (Qwen3.5's hybrid
  gated-delta layers) during the training forward pass: 14.55 GB in use
  on a 14.56 GB T4. Batch 2 × sequences up to 512 tokens with all-linear
  targets pushed training activations past the budget that the 8.03 GB
  weight-only probe suggested.
- Numbers banked: probe row only (`merge_metrics.csv`: probe,
  Qwen/Qwen3.5-9B, 8.03 GB). No expert trained, no eval numbers, no
  merged adapters. Nothing here supports or refutes the paper's claim.

## Resume state (exact)

- Session torn down: server reports no active sessions. Rotation state
  already stands at `2026-11-05 kyu009009`, so the next training day
  picks `cyc236ha`.
- Driver (staged, untested past the probe): `colab_merge_ranks_driver.py`
  in this session. Rerun with `colab.py train --driver <this file>
  --rotate --results /content/merge_metrics.csv
  /content/merge_details.csv --done-marker MERGE_DONE`.
- Fix order (floors: shrink the experiment, never the model):
  1. Cap training sequences at 256 tokens and drop the batch to 1 with
     gradient accumulation 8 (same effective batch), enable gradient
     checkpointing, and set
     `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.
  2. If the 9B training forward still OOMs, fall back to Qwen/Qwen3-8B
     (the proven training model) with the same driver and record the
     fallback in the session README.
- Do NOT rerun the driver unchanged: the OOM is deterministic at step 0
  for this configuration, not a transient reclamation.

## Honest limits

This day's training track produced no merging numbers. The BLOCKED
verdict is about this run's configuration, not about the paper: the
paper's 36-expert, 3-domain, 3-scale study was never within a one-day
T4 budget, and the bounded 4-expert version above remains untested
until the fixed driver runs.

## Fallback rerun (2026-11-05, automatic model fallback)

Run under Peter's 2026-11-05 standing rule: a BLOCKED training run gets
one automatic fallback attempt with a different agent-picked model plus
the shrink fixes for the failure cause. Colab session `merge-1105-fb`,
account `kyu009009` (`--account`, not `--rotate`; rotation untouched,
still `2026-11-05 kyu009009`), T4, isolated registry
`hidden_files/mergeranks-1105-rerun-registry.json`. Monitor outcome:
**RECLAIMED**. One attempt only — no further relaunch under the rule.

- Driver: `colab_merge_ranks_driver_fallback.py` — Qwen/Qwen3-8B (the
  9B OOM was in Qwen3.5's hybrid gated-delta layers, so a different
  proven architecture was the pick), training sequences capped at 256
  tokens, batch 1 with gradient accumulation 8, gradient checkpointing
  (+ `enable_input_require_grads`), `PYTORCH_CUDA_ALLOC_CONF=
  expandable_segments:True`. Everything else identical to the original
  driver. py_compile + import smoke test passed before launch.
- Progress before reclamation (far further than the first attempt):
  all 4 experts trained to completion — the shrink fixes + 8B switch
  PROVED past step 0 (qa_r32: 19/19 optimizer steps, ~20 s/it,
  train_loss 0.0359). Delta norms (the paper's mechanism diagnostic):
  math_r8 1.59, math_r32 2.85, qa_r8 1.78, qa_r32 2.94 — the r=32
  experts carry ~1.7x the delta magnitude of the r=8 experts on both
  domains, consistent with the paper's warning that magnitude-based
  selection drops the small expert. Base evals (Qwen3-8B, n=60 each):
  math 0.0167 (1/60), qa 0.10 (6/60).
- The session was reclaimed during the single-expert evals, after the
  base evals and before any expert-eval row. No expert evals, no merge
  evals, no uniform-vs-nonuniform comparison — C1 and C2 remain
  untested. **Verdict stays BLOCKED.**
- Data loss note: the wrapper downloaded nothing (`downloaded: []`) and
  produced no mid-run snapshots, so `/content/merge_metrics.csv` and
  `/content/merge_details.csv` died with the VM. The numbers above are
  recovered from the monitor's log tail in `train-run-fallback.json`
  (the only surviving record) and preserved as
  `merge_metrics_fallback_recovered.csv` (7 verifiable rows only: the
  qa_r32 train row, 4 delta-norm rows, 2 base-eval rows). Train losses
  for math_r8 / math_r32 / qa_r8 scrolled out of the log tail and are
  NOT recoverable. The 4 trained adapters (`/content/adapters/`) were
  never downloaded and are lost with the VM.
- Monitor note: the host-side monitor was killed by SIGTERM ~22 min in
  (runtime restart); the driver kept running detached on the VM and was
  re-attached with `train --watch-only --pid 673 --driver
  <fallback-driver>` (same session/registry/results/out-dir/
  done-marker) — no progress lost to the re-attach.

## Resume state (exact, updated)

- Session `merge-1105-fb` reclaimed; server reports no active sessions.
  Rotation state stands at `2026-11-05 kyu009009`, so the next training
  day picks `cyc236ha`.
- The fallback driver is PROVEN past training: all 4 experts complete
  in ~25 min of training; a future run needs ~1.5–2 h total including
  evals. Rerun with `colab.py train --driver
  colab_merge_ranks_driver_fallback.py --rotate --results
  /content/merge_metrics.csv /content/merge_details.csv --done-marker
  MERGE_DONE`. A future run must retrain all 4 experts from scratch
  (adapters lost) and should download adapters + metrics mid-run — this
  run produced no mid-run snapshots, so do not rely on the downloader
  alone; the incremental CSV writes only help if they are downloaded
  before reclamation.
- Active account note: kyu009009 was active at launch; by the end of
  the run the global active account read ppoppo205 (changed by another
  agent's live work while this run was in flight, not by this run —
  this run never switched accounts). Left as-is; do not treat
  ppoppo205 as this run's account.

## 4B rerun (2026-11-05 evening, Peter's order: "Re run on smaller model")

One-off exception to the standing floor ("never shrink the model",
2026-10-27): Peter's explicit instruction overrides the floor for
this rerun only. Colab session `merge-1105-4b`, T4, isolated registry
`hidden_files/mergeranks-1105-4b-registry.json`. Monitor outcome:
**RECLAIMED** — ~3 minutes after creation, during the Qwen3-4B weight
download (log tail ends at `Fetching 3 files: 33%`). One attempt only —
no relaunch.

- Driver: `colab_merge_ranks_driver_4b.py` — Qwen/Qwen3-4B (same Qwen3
  family as the proven Qwen3-8B, one class down, current generation,
  NF4 QLoRA; retired 1.5B/0.5B classes NOT used), shrink fixes kept
  (seq 256, batch 1 + accum 8, gradient checkpointing +
  `enable_input_require_grads`, `expandable_segments:True`), plus
  per-expert adapter zips (`/content/adapters/<tag>.zip`, merged
  adapters likewise) for mid-run download. py_compile + import smoke
  test passed before launch; the diff vs the fallback driver was
  exactly the intended changes.
- Account path (wrapper cross-account retry on gpu-unavailable):
  kyu009009 gpu-unavailable → ppoppo205 gpu-unavailable → created on
  **cyc236es** (rotation pool, legitimate fallback; cyc236de/cyc236hk
  never touched). The run therefore did NOT run on kyu009009 despite
  `--account kyu009009`. Rotation state untouched (`2026-11-05
  kyu009009`), so the next training day still picks cyc236ha.
- Mid-run downloader: a host-side loop
  (`hidden_files/mergeranks-1105-4b/midrun_download.sh`, 10-min polls)
  was running — it polled once at 10:50 UTC and downloaded nothing,
  because no result file existed yet when the session vanished. The
  wrapper's `train` downloads `--results` only on DONE/DIED/TIMEOUT,
  never on RECLAIMED, so `downloaded: []` again. The 4B driver itself
  is unproven past pip install: no probe row, no expert trained, no
  eval numbers, no adapters. Nothing here supports or refutes the
  paper's claim. **Verdict stays BLOCKED.**
- Teardown: session reclaimed by the server; isolated registry file
  is empty (`{}`); the one `[?]` session in the global list is another
  agent's live session on ppoppo205 (different endpoint), not touched.
  Active account restored to ppoppo205 (the pre-run value).

## Resume state (exact, updated again)

- All three 2026-11-05 attempts are recorded above. The 4B driver
  (`colab_merge_ranks_driver_4b.py`) is staged and verified
  (py_compile + import smoke test) but has never run past install —
  a future run retrains all 4 experts from scratch on Qwen3-4B
  (~1–1.5 h expected, shorter than the 8B run) with the adapter-zip +
  mid-run downloader protection already in place. The 8B fallback
  driver remains the proven-through-training option.
- Rotation state stands at `2026-11-05 kyu009009`, so the next
  training day picks `cyc236ha`. Any further retry needs Peter's
  fresh instruction — the one-attempt authorization for this rerun
  is spent.

## Account-switch attempt (2026-11-05 evening, Peter's order: "Figure out reason of being blocked. Then try to switch accounts")

Diagnosis (verified live before launching): the blocker is no longer
the model or the driver — it is free-tier capacity in the evening.
Attempt 1 (Qwen3.5-9B) died of a deterministic CUDA OOM at step 0
— a model/config problem, fixed by the 8B switch + shrink fixes.
Attempt 2 (Qwen3-8B) trained all 4 experts, then was RECLAIMED in
evals — free-tier preemption, not a driver problem. Attempt 3
(Qwen3-4B) was RECLAIMED ~3 min after creation during weight
download, after kyu009009 AND ppoppo205 both went gpu-unavailable
at creation. Switching accounts addresses the quota part only; it
cannot guarantee survival.

Account check (live, `colab.py sessions` per account, active account
restored after each check):
- ppoppo205: another agent's live session (weight-space) + gpu-
  unavailable at creation 18:46 — not usable.
- kyu009009: no live session listed, but gpu-unavailable at creation
  18:46 after two sessions today — daily GPU quota spent, not a
  live-session conflict.
- cyc236es: server list showed a `[?]` T4 record whose endpoint
  matches merge-1105-4b exactly — a stale record of the reclaimed 4B
  session (`status -s merge-1105-4b` says "not found").
- cyc236ha: server list showed a `[?]` T4 record whose endpoint
  matches the FRA session fra-1105 (created on ppoppo205, RECLAIMED
  ~18:49 HKT today) — the same record also appears under ppoppo205's
  own listing, so the list is not strictly per-account and the record
  is most likely stale, minutes after reclamation. kyu009009's list
  is empty although it had two reclaimed sessions today (records
  expire from the list within hours). This was treated as "not
  verified free", and the creation attempt itself was the definitive
  GPU-availability test.
- kyu008008: excluded by the rotation rule (used 2026-11-04; using
  it today would be two days in a row).

Launch: session `merge-1105-acct`, account `cyc236ha` (`--account`,
`--max-accounts 1` so creation could NOT spill onto any other
account), driver `colab_merge_ranks_driver_8b_adapters.py` — Qwen3-8B
UNCHANGED (the variable tested is the account, not the model) plus
the per-expert adapter-save+zip code ported from the 4B driver.
py_compile + import smoke test passed; diff vs the fallback driver
is exactly the intended changes. Monitor outcome:
**PROVISION_FAILED** — cyc236ha gpu-unavailable at creation, the
only candidate tried. No session created, no driver uploaded, no
numbers, no adapters. The `[?]` record on cyc236ha is therefore
either a live runtime after all or the account's GPU quota is spent
— either way, no GPU was available on cyc236ha tonight. One attempt
only — no relaunch.

Rotation state: deliberately UNCHANGED at `2026-11-05 kyu009009`.
The verify5 rule writes the state to the account that actually ran;
nothing ran here, so cyc236ha never consumed its turn and tomorrow's
daily run still picks cyc236ha. Active account restored to ppoppo205
(the pre-run value). Mid-run downloader was started and then killed
after the fast PROVISION_FAILED — it downloaded nothing.

**Verdict stays BLOCKED.** Resume state: the 8B-adapters driver
(`colab_merge_ranks_driver_8b_adapters.py`) is staged and verified
(py_compile + import smoke test) but has never run — a future run
retrains all 4 experts from scratch on Qwen3-8B with the adapter-zip
+ mid-run downloader protection already in place. Any further retry
needs Peter's fresh instruction.
