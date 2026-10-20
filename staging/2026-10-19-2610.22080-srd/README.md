# SRD 2610.22080 (Shared Residual Distillation) — status as of 2026-10-19

Paper: arXiv 2610.22080, Shared Residual Distillation (TUNE 5).
Picked 2026-10-19 06:44 HKT on Peter's order as an *additional* training-track
test, alongside (not replacing) the 12:20 daily cron.

## Verdict: PARTIAL / BLOCKED (Colab reclaimed before the training ran)

What was banked (all verified):
- CPU pre-check 29/29 passed: minmod shared-residual construction (Eqs. 6-8),
  q0-centering, target normalization, pointwise-clipped KL (Eq. 9), uniform
  peer split, synthetic math generator + `\boxed{}` checker, faithful C.4
  student/teacher prompts (`cpu_precheck.py`, `srd_common.py`).
- Colab T4 (kyu008008), testbed Qwen3-8B (the paper's own model family;
  Qwen3.5-9B abandoned: Mamba layers ~3x slow without fused kernels, and the
  `causal_conv1d` source build hangs on Colab). Thinking disabled per C.1.
- Base eval greedy pass@1 = 4/24 = 0.167, measured THREE times, identical.
- Phase 1 (96 peer rollouts) completed once (run clock 4466.8s) before the
  driver died entering phase 2 (third silent process death, no traceback).

What was NOT measured: the SRD training delta and the final eval. The
`sr-d-test` session was gone by 12:30 (reclaimed, /content wiped). The resume
checkpoints (`/content/srd_rollouts.json`, `/content/srd_scored.pt`) existed
only in the patched driver — the completed 96 rollouts lived in the OLD
driver's memory and were lost.

## Honest limits
- Base 0.167 is on 24 synthetic held-out prompts; the training claim
  (peer-distilled student beats base) is untested.
- Three silent process deaths during compute-heavy phases: VM/GPU flakiness,
  not a diagnosed code bug (minimal repros of the suspect paths all passed).
- Write-up as PARTIAL/BLOCKED is the faithful record; any retry is Peter's call.

## Files
- `cpu_precheck.py`, `srd_common.py` — CPU verification (29/29).
- `colab_srd_9b.py` — full Colab driver (base eval -> 96 peer rollouts ->
  teacher scoring -> SRD training -> final eval), resume-aware.
- `colab_driver_body.py`, `build_driver.py` — driver assembly pieces.

## Colab retry (2026-10-20, Peter-approved "三個都跑")

**RECLAIMED — partial, same story.** Account kyu008008, T4,
`colab_srd_9b.py` uploaded as `/content/srd_run.py`, launched detached.
Phase 1 (24 problems × 4 peer rollouts) completed and checkpointed
(`[2675.3s] rollouts 24/24`, `srd_rollouts.json` + `srd_metrics.csv` on
`/content`). Phase 2 (shared-residual teacher targets, minmod Eqs. 6–8)
scored 8/24 targets (`[3803.7s] targets 8/24 scored=128`, ~140s/target)
before the VM was reclaimed ~09:45–09:50 HKT — the driver never reached
SRD training or the final eval. Nothing was downloadable at harvest
(session 404, /content gone); the checkpoint files died with the VM.

Honest read: this is the second reclamation on this track in two days,
and the phase-2 teacher scoring (~1h for 24 targets) is now the measured
bottleneck — even a full morning window is marginal for end-to-end SRD
on a free T4. The `drive_ckpt.py` port noted below was NOT done for this
retry (SST's run proved the no-token fallback works, but a token was
never minted, so it would have bought nothing). The SRD training delta
remains unmeasured; verdict stays PARTIAL/BLOCKED pending a retry with
(a) `drive_token.json` minted, or (b) a paid/stable GPU.

## Rerun note (if Peter approves)
The driver already checkpoints rollouts/scored targets to /content; for
reclamation survival, port `drive_ckpt.py` from the SST session
(`sessions/2026-10-16-2610.22293-sst/`) into this driver the same way
`colab_sst_9b.py` does (restore at startup, sync in `mark()`). The one-time
prerequisite is a `drive_token.json` on the VM (see `mint_drive_token.py`).
