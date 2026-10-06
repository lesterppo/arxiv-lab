# FTW vs GRPO A/B run status (2026-10-05, subagent run)

## Outcome
- Arm1 (GRPO): FAILED — 2 attempts, both lost to Colab session loss. No training steps completed.
- Arm2 (FTW): NOT STARTED.

## Fixes applied to driver (local file only, NOT pushed)
1. `colab_ftw_grpo_9b.py`: 3 call sites passed `tok` as `prompt_ids` to
   `seq_logps(model, prompt_ids, comp_ids, ...)` -> TypeError at train_grpo line 220.
   Fixed to `seq_logps(model, pids, cids[, no_grad=False])`. Syntax re-verified.
2. Env: pinned `transformers>=4.52,<5.0` does NOT recognize Qwen3.5 (`qwen3_5`
   model type unknown -> ValueError on load). Working env: transformers from
   git main (5.19.0.dev0) + peft 0.21.0 + accelerate + bitsandbytes, torchao uninstalled.
   Both arms need this env.

## Findings
- `Qwen/Qwen3.5-9B-Instruct` does NOT exist on HF Hub (probe MODEL_ID_FAIL).
  `Qwen/Qwen3.5-9B` (base) resolves OK — both arms would train the base model.
- Arm1 attempt 1 (kyu009009): heldout BEFORE = 0.250 (24 greedy eval rollouts OK),
  then session reclaimed ~1.5h in during train_grpo (zero step lines completed).
- Arm1 attempt 2 (cyc236de): session lost ~10 min in during 19GB model download.
- The shared token.json symlink was switched externally twice mid-run
  (to kyu008008, then ppoppo205) — concurrent Colab use hazard is live.
- Capacity thin tonight: backend-busy (cyc236de, ppoppo205, kyu009009 post-reclaim),
  gpu-unavailable (cyc236es, kyu008008).

## Checkpoint-resume rebuild (2026-10-06 ~00:10 HKT, Peter-approved)
Driver rewritten (534 lines, syntax-verified): aria2c self-provisioning,
CKPT_EVERY=1 to /content/ckpt.pt (LoRA+optimizer+RNG+curve+replay+h_before),
resume from /content/ckpt_resume.pt. Checkpoints mediated by orchestrator
(Drive API 403 for colab-cli OAuth client — abandoned).
Run IDs: ftw-grpo-20261005, ftw-ftw-20261005.

## Resume PROVEN (2026-10-06 ~02:12 HKT)
- ftw-r3 (cyc236ha): FRESH_START, heldout BEFORE=0.250, steps 1-2 (r=0.250,0.250),
  CKPT_SAVED step=1,2. Checkpoint (168MB) downloaded to .ckpts/.
- Deliberate kill test: stopped ftw-r3 mid-run.
- ftw-r5 (cyc236ha): deps+weights (aria2c -x16: ~55MB/s peak, 19GB in ~6min),
  checkpoint re-uploaded via 5×40MB split parts (single 168MB upload hits
  network errors), reassembled to /content/ckpt_resume.pt.
- Driver printed RESUMED_FROM_STEP=2 (FRESH_START=False). Resume path WORKS.
- Note: transformers pip pulled 5.18.0 (no qwen3_5) — forced --upgrade to
  5.19.0.dev0; bitsandbytes needed --upgrade to 0.50.2 (>=0.46.1 required).
- Account verified cyc236ha before every mutating call; no external switches seen.
- 2026-10-06 ~02:40 HKT: ftw-r5 reclaimed after steps 3-5 (r=0.417,0.083,0.417).
  Step 4 ckpt secured (169MB); step 5 partial (4/5 parts). cyc236ha backend-busy
  → fallback kyu009009 (1 authorized). ftw-r6: resumed from step 4.
- 2026-10-06 ~03:00 HKT: ftw-r6 reclaimed after ~8 min (no new steps).
  ftw-r7 (cyc236ha): resumed from step 4 (RESUMED_FROM_STEP=4 confirmed),
  reclaimed after ~18 min.
- 2026-10-06 ~03:30 HKT: both cyc236ha and kyu009009 backend-busy. Stopping.
  GRPO: 5/12 steps complete. FTW: not started. Status: BLOCKED on capacity.

## To retry
Parent: review driver fix, ensure EXCLUSIVE Colab account access (no concurrent
switching), prefer off-peak. Recipe per arm: new session -> git-transformers deps
(see above) -> upload fixed driver -> FTW_MODE=<mode> nohup launch.

## Retry attempt (2026-10-05 ~21:08 HKT, cyc236ha)
- Session `ftw-grpo` created OK on cyc236ha (T4, READY); deps installed OK
  (transformers 5.19.0.dev0); driver uploaded OK (~14KB, took 96s).
- Console launch failed "not-found": the shared token.json symlink was
  switched EXTERNALLY to cyc236de at 21:09 mid-run (3rd occurrence tonight).
- After switching back to cyc236ha, session `ftw-grpo` is gone entirely —
  reclaimed (or orphaned) during the switch window. Recreating.

## Retry (2026-10-05 ~21:08-22:30 HKT, subagent run, cyc236ha + kyu009009)
- Attempt 3 (cyc236ha): session created READY, deps OK (5.19.0.dev0), driver
  uploaded. Launch failed not-found — token symlink switched externally to
  cyc236de at 21:09 (3rd occurrence). Session gone after switching back.
- Attempt 4 (kyu009009, single fallback): T4 READY. Download 19GB (~50 min,
  throttled), model load 427/427, heldout BEFORE 0.250 (reproduces att.1),
  GRPO steps 1-3/12 done: rewards 0.250, 0.250, 0.417 (loss -0.0000).
  Peak VRAM 12,865 MiB. Reclaimed with 9 steps left; no results JSON.
- FTW arm still unstarted. Poll via exec+tail (logs cmd hangs under load).
- Per task bounds: no further accounts tried. Teardown: server shows no
  active sessions. Active account left as cyc236ha per instruction.

## Fast-download attempt (2026-10-05 ~22:45-23:25 HKT, cyc236ha, session ftw-fastdl)
- KEY FINDING 1: `hf_transfer` is DEPRECATED in current huggingface_hub (FutureWarning:
  "HF_HUB_ENABLE_HF_TRANSFER is deprecated as hf_transfer is not used anymore").
  The documented replacement is `hf_xet` + `HF_XET_HIGH_PERFORMANCE=1`.
- KEY FINDING 2: `HF_XET_HIGH_PERFORMANCE=1` OOM-KILLED the 12GB Colab VM (dmesg:
  anon-rss 11GB, oom_kill_process on python3). Xet buffers aggressively in RAM.
  DO NOT use high-performance Xet on 12GB VMs.
- KEY FINDING 3 (the fix): `aria2c -j 4 -x 8 -s 8 -k 1M` against
  huggingface.co/.../resolve/main/ URLs: 5.0GB in ~3 min (~28 MB/s sustained,
  112 MB/s peaks). Full 19GB est. ~12 min (vs 50 min throttled before).
  Streams to disk, negligible RAM. Install via `apt-get install -y aria2`.
- Driver: added MODEL_DIR env override (local path short-circuits Hub probe)
  + MODEL_LOAD_DONE timing print. Syntax verified.
- Session LOST: ftw-fastdl unreachable after external agent activity
  (token symlink switched to ppoppo205 4th time tonight; sessions.json registry
  wiped to 0 entries at 15:11 UTC). `colab.py recover` re-attached the server
  session but exec/status still return not-found. Aria2 download was at 5GB+
  when contact lost. 7th account cyc236hk@gmail.com appeared (parent added).
- Training-time math (from attempt 4): ~5 min/GRPO-step -> 12 steps ~= 60 min
  per arm; both arms ~= 2h even with fast download. Exceeds typical free-tier
  30-60 min lifetimes — this is now the binding constraint, not download.

## Fast-download attempt 2 (2026-10-05 ~23:20-23:55 HKT, kyu009009, session ftw-k9)
- Deps: transformers 5.19.0.dev0 + peft 0.21.0 + accelerate + bitsandbytes,
  torchao uninstalled, aria2 1.37.0 (apt). ~1 min.
- Download: 4 safetensors shards via 4 parallel aria2c (-x 8 -s 8 -k 1M,
  explicit -o filenames) to /content/model/. ~19GB in ~12 min (~27 MB/s
  sustained, 112-160 MB/s peaks). Small files via curl. NOTE: multi-`-o`
  aria2c arg order is fragile — one shard initially saved under a hash name;
  per-file aria2c invocations with explicit -o are reliable. tokenizer.json
  got clobbered by the index content in the first attempt; re-downloaded clean.
- GRPO arm launched 15:45:29 UTC with MODEL_DIR=/content/model (no Hub I/O).
  Model load 427/427 in 69s. Then RECLAIMED ~25 min in (during heldout BEFORE
  eval or step 1; no step lines observed). No results recoverable.
- Download-speed verdict: PROVEN — aria2c is the fix (12 min vs 50 min).
- Training verdict: still BLOCKED — 12 GRPO steps need ~60 min/arm; free-tier
  sessions tonight die in 25-75 min. Both authorized accounts now exhausted
  (cyc236ha: registry wipe; kyu009009: reclaim).

## GRPO completion (2026-10-06 ~05:50-06:30 HKT, session ftw-resume, cyc236ha)
- Resume authorized by Peter ("Yes"). Active account set to cyc236ha; session
  ftw-resume created OK (T4 READY). Deps + aria2c; weights in ~6 min
  (aria2c -x16 -s16). Checkpoint parts ckpt_p4_aa..ae re-uploaded,
  reassembled on-VM: 176,449,533 bytes verified, step=4, all keys present.
- Relaunched RUN_ID=ftw-grpo-20261005 FTW_MODE=grpo -> RESUMED_FROM_STEP=4
  confirmed. Steps 5-12 completed (~5 min/step):
  0.250, 0.250, 0.417, 0.083, 0.417, 0.417, 0.500, 0.250, 0.417, 0.333,
  0.250, 0.750.
- heldout BEFORE 0.250 -> heldout AFTER 0.7917 (delta +0.5417).
  peak_vram_gb = 12.75. 144 rollouts. RESULTS_OK observed.
- ftw_grpo_results.json + ftw_grpo_curve.csv downloaded to session dir.

## FTW arm (2026-10-06 ~06:30-08:30 HKT)
- INCIDENT: first FTW launch picked up stale /content/ckpt_resume.pt (GRPO's)
  — driver does NOT validate run_id/mode on resume. Killed immediately,
  stale file deleted. LESSON: always rm ckpt_resume.pt before a different arm.
- OOM: second launch crashed (bitsandbytes modules to CPU/disk) — lingering
  python held 7749 MiB VRAM. pkill -9 -f ftw_driver.py -> 0 MiB, relaunch.
- FRESH_START confirmed, MODEL_LOAD_DONE. Steps 1-7 (r=0.083, 0.250, 0.167,
  0.083, 0.167, 0.417, 0.333; loss 0.0926->0.0831). Session reclaimed in step 8.
  Step-1 + step-2 checkpoints backed up locally; later downloads flaked.
- New session ftw-ftw on cyc236hk (7th account; cyc236ha + kyu009009
  gpu-unavailable; task allowed 2 fallbacks). Fresh FTW run, steps 1-7
  reproduced the earlier trajectory almost exactly.
- OOM at step 8: torch.OutOfMemoryError (tried 260 MiB; 13.6 GB busy).
  ROOT CAUSE: FTW loop held one forward graph per winner in a list, then one
  stacked backward (18 winners at step 7-8). FIX: per-winner micro-batched
  backward (grads accumulate; mathematically identical to mean-loss backward).
  Driver edited locally, syntax-verified, uploaded as /content/ftw_driver.py.
- Resume: step-7 ckpt verified on-VM (step=7, mode=ftw), copied to
  /content/ckpt_resume.pt, relaunched -> RESUMED_FROM_STEP=7 confirmed.
- Steps 8-12 completed, no OOM: 0.417, 0.750, 0.583, 0.750, 0.583
  (buf 96->144, winners 23->36, loss 0.0836->0.0755).
- heldout BEFORE 0.250 -> heldout AFTER 0.750 (delta +0.500).
  peak_vram_gb = 13.6. 144 rollouts. RESULTS_OK /content/ftw_ftw_results.json.
- ftw_ftw_results.json + ftw_ftw_curve.csv downloaded to session dir.

## Verdict
- Claim 1 (FTW matches GRPO): SUPPORTED — FTW 0.750 vs GRPO 0.792 heldout
  after, both +0.50 over 0.25 baseline (within run noise).
- Claim 2 (FTW < GRPO memory): NOT SUPPORTED — FTW 13.6 GB vs GRPO 12.75 GB
  peak. CPU replay buffer doesn't lower GPU peak vs GRPO in this impl.
  (Paper's stronger claim is vs critic-based PPO, untested here.)
- Findings 18-21 appended to README. Session teardown + account reset pending.
