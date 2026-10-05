# Session: Follow the Winners (2610.03361) — FTW vs GRPO A/B

**Paper:** https://arxiv.org/abs/2610.03361 — "Follow the Winners: Conservative
Policy Improvement with the Cross-Entropy Method for Critic-Free RFT"

**Status: BLOCKED (infrastructure).** No training steps completed on either arm.
No numbers were faked; everything below is what actually happened.

> Note: the arXiv abstract/PDF could not be read directly during this session
> (browser service upstream 502). The implementation follows the paper summary
> in the task brief: critic-free RFT for agentic LLMs; replay buffer on CPU;
> ordinal filter keeps top-k samples by return; CEM-style policy update toward
> elites; derived via control-as-inference; claims parity with GRPO/PPO on
> Sokoban and Search-R1 with lower GPU memory (no critic, no per-prompt group
> rollouts, buffer on CPU).

## Why picked (model-tuning track)

Critic-free RFT is directly on the training track's path: the daily
`soup-daily-finetune` runs QLoRA on T4, and the track has been moving from
0.5B toys to 7B/9B GRPO-style fine-tuning. FTW's promise — GRPO-class results
without group rollouts or a value model, trading GPU memory for CPU memory —
is exactly the kind of mechanism worth validating before adopting. Repo fit:
training track (`soup-daily-finetune`, `hermes-gi-egtkg-finetune`); mechanism
would graduate to `arxiv-lab` training modules if verified.

## What was built

`colab_ftw_grpo_9b.py` — Colab-T4 driver (runs ON the VM via
`colab.py upload` + `console` nohup; no tunnel), two arms, equal budgets:

- **GRPO baseline** (`--mode grpo`): G=4 rollouts/prompt, per-prompt
  group-mean baseline, standardized advantages, clipped policy-gradient
  update (clip 0.2), 1 epoch/step.
- **FTW** (`--mode ftw`): 1 rollout/prompt appended to a CPU replay buffer
  (FIFO, cap 256); each step an **ordinal filter** keeps the top 25% by
  return; CEM-style weighted MLE toward the elite winners (r > 0 only),
  1 epoch/step.

Common: Qwen3.5-9B NF4 QLoRA (r=16/α=32, all 7 linears, dropout 0.0), bf16,
lr 2e-5, AdamW8bit, grad checkpointing, eval-mode rollouts, 12 steps/arm,
144 rollouts/arm. Task: 48 train / 24 held-out synthetic multi-step integer
arithmetic problems (6 templates, answers computed exactly, `\boxed{}`
grading, binary reward). Held-out greedy accuracy before/after + peak VRAM
(`torch.cuda.max_memory_allocated`) per arm.

## What happened (2026-10-05 evening HKT)

| Arm | Outcome |
|---|---|
| GRPO, attempt 1 (kyu009009) | Held-out BEFORE eval completed: **0.250** (24 greedy). Session reclaimed ~1.5 h in, during `train_grpo`, zero step lines. No partial CSV recoverable. |
| GRPO, attempt 2 (cyc236de) | Session lost ~10 min after creation, during the ~19 GB model download. |
| FTW | Not started (sequential plan; arm 1 never completed). |

## Retry (2026-10-05 ~21:08–22:30 HKT, new sixth account)

Peter provided a sixth Colab account (`cyc236ha`) after the block. Retry:

| Arm | Outcome |
|---|---|
| GRPO, attempt 3 (cyc236ha) | Session `ftw-grpo` created OK (T4 READY); deps installed OK
  (transformers 5.19.0.dev0); driver uploaded OK. Console launch failed
  "not-found": the shared token symlink was switched **externally to
  cyc236de at 21:09 mid-run** (3rd occurrence tonight). After switching
  back, the session was gone — reclaimed/orphaned in the switch window. |
| GRPO, attempt 4 (kyu009009, single authorized fallback) | T4 secured. Full
  pipeline ran: 19 GB download (~50 min, throttled unauthenticated HF),
  model load (427/427 weights), held-out BEFORE = **0.250** (reproduces
  attempt 1 exactly), then **3/12 GRPO steps completed before reclaim**:
  step rewards **0.250 → 0.250 → 0.417** (loss −0.0000 throughout).
  Peak VRAM observed **12,865 MiB** (~12.6 GB). Session reclaimed with
  9 steps remaining; no results JSON written (only on clean finish). |
| FTW | Still not started. |

**Verdict unchanged: BLOCKED on both claims.** The partial GRPO curve
(3 steps, reward rising 0.250→0.417) is real but far too short to compare
against FTW — no verdict on either claim is possible. Notable: this is the
first time any training step completed in this session's history; the driver
and env recipe are now proven end-to-end through step 3.

**New findings:**
6. The external account-switcher is still active (21:09 switch to cyc236de
   mid-run, unprompted). Any future retry must assume the symlink can move
   at any time; check `colab-use-account list` before every mutating call.
7. `colab.py logs` hangs when the session is under load; `exec --code`
   with an inline `tail` of the log file is the reliable poll method.
8. Unauthenticated HF Hub downloads are throttled (~790 s/file for the
   19 GB model) — roughly half the session lifetime went to download.
   An HF_TOKEN would cut this substantially (future work, needs Peter's
   token via secure entry).

## Fast-download attempt (2026-10-05 ~23:20–23:55 HKT, kyu009009)

Motivation: attempts 1–4 lost ~50 min each to throttled HF downloads. This
attempt tested a parallel-download approach to fit the run into one session.

**Download findings (all measured live):**
9. `hf_transfer` is **deprecated** in current `huggingface_hub`
   (FutureWarning: "not used anymore"); the replacement is `hf_xet`.
10. `HF_XET_HIGH_PERFORMANCE=1` **OOM-kills a 12 GB Colab VM** (dmesg:
    11 GB anon-rss, oom_kill on python3). Never use on T4 runtimes.
11. Root cause of the slowness: `Qwen/Qwen3.5-9B` is stored on HF's **Xet**
    backend — `/resolve/main/` redirects to signed `xet-bridge-us` CDN URLs.
    The default Python client crawls (~6 MB/s); parallel segmented fetch
    flies.
12. **Working fix: `aria2c -j 4 -x 8 -s 8 -k 1M`** (apt-installed), one
    process per safetensors shard with explicit `-o` filenames, small files
    via curl. Result: **~19 GB in ~12 min (~27 MB/s sustained, 112–160 MB/s
    peaks)** — 4× faster than the throttled client. Streams to disk,
    negligible RAM. (Caveat: multi-`-o` in one aria2c invocation mislabels
    files — one shard initially saved under a hash name and tokenizer.json
    got clobbered by index content; per-file invocations with explicit `-o`
    are reliable. Always re-verify small JSON files parse.)
13. Driver now supports `MODEL_DIR` env override (local path short-circuits
    the Hub probe) and prints `MODEL_LOAD_DONE dl_time_min=...`.

| Arm | Outcome |
|---|---|
| GRPO, attempt 5 (kyu009009, authorized fallback) | Deps ~1 min; aria2c
  download ~12 min; model load 427/427 in 69 s (`MODEL_LOAD_DONE
  dl_time_min=1.2`). **Session reclaimed ~25 min in**, during held-out
  BEFORE eval — no step lines, nothing recoverable. |

**Verdict: BLOCKED on both claims** (unchanged). The download problem is now
solved (12 min vs 50 min), but the binding constraint moved: 12 GRPO steps
need ~60 min/arm (≈5 min/step measured), and free-tier sessions tonight die
in 25–75 min. Both authorized accounts are exhausted. Realistic paths
forward: (a) Drive-checkpointed resume across sessions (soup-daily-finetune
pattern); (b) Colab Pro; (c) reduced step count (weaker evidence).

**Verdict: BLOCKED on both claims** (performance parity, memory savings) —
zero training steps completed, so there is no evidence for or against the
paper. The single real datapoint: base `Qwen/Qwen3.5-9B` scores 0.250 greedy
on the synthetic arithmetic held-out (n=24), i.e. the task has headroom.

## Findings for the retry (all verified live)

1. **Driver bug (fixed, reviewed):** all 3 `seq_logps` call sites passed the
   tokenizer as `prompt_ids` (`seq_logps(model, tok, pids, cids)` vs
   signature `seq_logps(model, prompt_ids, comp_ids, no_grad=True)`) —
   `TypeError` before any training. Fixed to `seq_logps(model, pids, cids[,
   no_grad=False])`; syntax re-verified.
2. **transformers pin wrong for Qwen3.5:** `transformers>=4.52,<5.0` does not
   recognize the `qwen3_5` model type (`ValueError` on load). Working env
   (verified through model load + held-out eval): transformers from git main
   (5.19.0.dev0) + peft 0.21.0 + accelerate + bitsandbytes, torchao
   uninstalled. Docstring updated with the working recipe.
3. **Model ID:** `Qwen/Qwen3.5-9B-Instruct` does **not** exist on the Hub;
   `Qwen/Qwen3.5-9B` (base) resolves. Both arms would train the base model
   (driver probes both IDs automatically).
4. **Concurrent-use hazard is live:** the shared `~/.config/colab-cli/token.json`
   symlink was switched externally twice mid-run (to kyu008008, then
   ppoppo205). Retry needs exclusive Colab access (no concurrent switching).
5. **Capacity thin tonight:** `backend busy` on cyc236de/ppoppo205/kyu009009,
   `gpu-unavailable` on cyc236es/kyu008008; aggressive reclaims (matches the
   2026-10-03 free-tier pattern). Prefer off-peak.

## Checkpoint-resume rebuild (2026-10-06 ~00:10–03:30 HKT, Peter-approved)

Peter approved a checkpoint-resume redesign to make reclaims non-fatal.
Driver rewritten (534 lines, syntax-verified):
- Weights via aria2c self-provisioning (skips if present); `MODEL_DIR` override kept.
- Checkpoint every step (`CKPT_EVERY=1`) to `/content/ckpt.pt`: LoRA adapter +
  optimizer state + RNG states + reward curve + replay buffer + h_before +
  peak/wall accumulators. Prints `CKPT_SAVED step=N`.
- Resume: orchestrator uploads latest checkpoint as `/content/ckpt_resume.pt`;
  driver restores and prints `RESUMED_FROM_STEP=N` or `FRESH_START`.
- Google Drive abandoned: Drive API v3 returns 403 `accessNotConfigured` for
  the colab-cli OAuth client — checkpoints mediated by orchestrator via
  `colab.py download` to local `.ckpts/` (never pushed).
- Run IDs: `ftw-grpo-20261005`, `ftw-ftw-20261005`.

**Resume PROVEN (explicit kill test):**
- ftw-r3 (cyc236ha): FRESH_START, heldout BEFORE=0.250 (4th reproduction),
  steps 1–2 (r=0.250, 0.250), CKPT_SAVED step=1,2. Checkpoint (168MB)
  downloaded to `.ckpts/`.
- Deliberate kill: stopped ftw-r3 mid-run.
- ftw-r5 (cyc236ha): deps + weights (aria2c `-x16 -s16`: ~55 MB/s peak,
  19GB in ~6 min), checkpoint re-uploaded via 5×40MB split parts (single
  168MB upload hits network errors), reassembled to `/content/ckpt_resume.pt`.
- Driver printed **RESUMED_FROM_STEP=2** (FRESH_START=False). Resume WORKS.
- ftw-r5 continued: steps 3–5 (r=0.417, 0.083, 0.417), checkpoints saved.
  Step 4 checkpoint secured (169MB). Reclaimed after ~50 min.
- ftw-r6 (kyu009009, authorized fallback): resumed from step 4
  (RESUMED_FROM_STEP=4 confirmed), reclaimed after ~8 min, no new steps.
- ftw-r7 (cyc236ha): resumed from step 4, reclaimed after ~18 min.
- External switcher hit 5th time (to cyc236de mid-run); switched back.
- Both accounts backend-busy at 03:30 HKT. Stopping.

**GRPO partial results (5/12 steps, all real):**
- Step rewards: 0.250 → 0.250 → 0.417 → 0.083 → 0.417
- Heldout BEFORE: 0.250 (n=24, reproduced 4×)
- Peak VRAM: 12,865 MiB (from attempt 4; not re-measured in resume runs)
- The model learns (0.250→0.417) but with high variance (drop to 0.083).

**Verdict: BLOCKED on both claims** (unchanged). GRPO arm incomplete (5/12),
FTW arm never started. The resume mechanism is proven and the driver is
production-ready; the blocker is Colab free-tier capacity/instability, not
the science. Checkpoints for steps 2, 3, 4 are archived in `.ckpts/` (local
only) — a future run can resume from step 4.

**New findings:**
14. aria2c `-x 16 -s 16 -k 2M`: ~55 MB/s peak, 19GB in ~6 min (2× faster
    than `-x 8`). The Xet CDN handles aggressive parallelism.
15. transformers pip may pull 5.18.0 (no `qwen3_5`) — force
    `--upgrade --force-reinstall` to get 5.19.0.dev0. bitsandbytes needs
    `--upgrade` to ≥0.46.1 (got 0.50.2).
16. Single 168MB `colab.py upload/download` hits network timeouts; split
    into 40MB parts for reliable transfer.
17. The checkpoint-resume design is sound: RESUMED_FROM_STEP proven twice,
    training continues correctly from restored optimizer state.

## Files

- `colab_ftw_grpo_9b.py` — driver (fixed, Colab-ready; retry recipe in docstring)
- `RUN_STATUS.md` — subagent run log with fixes and findings
- `README.md` — this file

## Honest limits

- Proxy task (synthetic arithmetic), not the paper's Sokoban/Search-R1.
- QLoRA r=16, not full fine-tuning; 12 planned steps (3 executed on GRPO
  attempt 4 before reclaim — partial curve only, no comparison possible).
- Base model, not instruct (Instruct repo doesn't exist).
- Binary reward; no KL-vs-reference term (paper's "conservative" element
  approximated by 1-epoch small-lr elite updates — deviation noted).
- Single planned seed; no repeats.
- GRPO peak VRAM 12,865 MiB observed (n=1 snapshot series); FTW arm never
  ran, so the paper's memory-savings claim is untested.
