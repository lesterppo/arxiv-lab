# 2026-10-30 — WISD: Weakening-Induced On-Policy Self-Distillation (2610.38681)

- Article: "Weaken to Strengthen: Weakening-Induced On-Policy Self-Distillation
  for Large Reasoning Models" (Wu / Han / Cai, MIT / NVIDIA).
  Link: https://arxiv.org/abs/2610.38681
- Track: training (Colab T4). Session dir:
  `sessions/2026-10-30-2610.38681-wisd/`
- **Verdict: PARTIAL** — weakening operator verified end-to-end on the real
  model, frozen-pair scoring completed with measured numbers, base eval
  measured; the WISD-vs-control comparison itself was NOT measured because
  the Colab session was reclaimed by Google during WISD training.

## Why picked + repo fit

WISD is a fine-tuning method directly adaptable to the daily NF4-QLoRA
harness (`soup-daily-finetune` pattern: 8B NF4-QLoRA on a free Colab T4):
it needs no external reward model and no labelled trajectories — the
training signal comes from the model's own trajectories scored by a frozen
(original, weakened) pair. A positive result would be a drop-in training
method for the daily harness; a clean negative would rule out a cheap
self-improvement recipe. Either way it is TUNE-relevant.

## Paper method (as implemented)

- Original model pi0; weakened copy pi_w = W(theta0) via training-free
  weight degradation. Contrast r_w(a|s) = log pi0(a|s) - log pi_w(a|s).
  Tilted target q_w(a|s) ∝ pi0(a|s) * exp(r_w(a|s)/alpha), alpha = 2.
  Loss L = alpha * E[KL(pi_theta || q_w)] over states from pi0's own
  trajectories. Student initialised from theta0; frozen pair scores fixed
  cached trajectories; student used alone at inference.
- Weakening arm used here: magnitude pruning ONLY (paper's stronger arm,
  +5.5 avg). tau = 1/6 group-relative threshold (groups of 128 input
  columns), then each output row's L2 norm restored. Applied ONLY to
  attention + feed-forward projection weights (Qwen3-8B is dense, so no MoE
  experts exist to touch); embeddings, output head, norms untouched.
  The quantisation arm was not run (brief: only if the pruning arm finished
  with ample time left; it did not).

## Chosen model + why + probe numbers

- **Qwen/Qwen3-8B ONLY**, NF4 QLoRA r=16/alpha=32, bf16,
  transformers 4.57.6 (< 5.0), peft + bitsandbytes. This is the proven
  training default. Qwen3.5-9B was NOT retried: its QLoRA path is not proven
  end-to-end (a 2026-10-30 probe OOM'd at the first backward), per the brief.
- Probe numbers (live, wisd-1030b, Tesla T4 14.56 GiB):
  - pi0 NF4 load: 5.66 GiB VRAM; packed-param count 4.72B (bitsandbytes
    counts packed uint8 storage, hence below 8B — not a model mismatch;
    the checkpoint is Qwen/Qwen3-8B, 5 shards, 36 layers).
  - Two frozen NF4 copies resident during scoring: 11.62 GiB; peak after
    pruning 12.78 GiB; peak after scoring 12.78 GiB (fits the T4).
  - LoRA student: 43,646,976 trainable params.
  - Throughput (measured, the reason the run ran long): sampling
    generation ~114 s/batch of 8 (192-token cap, NF4 decode-bound);
    rep 1 = 5006 s; rep 2 = 9516 s cumulative; frozen scoring 2932 s for
    700 trajectories (~4.2 s each for the pair).

## What was built

- `test_wisd_cpu.py` — CPU check (stdlib + numpy, no GPU), run on this VM
  BEFORE the Colab launch: pruning operator (group-relative threshold +
  row-norm restore), tilted-target math (normalisation, alpha -> infinity
  recovers pi0, logZ identity), top-K cache with force-included sampled
  token + renormalised KL, GSM8K prompt formatting + #### / last-number
  parsing. Result: `CPU_CHECK_ALL_PASS` (see `cpu_check_output.txt`; the
  check caught and fixed one real parser bug for negative answers).
- `colab_wisd_driver.py` — the as-run (v2) Colab driver. Self-contained
  (pip installs at top, `pip uninstall -y torchao` right after),
  prints `TRAIN_DONE` as its last line on completion, writes metrics
  CSVs to /content. Arms: (1) BASE eval only; (2) WISD student (KL to
  tilted target, alpha = 2); (3) CONTROL student (KL to pi0 itself,
  alpha -> infinity) — same cached trajectories, steps, LR. The control
  is essential: without it a gain could be plain self-training.
- `colab_wisd_driver_v1_nocache.py` + `train_run_v1.log` — the v1 driver
  and its partial log (see Deviations).

## Results (measured on the live T4; nothing below is estimated)

| Measurement | Value |
|---|---|
| Trajectories generated (GSM8K TRAIN questions only, no correctness filtering) | 700 = 350 prompts x 2 responses, temp 0.8, top_p 0.95, max 192 new tokens, 0 dropped as empty |
| Mean response length | 192.0 tokens (min 184, max 192) — see Limits |
| BASE eval, GSM8K TEST, greedy, exact match | **0.2300 (46/200)**, hash_rate 0.475, mean gen tokens 192.0 |
| Pruned modules | 252 / 252 expected (36 layers x 7 projections) |
| Pruned fraction (tau=1/6, group 128) | mean 0.4013 (min 0.371, max 0.736) |
| Pruning verification (3 modules re-dequantised) | max row-norm rel err 0.004, zero fraction 0.380 |
| Frozen-pair scoring, 134,392 response positions | mean KL(pi0 \|\| q_w) = 0.1200 (full vocab, alpha = 2); mean top-64 mass under q_w = 1.0000 |
| WISD student training | STARTED (LoRA attached, 43.6M trainable) — no optimiser steps, loss, or eval recorded before reclamation |
| CONTROL student | NOT STARTED |
| WISD / CONTROL eval accuracies | NOT MEASURED |

The base rate (23%) is far from ceiling, so the comparison, had it run,
could have discriminated. The tilted target is real but modest in
magnitude (mean KL 0.12 per position) and extremely peaked (top-64 mass
1.0000), so the top-K truncation loses negligible mass — that
approximation is not what limits this run.

## What was NOT measured (honest gaps)

- No WISD accuracy, no control accuracy, no WISD-minus-control delta.
  Any claim about whether weakening-induced tilting helps would be
  fabrication at this point; none is made.
- No training loss trace (the loss CSV downloaded contains only its
  header — training had just started).
- No adapter weights survived (adapters are saved only after training;
  the VM was reclaimed first).

## Deviations from the brief (all recorded, none silent)

1. **Account switch**: the brief asked for a manual switch before
   `train --rotate`. The wrapper's `--rotate` itself reads the rotation
   file and switches, so a manual switch first would have double-rotated
   (kyu008008 -> kyu009009 -> cyc236ha). Instead the pre-switch account
   (kyu008008) was recorded, `train --rotate` performed the single
   rotation to kyu009009, the rotation file was updated by the wrapper,
   and kyu008008 was restored at the end (verified). Accounts cyc236de /
   cyc236hk were never touched.
2. **v1 use_cache bug -> v2 fix**: v1 set `use_cache=False` globally, so
   `generate()` recomputed the full prefix per token; v1 burned 65+ min
   of GPU at 100% with zero generation output lines and was killed.
   v2 sets `use_cache=True` (plus per-batch progress prints and a
   progress JSON per generation rep). Design, seed, prompts, and counts
   are unchanged between v1 and v2.
3. **LR**: paper LR 1e-6 is full-parameter; both LoRA students use
   LR 5e-5 (same schedule for both arms), cosine with 10 warmup steps,
   micro-batch 1, grad accum 8, 1 epoch.
4. **Scale**: 350 prompts x 2 responses (not the paper's 3200 x 4);
   eval n = 200 (not the paper's AIME/HMMT/LiveCodeBench suites).
   GSM8K TEST only, per the brief. Experiment shrunk, model not shrunk.
5. **Top-K truncation**: per-position reverse KL is computed over the
   top-64 set of q_w (sampled token force-included), with student and
   target distributions renormalised over that set — not the full
   151,669-token vocabulary. Justification: measured top-64 mass is
   1.0000, so the truncation is near-lossless here. The frozen pair
   itself was never computed from student logits.
6. **Monitor windows**: after the v1 monitor process was SIGTERM'd at
   ~57 min (cause unknown, piped output lost), the v2 run was watched in
   50-minute `train --watch-only` windows (TIMEOUT leaves the session
   up; re-attach). Five windows completed cleanly; the sixth ended
   RECLAIMED after ~7 min.
7. **Session names**: v1 `wisd-1030` (killed, torn down); v2 `wisd-1030b`
   (reclaimed). The v2 isolated registry file lives under goal
   hidden_files (not /tmp — the v1 /tmp registry file vanished).
8. **Registry recovery (operational note for future runs)**: after the
   v1 monitor died, the VM was recovered via `colab.py recover`, which
   wrote a `variant` of int 1 into sessions.json; pydantic's
   SessionState requires a string, so every session read as "not found"
   until the variant was fixed to the string "GPU" in place. The store
   then validated and the session resolved. Do not hand-edit session
   entries beyond this known fix; the store is regenerable via recover.

## Limits

- **Truncated trajectories**: mean response length equals the 192-token
  cap (min 184), i.e. the base model almost never emits EOS under this
  prompt. Training trajectories are therefore cut-off reasoning chains,
  and eval generations usually lack `####` (hash_rate 0.475), so the
  parser falls back to the last number. Both arms share the same
  trajectories and the same parser, so the comparison would still have
  been fair — but absolute accuracies are depressed by this, and a
  follow-up should consider a prompt/format the base model terminates.
- **Base-model weakness**: Qwen3-8B (base, not instruct) scores 23% on
  GSM8K TEST under this prompt. That is a property of the base model +
  prompt, not of WISD; it leaves headroom but also means small deltas
  need the full n = 200 to resolve (a 5-point delta is ~2.2 standard
  errors at n = 200, p ~ 0.23 — suggestive, not conclusive).
- **Single seed, single epoch, LoRA (not full-parameter)**: the paper's
  full-parameter, multi-domain result is not being replicated at full
  fidelity; this is a bounded-scale directional test, as designed.
- **No checkpoint survived**: the driver writes metrics incrementally
  but saves adapters and the summary only at phase end; the top-K cache
  (`wisd_cache.pt`, ~118 MB) lived only on the VM and was not in the
  downloaded set. A rerun regenerates everything.

## Resume state (exact)

- Status at loss: session `wisd-1030b` (endpoint
  gpu-t4-s-usw1b0-1seumqzchk2m6), account kyu009009, driver PID 2020,
  reclaimed by Google during WISD training (phase: TRAIN wisd, model
  loaded, LoRA attached, no recorded optimiser steps). Verified gone:
  no active sessions under either the global or the registry-file
  listing; active account at check time was still kyu009009 (no
  interference from another agent). Nothing remained to tear down.
- Completed and preserved locally: generation (700 trajectories),
  base eval (200 predictions in `wisd_eval_base.jsonl`), pruning +
  verification, frozen-pair scoring probe numbers (in
  `wisd_metrics.csv`). The underlying trajectories and top-K cache
  were NOT preserved (VM-only).
- To resume: re-run `colab_wisd_driver.py` from scratch on a fresh
  session (same account rule applies: rotation file says 2026-10-30
  kyu009009, so the next training day rotates to cyc236ha; a same-day
  retry stays on kyu009009). Expected duration at measured throughput
  is ~6 h end-to-end (generation ~2.8 h, base eval ~0.5 h, scoring
  ~0.8 h, two trainings + two evals ~2 h), so a retry should either
  start early in the day or shrink the experiment (fewer prompts /
  shorter cap) — that shrink needs a fresh decision, it is not
  authorised by this run. A cheap robustness fix for a retry: add the
  top-K cache and the trajectory list to the downloaded results and
  flush adapter checkpoints per phase, so a reclamation mid-run
  preserves partial work.
- Account state after this run: active account restored to kyu008008
  (verified via `colab-use-account list` + token symlink); rotation
  file left at `2026-10-30 kyu009009` (today ran on kyu009009).

## Files in this session dir

- `README.md` (this file)
- `colab_wisd_driver.py` — as-run v2 driver
- `colab_wisd_driver_v1_nocache.py` — v1 driver (superseded)
- `train_run_v1.log` — v1 driver log (downloaded before teardown)
- `train-run-launch-wisd-1030.json`, `train-run-launch-wisd-1030b.json`,
  `train-run.json` (final: RECLAIMED) — run records
- `test_wisd_cpu.py`, `cpu_check_output.txt` — CPU check + output
- `wisd_metrics.csv` — all measured probe/base numbers
- `wisd_eval_base.jsonl` — 200 base eval predictions
- `wisd_progress.json` — generation progress record (rep 1)
- `wisd_train_loss.csv` — header only (no training steps recorded)
- NOT present (lost with the VM): `wisd_summary.json`,
  `wisd_eval_wisd.jsonl`, `wisd_eval_control.jsonl`, `wisd_cache.pt`,
  adapter weights.

## Run timeline (HKT = UTC + 8)

- 12:36 HKT: v1 launched (`wisd-1030`, kyu009009 via `--rotate`).
- ~13:33 HKT: v1 monitor SIGTERM'd (~57 min, cause unknown); driver
  found alive on the VM, generating pathologically slowly
  (use_cache=False); registry recovered (variant fix); v1 killed,
  session stopped, log preserved.
- ~13:52 HKT: v2 launched (`wisd-1030b`, kyu009009, no re-rotation).
- Windows 1–5 (50 min each): generation (rep 1: 5006 s; rep 2 done at
  9516 s cumulative), base eval, pruning, scoring; all TIMEOUT with
  the session healthy.
- ~18:14 HKT: window 6 ended RECLAIMED ~7 min in, during WISD
  training. Session verified gone; account restored to kyu008008.
