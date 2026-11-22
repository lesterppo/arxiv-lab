# Session — 2026-11-22 — arXiv:2611.25926

## Article
**ROAM-GRPO: Enforcing Reasoning Diversity in GRPO via Routed Off-mode
Approach Mining**
Jeff Duan, Matt Thomson.
https://arxiv.org/abs/2611.25926

## Why picked + repo fit
AGENT 2, TUNE 5. GRPO collapses to a low-diversity mode; ROAM-GRPO
detects well-solved problems during training, routes them to an
"off-mode" track where the model is prompted for distinct solution
strategies (judged for validity/distinctness), and self-distills them
back into the policy. Paper: Qwen3.5-2B + 1,000 MATH lvl5, peak
pass@16 +4.7%, correct-embedding cosine distance ~1.5–2x. Fit:
soup-daily-finetune — an RLVR recipe directly adaptable to the daily
8B QLoRA harness. The paper's claim is about a training mechanism, so
it belongs on the Colab T4 track.

## What was built
- `colab_roam_grpo.py` — self-contained Colab driver (v3). Model
  Qwen/Qwen3.5-9B-Base (standing pick), NF4 QLoRA r=16/a=32, bf16,
  transformers==5.20.2. Data: EleutherAI/hendrycks_math (per-subject
  configs), Level 5 only; N_TRAIN=24, N_EVAL=20, G=4, EVAL_G=4,
  MAX_NEW=256. Single-update GRPO: group-relative advantage
  (std floor 1e-4, skip on all-equal rewards), per-token logprobs via
  generate(output_scores=True), clipped surrogate eps 0.2, no KL term,
  grads accumulated over 4 groups, AdamW lr 1e-5. ROAM arm adds
  off-mode mining (pass rate >= 0.75 → 4 samples at temp 1.0, keep if
  TF-IDF cosine distance > 0.30 to every kept correct solution) + one
  SFT epoch over the mined buffer (lr 2e-4). Eval: pass@4 + mean
  pairwise TF-IDF cosine distance among correct solutions.
  Metrics CSV + summary JSON to /content, flushed per phase.
  Prints TRAIN_DONE last.
- `test_roam_grpo_cpu.py` — deterministic CPU test, 52 checks pass
  (answer extraction incl. nested-brace boxed, advantage math, clipped
  surrogate, TF-IDF distinctness, routing/mining rules, metric schema,
  env defaults).
- `train_run_d.json` — the completed run's train-run.json (outcome
  DONE, driver log tail with all numbers). The metrics CSV and summary
  JSON were written to /content but LOST: the watch-only re-attach did
  not download them before the session was torn down (see Limits).

## Results (train-1122-roam-d, account cyc236ha, T4 — TRAIN_DONE)
Numbers are from the driver log tail in `train_run_d.json` (the metrics
files were lost; see Limits).

| Arm | pass@4 | mean pairwise cos dist | mean gen tokens |
|---|---|---|---|
| base | 0.500 | 0.583 | 249.4 |
| GRPO (adapter_grpo) | 0.450 | 0.661 | 248.6 |
| ROAM (adapter_roam) | 0.350 | 0.603 | 249.0 |

Training signal — **none, in either arm**:
- GRPO pass: `updates=0` (2,199 s). All 24 train problems had
  all-equal rewards (all 4 samples correct or all 4 wrong), so every
  group's advantage was zero and no gradient step was ever taken.
- ROAM pass: `updates=0`, `well_solved=0`, `mined=0`
  (2,198 s). No train problem reached pass rate >= 0.75, so the
  off-mode mining trigger never fired, the distillation buffer stayed
  empty, and no SFT step ran ("WARNING: distillation buffer empty").
- Combined with well_solved=0 and all-equal groups, every train
  problem was 0/4 correct in every sample — while the eval set (same
  distribution, test split) had pass@4 = 0.500 (10/20 solved at least
  once). That train/eval solve gap (0/24 vs 10/20) is unexplained:
  either the train split is genuinely much harder for these Level-5
  problems, or the train-mode sampling path degraded generations.
  The driver did not save train generations, so the two cannot be
  distinguished from this run (a design gap — see Resume).
- Both adapters are therefore untrained (fresh LoRA with B=0 init is
  the identity), so the eval differences (0.500 vs 0.450 vs 0.350 on
  n=20) are pure sampling noise between identical models, not a
  mechanism effect. The diversity numbers (0.583 / 0.661 / 0.603)
  likewise carry no training signal.

## Verdict
**INCONCLUSIVE.** The test design could not discriminate: the GRPO
update condition (mixed-reward groups) and the ROAM mining trigger
(well-solved problems) never fired at G=4 / N_TRAIN=24 on MATH Level 5,
so neither arm trained and no comparison between GRPO and ROAM-GRPO
was measured. This is not a NEGATIVE — the claim was never fairly
tested, because no weight update of any kind happened — and not
BLOCKED, because the run completed (TRAIN_DONE) with measured numbers.
The one mechanism-relevant observation is negative evidence about the
test design, not the paper: at this scale the reward signal is
degenerate (bimodal per-problem pass rates), so a GRPO-vs-ROAM
comparison needs larger groups, more problems, and pilot-filtered
intermediate-difficulty problems.

## Run history today (all on cyc236ha, T4)
- Launch 1 (train-1122-roam): DIED ~10 min. Model PROBE_OK (Paris,
  7.12 GiB) but `load_dataset("hendrycks/competition_math")` raises
  DatasetNotFoundError — that dataset does not exist on the Hub.
  Fixed to EleutherAI/hendrycks_math (per-subject configs).
- Launch 2 (train-1122-roam-b): dataset fix worked (level-5 train=64
  eval=48), but Phase 1 ran at ~240 s per eval problem (8x384 tokens),
  so the v1 design (~3.2 h for Phase 1 alone) could not finish inside
  the ~4 h reclaim horizon. The local monitor was SIGTERM'd at ~60 min
  (VM replacement); the session survived, the driver was verified alive
  (pid 887, GPU 99%), but the run was killed as unfinishable. Lesson:
  size training drivers from MEASURED per-problem seconds, not from
  generation estimates.
- Launch 3 (train-1122-roam-c, v2 shrink: 24/20 problems, EVAL_G=4,
  MAX_NEW=256): DIED ~36 min at the first GRPO training forward —
  `RuntimeError: GET was unable to find an engine to execute this
  computation` in Qwen3.5's Gated Delta Net depthwise bf16 conv1d
  (causal_conv1d_fn) under gradient checkpointing: cuDNN has no
  backward-capable engine for that op on this T4 stack (no-grad
  generation works). Fix: `torch.backends.cudnn.enabled = False` in
  main(). Phase 1 metrics were lost with the VM.
- Launch 4 (train-1122-roam-d, v3 cuDNN fix): fix WORKED, run
  completed (TRAIN_DONE). Outcome as above. The local monitor was
  SIGTERM'd at ~75 min (VM replacement); the session survived and the
  watch was re-attached (--watch-only, pid 1065), which completed the
  run but did not download the metrics files (see Limits).
- No model fallback was triggered: Qwen3.5-9B went GREEN on probe in
  launch 1 and the failures were driver/design issues, not model
  blocks. A smaller model would face the same degenerate-reward issue
  (arguably worse — it solves even fewer problems), so the automatic
  fallback to Qwen3-8B was not invoked; the verdict is INCONCLUSIVE
  on the test design, not on the model.

## Limits — read before reusing
- The metrics CSV and summary JSON were lost: the `train` command's
  watch-only re-attach completed the run (outcome DONE) but reported
  `downloaded: []`, and the session was torn down before the files
  could be fetched manually. The numbers above come from the driver
  log tail preserved in `train_run_d.json`. For future runs: download
  metrics from /content at every phase boundary (or have the driver
  upload them), and do not rely on the watch-only path to download.
- The train/eval solve gap (0/24 vs 10/20) is unexplained. The driver
  does not save train generations or per-problem pass rates, so it
  cannot be diagnosed from this run. It may be a genuine split-
  difficulty gap, or a train-mode sampling degradation (peft train()
  + gradient checkpointing + use_cache=False vs eval() +
  use_cache=True). Treat any rerun's train pass rates with suspicion
  until the driver logs them.
- The paper's judge (LLM validity/distinctness) is replaced here by
  exact-answer match + TF-IDF cosine > 0.30 (a lexical proxy); the
  paper's diversity metric uses correct-solution embeddings, here
  TF-IDF cosine. Neither proxy was exercised, because no traces were
  mined.
- No KL term in the GRPO objective (documented simplification); G=4,
  single GRPO pass, N_TRAIN=24 — a directional mechanism test at
  best, not a reproduction of the paper's headline numbers.

## Resume state (for a rerun)
- Driver: `colab_roam_grpo.py` (v3: cuDNN fix + v2 sizes), CPU test
  52 checks pass. Session: train-1122-roam-d (torn down after DONE).
  Account for the day: cyc236ha. Next rotation account: kyu008008.
- Fixes needed before a rerun: (1) larger G (8–16) and more train
  problems (64+), with a pilot pass-rate filter to keep
  intermediate-difficulty problems (mixed-reward groups are the GRPO
  signal); (2) save train generations + per-problem pass rates to the
  metrics CSV so the train/eval gap is diagnosable; (3) download or
  upload metrics at every phase boundary; (4) consider a higher
  sampling temperature or a weaker base for the train split if the
  reward signal stays degenerate.
