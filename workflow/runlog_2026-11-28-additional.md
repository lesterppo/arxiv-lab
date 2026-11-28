# Additional run — 2026-11-28 (second run of the day)

Operator: Muse (side chat), at Peter's request ("Add additional run by
yourself today", 2026-11-28 15:21 HKT). The first daily run (by the
daily-run agent) had already completed both tracks for 2026-11-28:
DART 2611.34928 (agent, NEGATIVE) and skill-composition 2611.34398
(training, SUPPORTED), pushed to origin/main at 393bc97. This run picks
two different papers; nothing already covered was re-tested.

## Picks

- Agent: arXiv 2611.34327 MeterRouter (AGENT 5, TUNE 2). Budget-aware
  incremental multi-agent routing; CPU-testable mechanism, quantitative
  claims. Near-misses: 2611.34751 Theory Worth (AGENT 4 — needs a live
  LLM solver; no NVIDIA API key in this VM and the NIM client from the
  DART run is gone from /tmp, so the live leg was not feasible today),
  2611.34762 TASK Memory (AGENT 4 — needs activation steering on a real
  model), 2611.34112 (review, AGENT 3, not testable).
- Training: arXiv 2611.33972 WSP, closed-form fine-tuning on a frozen
  RoBERTa-large encoder (AGENT 1, TUNE 4). Near-misses: 2611.34275
  LatentCP-DPO (implementation risk), 2611.34707 bandits (complex
  proxy/target setup), 2611.34635 ZeroHero (environments too heavy).

## Agent track — MeterRouter — DONE, verdict SUPPORTED

- Module `src/arxiv_lab/agent/meter_router.py`, CPU test
  `tests/test_meterrouter_cpu.py` (E1–E8, all pass, n = 500).
- At B = $50×10⁻³: router acc 0.782 vs strongest MAS 0.798 at
  $6.24×10⁻³ vs $10.74×10⁻³ per query; overrun rate 0.000 at all
  budgets; random-tier ablation loses 6.6–21.0 pts; hard-query budget
  gain +0.242 vs easy +0.209.
- Rust second stage in the session `rust/`: `cargo test --release`
  2/2 passed (parity 1140 cases, 0 mismatches); `cargo run --release
  --bin verify` all checks passed (router 0.764 vs strongest 0.780).
- Session: `sessions/2026-11-28-2611.34327-meterrouter/`.

## Training track — WSP — DONE, verdict PARTIAL

- CPU test `tests/test_wsp_cpu.py` passes (all checks).
- Driver `sessions/2026-11-28-2611.33972-wsp/colab_wsp_driver.py`.
- Colab: account cyc236es only (rotation pool tokens do not exist in
  this VM; deviation recorded in the session README).
- Attempt 1, session `wsp-1128`: PROVISION_FAILED (timeout); datasets≥3
  no longer resolves legacy script ids and the ag_news mirror schema
  differed — fixed in the driver; artifacts kept as
  `wsp_results_attempt1.json` / `wsp_metrics_attempt1.csv` /
  `train-run-1128a.json`.
- Attempt 2, session `wsp-1128b`, T4, pid 7010: DONE in 31.2 min.
  First monitor killed by a runtime restart 35 s after launch;
  re-attached with `train --watch-only --pid 7010` (watch-only still
  requires `--driver`, despite WORKFLOW.md §4.5's wording).
- Results (eval acc, SST-2 n=872, AG News n=1000): WSP 0.8658 / 0.9094
  / 0.9140 / 0.9232 (SST-2) and 0.8530 / 0.8550 / 0.8510 / 0.8910
  (AG News) at budgets 16/64/128/512 — 1.2–1.3 pts below the paper at
  512 labels; cross-fitting picks the RBF kernel head in 7 of 8 cells.
- FT/LoRA arms collapsed (SST-2 ~0.51, AG News ~0.24, i.e. majority
  rate) because `train_classifier` loads the classifier in pure bf16
  with a raw AdamW loop and no fp32 master weights, so lr-1e-5 updates
  round to zero. One non-collapsed trained cell: AG News LoRA 0.878 at
  512 vs WSP 0.891. The 48–53 pt "lead" over FT/LoRA measures the
  broken baseline, not the paper — hence PARTIAL, not SUPPORTED.
- Resume state: fix `train_classifier` to fp32 master weights (or
  autocast with fp32 optimizer states) and re-run the FT/LoRA arms
  only; WSP numbers are banked. The one allowed fallback was not
  spent on this (the run ended PARTIAL, not BLOCKED).

## Infra notes (worth keeping for the next run)

- The hermes-colab-cli wrapper must run with `PYTHONNOUSERSITE=0`
  (the agent shell hides user-site `colab_cli`).
- google-colab-cli 0.8.0 is unpatched for HTTP 411 on /tun/m/assign;
  `patch_colab_cli.py` aborts on it (execution.py has 2 prune blocks,
  not 3 — version drift). The 411 fix was applied manually to
  `~/.local/lib/python3.12/site-packages/colab_cli/client.py`
  (backup `client.py.bak`): `data="{}"` on the two POST calls in
  `_post_assignment` and `unassign`. The full patch (incl. 404/401
  token-refresh) is still NOT applied on 0.8.0.
- One creation attempt produced a phantom session (kernel never
  answered); stopped, retry succeeded.

## Follow-up (same day) — WSP FT/LoRA arms re-run with fp32 fix

- Fix (one change in `train_classifier`): classifier loaded in
  fp32 instead of pure bf16; nothing else changed. New env gate
  `WSP_ONLY_ARMS=ft,lora` so only the 16 gradient arms ran.
- Colab session `wsp-1128c`, account cyc236es, T4, pid 755,
  endpoint gpu-t4-s-euw4c1-1gsp0hpgoxolb, DONE in 24.3 min
  (driver elapsed). No provision failure this time; the 411 fix
  and PYTHONNOUSERSITE=0 from the first run still applied.
- Results (merged into `wsp_results.json`; WSP arms verified
  byte-identical to the pre-follow-up backup): FT at 512 labels
  now reproduces the paper (SST-2 0.9232 vs paper FT 0.9285;
  AG News 0.905 vs 0.9011) — the original collapse was the bf16
  bug, confirmed fixed. But FT/LoRA still collapse at 16/64
  total labels (8–24 optimizer steps at lr 1e-5 cannot move a
  355M model), and SST-2 LoRA b512 stays at 0.5837 with a flat
  loss while AG LoRA b512 reaches 0.886 (same split as attempt
  2; likely cause: peft default freezes the classifier head).
  At 512 labels WSP vs FT is a tie on SST-2 and −1.4 pts on
  AG News — near-parity as the paper says.
- Verdict stays PARTIAL: the low-budget lead (the paper's
  discriminating claim) is still measured against collapsed
  baselines. Per the one-fix rule, no further tuning was done.
  A fair low-budget test would need more steps at tiny budgets
  and a LoRA setup that trains the classifier head — a new run.
- Files: `wsp_results_followup.json`,
  `wsp_metrics_followup.csv`, `train-run-followup.json`,
  `wsp_results_pre_followup.json` (backup),
  `wsp_metrics_pre_followup.csv` (backup),
  `registry_followup.json`. NOT committed/pushed (parent's step).

## Repo update — DONE, verified

- No git login in this VM, so the push went through the GitHub git
  Data API (`workspace/tools/push_arxiv_lab.py`, custom.github
  credential surrogate): 2 commits, 25 files, main 393bc97 ->
  c797f8f (daily-run agent's 3 train-run records, 07:11–07:12 UTC) ->
  7affd8a (DART README fix) -> 4bd6cfd (additional run).
- Local clone re-synced by rebuilding the 5 commits with git plumbing
  (`workspace/tools/sync_arxiv_lab_local.py`); every rebuilt SHA
  matches remote. Local main = origin/main = 4bd6cfd.
- `workflow/verify_push.py` on the pushed paths: secrets check clean,
  all 25 files match remote blobs (only diffs are rust/target/
  build artefacts, deliberately not pushed).
- Repo-wide `verify_push.py` still flags pre-existing false positives
  in old files (e.g. `workflow/covered_ids.py`,
  `sessions/2026-10-16-2610.22293-sst/drive_ckpt.py`,
  `mint_drive_token.py`); none are from this run.

## Follow-up push (same day, ~17:00 HKT)

- WSP follow-up files pushed via the GitHub API (same route as above):
  modified driver/README/results.json/run-log, new
  wsp_results_followup.json, wsp_metrics_followup.csv,
  train-run-followup.json, wsp_results_pre_followup.json,
  wsp_metrics_pre_followup.csv, registry_followup.json.
- Local clone re-synced with sync_arxiv_lab_local.py (SHA-verified
  rebuild); local main = origin/main after the push.
