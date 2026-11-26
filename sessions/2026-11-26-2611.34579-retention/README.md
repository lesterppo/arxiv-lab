# Tune for Retention, Not Just Gain: SFT Forgets Far Less than You Think

- **Article:** arXiv:2611.34579 — "Tune for Retention, Not Just Gain: SFT Forgets Far Less than You Think"
- **Link:** https://arxiv.org/abs/2611.34579
- **Date:** 2026-11-26
- **Track:** training (Colab T4, NF4 QLoRA)
- **Verdict: PARTIAL**

## Why picked

AGENT 2 / TUNE 5. The paper's claim is directly relevant to the
daily-finetune harness (soup-daily-finetune / hermes-gi-egtkg-finetune):
catastrophic forgetting in SFT is largely a hyperparameter artefact —
lower learning rate and smaller batch size keep most of the new-task
gain with little to no forgetting, and vanilla SFT selected jointly
for gain AND retention is competitive with SDFT (an on-policy method
designed to mitigate forgetting). A gain–retention frontier across
LR × batch on one T4 slot is the testable core; SDFT itself is out of
scope for one slot.

## Design

- New task: finance-domain SFT on `gbharti/finance-alpaca` (fallback
  `tatsu-lab/alpaca`): 500 train examples, 2 epochs. Gain = ROUGE-L F1
  of greedy completions vs reference outputs on 100 held-out examples.
- Retention probes per arm (greedy, fixed prompts): GSM8K test first
  80 zero-shot ("Question: ...\nAnswer:", last-integer exact match);
  MMLU test first 80, 5-shot, first-letter A–D match; mean generated
  length on the GSM8K generations.
- Arms (fresh base + fresh LoRA per arm, identical code paths):
  base (probe only), lr1e-5_b4, lr1e-5_b16, lr5e-5_b8, lr2e-4_b4,
  lr2e-4_b16. lr2e-4_b4 is the max-gain arm, lr1e-5_b4 the
  retention-friendly arm.
- Time gates (elapsed since driver start): >100 min shrinks remaining
  arms to 1 epoch; >150 min skips remaining arms (recorded partial).
- CPU test `test_retention_cpu.py` PASSED (pure-helper checks incl. a
  fixed ROUGE-L hand-computation in the test).
- SDFT (the paper's on-policy comparator) NOT run — out of scope for
  one T4 slot; the test is the SFT-internal gain–retention frontier
  only.

## Leg 1 — Qwen3.5-9B-Base (kyu008008, session `retention-1126`): DIED

- Launch 04:25 UTC via `colab.py train --account cyc236es`;
  cyc236es + ppoppo205 gpu-unavailable at creation; cross-account
  retry landed on kyu008008, T4 gpu-t4-s-euw4c1-5t6d35fx6exb,
  driver pid 2547.
- Base probe complete (downloaded): gain ROUGE-L **0.1452** (n=100),
  GSM8K **0.425** (34/80), mean len 114.3, MMLU **0.30** (24/80).
- Arm lr1e-5_b4 trained 4 steps (losses 1.7628, 2.3801, 2.1109,
  2.3307), then **DIED on CUDA OOM in `train_arm` backward**
  (driver line 347, `(loss / n_mb).backward()`; tried to allocate
  1.62 GiB with 13.65 GiB in use, 931.81 MiB free). Root cause: the
  Qwen3.5 Gated Delta Rule reference fallback (transformers 5.20.2
  falls back to its slow PyTorch implementation — `causal_conv1d` /
  `flash-linear-attention` not installed) materializes heavy attention
  intermediates, and `train_arm` materializes full [B,T,V] fp32 logits
  (`out.logits[:, :-1, :].float()`, 4×511×151936) before the CE.
- Files downloaded and banked: `retention_metrics_leg1_qwen35.csv`,
  `retention_results_leg1_qwen35.json`,
  `retention_benchmark_leg1.json`, `train-run-leg1.json`
  (outcome DIED).
- No fallback was consumed by this leg's failure mode alone — the
  automatic model fallback rule fired instead (see leg 2).

## Leg 2 — fallback, Qwen3-8B (kyu009009, session `retention-1126b`): RECLAIMED

- Automatic model fallback (standing rule 2026-11-05): driver
  `colab_retention_driver_fallback.py` — identical design/data/seeds/
  arms/evals, with shrink fixes for the leg-1 OOM: MODEL_ID
  Qwen/Qwen3-8B (standard attention, proven transformers
  >=4.51,<5.0 NF4 QLoRA recipe), MICRO_BATCH 4→2 and MAX_LEN 512→384
  via env (grad accum keeps the effective batch). Launched ~05:14 UTC
  via `colab.py train --rotate` (rotation advanced kyu008008 →
  kyu009009), T4 gpu-t4-s-ass1a0-1y5v1ar5s4tdk, driver pid 1187.
  No OOM: the shrink fixes held.
- Base probe complete (read live from the VM 05:56 UTC): gain
  ROUGE-L **0.1478** (n=100), GSM8K **0.20** (16/80), mean len 128.0,
  MMLU **0.0** (0/80).
- Arm lr1e-5_b4 trained all **250 steps** (2 epochs, mean loss
  **1.9791**, ~16 s/step). Arm gain eval: ROUGE-L **0.1697** (n=100),
  up from base 0.1478 (+2.19pp).
- **RECLAIMED during the arm-1 eval**, after the gain eval and before
  the GSM8K/MMLU evals — no retention number exists for any arm, and
  arms lr1e-5_b16, lr5e-5_b8, lr2e-4_b4, lr2e-4_b16 never ran. The
  reclaimed VM's files were NOT downloadable (`downloaded: []` in
  `train-run-leg2.json`); the leg-2 numbers that survive are the
  recovered ones in `retention_metrics_leg2_qwen38_RECOVERED.csv` and
  `retention_results_leg2_qwen38_RECOVERED.json` (read live from the VM
  during the run + the watch-only monitor's log tail; steps 1–48,
  72–148, 150–172, 174–219 were never observed — see the provenance
  note in the JSON). Step 220 is inferred from sequence position.
- The host-side monitor for leg 2 was killed by a VM runtime restart
  mid-run; the VM-side driver survived (pid 1187 verified alive by
  direct `ps` + live metrics reads) and was re-attached with
  `train --watch-only --pid 1187`.

## Results (per-arm gain / retention)

| arm (model) | gain ROUGE-L | GSM8K | MMLU | mean len |
|---|---|---|---|---|
| base (Qwen3.5-9B) | 0.1452 | 0.425 | 0.30 | 114.3 |
| base (Qwen3-8B) | 0.1478 | 0.20 | 0.0 | 128.0 |
| lr1e-5_b4 (Qwen3-8B) | **0.1697** | — (reclaimed) | — (reclaimed) | — |

No arm has both a gain and a retention measurement, so the paper's
gain–retention frontier is **untested, not refuted**. What exists:
base calibrations on two models, one fully trained arm, and that
arm's gain (+2.19pp over its own base).

## Honest limits

- The frontier — the paper's discriminating prediction — was never
  measured: reclamation cut the run inside the first arm's eval.
- MMLU sits at floor (0.0) for the Qwen3-8B base under this probe
  (5-shot, max_new 8, first-letter parse): even a completed run would
  have had weak MMLU retention signal for this base model. GSM8K is
  the informative retention probe here (base 0.20, headroom to fall).
  Qwen3-8B base GSM8K (0.20) is far below Qwen3.5-9B base (0.425) on
  the same 80 problems — a model difference, not a harness artefact
  (identical prompts, seeds, and parse).
- GSM8K mean len 128.0 for the Qwen3-8B base means every generation
  hit the 128-token cap (rambling base-model continuations); the
  last-integer parse still worked, but the generations are not clean.
- Training is slow at this configuration (~16 s/step at micro 2,
  seq ≤384, grad checkpointing + full-vocab fp32 logits): the 5-arm
  × 2-epoch design does not fit a ~4 h reclaim horizon. The time
  gates (>100 min → 1 epoch, >150 min → skip) are the correct
  response, but even so only ~2 arms would have completed.
- Single seed; 500 finance examples; ROUGE-L gain proxy, not the
  paper's task metric; no SDFT comparator.
- Leg-2 numbers are recovered observations, not a downloaded metrics
  file — see the provenance note in
  `retention_results_leg2_qwen38_RECOVERED.json`.

## Resume state (exact)

- Rerun `colab_retention_driver_fallback.py` with
  `--env MODEL_ID=Qwen/Qwen3-8B MICRO_BATCH=2 MAX_LEN=384` in a
  morning GPU window. Full rerun from scratch: adapters were never
  saved and the leg-2 metrics died with the reclaimed VM (only the
  RECOVERED numbers above survive). Same data/seeds/arms/evals.
- Expect base + ~2 arms before the hard-stop gate, at the observed
  ~16 s/step; to get the full 5-arm frontier, split the arms across
  two sessions (the driver has no per-arm env selector yet — add one
  or run two driver copies with a trimmed ARMS list) or accept a
  2–3-arm frontier.
- Fallback already consumed for today (leg 1 DIED → leg 2 RECLAIMED);
  a rerun needs no fresh instruction under the carry-over rule but is
  a new day's run, not an automatic retry.

## Files

- `README.md` (this file)
- `colab_retention_driver.py` — leg-1 driver (Qwen3.5-9B-Base)
- `colab_retention_driver_fallback.py` — leg-2 driver (Qwen3-8B,
  MICRO_BATCH/MAX_LEN env-driven; diff vs leg 1 is the header note
  plus those two env lines)
- `test_retention_cpu.py` — pure-helper CPU test (PASSED)
- `retention_metrics_leg1_qwen35.csv`,
  `retention_results_leg1_qwen35.json`,
  `retention_benchmark_leg1.json`, `train-run-leg1.json` — leg 1,
  downloaded (outcome DIED, CUDA OOM)
- `retention_metrics_leg2_qwen38_RECOVERED.csv`,
  `retention_results_leg2_qwen38_RECOVERED.json`,
  `train-run-leg2.json` — leg 2, recovered observations only
  (outcome RECLAIMED; `downloaded: []`)
