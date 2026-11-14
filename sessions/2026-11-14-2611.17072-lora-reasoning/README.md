# 2611.17072 — Reasoning Supervision and Capability Retention in LoRA Domain Adaptation

- **Article:** https://arxiv.org/abs/2611.17072 (Ashkboos, Desai, Ghita; submitted 2026-11-12)
- **Date tested:** 2026-11-14
- **Verdict: SUPPORTED**

## Why picked + repo fit

The paper fine-tunes Qwen3-8B with LoRA on financial data and compares three
regimes: (1) response-only LoRA (label alone in the loss), (2) single-adapter
reasoning (teacher traces + labels in the loss), (3) dual-adapter composite
(separate adapters on traces and labels). Headline claims: response-only
improves in-domain and leaves general capabilities near frozen; single-adapter
reasoning stays behind response-only in-domain while MMLU-Pro can rise slightly
and other retention benchmarks stay within ordinary variation; dual-adapter
gives no further gain; shorter (summarized) traces move calibration toward
response-only without reaching it.

This is the daily training track's home turf (`soup-daily-finetune`): the
paper's exact model (Qwen3-8B) under the proven NF4-QLoRA recipe, on a finance
task (Banking77 intent classification), with a retention check. The bounded
test covers claims (1)+(2) — the paper's core comparison.

## What was built

- `colab_lora_reasoning_driver_v4.py` — self-contained Colab driver (v4 is the
  run that produced the verdict; v1–v3 are the earlier attempts, see history).
  Pipeline: base eval → trace generation (or reuse) → train arm R
  (response-only) → eval R → rebuild fresh model → train arm T
  (trace+label, single adapter) → eval T. Prints `TRAIN_DONE` on completion.
- `test_lora_reasoning_cpu_v4.py` — 42 CPU checks, all passing
  (prompt construction, trace parsing, `parse_pred`, ECE, macro-F1, arm
  construction, trace reuse).
- `colab_out_v2/traces.jsonl` — 193 teacher traces (320 attempted, 60.3%
  yield), md5 `38b76d2e3641e28a0294c3c0f8b9ed02`, reused by v4.
- `colab_out_v4/lora_reasoning_metrics_v4.csv` /
  `lora_reasoning_results_v4.json` — v4 numbers, **reconstructed from the
  driver's own printed log lines** in `colab_out_v4/train-run.json`
  (the session was reclaimed before the driver wrote its metrics files;
  no number was invented).

## Experiment

- **Task:** Banking77 (PolyAI/banking77, 77-way finance intent
  classification; train 10,003 / test 3,080). Prompt identical for train +
  eval: full 77-label list + message + `Category:`.
- **Arms (matched):** same 128 items, same N, 1 epoch, lr 2e-4, micro-batch 2
  × accum 2 (eff. batch 4), fresh model rebuild between arms.
  - **R (response-only):** completion = ` <label>`.
  - **T (single-adapter reasoning):** completion =
    ` <short trace>\nCategory: <label>`. Teacher = the **base model itself**
    (Qwen3-8B, few-shot greedy, kept only when the trace's final Category
    equals the gold label) — NOT an external larger teacher as in the paper;
    recorded as a limitation. Trace completion mean 111 chars, max 342.
- **Eval:** Banking77 held-out test n=200 (greedy; accuracy, parse-fail,
  macro-F1, ECE from first-generated-token probability, 5 bins) + retention:
  strided MMLU validation subset n=100, logprob-scored A/B/C/D. Eval
  generation cap 96 tokens with Category-line parsing (v3 fix), applied
  identically to all arms.
- **Not tested:** dual-adapter composite (claim 3) — composition rule
  unspecified in the abstract, full text unavailable (no HTML version yet;
  ar5iv redirects to the abs page as of 2026-11-14). Trace summarization
  (claim 4) — not tested. Both recorded, not silently dropped.

## Results (v4, fair eval — the verdict numbers)

| arm | Banking77 acc (n=200) | parse_fail | macro-F1 | ECE | MMLU (n=100) |
|-----|----------------------|------------|----------|-----|--------------|
| base | 0.6650 (133/200) | 0.0350 | 0.6170 | 0.1648 | 0.7700 |
| R response-only | **0.7500 (150/200)** | 0.0650 | 0.7278 | 0.2206 | 0.7400 |
| T reasoning | 0.6700 (134/200) | 0.0850 | 0.6038 | 0.3281 | **missing** (see below) |

- **R beats T in-domain by 8.0 pp** (150 vs 134 of 200), beyond the ~±3–4 pp
  noise at n=200. T is only +0.5 pp over base — essentially no in-domain gain
  from reasoning supervision at this scale.
- The ordering holds excluding parse-fails: R 150/187 = 0.8021 vs
  T 134/183 = 0.7322.
- **Retention flat:** base MMLU 0.7700, R 0.7400 (−3 pp), T 0.7500 (−2 pp).
  The T figure is v2's valid measurement of the identical recipe (same 193
  traces, same 128 items, same hyperparams); v4's T MMLU was lost when the
  session was reclaimed minutes before it ran.
- **Calibration diverges from the paper:** ECE *worsened* for both arms
  (R 0.2206, T 0.3281 vs base 0.1648) — the reasoning arm is the most
  miscalibrated. Caveat: first-token ECE measures label-token confidence for
  R/base but reasoning-onset confidence for T, so the T ECE is not
  apples-to-apples.
- Training: R 32 steps / 1337 s / peak VRAM 8.64 GiB; T 32 steps / 1361 s /
  peak VRAM 8.65 GiB. Step-25 losses: R 0.7235, T 0.3919 (per-batch, not
  averages). T4 forward rate only ~80–120 tok/s on this stack; base eval
  (200 items, 96-token cap, batch 8) took 33.5 min.

## Verdict: SUPPORTED

Claims (1)+(2) are reproduced directionally at this scale under a fair eval:
response-only improves in-domain (+8.5 pp over base) while retention stays
within ordinary variation, and single-adapter reasoning stays clearly behind
response-only in-domain (8.0 pp) with retention also flat. Not VERIFIED:
small scale (N=128, 1 epoch, eval n=200), MMLU validation subset instead of
MMLU-Pro, self-teacher instead of an external larger teacher, T's MMLU cell
filled from v2 rather than v4, and the calibration direction diverges from the
paper's claim. Claims (3) dual-adapter and (4) trace summarization untested.

## Model — why Qwen3-8B, probe numbers

The paper's exact model. Qwen3-8B under NF4 QLoRA (r=16/alpha=32, bf16,
transformers<5.0, grad checkpointing, target q,k,v,o,gate,up,down;
43,646,976 trainable params, 0.5301% of 8,234,382,336). Proven T4 QLoRA
carrier in this repo (ToolSelect verify3/verify5, AOPD run f, MCPO arm 1).
Qwen3.5-9B probe GREEN 2026-10-30 but its full trainer OOM'd, so 8B is the
proven carrier; no 9B arm is required — the claim is about Qwen3-8B.
Budget pressure was absorbed by shrinking the experiment (N=128, 1 epoch,
eval n=200), never the model.

## Run history

- **v1** (`lora-reasoning-1114`, kyu009009, PID 730) — killed by operator
  05:14 UTC in trace generation after timing diagnosis (320×2-epoch arms
  would need ~2 h each). Base numbers preserved in
  `colab_out/lora_reasoning_metrics_v1.csv`: Banking77 189/308 = 0.6136
  (parse_fail 0.0617, macro-F1 0.5982, ECE 0.2075), MMLU 0.7700.
- **v2** (`lora-reasoning-1114b`, kyu009009, PID 4975) — DONE 08:04 UTC.
  N=128, 1 epoch, eval n=200. Trace gen: 320 attempted, 193 kept (60.3%).
  Base 0.6550 / R 0.7100 / T 0.0850 with 89% parse-fail — the T in-domain
  number is **DISCARDED as a measurement artifact** (16-token eval cap
  truncated the reasoning arm before its Category line + first-line-only
  parsing). Valid v2 numbers kept: base/R in-domain and all three MMLU
  (base 0.7700 / R 0.7600 / T 0.7500).
- **v3** (`lora-reasoning-1114c`, kyu009009, PID 391) — DIED ~08:19 UTC,
  CUDA OOM in `sdpa_attention_forward` during base eval (13.33 GiB
  allocated, 180 MiB alloc failed). Cause: the v3 eval fix (96-token
  generations with `output_scores=True` at batch 24) stores ~1.4 GB of
  score tensors + ~1.9 GB attention matrix on top of an already-near-edge
  eval. No metrics existed; only the uploaded traces copy was downloaded.
- **v4** (`lora-reasoning-1114d`, kyu009009, PID 1652, T4
  gpu-t4-s-usw1b2-cctmdupilnh2) — the one automatic retry the standing
  fallback rule allows after a BLOCKED run. Fix: Banking77 eval batch
  24→8 (memory stayed ~9.5 GiB, no OOM); **no model change** — v2 proved
  Qwen3-8B fits end-to-end and the OOM cause was eval batching, not the
  model. CPU checks 42/42. Base eval 08:24:45–08:58:19, GEN reuse confirmed
  09:00:38 (193 traces, md5-verified), R trained 09:00:38–09:22:55, R eval
  09:34:47/09:37:11, T trained 09:38:28–10:01:10, **T Banking77 landed
  10:19:56 — then the session was RECLAIMED ~10:20 UTC**, before T MMLU
  and before the driver wrote its metrics files (train-run.json outcome
  RECLAIMED, downloaded []). The headline comparison survived in the
  driver's log lines; v4 metrics were reconstructed from them.
  Three runtime restarts during the day killed monitor shells; the driver
  survived detached each time and was re-attached with
  `train --watch-only --pid <pid>`.

## Resume state

No further automatic attempt: the one-retry rule is exhausted (v3 BLOCKED →
v4 RECLAIMED). The v4 T adapter weights died with the VM (adapters were
never downloaded) and v4's T MMLU was never measured. A resume would need a
fresh session and a retrain of arm T from scratch (~27 min train + ~3 min
MMLU, or a full rerun for a clean DONE) — **not authorized**; it needs
Peter's fresh instruction. Rotation state: 2026-11-14 kyu009009 (next:
kyu009009 → cyc236ha in pool order ppoppo205 → cyc236es → kyu008008 →
kyu009009 → cyc236ha). Active account restored to kyu008008 after the run.

## Limits (honest)

- Abstract-only paper (full text unavailable); claims 3 and 4 untested.
- Self-teacher (base Qwen3-8B), not an external larger teacher; traces are
  short (mean 111 chars) and rejection-sampled on the gold label, so they
  carry less signal than the paper's teacher traces.
- Small scale: N=128, 1 epoch, eval n=200, MMLU n=100 (validation subset,
  not MMLU-Pro).
- v4 ended RECLAIMED, not DONE; T MMLU is v2's measurement of the identical
  recipe, not v4's own.
- ECE is first-token ECE; for T it measures reasoning-onset confidence,
  not label confidence.
- Single seed, no confidence intervals beyond the n=200 sampling noise
  (~±3–4 pp).
