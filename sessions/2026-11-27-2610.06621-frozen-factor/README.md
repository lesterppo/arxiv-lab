# Frozen Factor or Spectral Band? In LoRA, the Factor Matters More at Low Rank

- Article: https://arxiv.org/abs/2610.06621
- Track: training / fine-tuning (Colab T4)
- Date: 2026-11-27
- **Verdict: SUPPORTED** — the paper's headline claim reproduced
  directionally on a real downstream task at rank 2: which LoRA
  factor is frozen matters more than which spectral band it is
  frozen to, and the dominant band beats the minor band within
  either factor. The OBQA leg was not run (session reclaimed).

## Why picked (AGENT 1, TUNE 5)

A LoRA initialisation paper that is directly testable on the
daily 8B NF4-QLoRA harness: it asks whether freezing one LoRA
factor to a spectral band of the pretrained weight (dominant vs
minor singular directions) changes downstream accuracy, and
whether the factor (A vs B) matters more than the band. That
maps one-to-one onto the soup-daily-finetune / hermes-gi-egtkg
pattern (daily 8B LoRA on Colab T4, finance domain), so a
verified answer changes how those harnesses initialise LoRA.

## What was built

- `colab_frozenfactor_driver.py` — self-contained Colab driver:
  Qwen/Qwen3.5-9B-Base, transformers 5.20.2, NF4 QLoRA, bf16,
  AdamW8bit, lr 1e-4 constant (3-step warmup), 1 epoch/arm,
  MICRO_BATCH 2, MAX_LEN 384, gradient checkpointing, chunked
  cross-entropy (the fix for the 2026-11-26 leg-1 CUDA OOM,
  a full fp32 [B,T,V] logits blow-up). LoRA on attention
  projections only (q/k/v/o), so the band init is computed
  exactly via SVD of the dequantised weights. Init scheme, all
  with zero initial update: freezeA-dom A = Vh top-r frozen /
  B = 0 trainable; freezeA-minor A = Vh bottom-r; freezeB-dom
  B = U top-r / A = 0; freezeB-minor B = U bottom-r; standard
  arm Kaiming A / zero B, both trainable. lora_alpha = 2r.
  One base load, fresh LoRA per arm, `unload()` between arms.
- `test_frozenfactor_cpu.py` — CPU test, PASSED before launch
  (data gen, JSON scoring, SVD band-init assembly incl.
  orthonormality + zero initial update, arm table).
- Tasks: format task (synthetic finance-news -> exact JSON
  `{"ticker","sentiment","impact"}`, seeded RNG, 240 train /
  120 eval; metrics format-exact + content accuracy) and
  OpenBookQA (`allenai/openbookqa` "main", 256 train / 200 eval,
  0-shot logprob letter scoring).

## Results (live, Colab T4, account cyc236ha, session frozenfactor-1127)

Base probe: format_exact 0.0 / content_acc 0.0 (the base model
cannot do the format at all — it emits `<think>` traces and
wrong-schema JSON; good headroom), obqa_acc **0.91** (near
ceiling — the OBQA arms had at most +0.09 headroom).

Format task, content_acc (n = 120), train mean loss in brackets:

| arm | r | freeze | band | content_acc | format_exact | mean loss |
|---|---|---|---|---|---|---|
| standard | 2 | — | — | 0.7333 | 1.0 | 1.1740 |
| freezeA-dom | 2 | A | dominant | **0.7833** | 1.0 | 1.1100 |
| freezeA-minor | 2 | A | minor | 0.5833 | 1.0 | 1.7838 |
| freezeB-dom | 2 | B | dominant | 0.5167 | 0.8167 | 2.0348 |
| freezeB-minor | 2 | B | minor | 0.3583 | 0.725 | 2.0184 |
| freezeA-dom | 8 | A | dominant | 0.775 | 1.0 | 0.8168 |
| freezeB-dom | 8 | B | dominant | — (trained, loss 1.2486; eval unrun) | — | 1.2486 |

Contrasts at r = 2: factor gap at the dominant band
(freezeA-dom vs freezeB-dom) = **26.7 pts**; at the minor band
= 22.5 pts. Band gap within A = 20.0 pts; within B = 15.8 pts.
Both dimensions matter, the factor more — the paper's headline.
At r = 8, freezeA-dom (0.775) is essentially the same as at
r = 2 (0.7833): no big rank effect for freezeA-dom here.

## Honest limits

- **OBQA leg not run.** The Colab session was reclaimed at
  ~06:40 UTC (2h05m in), after arm 7 trained but before its
  eval. The OBQA r2 pair and OBQA r8 pair never ran; the
  90-min time gate had already dropped the OBQA r8 pair.
  `frozenfactor_results.json` / `frozenfactor_metrics.csv`
  were lost with the VM — the per-arm numbers above were read
  live from the driver log and are the record; per-step
  metrics are unrecoverable. No fallback run: the core test
  completed fairly, so the run is SUPPORTED, not BLOCKED (see
  the run log for the reasoning).
- **Attention-only LoRA, and only 8 layers.** Qwen3.5-9B-Base
  is a hybrid model (32 text layers: 8 full-attention + 24
  linear-attention). The SVD cache held exactly the 8
  full-attention layers' q/k/v/o (32 modules, exact SVD;
  `frozen_params_set=32` on every freeze arm confirms peft
  wrapped exactly those). The 24 linear-attention layers are
  not adapted in any arm. The factor x band contrast is clean
  and consistent across arms, but it is a statement about
  attention-projection LoRA, not about all linears.
- **format_exact saturates** (1.0 for every A arm and the
  standard arm), so content_acc is the discriminating metric.
  The freezeB arms are the only ones with format_exact < 1.0
  (they loop "News: ... JSON: ..." or emit a wrong schema).
- **Single seed, 1 epoch, lr fixed 1e-4**, synthetic format
  task (240 train), OBQA-200 proxy unused. The format task is
  a real downstream-style task, but the numbers are one seed
  on one model (Qwen3.5-9B-Base).
- **Eval slowness is diagnostic, not a bug:** freeze arms emit
  the JSON first and then a `<think>` trace, burning the full
  64-token cap on every prompt (no early EOS), so their evals
  took ~11 min vs ~3 min for the standard arm.

## Resume state (carried over)

- Session `frozenfactor-1127` (cyc236ha, T4
  gpu-t4-s-usw4a2-2idli3g0qxckz) reclaimed ~06:40 UTC; nothing
  to tear down. Driver + CPU test are in this session dir.
- To finish the OBQA leg: re-run
  `colab_frozenfactor_driver.py` with the OBQA arms only
  (obqa_r2_freezeA_dom, obqa_r2_freezeB_dom, and the r8 pair
  if budget allows) on the next rotation account; base OBQA is
  0.91, so expect a ceiling effect. Arm 7's eval
  (fmt_r8_freezeB_dom) can be re-run the same way.

## Repo fit

Training-track method for the daily 8B NF4-QLoRA harnesses
(soup-daily-finetune, hermes-gi-egtkg-finetune): freeze-A-to-
dominant-band is the init to try first at low rank; freeze-B
is the one to avoid.
