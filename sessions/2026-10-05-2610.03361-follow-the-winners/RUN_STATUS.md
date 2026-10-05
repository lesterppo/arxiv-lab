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

## To retry
Parent: review driver fix, ensure EXCLUSIVE Colab account access (no concurrent
switching), prefer off-peak. Recipe per arm: new session -> git-transformers deps
(see above) -> upload fixed driver -> FTW_MODE=<mode> nohup launch.
