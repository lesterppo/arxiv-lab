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

## Files

- `colab_ftw_grpo_9b.py` — driver (fixed, Colab-ready; retry recipe in docstring)
- `RUN_STATUS.md` — subagent run log with fixes and findings
- `README.md` — this file

## Honest limits

- Proxy task (synthetic arithmetic), not the paper's Sokoban/Search-R1.
- QLoRA r=16, not full fine-tuning; 12 planned steps (none executed).
- Base model, not instruct (Instruct repo doesn't exist).
- Binary reward; no KL-vs-reference term (paper's "conservative" element
  approximated by 1-epoch small-lr elite updates — deviation noted).
- Single planned seed; no repeats.
