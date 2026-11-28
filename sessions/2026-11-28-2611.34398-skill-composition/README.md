# When Skills Fail to Compose — training-track repro

- **Article:** arXiv [2611.34398](https://arxiv.org/abs/2611.34398) — "When Skills Fail to Compose: Compatibility and Retention in Language Models" (Yan Lin et al., submitted 2026-11-25)
- **Code (paper):** https://github.com/YanLin-Quinne/SkillCompatibility — **404s at run time** (repo not publicly reachable); task definitions below are taken from the paper text itself (App. A.2), not from the repo.
- **Date / run:** 2026-11-28, daily arXiv read-and-test, training track (TUNE 5)
- **Verdict: SUPPORTED** (see "Why SUPPORTED, not VERIFIED")

## Why picked + repo fit

Soft-prompt skill composition on frozen LMs, digit tasks, directly testable on a
free Colab T4 — a fine-tune method adaptable to the daily-LoRA harness
(`soup-daily-finetune`, `hermes-gi-egtkg-finetune`): if independently learned
skill blocks don't compose in one call, a skill library for self-evolving
agents must be evaluated for compatibility, not just per-skill proficiency.
The paper's claim is a clean, falsifiable number: one-call composition
accuracy at most 7.6% vs 88.2–100% over two calls.

## What was built

Soft-prompt (prompt-tuning) skills on frozen Qwen3 bases, following the
paper's App. A.2 recipe: each skill = 20 trainable embedding rows, all other
parameters frozen, cross-entropy, greedy decoding, exact match on the first
gold-length span, diagnostic subset where `x`, `A(x)`, `B(A(x))` are pairwise
distinct. Skills used are the paper's own worked-example pair:

- `[SHIFT_LEFT]` (SL): move first digit to end (`26647378` → `66473782`)
- `[SWAP_PAIRS]` (SWAP): exchange adjacent digits (`66473782` → `66747328`)

Conditions per arm (n=200 held-out length-8 strings each): single skill,
one-call composition (both 20-token blocks in a single call), two-call
composition (sequential calls with explicit intermediate output), both orders.
Partner-replacement check: an independently learned SWAP-B block substituted
for SWAP in the one-call composition.

Simplifications vs the paper (stated limits, not hidden): skills trained
alone rather than next to pretrained operations (no further-pretrained digit
base exists for Qwen3); 2,000–3,000 examples/skill × 2 epochs instead of
99,999 × 3 epochs; one seed per skill instead of three; 2 of the paper's 6
skills; length-8 only; Qwen3 bases (paper: Qwen2.5-0.5B/1.5B/7B, Gemma-3-1B).
Training loss is computed only on the 8 target positions (hidden states
sliced before `lm_head` — the 152k vocab makes full-sequence logits the
step-time bottleneck).

## Results

| Arm | Single SL | Single SWAP | One-call SL→SWAP | Two-call SL→SWAP | One-call SWAP→SL | Two-call SWAP→SL |
|---|---|---|---|---|---|---|
| Qwen3-0.6B (n=200) | 1.000 | 0.975 | **0.000** (A-present 0.170) | 0.965 | **0.000** (A-present 0.000) | 0.975 |
| Qwen3-4B (n=200) | 1.000 | 0.995 | **0.000** (A-present 0.045) | 1.000 | **0.000** (A-present 0.000) | 0.995 |

Paper for comparison: digit one-call 0.9–6.4%, two-call 88.2–100%.

Partner replacement (0.6B arm only): SWAP-B alone 0.970 (n=100);
one-call SL→SWAP-B 0.000 — the same collapse as with SWAP-A. (Uninformative
in one direction since the baseline is already 0.000, but consistent: an
independently learned block does not rescue the composition.)

The core claim reproduces on both completed arms, in both orders, with
decisive margins: individually proficient skills (≥97.5% alone) compose at
0.0% in one call and 96.5–100% over two calls.

## Why SUPPORTED, not VERIFIED

The numbers reproduce the claim end-to-end on two bases, but the test is
bounded: one skill pair, one seed, simplified training (see above), and the
mandatory 8–9B arm did **not** complete (BLOCKED on infra, below). VERIFIED
would overclaim; SUPPORTED states the bound honestly.

## The 8B arm — BLOCKED (infra, not a claim failure)

- Attempt 2 (`skillcomp-1128b`, cyc236es): 0.6B arm completed, then
  Qwen3-8B NF4 OOMed on the first training step. `py-spy` showed the process
  stuck in bitsandbytes `_dequant_linear_fallback` — the fused 4-bit kernel
  path was not taken on this Colab stack, so each step took minutes and
  memory blew past 14 GB. Driver killed by the operator; monitor outcome
  DIED; 0.6B results downloaded.
- Attempt 3 (`skillcomp-1128c`, kyu008008): fallback to Qwen3-4B bf16, which
  OOMed on the first backward for a different, fixable reason — the driver
  never froze the base (`from_pretrained` leaves `requires_grad=True`, so
  backward allocated `.grad` for all 4B weights, ~8 GB on top of the 8 GB
  weights). Fixed in `colab_skillcomp_4b_fix.py` (explicit freeze loop).
- Attempt 4 (`skillcomp-1128d`, kyu009009): Qwen3-4B bf16 with the freeze
  fix — VRAM held at 7.5 GB, full main eval completed (numbers above).
  Monitor outcome TIMEOUT (90-min budget) during the 4B replacement
  training; the 4B replacement check is therefore **not run** (0.6B
  replacement is done and recorded above).
- Attempt 1 (`skillcomp-1128`, ppoppo205) was killed ~10 min in for
  slowness (2.3 s/step with full-vocab loss) before any metric was written;
  the driver was then optimised (target-position loss slicing, fewer
  examples) and relaunched.

Resume state if the 8B arm is retried: use `colab_skillcomp.py` with the
freeze fix from `colab_skillcomp_4b_fix.py` applied to `load_model`, and
either fix the bitsandbytes kernel path (pin a known-good bnb/torch pair)
or run the 8B arm in bf16 with CPU offload — NF4 prompt-tuning backward on
this stack is not trusted. Trained 0.6B/4B blocks are banked in
`prompt_blocks.pt` / `prompt_blocks_4b.pt` (20×hidden fp32 tensors).

## Colab runs

| Attempt | Session | Account | Outcome | Note |
|---|---|---|---|---|
| 1 | skillcomp-1128 | ppoppo205 | killed by operator | slow driver (2.3 s/step), no metrics |
| 2 | skillcomp-1128b | cyc236es | DIED | 0.6B arm complete; 8B OOM (bnb fallback) |
| 3 | skillcomp-1128c | kyu008008 | DIED | 4B OOM (unfrozen base grads); driver bug, fixed |
| 4 | skillcomp-1128d | kyu009009 | TIMEOUT | 4B main eval complete; replacement cut off |

Rotation file ends 2026-11-28 on kyu009009. Full logs:
`~/workspace/goals/arxiv-daily-read-and-test/hidden_files/train-1128-skillcomp-launch{,2,3,4}.log`
and `train-1128-skillcomp-out{,2,3,4}/train-run.json`.

## Files

- `colab_skillcomp.py` — main driver (0.6B + 8B arms; 8B arm hit the bnb issue)
- `colab_skillcomp_4b.py` — 4B fallback driver (attempt 3, unfrozen-base bug)
- `colab_skillcomp_4b_fix.py` — 4B fallback driver with the freeze fix (attempt 4)
- `skillcomp_metrics.csv` — 0.6B metrics (10 rows)
- `skillcomp_metrics_4b.csv` — 4B metrics (8 rows)
- `prompt_blocks.pt` / `prompt_blocks_4b.pt` — trained soft-prompt blocks
  (local only, never pushed: binary artifacts)
- `results.json` — merged machine-readable summary
- `train-run-06b.json`, `train-run-4b.json`, `train-run-4b-attempt3.json` —
  wrapper outcomes (local only, not pushed: they embed full VM log tails)

## Honest limits

- One skill pair, one seed, simplified training; no statistical test across
  seeds (paper: 3 seeds, Holm-corrected pair tests).
- No 8B numbers: the claim is verified at 0.6B and 4B scale only.
- 4B replacement check not run (timeout); 0.6B replacement is a single
  observation.
- "A present" (first operation's effect visible in the one-call output) was
  low (0–17%): failures are not clean omissions of the second operation.
- No neutral-insertion training, repair, or retention/transfer tests from
  the paper's later sections — out of scope for this run.
