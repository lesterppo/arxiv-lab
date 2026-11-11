# Margin-Guided Token Credit for RL with Verifiable Rewards (MCPO)

- **Article:** arXiv [2611.14015](https://arxiv.org/abs/2611.14015) — "Margin-Guided Token Credit for Reinforcement Learning with Verifiable Rewards" (submitted 10 Nov 2026)
- **Verdict: BLOCKED** (no fair test of the claim was possible — see "Why BLOCKED")
- **Track:** training track — Google Colab T4 via `hermes-colab-cli` (`colab.py train`)

## Why picked

- **AGENT 2 / TUNE 5.** A minimal, elegant RL mechanism: in GRPO every token of a solution gets the same credit, so MCPO places the credit where the model hesitated — the top-2 margin M = p1 − p2 of the next-token distribution, mapped to a score with two fixed thresholds, spread so each solution's advantage keeps its mean — at no extra model-evaluation cost (the margin comes from a forward pass the trainer already performs). The paper reports +6.55 macro avg@32 over GRPO (MiniCPM5-2B-Base + LoRA, 200 steps on 2,048 DAPO-Math problems) and +4.75 on a Qwen3-4B replication.
- **Repo fit:** directly adaptable to the `soup-daily-finetune` daily-QLoRA harness — a drop-in change to GRPO's credit assignment, no new infrastructure.
- **Testable within budget:** one paired comparison (GRPO vs MCPO, same prompts, same steps, same code path) on a Colab T4.

## What was built

- `colab_mcpo_driver.py` — the Qwen3-8B driver (NF4 QLoRA r=16/a=32, bf16; DAPO-Math-17k train pool 512 / held-out eval 80; paired prompts; 24 steps/arm, 4 prompts/step, group 6, max_new 320, lr 1e-5; base eval first; both arms in one driver with identical code paths — the only difference is the token-credit weighting). Pure helpers (margin weighting, advantage spreading, boxed-answer extraction) were unit-tested locally before launch.
- `colab_mcpo_driver_4b.py` — the fallback driver (Qwen3-4B + the chunked-loss fix; see "Run history").
- `recovered_metrics.csv` — all measured numbers that survived (see "Results").
- `train-run-mcpo-1111d.json` — the wrapper's final outcome record for the fallback run (RECLAIMED).

## Run history

1. **Launch 1 (mcpo-1111)** DIED ~2 min: driver bug, `get_device_properties(0).total` (the attribute is `total_memory`) — the same one-line bug as the 2026-11-01 SCARL run. Fixed; no fallback triggered (driver bug, not a model failure).
2. **Launch 2 (mcpo-1111b)** DIED ~20 min, killed by the agent: **data bug** — the driver extracted `\boxed{}` "golds" from the prompt text. The parquet's real schema is `prompt=[{content}]` / `reward_model={ground_truth}` (1,791,700 rows, not 17k); only 2,500/1.79M rows had a boxed expression in the problem statement, so rewards would have been garbage. Killed mid-base-eval (model loaded, peak 6.63 GiB). Fix validated locally against the real parquet (299 MB): prompt parsing (list/ndarray of message dicts), ground-truth extraction, `extract_answer` (boxed OR the prompt's own "Answer:" line), fraction-aware answer comparison. Local validation: 19,982/20,000 sampled rows usable, all numeric golds.
3. **Launch 3 (mcpo-1111c, Qwen3-8B)** DIED (OOM): base eval completed — **Qwen3-8B zero-shot 1/80 (0.0125)**, mean completion 215.9 words — then CUDA OOM at GRPO step 1, driver line 366: `log_softmax` over the full `[B, T, V]` fp32 logits (V=151,936; 11.15 GiB allocated, 1.74 GiB alloc failed). The local `train` monitor was SIGTERM'd by the runtime background limit after ~34 min; the VM driver was already dead (PID 786 zombie, GPU idle). Session stopped.
4. **Launch 4 — automatic fallback (mcpo-1111d, Qwen3-4B, cyc236es)**: ppoppo205 was GPU-unavailable at creation, so the wrapper fell back to cyc236es. The chunked-loss fix worked (peak 7.06 GiB, no fatal OOM; one transient allocation-failure warning at 07:17 was recovered from). The local monitor was SIGTERM'd after ~1.9h; the VM driver survived and monitoring was re-attached with `train --watch-only --pid 723` (note: `--driver` is still required in watch-only mode). The VM was **RECLAIMED** during GRPO step 22/24. Fallback spent — no further automatic attempts today; a rerun needs Peter's fresh instruction.

## Results

| measurement | value |
|---|---|
| Qwen3-8B base, held-out DAPO n=80 | 0.0125 (1/80) |
| Qwen3-4B base, held-out DAPO n=80 | 0.0 (0/80) |
| GRPO train reward, steps 1–22 (4B) | 0.000 on 18/22 steps; 0.042 on steps 9, 12, 21; 0.125 on step 13 |
| Correct rollouts in GRPO training | 6 of 528 |
| GRPO eval / MCPO arm | never ran (reclaimed at step 22/24) |
| Step time (4B, T4) | ~300 s/step (~2 h per 24-step arm) |
| Peak VRAM (4B, chunked loss) | 7.06 GiB |

Full per-step rows: `recovered_metrics.csv` (steps 20–22 reconstructed from the watch monitor's log tail; the rest downloaded mid-run before reclamation).

## Why BLOCKED

- The MCPO-vs-GRPO comparison never ran: the GRPO arm reached 22/24 steps with no eval, and the MCPO arm never started, because the VM was reclaimed. No fair test of the paper's claim was possible.
- Separately, the partial data shows the test design itself has a problem that a same-design retry would hit again: DAPO competition problems are near-unsolvable zero-shot for a 4B (0/80) and 8B (1/80) base model, so the group-relative advantage is ~0 on most steps (6 correct rollouts in 528) and the credit-assignment comparison has almost no signal to discriminate on. This is recorded here as a design finding, not as evidence about MCPO — the paper's +6.55 came after 200 steps on a model/problem mix that evidently produced far more solved rollouts.

## Resume state / retry fixes (for a future run on Peter's say-so)

- Drivers are staged and proven to launch: `colab_mcpo_driver_4b.py` (chunked loss — port the chunked-loss fix back to the 8B driver before any 8B retry; the 8B driver as written still OOMs at line 366).
- Data loading is validated against the real parquet (schema, prompt parsing, ground-truth extraction, `Answer:`-line parsing, fraction comparison).
- Retry design changes needed (the reclamation alone does not explain the lack of signal):
  1. **Easier problems or a stronger starting point** — GSM8K-style problems, an instruct base, or an SFT warmup — so a meaningful fraction of groups has non-zero reward; otherwise MCPO vs GRPO cannot be discriminated at any step count.
  2. **Time budget** — ~300 s/step means a two-arm 24-step comparison needs ~4.5 h of uninterrupted GPU; split the arms across sessions (one arm per session, same paired prompts) or cut to 12 steps/arm, and download metrics mid-run (the TOPSD lesson — the step 1–19 snapshot is what survived here).
  3. Morning window (free-tier sessions live longest then).
- No adapters exist (no arm finished). Rotation state file stays `2026-11-11 ppoppo205`; tomorrow `--rotate` picks cyc236es.

## Limits

- The MCPO margin thresholds (0.10, 0.50) and weights (1.0, 0.5, 0.1) instantiate the abstract's "two fixed thresholds" — the paper does not publish the exact values, so a retry tests the mechanism family, not the exact paper configuration.
- Answer matching is exact-string/numeric (with `\frac` handling); symbolic answers that are algebraically equal but textually different count as wrong. Both arms share this, so it cannot bias the comparison, but it depresses absolute accuracy.
- Base evals are n=80 greedy; the 8B 1/80 and 4B 0/80 figures have wide uncertainty and are baselines only.
