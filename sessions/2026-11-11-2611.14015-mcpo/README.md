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

- Drivers are staged and proven to launch: `colab_mcpo_driver_4b.py` (chunked loss) and `colab_mcpo_driver.py` (8B — chunked-loss fix ported back 2026-11-11 after the run, micro 6 → 4 to match the proven 4B config; CPU equivalence check vs the full-vocab computation: tok_logp exact, margin max diff 2.8e-09, backward grad finite; `py_compile` clean on both drivers. The ported 8B driver has not yet run on Colab — the fix removes the step-1 OOM, it does not guarantee a full run survives reclamation).
- Data loading is validated against the real parquet (schema, prompt parsing, ground-truth extraction, `Answer:`-line parsing, fraction comparison).
- Retry design changes needed (the reclamation alone does not explain the lack of signal):
  1. **Easier problems or a stronger starting point** — GSM8K-style problems, an instruct base, or an SFT warmup — so a meaningful fraction of groups has non-zero reward; otherwise MCPO vs GRPO cannot be discriminated at any step count.
  2. **Time budget** — ~300 s/step means a two-arm 24-step comparison needs ~4.5 h of uninterrupted GPU; split the arms across sessions (one arm per session, same paired prompts) or cut to 12 steps/arm, and download metrics mid-run (the TOPSD lesson — the step 1–19 snapshot is what survived here).
  3. Morning window (free-tier sessions live longest then).
- No adapters exist (no arm finished). Rotation state file stays `2026-11-11 ppoppo205`; tomorrow `--rotate` picks cyc236es.

## Redesigned retry (staged 2026-11-11, NOT launched — launch timing is Peter's call)

**Decision: easier problems only — GSM8K train/test, no SFT warmup.**
Measured evidence (all numbers below were measured, not guessed):

- **DAPO has no signal to recover.** Qwen3-8B base 1/80, Qwen3-4B base 0/80
  on held-out DAPO; the 4B GRPO run produced 6 correct rollouts in 528, so
  the group-relative advantage was exactly 0 on 18/22 steps — only ~4/88
  training groups carried any signal. A same-design rerun cannot
  discriminate MCPO from GRPO at any step count.
- **Within-DAPO difficulty filtering is not possible.** The parquet's
  `extra_info` carries only an `index` UUID (checked live via the
  datasets-server rows API, 2026-11-11); `ability` is just "MATH". There
  is no difficulty/level field to filter on, so any filter would need an
  unvalidated proxy (e.g. prompt length).
- **GSM8K restores signal.** Qwen3-8B base scores 0.83–0.90 on GSM8K
  (Qwen3 Technical Report Table 7: 89.84 few-shot; AgentArk: 88.17;
  independent paper: 0.83). At group=6 the mixed-group fraction is 0.88
  at p=0.7, 0.74 at p=0.8, 0.47 at p=0.9 — i.e. roughly half to
  three-quarters of training groups carry non-zero advantage, ~10–19x
  denser signal than DAPO's 0.045. Ceiling risk is bounded: even at
  p=0.9, ~45 of 96 training groups still discriminate, and the base eval
  (greedy zero-shot, not few-shot) will land below the published
  few-shot figures.
- **GSM8K parses cleanly.** Measured on the real files 2026-11-11:
  gold (`#### N`) parses 7,473/7,473 train + 1,319/1,319 test, all
  numeric; questions mean 45 words (max 183), solutions mean 52 words
  (p95 102, max 216) — comfortably inside MAX_PROMPT_TOKENS=512 and
  MAX_NEW=320.
- **No SFT warmup.** GSM8K alone restores signal; a warmup phase would
  add a full training stage, ~1–2 h, and new failure modes with no
  evidence it is needed. It is added only if the redesigned run's base
  eval comes back <0.05 or per-step frac_solved stays ~0 (i.e. GSM8K
  also yields no signal — contrary to the published numbers).

**What changed vs the original drivers:**
- New files `colab_mcpo_driver_gsm8k_grpo.py` / `colab_mcpo_driver_gsm8k_mcpo.py`
  (one arm each; `diff` between them shows ONLY arm name, MCPO flag, done
  marker, output paths, docstring labels). Everything else is identical:
  same model (Qwen3-8B NF4 QLoRA r=16/a=32, bf16), same seed, same
  hyperparameters, same chunked-loss fix (micro 4, CH=64,
  expandable_segments), same margin thresholds/weights. The only
  difference between arms is the token-credit weighting, as in the paper.
- Data: GSM8K train pool 512 / GSM8K **test** eval 80 (true held-out,
  unlike the original driver whose eval came from the same sampled pool);
  paired prompts are train_pool[:96], selected deterministically so both
  arms train on the same prompts in the same order and evaluate on the
  same 80 problems. Prompt is one fixed instruction + the raw question,
  final answer as `Answer: $N`; gold is the `#### N` number.
- Loss/leak fixes from the 4B run carried over; new: mid-run adapter
  checkpoint + zip at step 12 (a reclamation at step 22 must not lose
  the whole arm), results.json rewritten after every phase, metrics CSV
  flushed per row, and a `SIGNAL_CHECK` line after the base eval (flags
  it if base acc is <0.05 or >0.98).
- `test_mcpo_redesign_cpu.py`: **71/71 checks pass** — helper identity
  across arms, config identity except arm-identity keys, margin/advantage
  helpers, answer extraction, GSM8K gold parsing on the real files
  (7,473/7,473 + 1,319/1,319), pool determinism (identical paired
  prompts/eval sets across arms, no train/eval overlap), signal math,
  and the driver-pitfall checklist in both files. `py_compile` clean on
  both drivers.
- Verdict stays **BLOCKED** — nothing about the claim has been tested
  under the redesign yet.

**Launch plan (for Peter's timing pick; do NOT launch without it):**
- Arm order: GRPO first (`mcpo-1111-grpo`), then MCPO (`mcpo-1111-mcpo`).
  Each arm is its own Colab session; the two arms must NOT share a
  session (a single 4.5 h session is what reclamation killed last time).
- Account: **ppoppo205** (today's rotation account; tomorrow's 12:20
  daily run picks cyc236es, so no collision). Cross-account retry stays on.
- Wall time: ~300 s/step × 24 steps ≈ 2 h/arm + base eval + model load
  ≈ 2.2–2.5 h per arm, ~4.5–5 h total across the two sessions.
  Morning window (free-tier sessions live longest then). Stage a host-side
  mid-run downloader per arm (TOPSD pattern) polling the metrics CSV and
  the step-12 adapter zip — the wrapper's `--results` only downloads at
  completion.
- Exact commands (run from this session dir; registry/out-dir under
  `~/workspace/goals/arxiv-daily-read-and-test/hidden_files/mcpo-1111-redesign/`):

```bash
# Arm 1 — GRPO
python3 ~/workspace/hermes-colab-cli/colab.py train \
  --driver colab_mcpo_driver_gsm8k_grpo.py --account ppoppo205 \
  --session mcpo-1111-grpo --gpu T4 --monitor-timeout 14400 \
  --done-marker MCPO_GRPO_DONE \
  --results /content/mcpo_gsm8k_metrics_grpo.csv /content/mcpo_gsm8k_results_grpo.json \
            /content/mcpo_gsm8k_adapter_grpo.zip /content/mcpo_gsm8k_adapter_grpo_mid.zip \
  --out-dir ~/workspace/goals/arxiv-daily-read-and-test/hidden_files/mcpo-1111-redesign/grpo \
  --registry-file ~/workspace/goals/arxiv-daily-read-and-test/hidden_files/mcpo-1111-redesign/registry.json \
  --json

# Arm 2 — MCPO (only after arm 1 completes; same paired prompts by construction)
python3 ~/workspace/hermes-colab-cli/colab.py train \
  --driver colab_mcpo_driver_gsm8k_mcpo.py --account ppoppo205 \
  --session mcpo-1111-mcpo --gpu T4 --monitor-timeout 14400 \
  --done-marker MCPO_MCPO_DONE \
  --results /content/mcpo_gsm8k_metrics_mcpo.csv /content/mcpo_gsm8k_results_mcpo.json \
            /content/mcpo_gsm8k_adapter_mcpo.zip /content/mcpo_gsm8k_adapter_mcpo_mid.zip \
  --out-dir ~/workspace/goals/arxiv-daily-read-and-test/hidden_files/mcpo-1111-redesign/mcpo \
  --registry-file ~/workspace/goals/arxiv-daily-read-and-test/hidden_files/mcpo-1111-redesign/registry.json \
  --json
```

## Limits

- The MCPO margin thresholds (0.10, 0.50) and weights (1.0, 0.5, 0.1) instantiate the abstract's "two fixed thresholds" — the paper does not publish the exact values, so a retry tests the mechanism family, not the exact paper configuration.
- Answer matching is exact-string/numeric (with `\frac` handling); symbolic answers that are algebraically equal but textually different count as wrong. Both arms share this, so it cannot bias the comparison, but it depresses absolute accuracy.
- Base evals are n=80 greedy; the 8B 1/80 and 4B 0/80 figures have wide uncertainty and are baselines only.
