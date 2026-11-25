# InstructReplay — 2611.32918 (2026-11-25)

**Verdict: INCONCLUSIVE** — the test design could not discriminate.
The SFT arm showed no forgetting at this scale, so the paper's core
claim (replay reduces forgetting) was never actually tested. This is
not a refutation.

## Article

- arXiv:2611.32918 "InstructReplay: Mitigating Catastrophic
  Forgetting in LLM Fine-Tuning via Self-Generated Instruction Replay"
- Core claim: fine-tuning on new facts erodes general abilities
  (standard SFT lowers five general benchmarks ~35 points on average,
  tracking an answer-length collapse); replaying the initial model's
  own answers to general instructions during fine-tuning cuts the
  average decline from 28.1 to 2.4 points (28.07 → 2.44) while
  learning facts as well as standard SFT. Replay uses forward KL
  against the frozen initial model's full next-token distribution,
  λ = 0.25, mixed 1:1 with fact sequences.
- Why picked: TUNE 5 / AGENT 3. Directly relevant to the
  training-track question of how to teach a model new facts without
  destroying it — the same failure mode this track keeps hitting
  (fact-injection runs that collapse general ability).

## What was built

- `colab_instructreplay_driver.py` — Colab driver (Qwen3.5-9B-Base,
  NF4 QLoRA r=64/a=128, bf16, lr 2e-5, micro-batch 4 × accum 2).
  80 fictional drugs (invented names + random doses, label-style
  docs); eval 40 taught drugs (exact dose + first-dose) + 20
  untrained controls. Replay: base model answers 80 general
  instructions once (60 GSM8K train questions + 20 hand-written
  general instructions), greedy, max_new 160; replay arm trains
  facts + replay 1:1 per epoch. Evals per arm (greedy): fact
  injection, GSM8K test first 60 zero-shot last-integer match,
  answer length (mean generated tokens on GSM8K + 20 held-out
  hand-written general instructions, disjoint from the replay set).
- `test_instructreplay_cpu.py` — CPU test, 5/5 passed (fact
  uniqueness, dose matching, GSM8K scoring, 1:1 replay mix,
  replay/length set disjointness).
- `instructreplay_metrics.csv`, `instructreplay_results.json`,
  `instructreplay_replay.json` (80 replay pairs),
  `instructreplay_benchmark.json` (fact/control drug lists),
  `train-run.json` (Colab outcome DONE).

## Results (Colab T4, cyc236ha, 107 min, DONE)

| metric | base | SFT (facts only) | Replay (facts + own answers) |
|---|---|---|---|
| inject_exact (n=40) | 0.0 | 0.025 (1/40) | 0.05 (2/40) |
| inject_first | 0.025 | 0.025 | 0.05 |
| control_exact (n=20) | 0.0 | 0.0 | 0.0 |
| GSM8K acc (n=60) | 0.4167 (25/60) | 0.4833 (29/60) | 0.4667 (28/60) |
| GSM8K mean len | 113.5 | 114.7 | 110.9 |
| general mean len (n=20) | 156.6 | 153.3 | 154.6 |

Training: SFT 20 steps, mean loss 1.2892; replay 40 steps, mean
loss 0.6285 (replay sequences are the model's own fluent outputs,
so they fit easily). Replay corpus: 80 pairs, mean answer 135.4
tokens.

## Why INCONCLUSIVE

- **No forgetting occurred in the SFT arm.** GSM8K went UP
  (0.4167 → 0.4833) and answer lengths stayed flat (general 156.6 →
  153.3 tokens; GSM8K 113.5 → 114.7). The paper's effect — SFT
  collapsing general ability ~28 points and halving answer length —
  needs a fact intervention strong enough to cause forgetting; 80
  facts over 2 epochs of LoRA did not. With nothing to rescue, the
  replay arm cannot demonstrate a rescue: replay ≈ SFT on every
  metric (GSM8K 28 vs 29 correct; lengths within 2–4 tokens).
- **Fact injection itself was at floor.** SFT learned 1/40 doses
  exactly, replay 2/40 — a 1-drug difference, noise. The models
  answer dose questions confidently with wrong doses (e.g. Prilal
  true dose 250 mg every 12 hours; both arms answer "10 mg once
  daily"). The "learns facts as well as SFT" leg is therefore also
  uninformative at this scale.
- The secondary observation is consistent with the paper (replay
  does not hurt fact learning or general ability), but that is not
  the headline claim.

## Operationalization differences (honest)

- Replay uses plain SFT (cross-entropy) on the base model's own
  generated answers, **not** the paper's forward-KL soft-target loss
  (λ = 0.25). Justified by the paper's own controlled comparison
  (Appendix E / Sec. 5): soft vs sampled (hard CE) targets perform
  similarly — the gain comes from replaying the model's OWN answers,
  not the loss form. The own-answers part is implemented exactly
  (base model generates the replay corpus once, greedy, no
  correctness filtering).
- Replay generation is greedy (temp 0), max_new 160; the paper uses
  temp 1, top-p 1, max 512 new tokens on 9,600 instructions. Ours is
  80 instructions — a much smaller replay set.
- Model is Qwen3.5-9B-Base (standing-rule 8–9B class); the paper's
  models are 3–4B (SmolLM3-3B, Qwen3-4B, Phi-4-mini-instruct).
  LoRA r=64/a=128, lr 2e-5; the paper's LoRA-SFT is r=16/a=32,
  lr 1e-4.
- Benchmarks are GSM8K + answer length only, not the paper's five
  (IFEval, MMLU, MATH-500, BBH, HumanEval+).

## Run notes

- Two `new-failed` PROVISION_FAILEDs before launch: the harness sets
  `PYTHONNOUSERSITE=1`, so the raw `colab_cli` import failed
  (`ModuleNotFoundError: No module named 'colab_cli'`). Fixed by
  exporting `PYTHONPATH="$HOME/.local/lib/python3.12/site-packages"`
  for every `colab.py` invocation (known 2026-11-24 env break,
  re-confirmed). Third attempt on cyc236ha ran clean.
- The host-side train monitor died in a runtime restart at ~06:51
  UTC; the VM-side driver survived (detached) and was re-attached
  with `train --watch-only --pid 552`. Outcome DONE, all four result
  files downloaded.
- Two non-fatal `CUDACachingAllocator` OOM warnings during replay
  training (steps ~4 and ~32, 20 MB mapping failures, free ~10 MB);
  the allocator recovered and training continued — peak VRAM pressure
  from the mixed batches with 160-token replay sequences.
- Qwen3.5 Gated DeltaNet reference-kernel fallback warnings
  (`causal_conv1d` / `chunk_gated_delta_rule` /
  `fused_recurrent_gated_delta_rule` not installed): generation is
  slow but correct. Replay generation (80 prompts) took ~17 min;
  each eval generation pass ~10–12 min.
- One replay-arm GSM8K generation degenerated into repetition
  ("Answer: A robe takes 2 bolts..." looped) — single-item anecdote,
  not a finding.

## Resume state (if retried)

The verdict needs a setup where SFT actually forgets first. Increase
the fact load until the SFT arm shows a measurable GSM8K drop and
length collapse (e.g. 300 facts, more epochs, and/or higher LR), then
re-run the replay arm against the same SFT arm. The driver already
supports this via `N_TRAIN_DRUGS` / `EPOCHS` env; the replay corpus
generation and eval harness are unchanged. No fallback was consumed
(leg 1 DONE, not BLOCKED).
