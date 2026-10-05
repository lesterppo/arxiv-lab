# Session 2026-10-05 — Prospective Hindsight (arXiv:2610.02740)

**Article:** "Prospective Hindsight: Self-Calibrating Reinforcement Learning
via Prediction-Reality Gaps" ([abs](https://arxiv.org/abs/2610.02740))

**Why picked:** most prospective *training* article in the 2026-10-02→05
window (AGENT 3/5, TUNE 4/5). It gives a concrete, small modification to the
RL post-training loop Peter's training track is growing toward
(soup-daily-finetune runs SFT/QLoRA today; GRPO-style post-training is the
natural next step): weight each rollout's advantage by the *surprise* —
|prospective prediction − retrospective outcome| — so the gradient focuses
on the agent's remaining blind spots, with calibration emerging as a
byproduct. The primitive is 5 lines; the claim is testable on CPU and its
premise (miscalibrated prospective predictions at 9B) is checkable on the
Colab-deployed Qwen3.5-9B.

**Paper claims:** (i) PH improves task performance *and* calibration under
GRPO, on-policy distillation, and their combination, across model scales;
(ii) minimizing the surprise residual gives a descent pathway on the
miscalibration rate — calibration emerges as a byproduct, not an added
objective; (iii) the prospective predictor shares parameters with the
policy, so the two co-evolve; (iv) dominant miscalibration mode is
overconfident failures (single-turn) / underconfident successes (multi-turn).

## What was built (training track)

- Reusable mechanism: `src/arxiv_lab/training/prospective_hindsight.py`
  (+ `__init__.py`, new `training` area) — `surprise_weights()`
  (w = 1 + λ|p − r|), `ph_advantages()`, `miscalibration_rate()`,
  `expected_calibration_error()`. Test: `tests/test_training_cpu.py`.
- Session experiment: `test_session_cpu.py` (snapshot) + `results.json` +
  `colab_premise.py` (Colab driver, premise validation on the deployed 9B).

## Results

### CPU mechanism test (this VM, deterministic, seeds 11/23/37)

Contextual bandit; policy head + prospective value head **share** the feature
extractor (claim iii); GRPO-style batch-normalized advantages as the base
method; PH multiplies the normalized advantage by (1 + 1.5·|V − r|).

| seed | base return | +PH return | base miscal | +PH miscal |
|---|---|---|---|---|
| 11 | 0.692 | 0.722 | 0.453 | 0.444 |
| 23 | 0.770 | 0.802 | 0.418 | 0.411 |
| 37 | 0.655 | 0.685 | 0.436 | 0.434 |

Mean: miscalibration **−0.0061** (3/3 seeds improved), return **+0.031**
(3/3 seeds no worse). Direction matches claims (i)+(ii): calibration
descends as a byproduct while task performance improves.

Design notes from two failed prototypes (kept honest): a per-step
actor-critic with unnormalized advantages showed *zero* PH effect because
the policy never learned enough to change behavior (arm sequences identical
across arms — degenerate experiment); an unbatched shared-extractor version
destabilized the actor (return collapsed). The GRPO-style batched,
normalized-advantage version is the faithful setting and the one reported.

### Colab premise validation (Qwen3.5-9B, `qwen3.5-9b-q4` via deploy.py)

**BLOCKED — transient free-tier T4 capacity.** Three deploy attempts on the
active account (kyu008008) all failed at step [1/6] "Creating session" with
`gpu-unavailable` (subagent ×2 with a 5-min wait between, parent ×1 ~15 min
later). No account switch was made (task constraints; cyc236es also showed
transient `gpu-unavailable` earlier today). Nothing was deployed, nothing
needed undeploying, no partial state. This matches the known free-tier
instability pattern (2026-10-03 evening: 3 T4 sessions reclaimed in hours).

**Not faked:** no premise numbers exist. The planned check (~24 verifiable
math questions; model states prospective confidence 0–100 before solving;
grade retrospective correctness; accuracy, ECE, mean confidence,
overconfident-failure rate, per-sample surprise) is staged as
`colab_premise.py` in this session — stdlib-only, takes
`<out.json> <ollama-base-url>`. Re-run it against a fresh `qwen3.5-9b-q4`
deploy when T4 capacity frees up.

## Verdict: GREEN (mechanism) / BLOCKED (Colab premise)

The PH primitive and its headline effect (miscalibration descends as a
byproduct, return unharmed) reproduced on CPU across 3 seeds. The reusable
mechanism graduated to `src/arxiv_lab/training/`. The Colab premise leg is
honestly blocked on transient T4 capacity — recorded, not faked, with a
ready driver for re-run.

## Honest limits

- Synthetic bandit, not LLM post-training: no language, no real GRPO, no
  9B weights in the loop. Effect sizes are small (−0.006 miscalibration)
  though direction-consistent across seeds.
- ECE did not improve consistently (10-bin ECE on binary outcomes is noisy);
  the asserted metric is the miscalibration rate, matching the paper's
  "surprise residual" framing.
- Full GRPO-with-PH at the 9B class on Colab remains future work (needs a
  transformers-compatible 9B base + TRL GRPO on T4; bounded but not
  attempted in this daily run).
- Colab premise leg blocked by transient T4 capacity (3 attempts,
  `gpu-unavailable`); recorded honestly with a ready re-run driver.
