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

**BLOCKED (morning) — transient free-tier T4 capacity.** Three deploy attempts on the
active account (kyu008008) all failed at step [1/6] "Creating session" with
`gpu-unavailable` (subagent ×2 with a 5-min wait between, parent ×1 ~15 min
later). No account switch was made (task constraints; cyc236es also showed
transient `gpu-unavailable` earlier today). Nothing was deployed, nothing
needed undeploying, no partial state. This matches the known free-tier
instability pattern (2026-10-03 evening: 3 T4 sessions reclaimed in hours).

**COMPLETED (evening, ~18:10–18:40 HKT) — INCONCLUSIVE.** After Peter
confirmed 5 Colab accounts, deployed `qwen3.5-9b-q4` (Ollama tag
`qwen3.5:9b`, Q4_K_M, ~7GB VRAM) on the first tried account **cyc236de**
(T4, instant capacity; one runtime approval for session creation). Ran
`colab_premise.py` (24 verifiable math questions; model states prospective
confidence 0–100, then answers; grades retrospective correctness) against
the tunnel endpoint. Results in `colab_premise_results.json`:

- n_graded 14/24, accuracy **1.0**, ECE_10bin **0.0**, mean_confidence
  **100.0**, overconfident_failure_rate **null** (no failures),
  mean_surprise **0.0**
- 10/24 lost to transient Cloudflare tunnel 502/530 errors; the tunnel
  died entirely mid-run (error 1033) — Colab reclaimed the VM. Session
  undeployed cleanly afterwards (server-side already gone).

**Verdict on the premise: INCONCLUSIVE, not disconfirming.** The question
set was too easy for the 9B model — 14/14 correct at confidence 100 means
zero prediction-reality gaps to measure, so the premise cannot discriminate
the paper's overconfidence claim either way. A discriminating premise needs
questions hard enough to produce failures (e.g. multi-step word problems
near the model's competence boundary). The endpoint, harness, and scoring
all worked; the limitation is question difficulty, not infrastructure.

**v2 re-run (same evening, ~18:35–18:50 HKT) — harder set, still
INCONCLUSIVE.** After Peter approved a harder re-run, redeployed
`qwen3.5-9b-q4` on **cyc236de** (fresh session `premise-hard`; the deploy's
runtime approval gate fired again and was approved). New driver
`colab_premise_hard.py`: 16 harder computation items (47×83, 17³,
GCD/LCM, 2¹⁶, primes<100, clock-strike sums, trailing zeros of 10!,
handshakes) + 8 classic cognitive-reflection traps (months-with-28-days,
bat-and-ball, lily pads, digit-9 counting, 10th prime, apple takeaway) —
all answers verified integers. Results in `colab_premise_hard_results.json`:

- n_graded 11/24, accuracy **1.0**, ECE_10bin **0.0**, mean_confidence
  **100.0**, overconfident_failure_rate **null**, mean_surprise **0.0**
- 13/24 lost to transient tunnel 502/530s; the tunnel died again and Colab
  reclaimed the VM mid-run (session auto-pruned locally afterwards)
- All 11 graded items were computation questions — answered correctly at
  confidence 100. **The 8 reflection traps never got a clean shot** (all
  errored on the dead tunnel), so the discriminating part of v2 is
  untested, not failed.

**Verdict on v2: still INCONCLUSIVE.** Harder arithmetic still ceilinged;
the traps — the items most likely to expose overconfident failures —
remain unmeasured. The pattern across both runs: a 9B instruct model at
temperature 0 states confidence 100 and is right on everything it answers,
so this single-turn confidence-elicitation format cannot produce the
prediction-reality gaps the paper's claim needs. Either the premise needs
genuinely adversarial items (trick wording, multi-hop traps at the
competence boundary) in a short trap-only run, or the overconfidence mode
the paper targets doesn't manifest in this elicitation format at 9B.

**Not faked:** no premise numbers exist. The planned check (~24 verifiable
math questions; model states prospective confidence 0–100 before solving;
grade retrospective correctness; accuracy, ECE, mean confidence,
overconfident-failure rate, per-sample surprise) is staged as
`colab_premise.py` in this session — stdlib-only, takes
`<out.json> <ollama-base-url>`. Re-run it against a fresh `qwen3.5-9b-q4`
deploy when T4 capacity frees up.

## Verdict: GREEN (mechanism) / INCONCLUSIVE (Colab premise)

The PH primitive and its headline effect (miscalibration descends as a
byproduct, return unharmed) reproduced on CPU across 3 seeds. The reusable
mechanism graduated to `src/arxiv_lab/training/`. The Colab premise leg ran
twice on real 9B endpoints (cyc236de T4) but is inconclusive both times:
v1's 24-question set was too easy (14/14 correct at confidence 100 → no
gaps to measure); v2's harder set still ceilinged on the 11 graded items
(all computation, all correct @100) while the 8 reflection traps — the
discriminating items — all died on the tunnel before being answered. Both
runs lost samples to free-tier tunnel flakiness and had their VMs reclaimed
mid-run. Honestly recorded with results JSONs, not faked.

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
- Colab premise leg: morning blocked by transient T4 capacity (3 attempts,
  `gpu-unavailable`); evening v1 run on cyc236de completed 14/24 but
  inconclusive — questions too easy for 9B (ceiling), tunnel died mid-run
  (VM reclaimed). v2 harder-set re-run (same evening, cyc236de): 11/24
  graded, still all correct @100; the 8 reflection traps all died on the
  tunnel before being answered — discriminating items remain unmeasured.
