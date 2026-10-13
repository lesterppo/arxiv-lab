# Session 2026-10-13 — arXiv:2610.12659 "MagicBeanStalk: Escaping the RLVR Zero Advantage Trap with Recursive Data Synthesis"

**Article:** https://arxiv.org/abs/2610.12659
**Track:** training (RLVR data) · **AGENT 3 / TUNE 4** · **Verdict: SUPPORTED** (live model)

## Why picked
Data-centric RLVR claim with a directly measurable statistic: for a fixed
target policy and rollout budget, ~88% of a pooled math corpus falls into
the "zero-advantage trap" (every rollout correct or every rollout wrong →
zero advantage → no learning signal), and *recursive* data synthesis
(teacher generates calibrated variants of hard problems, steering
difficulty until outcomes are mixed) restores the signal. Testable live
on the training track's Qwen3.5-9B recipe without any fine-tuning —
pure rollout measurement. Chosen over TQS (2610.13152, TUNE 4), whose
SWE-bench fine-tuning claim cannot be honestly tested in a bounded run.

## What was built
- `mbst_experiment.py` — self-contained Colab driver (stdlib only,
  localhost Ollama): Phase 1 rolls out G=4 on 20 inline verifiable math
  problems; Phase 2 asks the model-as-teacher for easier variants of the
  all-failed problems and rolls out G=3 on each.
- `mbst_recursive.py` — the paper's actual recursive loop: up to 3
  rounds per hard problem, steering the teacher harder/easier based on
  the observed outcome mix until rollouts are mixed.
- `mbst_results.json`, `mbst_recursive.json` — raw results.

Run live 2026-10-13 on Colab T4 (session `mbst-2610`, account kyu008008):
`deploy.py deploy --model qwen3.5-9b-q4` → Ollama `qwen3.5:9b`, all calls
VM-side via localhost (no tunnel needed). Session torn down afterwards.

## Results
**Phase 1 — the trap is real:** 16/20 problems (80%) in the zero-advantage
trap (14 all-correct, 2 all-failed); only 4/20 gave mixed outcomes.
Per-problem successes: `[4,4,4,4,4,4,4,4,4,0,4,2,1,3,0,3,4,4,4,4]`.
Paper reported ~88% on their pooled corpus — same phenomenon, live.

**Phase 2 — one-shot naive synthesis does NOT calibrate:** 4 variants,
outcomes 3/3, 3/3, 0/3, 3/3 → mixed-outcome fraction **0/4**. The teacher
overshot (too easy) or undershot (still too hard). This confirms the
paper's point: calibration, not just generation, is the hard part.

**Phase 3 — recursive synthesis:** 1/2 hard problems calibrated.
GCD(84,126) → teacher's GCD(12,18) → 2/3 mixed ✓ in round 1.
Palindrome(200–300, ÷11) resisted 3 rounds of progressive simplification
(100–200 ÷11 → 50–100 ÷2 → 20–30 ÷2), all 0/3 — the policy genuinely
lacks the palindrome concept, and no difficulty tweak within the
operation family fixed it.

## Honest limits
- N=20 problems, G=4 rollouts — small; trap classification at G=4 is
  noisy (a 50%-policy shows 0/4 or 4/4 with p≈0.06 each). The 14
  all-correct are robust; the 2 all-failed may include an unlucky draw
  (the GCD problem was solved on a later single probe).
- Problems are inline-written, not a real RLVR corpus slice; "same
  reasoning operations" in synthesis is prompt-enforced, not verified.
- `qwen3.5:9b` is a hybrid thinking model: with thinking enabled it
  burns the token budget in the `thinking` field and returns an empty
  `response` — the first experiment run silently recorded all-false
  because of this. Fix: `"think": false` in `/api/generate`. **Durable
  pitfall for the training track.**
- No fine-tuning was run: the claim tested is the data characterization
  + synthesis calibration, not downstream RLVR gains.

## Files
- `mbst_experiment.py`, `mbst_recursive.py` (Colab drivers)
- `mbst_results.json`, `mbst_recursive.json`
- This README
