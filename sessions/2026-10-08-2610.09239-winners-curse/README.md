# Session 2026-10-08 — The Winner's Curse in LLM Self-Improvement Loops (arXiv:2610.09239)

**Article:** "The Winner's Curse in LLM Self-Improvement Loops: Selection
Noise, Lock-in, and Acceptance Rules"
([abs](https://arxiv.org/abs/2610.09239))

**Why picked:** most prospective *agent* article in the 2026-10-08 cs.AI
window (AGENT 5/5). The paper formalizes what every keep-if-better
self-improvement loop is actually doing: selection under measurement noise
on a small eval set. Its four quantitative claims are unusually testable
without any LLM: (1) with K candidates sharing one selection set,
candidate errors are correlated within the decision; (2) in native loops
most proposals after the first are truly harmful; (3) the greedy loop's
final selection-set score exceeds held-out accuracy by 13–20 points at
n=16 items (1–5 at n=256), and the tested acceptance rules did not beat
greedy over whole runs; (4) scoring start and current on 64 items never
used for selection removes the average bias of the reported gain, though
single estimates stay off by ~6 points. Direct repo fit: `deepworld`
(any contribution-loop / prompt-evolution mechanism that keeps-if-better
on a small eval set inherits this bias) and `arxiv-lab` as the mechanism
library.

## What was built

- Reusable mechanism: `src/arxiv_lab/loops/winners_curse.py` (new `loops`
  area) — item-level statistical model of the paper's selection mechanism:
  true quality q (points), per-item Bernoulli outcomes with a per-set
  difficulty component shared across candidates (correlated errors),
  proposal distribution N(q_cur − 2, 3) (mostly harmful), greedy
  keep-if-better reusing one selection set across generations, plus two
  acceptance rules (`fresh64`: commit only if the winner also beats the
  incumbent on 64 fresh never-used-for-selection items; `conservative`:
  commit only if the selection-score gap exceeds a one-SE significance
  bar). Public API: `make_selection_set`, `score_candidates`,
  `heldout_score`, `generate_candidates`, `error_correlation`,
  `winners_curse_gap`, `run_greedy_loop`, `run_loop_with_rule`,
  `selection_bias`. numpy-guarded, stdlib otherwise, no network.
- CPU test: `tests/test_loops_cpu.py` — 14 checks, all passing.
- This session: `experiment_winners_curse.py` (verbatim snapshot, runnable
  from the repo root) and `results_winners_curse.json` (300 seeds).

## Results (300 seeds; greedy loop, 8 generations x 6 candidates)

**Claim 3 — bias table (final selection-set score minus held-out accuracy):**

| rule \ n_items | 16 | 64 | 256 |
|---|---|---|---|
| greedy | **13.70** | 6.84 | **2.82** |
| fresh64 acceptance | 5.68 | 3.20 | 1.35 |
| conservative | 9.23 | 5.27 | 2.40 |

Paper's bands: 13–20 pts at n=16, 1–5 pts at n=256. The sim lands inside
both with no tuning beyond the Bernoulli item model (one-shot argmax gap:
16.98 / 8.63 / 3.98). Bias decreases monotonically with n; bias(16) is
~5x bias(256).

**Claim 2 + lock-in (greedy, n=16):** harmful-proposal fraction 0.74;
harmful-commit fraction 0.64 — a majority of commits are truly harmful.
Mean reported gain **+9.06** pts while mean true held-out gain is
**−5.54** pts: the loop reports improvement while true quality declines.
True gains grow with selection-set size (−5.54 → −0.05 → +6.82), matching
the paper's TREC direction.

**Claim 1:** corr(candidate errors within a decision) = +0.12 (positive,
from the shared set difficulty).

**Claim 4:** scoring start and current on 64 fresh items gives mean bias
−0.07 pts (≈ 0, "removes the average bias") with mean single-estimate
absolute error 7.5 pts (≈ the paper's ~6).

**Acceptance rules vs greedy — true held-out gain:**

| rule \ n_items | 16 | 64 | 256 |
|---|---|---|---|
| greedy | −5.54 | −0.05 | **6.82** |
| fresh64 acceptance | **0.15** | **2.38** | 5.19 |
| conservative | −2.44 | 1.35 | 5.54 |

The paper's "acceptance rules did not beat greedy over whole runs"
**replicates only in the low-noise regime (n=256)**, where greedy ≥ both
rules. At n=16/64 the ranking inverts: greedy over-commits to noise and
is the worst of the three (fresh64 acceptance loses the least). Honest
read: the claim is regime-dependent in this model, not universal.

## Verdict

**Replicates** claims 1, 2, 3 (bias magnitudes), and 4 quantitatively in
the statistical model. Claim 3's acceptance-rule sub-claim replicates
only at n=256. The headline mechanism — keep-if-better on a small eval
set is selection under noise, and the reported gain is mostly winner's
curse at n=16 — comes through cleanly and with the paper's own numbers.

## Honest limits

> **Simulation honesty box.** The quality/noise model is a scripted
> statistical stand-in (Gaussian proposals, Bernoulli items), not LLMs.
> It tests the paper's *selection mechanism* (winner's curse bias,
> lock-in dynamics, acceptance rules), not LLM rewriting behavior. In
> particular: real proposal distributions are not Gaussian around the
> incumbent; real eval noise is not iid Bernoulli; the "mostly harmful"
> rate (delta=2, sigma=3) is a modeling choice implementing claim 2, not
> a measured LLM property. The n=256-only replication of the
> acceptance-rule claim may be an artifact of this noise model rather
> than a fact about LLM loops.

- Single-run variance is large (bias std 8.9 pts at n=16); the paper's
  bands are about means.
- The fresh-64 acceptance rule re-draws its 64 items every generation;
  a fixed acceptance set would just move the curse.
- No deepworld wiring: the mechanism is statistical, not a plug-in
  component; the takeaway for deepworld is diagnostic (distrust
  small-eval-set keep-if-better reports; re-score on fresh items).
