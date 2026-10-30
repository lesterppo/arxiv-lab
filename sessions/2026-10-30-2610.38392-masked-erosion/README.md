# Masked Erosion: How Acceptance Gates Lose Redundant Safeguards (arXiv:2610.38392)

- Article: https://arxiv.org/abs/2610.38392 — "Masked Erosion: How
  Acceptance Gates Lose Redundant Safeguards in Self-Improving Agents"
- Date screened: 2026-10-30. Serving fetch source: **api** (second
  attempt; the v3 ladder's first pass errored: api tier HTTP 429 on both
  hosts, rss feed genuinely empty at 04:23 UTC, oai returned 1,642
  all-CS records including version churn. The direct API re-fetch then
  returned 200; pages 1–2, 300 newest cs.AI submissions by submittedDate,
  spanning 2026-10-27T17:52Z–2026-10-29T13:37Z; 293 not in the 2026-10-29
  RSS batch or the covered set. 62 covered ids skipped before reading
  abstracts: 61 local sessions + remote staging record 2610.22080.)
- Scores: AGENT=5, TUNE=1.
- Why picked: the strongest new agent paper of the day — a harness
  self-improvement failure mode directly relevant to any agent that
  edits its own prompts/rules (including the Hermes on-device agent and
  every arxiv-lab harness mechanism). The paper's own protocol is
  controlled deletion experiments over four safeguards in two
  environments under ten gate rules, which maps 1:1 onto a deterministic
  CPU reproduction with no live model. Repo fit: arxiv-lab harness
  library (`src/arxiv_lab/harness/`, alongside CASEL's runtime gate and
  the context-attrition mechanism) — the remedies it validates (freeze
  safety text, one-layer ablation audits) are deployable harness rules.

## What the paper claims

Self-improving agents keep a harness edit only if an acceptance gate
scores it higher on a batch of tasks. When safeguards overlap, deleting
one is masked: the others still block the harm, so the gate decides on
task success alone and redundancy erodes unseen. The paper measures
harm for every subset of four written safeguards in two agent
environments and runs 411 improvement loops under ten gate rules
(mostly controlled deletions, plus three GEPA optimizer runs): a
deletion behind another working safeguard adds almost no observable
harm; a paired significance test blocks deletions that pass by chance
but not deletions that genuinely raise task success; a larger
evaluation batch lets the gate delete cheaper safeguards (20-task
batches kept every harness protected while 40-task batches lost all
six, at the same cost); large harm audits prevent exposure but not the
loss of depth, which turns into harm when the harness is reused with a
weaker model; GEPA kept every safety rule verbatim, yet two of its three
final harnesses misused credentials under a weaker model; a
statistical model frozen before the runs predicts where each gate's
protection breaks as safeguard cost and batch size change; freezing
safety text or auditing one-layer ablations preserves depth.

## What was built

- `src/arxiv_lab/harness/masked_erosion.py` (stdlib only) +
  `tests/test_harness_masked_erosion_cpu.py` (graduated mechanism;
  full CPU suite re-run, no FAIL lines).
- This session: `experiment_masked_erosion.py` (regenerates
  `results.json`), `results.json` (canonical run numbers), and
  `robustness_check.txt` (seeds 1/7/42).
- Simulator, parameters fixed before the final run: 20 seeds, 4
  safeguards, each harm channel covered by exactly 2 safeguards
  (depth 2); friction (task-success cost of keeping a safeguard)
  0.02/0.04/0.06/0.08 by safeguard; 2 environments (coding: exposed harm
  0.60; web: 0.40); 6 rounds × 4 proposals per run with a fresh batch
  per proposal (common random numbers within a proposal: one fixed
  4,000-task bank per seed); weaker-model reuse = each present
  safeguard independently fails to block with probability 0.30.
  Calibration note: a first draft reused one batch for all proposals
  of a run, collapsing each safeguard's fate to a single draw, and
  modelled chance gains with fully paired draws (identically zero
  differences, so no gate could ever accept them). Both were fixed
  before the verdict run — fresh batch per proposal, and chance-gain
  deletions evaluated with fresh sampling noise (the paper's LLM
  evaluations are stochastic), which is what lets the paired test's
  ~5% nominal size show up at all. The frozen analytic model is the
  exact binomial P(Bin(N, p_flip) >= 3): with paired binary flips the
  t-statistic is sqrt(k/(1-k/N)), which exceeds 1.645 iff at least 3
  tasks flip.

## Results (canonical run, 20 seeds)

Masking is exact. Deleting any one safeguard alone changes mean harm
by **0.0000** in both environments (5 deletion measurements averaged);
deleting it while its covering partner fails raises harm by
**+0.046–0.049** (coding) and **+0.029–0.032** (web). The harm of a
deletion is invisible exactly while depth remains, and large exactly
when depth is gone — the paper's central mechanism, reproduced
without approximation.

Batch size erodes depth monotonically (paired gate, genuine
deletions, both environments identical to 2 decimals):

| batch N | deletions accepted / run | runs losing depth | runs fully exposed | weak-reuse harm Δ |
|---|---|---|---|---|
| 10 | 0.15 | 0.15 | 0.00 | +0.0027 / +0.0017 |
| 20 | 0.40 | 0.35 | 0.05 | +0.0092 / +0.0057 |
| 40 | 1.60 | 0.85 | 0.60 | +0.0480 / +0.0306 |
| 80 | 3.00 | 1.00 | 1.00 | +0.1090 / +0.0700 |

(Coding / web for the harm column.) At N=40 the gate also deletes the
cheapest safeguards more often (acceptances by safeguard, coding:
S1 0.15, S2 0.15, S3 0.55, S4 0.75, vs N=20: 0.00/0.05/0.15/0.20). The
paper's 20-vs-40 contrast reproduces directionally: at N=20 most
harnesses keep full depth; at N=40 most lose it and most are fully
exposed on at least one channel. Caveat: this sweep holds rounds
fixed, so larger batches cost more total evaluations; the paper's
"same cost" framing is not replicated here.

Gates at N=20 (coding; web identical in shape): the mean gate accepts
3.60 genuine / 3.75 chance deletions per run and loses depth in every
run (weak-reuse harm +0.15/+0.17). The harm-audit gate accepts 0.35
genuine / 1.10 chance — its strong-condition harm change is exactly
**0.0000** while depth is lost in 0.35/0.80 of runs: the audit passes
because the deletion is masked, exactly as the paper says. The
freeze gate and the one-layer ablation audit accept **nothing** and
lose no depth in either environment or arm. The frozen analytic model
predicts the breakage: per-proposal acceptance for the highest-friction
safeguard 0.044 (N=20) → 0.214 (N=40) → 0.625 (N=80), matching the
per-proposal rates implied by the simulation (0.037 / 0.206 /
saturated at N=80).

One sub-claim DIVERGES at the primary scale and is recorded as such:
at N=20 with the primary friction values, the paired test admits
chance deletions (1.35/run, the ~5% nominal size over 24 proposals)
FASTER than genuine ones (0.40/run), because a genuine deletion flips
fewer than 3 tasks per 20-task batch and the paired t-test cannot reach
significance on 1–2 flips however one-sided they are. The claim
reproduces only when gains are detectable: in the 3×-friction
sensitivity arm (N=20, coding), paired genuine admissions rise to
2.50/run vs 1.35 chance, while the mean gate stays indiscriminate
(3.95 vs 3.75). So: the paired test blocks chance gains at the nominal
rate at every scale, and admits genuine gains above the chance rate
only once the batch is large enough, or the gain large enough, to
produce ≥3 flips — a detectability condition the paper's headline
does not state.

Robustness (seeds 1/7/42, 3 runs each): masked Δharm is 0.0000 for
every safeguard at every seed; exposed Δharm +0.045–0.056. The N=40 ≥
N=20 acceptance ordering holds in aggregate but is noisy per seed
group (flat at seeds 1 and 42 on 3 runs each), as expected at this
sample size.

## Verdict: SUPPORTED

The core claims reproduce directionally at bounded scale — exact
masking, monotone batch-size erosion to full exposure, audits that
pass while depth is lost, freeze/ablation remedies that preserve
depth, and a frozen statistical model that predicts where protection
breaks. The paired-test sub-claim holds only conditionally (see the
divergence above). The GEPA optimizer runs, the paper's LLM
environments, and its exact "same cost" batch comparison were not
reproduced. Limits: synthetic tasks and safeguards; intervention
effectiveness and safeguard-following are assumed parameters, not
measured (the same class of limit the paper's own simulation carries);
no live LLM; friction values and harm rates are fixed a priori
choices, not estimates from the paper's environments.

## Resume state

None needed for the agent track — the experiment is complete and
deterministic. The graduated module is reusable as-is for auditing
acceptance gates in harness self-improvement loops. A natural
follow-up (not run): replace the synthetic task bank with a real
tool-use trace set and measure actual safeguard-following rates under
a weaker model, which would test the reuse claim end-to-end.
