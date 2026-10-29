# CASEL: Consequence-Aware Selective Execution (arXiv:2610.36570)

- Article: https://arxiv.org/abs/2610.36570 — "CASEL: Consequence-Aware
  Selective Execution for Reliable Long-Horizon Tool-Using Agents"
- Date screened: 2026-10-29. Serving fetch source: **rss**
  (rss.arxiv.org/rss/cs.AI; the api tier errored: export.arxiv.org HTTP 429
  + arxiv.org connection closed). 407 papers in the 3-day window; 59
  covered ids skipped before reading abstracts.
- Scores: AGENT=5, TUNE=2.
- Why picked: strongest agent runner-up on 2026-10-28 ("reconsider first
  on a later agent day"); a runtime allocation gate for tool calls is a
  direct harness mechanism, and the paper's own protocol is a controlled
  simulation with a same-menu / same-cost comparison, which maps 1:1 onto
  a deterministic CPU reproduction. Repo fit: arxiv-lab harness library
  (`src/arxiv_lab/harness/`, alongside PrefillBudget/FinishGate and the
  2026-10-28 context-attrition mechanism); applicable in principle to the
  Hermes on-device agent's tool-call gate.

## What the paper claims

Most runtime safeguards allocate scrutiny by generic uncertainty scores or
coarse mutation heuristics. CASEL combines calibrated failure probability,
state-dependent consequence, reversibility, ambiguity, and evidence
freshness, and chooses per tool call among direct execution, verification,
sandboxing, clarification, approval, and abstention by minimizing a
per-mode cost. Direct execution is formalized as selective prediction;
the paper states and tests the validity conditions of a finite-sample
threshold and a trajectory risk ledger. In a controlled simulation (20
seeds, 120,000 test actions each, same intervention menu and same compute
cost for every method): harmful-side-effect trajectories 14.75%
(uncertainty-ranked gate) → 7.02% (CASEL); mean normalized harm 0.184 →
0.123. A product gate (failure probability × consequence) reaches similar
harm (5.82%, 0.120) but sends 45% more actions to human approval; removing
reversibility-aware routing raises the rate to 15.53%; a trajectory-level
risk budget increases harm at equal cost (11.82%). The paper's own limit:
intervention effectiveness is assumed, not measured — the results rank
allocation rules under the simulator's assumptions and do not establish
deployment effectiveness.

## What was built

- `src/arxiv_lab/harness/casel.py` (stdlib only) + `tests/test_harness_casel_cpu.py`
  (graduated mechanism; full CPU suite re-run, no FAIL lines).
- This session: `experiment_casel.py` (regenerates `results.json`) and
  `results.json` (canonical run numbers).
- Simulator, parameters fixed before the final run: 20 seeds × 10,000
  actions, trajectories of 20 actions (200,000 actions, 10,000
  trajectories), canonical seed 20261029, robustness seeds 1/7/42 (8 seeds
  × 5,000 actions each). Per action: calibrated failure-probability
  estimate (small unbiased noise), generic uncertainty score (large noise,
  blind to consequence/reversibility), consequence (70% low / 25% medium /
  5% high, state-dependent: ×(1 + 0.5 × prior harmful events in the
  trajectory)), reversibility (60% reversible, harm ×0.3 when failed),
  ambiguity and evidence-staleness components, a shared failure uniform
  (common random numbers — paired comparison), and an abstention value.
  Mode effects are the simulator's assumptions, as in the paper:
  verification refreshes stale evidence and catches 40% of harm,
  clarification resolves ambiguity, sandboxing contains reversible harm
  (×0.2) but leaks for irreversible side effects (×0.6), approval
  catches 95%, abstention executes nothing at the forfeited action value.
  Expected cost = mode cost + 20 × failure probability × planned harm
  (the harm weight puts harm and compute cost on one scale; without it no
  mode cost ever outweighs expected harm and the policy degenerates to
  "always direct"). Ranked gates (uncertainty, product) send a
  cost-matched top fraction to approval; the budget gate spends CASEL's
  own per-trajectory cost in trajectory order, then goes direct.
- Two simulation bugs were caught and fixed before the verdict run, and
  are recorded here rather than hidden: (1) the first draft used
  50-action trajectories at high base failure rates, saturating the
  harmful-trajectory rate at ~1.00 for every policy so the metric could
  rank nothing; (2) the first draft's flat abstention cost dominated
  approval for every risky action, degenerating the six-mode menu, and
  its unweighted costs made "always direct" optimal. Both are design
  flaws in the reproduction, fixed on principle (discriminating metric
  range; every mode sometimes optimal), not tuned to the paper's point
  estimates — the absolute numbers below differ from the paper's, as
  expected for a different simulator.

## Results (canonical seed 20261029)

| policy | harmful-trajectory rate | mean normalized harm | mean cost/action | approval rate |
|---|---|---|---|---|
| CASEL | 0.1434 (seed SD 0.0152) | 0.0028 | 0.0477 | 0.0002 |
| uncertainty-ranked gate | 0.3985 (SD 0.0239) | 0.0092 | 0.0500 | 0.1000 |
| product gate (p × consequence) | 0.3544 (SD 0.0245) | 0.0076 | 0.0500 | 0.1000 |
| CASEL, no reversibility | 0.2662 (SD 0.0238) | 0.0047 | 0.0310 | 0.0000 |
| trajectory risk budget | 0.1460 (SD 0.0146) | 0.0029 | 0.0472 | 0.0002 |

- Main ranking reproduces directionally: CASEL cuts the harmful-trajectory
  rate by 64% relative to the uncertainty gate (0.1434 vs 0.3985; paper:
  −52%, 0.0702 vs 0.1475) and mean normalized harm by 70% (0.0028 vs
  0.0092; paper: −33%). Robustness seeds 1/7/42 reproduce the ranking on
  both metrics (CASEL rates 0.1375/0.1390/0.1535 vs uncertainty
  0.4085/0.3990/0.3980).
- Reversibility ablation reproduces: dropping reversibility-aware routing
  raises the harmful rate to 0.2662 (paper: 0.1553 vs 0.0702).
- Formal conditions reproduce at bounded scale: finite-sample threshold —
  held-out direct-arm failure rate 0.0463 (n=16,457) against the 0.05
  target (τ=0.0090 on the risk score, Wilson 95% bound on a calibration
  split); risk ledger — planned expected harm tracks realized harm with
  mean absolute error 0.114 per trajectory, realized harm within
  ledger + 0.5 on 98.25% of 400 trajectories.
- Mode mix (diagnostic, 40,000 actions): direct 21,872; verify 15,253;
  clarify 2,825; sandbox 34; approve 8; abstain 8. Approval/sandbox are
  last resorts in this cost model — verification dominates sandboxing
  because it is cheaper and nearly as effective under the assumed
  effects. This is a property of the assumed cost/effect numbers, not a
  finding about real deployments.

## Divergences (stated plainly)

- **Product-gate equivalence did NOT reproduce.** The product gate's mean
  harm (0.0076) is 2.7× CASEL's (0.0028), not similar as the paper
  reports (0.120 vs 0.123). Under this simulator's assumed intervention
  effects, CASEL's cheap-mode menu (verify/clarify) beats any
  approve-only ranked gate by a wide margin. The approval direction did
  reproduce (product gate 10.0% approvals vs CASEL 0.02%), but the
  "similar harm" part did not, so the paper's product-gate claim is not
  supported by this reproduction.
- **Trajectory-budget harm increase is negligible here.** The budget gate
  (0.1460) beats CASEL (0.1434) by only 0.0026 — direction only. Because
  the budget equals CASEL's own per-trajectory spend, exhaustion binds
  only on the last actions; the paper's larger gap (0.1182 vs 0.0702)
  does not reproduce at meaningful size under this operationalization.

## Verdict

**SUPPORTED** — the core allocation ranking (CASEL > uncertainty gate;
reversibility matters; threshold and ledger conditions hold) reproduces
directionally at bounded scale with measured numbers, but two secondary
claims diverge as recorded above, and the reproduction inherits the
paper's own limit: intervention effectiveness is assumed, not measured;
no live LLM, no real tool calls, synthetic actions only. The mechanism
graduates to the harness library as a CPU-tested allocation rule, not as
a deployment-ready gate.

## Honest limits

- Synthetic corpus + deterministic rule realization; failure probabilities
  are drawn, not estimated by a model, and "calibrated" is built in
  rather than learned — calibration failure modes of real estimators are
  untested.
- Absolute harm numbers are simulator-scale (mean normalized harm 0.003
  vs the paper's 0.12) and are not comparable across simulators; only
  rankings and relative reductions transfer.
- Cost/effect numbers (mode costs, harm weight 20, sandbox/verify
  effectiveness, abstention values) are assumptions fixed a priori; a
  different cost model could reorder the secondary comparisons, as the
  product-gate divergence shows.
- Trajectory length 20 (paper's trajectory structure unstated); the
  budget-gate operationalization (front-loaded spend of CASEL's own
  budget) is one reading of "trajectory-level risk budget".
