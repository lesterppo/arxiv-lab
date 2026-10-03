# Session 2026-10-04 — Component Routing (arXiv:2610.01787)

**Article:** "Not All Experience Belongs in the Weights: Component Routing
for Self-Improving GUI Agents"
([abs](https://arxiv.org/abs/2610.01787))

**Why picked:** most prospective *agent* article in the 2026-10-03→04
window (AGENT 4/5). It attacks a design question at the heart of
Peter's local stack: DeepWorld's contribution loop writes agent
experience into weights (accepted contributions become tools), while
the prompt/context path carries the rest — the paper says the *unit*
of that decision should be the component (locator, procedure, state
fact, lesson), not the whole trajectory, and gives a two-property
routing rule. Directly implementable on this VM, no GPU needed.

**Paper claims:** (i) locators + lessons win in weights, procedures +
state facts in context; (ii) a rule fit on recurrence and
state-conditionality recovers a held-out backbone family's destination
in 24/24 cells, and routing by the rule beats every whole-trajectory
baseline (+3.5 pts avg); (iii) note readout drops after the same
component is written into weights (most for highest-recurrence
items), context gains rise with the information gap, weights gains
fall with the policy gap.

## What was built (agent track, this VM)

- Reusable mechanism: `src/arxiv_lab/routing/component.py` — `Component`
  (kind, recurrence, state-conditionality), `RoutingRule`
  (logistic P(weights | r, s), pure-stdlib fit), `route_components()`.
  Test: `tests/test_routing_cpu.py` (all pass).
- Session experiment: `sessions/2026-10-04-2610.01787-component-routing/experiment.py`
  (+ `test_session_cpu.py`, `results.json`). Synthetic world: 4 component
  types × 2 envs × 3 seeds; three "backbone families" with different gain
  scalings (memorizer 1.5/0.75, in-context 0.65/1.35, balanced 1.0/1.0);
  per-item gain-in-weights vs gain-in-context with noise. Rule fit on two
  families, tested on the held-out family's 24 cells — mirroring the
  paper's protocol. Claim (iii) as three toy demonstrations.

## Results (CPU, deterministic, seed 20261004)

**(i) Destination pattern** — P(weights wins) per type × family:

| family | locator | lesson | procedure | state_fact |
|---|---|---|---|---|
| memorizer | 1.000 | 1.000 | 0.133 | 0.058 |
| in-context | 0.971 | 0.896 | 0.000 | 0.000 |
| balanced (held-out) | 1.000 | 1.000 | 0.004 | 0.000 |

Pattern holds on all three families: locators/lessons → weights,
procedures/state facts → context.

**(ii) Routing rule** — fit coefs: `a_r=+7.633, a_s=−7.273, b=−0.300`
(recurrence pushes to weights, state-conditionality to context — the
paper's direction). **Held-out 24-cell recovery: 24/24.** Two
interventions move components toward the boundary: procedure with
recurrence boost |p−0.5| 0.463 → 0.140; lesson with state-conditionality
boost 0.485 → 0.339 (toward, not across — destinations did not flip).

Routing-by-rule vs whole-trajectory baselines (mean gain/item, held-out):

| rule | oracle | all→weights | all→context | random |
|---|---|---|---|---|
| **1.2393** | 1.2393 | 0.7437 | 0.7150 | 0.7293 |

Rule matches the per-item oracle exactly (100% of headroom over the best
single destination; margin +0.4956 in synthetic units).

**(iii) Mechanism toys** — note readout after weights-write (blend of m
recurring instances, exact note kept in context = 1.0):

| recurrence | 0.05 | 0.20 | 0.40 | 0.60 | 0.80 | 0.95 |
|---|---|---|---|---|---|---|
| writes m | 1 | 3 | 5 | 6 | 8 | 10 |
| readout quality | 1.0000 | 0.7896 | 0.7462 | 0.7308 | 0.7195 | 0.7120 |

Monotone decrease, worst for highest recurrence ✓. Context gain rises
linearly with the information gap (0 → 0.765 at retrieval quality 0.85);
weights gain falls monotonically with the policy gap (1.0 → 0.447).

**Verdict: SUPPORTED as a mechanism, with the synthetic doing heavy
lifting (see limits).** The two-property rule genuinely recovers
held-out destinations and beats whole-trajectory routing in this world;
the readout/gap toys move in the paper's directions.

## Honest limits

1. **Claim (i) is assumed, not discovered.** The gain equations were
   constructed so locators/lessons favor weights — the type pattern is
   baked in. What was actually *tested* is (ii): the rule generalizes to
   a held-out family — but even that is eased by construction, since all
   families share the same linear-boundary form with only modest scaling
   shifts. Real backbones could disagree more sharply.
2. **The synthetic is too clean.** Rule == oracle exactly (24/24,
   100% headroom) means classes separate with margin; the paper's +3.5
   pts is on noisy real data. Nothing here calibrates the effect size.
3. **The readout toy required a specific mechanism choice.**
   Formalizing recurring instances as *noisy copies of one truth* gave
   the opposite pattern (averaging helps); the claimed direction only
   appears under *diverse instances around a prototype* (the note's own
   contribution dilutes as more distinct instances blend in). The toy
   validates the claim under that interpretation, not in general.
4. **Gap toys are near-definitional** (info-gap curve is linear by
   construction). They are directional sanity checks, not evidence.
5. **Interventions moved toward but did not cross the boundary** —
   weaker than a flip demonstration.

## Integration note

`RoutingRule` is a candidate for DeepWorld's contribution loop: accepted
contributions are currently all written toward weights, but the paper
suggests routing per-component (stable locators/lessons → weights,
state-dependent procedures/facts → retrievable context). Wiring it into
the loop's accept step is a working-copy change — queued, needs Peter's
go-ahead like the harness integration. No secrets, no network, no model
calls were used anywhere in this session.
