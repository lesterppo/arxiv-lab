# How Should Planning Agents ReSHAPE Their Search Space?

- **Article:** arXiv [2611.16466](https://arxiv.org/abs/2611.16466) — "How Should Planning Agents ReSHAPE Their Search Space?" (submitted 12 Nov 2026)
- **Verdict: SUPPORTED** (directional reproduction at bounded scale; see "Why SUPPORTED, not VERIFIED")
- **Track:** agent track — CPU only, stdlib only, no network in the primary test
- **Code:** no public code link in the abstract; mechanism re-implemented from the paper's abstract-level task definition (ReSHAPE: judge a revision by whether its search space matches the acceptable plans and a planner finds one there)

## Why picked

- **AGENT 5 / TUNE 1.** The paper's lesson — treat a request as a hypothesis to test against the user's reactions to concrete plans — is a harness primitive for any planning agent: how a revision strategy reshapes the search space is exactly what a spec-revision / clarification loop should optimise. It maps directly onto the hermes agent's planning and spec-agent work (`src/arxiv_lab/agent/spec_agent.py`, `src/arxiv_lab/agent/reshape.py`): the choice between asking about preferences and showing candidate plans is a design decision every agent loop makes.
- **Repo fit:** `arxiv-lab` mechanism library — a paired comparison of two revision strategies (ASK vs SHOW) on the same instances is a reusable harness evaluation, complementary to yesterday's termination-lag stop-policy comparison.
- **Testable within budget:** the ReSHAPE task (revision quality = search-space match + planner success) maps onto a deterministic synthetic planning domain, so a faithful CPU proxy test is possible in one run.

## What was built

- `reshape.py` — the mechanism (stdlib only, deterministic):
  - Two synthetic domains, travel and shopping, each a 108-plan universe over 5 attributes. Each instance has a hidden user utility with ONE trade-off (a sacrificed dimension compensated by other attributes at their best levels), an acceptable set = the top 22 plans by that utility, and an initial spec S0 constraining 3 attributes to non-worst levels — so S0 excludes the trade-off plans.
  - Four revision strategies on the *same* instances with per-(instance, strategy) RNG streams (SplitMix64, same construction as the 2026-11-13 termination-lag session): `none` (S0 unrevised), `ask` (ask about each constrained attribute in the abstract; the user answers from stated marginal preferences — flexibility falls with the attribute's weight, so the trade-off's sacrifice dimension, which the user cares most about, is least likely to be relaxed; the agent relaxes the single most-flexible dimension, unconditionally), `show` (show 8 concrete plans from the single-relaxation neighbourhood of S0, learn accept/reject from the hidden utility, and fit the revision — relax dimension *d*, optionally requiring a condition on other attributes — that best explains those labels by F1), `oracle` (the one correct trade-off: S0 plus exactly the hidden trade-off plans; paper reference arm).
  - A budgeted sampling planner: draws K=2 plans uniformly without replacement from the revised space and succeeds iff any drawn plan is acceptable. Metrics: admission (|S∩A|/|A|), precision, F1 of the revised space vs the acceptable set, planner success rate.
- `test_reshape_cpu.py` — deterministic test, 13 pre-registered assertion groups (A1–A13); writes `results.json`.
- `rust/` — Rust second stage (standing rule, 2026-11-02): a line-for-line port of the mechanism (no external crates), `verify` binary mirroring the Python test's experiments and assertions, unit + deep + parity tests. See `rust/README.md`.

## Results

Main experiment (300 instances/domain, seed 20261114, planner K=2, show K=8):

| domain | strategy | admission | precision | F1 | planner success | mean \|S\| |
|---|---|---|---|---|---|---|
| travel | none | 0.448 | 0.438 | 0.440 | 0.673 | 22.5 |
| travel | ask | 0.533 | 0.365 | 0.424 | 0.583 | 33.3 |
| travel | show | 0.609 | 0.470 | 0.526 | 0.727 | 28.7 |
| travel | oracle | 0.754 | 0.483 | 0.583 | 0.747 | 35.2 |
| shopping | none | 0.458 | 0.436 | 0.443 | 0.670 | 23.1 |
| shopping | ask | 0.544 | 0.361 | 0.425 | 0.587 | 34.2 |
| shopping | show | 0.608 | 0.464 | 0.522 | 0.730 | 28.9 |
| shopping | oracle | 0.764 | 0.482 | 0.586 | 0.747 | 35.6 |

- The paper's headline reproduces in direction and, for the F1 gap, in size: showing plans beats asking by **+10.3 F1 points in travel** (0.526 vs 0.424) and **+9.7 points in shopping** (0.522 vs 0.425), against the paper's +9.7 points in travel. Planner success under SHOW is 1.25x ASK's (0.728 vs 0.585 pooled).
- The one correct trade-off admits **75.9%** of the acceptable plans pooled (paper: over 52%), vs 53.9% for ASK — the oracle arm confirms the trade-off is the single most valuable revision, and SHOW (60.9%) recovers most of the way there from concrete reactions alone.
- Search-space quality predicts what the planner finds: Pearson r = 0.893 between arm F1 and planner success across the 8 arm-domain points.
- Asking does not merely fail to find the trade-off — it actively dilutes the space: ASK's unconditional relaxation of the wrong dimension lowers precision (0.365 vs 0.438 unrevised in travel) and planner success (0.583 vs 0.673) *below* the unrevised baseline. Paired per-instance, SHOW rescues 132/600 instances the unrevised planner fails, and loses 98.
- Sensitivity: SHOW F1 rises with more shown plans (0.498 → 0.524 → 0.546 for K=4/8/16), while asking more questions does not help (ASK F1 0.435 at Q=1 vs 0.424 at Q=3) — concrete reactions scale, abstract questions do not. Planner success saturates with budget (SHOW 0.478 / 0.728 / 0.945 at K=1/2/4), which is why K=2 is the main budget.
- Robustness: SHOW F1 > ASK F1 in both domains at bonus fractions 0.15 and 0.35 as well as the main 0.25, so the ordering is not bonus calibration luck.

## Why SUPPORTED, not VERIFIED

- The user is a rule-based stand-in, not a model: acceptability is a drawn utility with one explicit trade-off, not a real user's judgments; the paper's simulated user is human-validated on real planning benchmarks (travel/shopping). The revision-strategy comparison and the search-space/planner accounting are the reproduced mechanism; the user side is a model of a user.
- Absolute admission levels differ from the paper's: our ASK admits 53.9% of acceptable plans pooled, vs the paper's "at most 32%" for current agents — our ASK is a generous one (it always relaxes the most flexible dimension when any flexibility is stated), while real agents' revisions are mostly refinements that never expand toward the trade-off. The paper's 4.6x planner lift is also a max over their real instances; ours is a mean lift of 1.25x (SHOW over ASK) and 1.11x (oracle over unrevised). The reproduced claims are the *ordering and the mechanism* (showing beats asking by ~10 F1 points; the trade-off alone admits most of A; space quality predicts planner success), not the 32%/52%/4.6x figures exactly.
- The miniature's planner is budgeted uniform sampling (K=2), not the paper's planner on real benchmarks; absolute planner success rates are miniature-scale (600 synthetic instances, 108-plan universes).
- The trade-off bonus fraction (0.25 of max base utility) is a modelling choice calibrated so S0 admits ~45% of A and the oracle ~76%; the paper does not publish an equivalent.

## Limits

- One trade-off per instance, one sacrifice dimension, 1–2 compensating attributes: real requests have several interacting trade-offs; the miniature's SHOW fit searches single-dimension relaxations only, so it cannot discover multi-dimensional trade-offs.
- The shown plans are drawn from the single-relaxation neighbourhood of S0 (nearby candidates a planner could produce); a real agent's shown plans come from its own planner and may be less informative.
- ASK's failure is partly by construction: stated flexibility is defined to fall with marginal weight, so the high-weight sacrifice dimension is rarely relaxed. The mechanism (marginal questions cannot reveal interactions) is the paper's, but the exact flexibility curve is a modelling choice.
- No model-backed validation: the mechanism was not re-run with an LLM agent (e.g. via NVIDIA NIM), so the quantitative gaps (ASK admission 54% vs the paper's ≤32%) are miniature-calibration differences, not model measurements.

## Files

- `reshape.py` — mechanism (stdlib only)
- `test_reshape_cpu.py` — deterministic CPU test (23 assertions, 23 passed)
- `results.json` — main experiment + sensitivity sweeps + assertion list
- `rust/` — Rust second stage
- Graduated: `src/arxiv_lab/agent/reshape.py` + `tests/test_agent_reshape_cpu.py` (same mechanism, same assertions; 9/9 passed)
