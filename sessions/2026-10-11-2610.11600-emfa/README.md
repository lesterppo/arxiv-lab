# Session 2026-10-11 — EMFA: Error-Propagation Modeling for Failure Attribution (arXiv:2610.11600)

**Article:** "Error-Propagation Modeling for Failure Attribution in LLM-Based
Multi-Agent Systems" ([abs](https://arxiv.org/abs/2610.11600))

**Why picked:** most prospective *agent* article in the 2026-10-08 cs.AI
window (AGENT 5/5) that is genuinely new vs the tested set. The decisive-error
formulation (the (agent, step) whose correction would recover the execution)
and the two-branch mechanism — cascading propagation tracing plus persistent
interaction loops, then counterfactual verification — are not covered by any
tested module: winner's-curse (selection bias), DAEDALUS (memory generation),
CMP (consolidation), reversible forgetting, IEC (intent checks), Sentry
(recovery), BDA (council). Direct repo fit: `deepworld` (multi-agent sim —
failure attribution over its contribution/negotiation trajectories) and
`arxiv-lab` as the mechanism library.

**Paper claims:** (i) the decisive error is the (agent, step) pair whose
correction recovers the failed execution; (ii) modeling cascading propagation
and persistent loops separates decisive errors from downstream carriers and
recovered deviations; (iii) counterfactual verification + earliest-decisive
rule identifies it; (iv) SOTA step-level attribution on Who&When (+3.45 /
+4.40pp over previous best).

## What was built

- Reusable mechanism: `src/arxiv_lab/attribution/emfa.py` (new `attribution`
  area) — `find_final_failure`, `build_dependencies` (H = step/subtask/
  producer-consumer/subtask-reuse), `backward_trace` (Ginfo),
  `annotate_propagation` (introduce / carry / correct / clean),
  `extract_loops` (Gloop: >=3 same-(agent, subtask, action) steps with
  identical taint = no progress), `screen_candidates` (dual branch, <=2 each,
  <=4 total: earliest *unrecovered* deviations + loop initiators, with the
  paper's ambiguous-instruction rule), `counterfactual_verify` (repair each
  candidate chronologically; earliest recovery wins), `emfa_attribute`
  (full pipeline), plus three naive baselines (`blame_final`,
  `blame_latest_wrong`, `blame_earliest_wrong`) and five seeded scenario
  builders (`make_cascade`, `make_recovered`, `make_loop`,
  `make_premature`, `make_redundant`).
  Test: `tests/test_attribution_cpu.py` — 22/22 pass (stdlib only).
- Session record: `test_session_emfa.py` (verbatim snapshot) +
  `results_emfa.json` (240 trajectories).

## Results (CPU, deterministic; 240 trajectories, 6 scenario configs x 40)

| method | step-level acc | agent-level acc |
|---|---|---|
| EMFA | **1.000** | **1.000** |
| blame_final | 0.167 | 0.167 |
| blame_latest_wrong | 0.000 | 0.000 |
| blame_earliest_wrong | 0.667 | 0.667 |

Per config (EMFA step acc): cascade 1.00, recovered 1.00, loop 1.00,
loop-noamb 1.00, premature 1.00, redundant 1.00. The recovered-deviation and
premature-stop configs are where every isolation baseline fails (0.00) and
only propagation-aware screening + counterfactual verification recover the
decisive pair; the redundant-path config confirms the earliest-decisive
tie-break.

## Verdict

**Supports the paper's mechanism claim in simulation.** Propagation-aware
screening (excluding recovered deviations and carriers) plus chronological
counterfactual verification identifies the decisive (agent, step) pair on
every trajectory (+33.3pp over the best isolation baseline), mirroring the
paper's direction of SOTA step-level gains from modeling propagation
explicitly. Absolute magnitudes are not comparable to Who&When (synthetic
world, known ground truth).

## Honest limits

> **Simulation honesty box.** The paper's LLM roles (dependency extraction,
> propagation annotation, loop identification, screening and verification
> judgments) are played here by deterministic rules over an explicit
> taint-tag world model (`simulate` / `repair_and_resimulate`). This tests
> the *attribution protocol*, not LLM judgment quality. Ground-truth
> decisive pairs are known by construction and validated by an oracle
> (repairing the GT step always recovers; it is the earliest such step).
> Real trajectories have ambiguous dependencies and no clean counterfactual;
> the 33.3pp margin will shrink there. One seed family; no real LLM agents
> in the loop (no local LLM on this VM).
