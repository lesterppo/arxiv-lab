# Session 2026-10-13 — arXiv:2610.13144 "Competence-Aware Orchestration for LLM-Based Multi-Agent Systems"

**Article:** https://arxiv.org/abs/2610.13144
**Track:** agent · **AGENT 5 / TUNE 2** · **Verdict: SUPPORTED** (mechanism miniature)

## Why picked
Two-stage framework with a crisp, implementable claim: (1) model the
backbone's task-conditioned competence for the current query, (2) adapt
role activation and the communication graph to that signal. The paper
reports 87.22→89.97 accuracy with −36.26% token cost on higher-competence
backbones and +4.58 pts on lower-competence ones. Directly relevant to
local agent-harness engineering (role orchestration is the harness's
job) and a plausible future fit for deepworld's multi-agent contribution
loop. Chosen over LongHarness (2610.13075, AGENT 5) because the
accuracy/cost tradeoff claim is sharper and measurable without an LLM.

## What was built
`src/arxiv_lab/agent/competence.py` (new `arxiv_lab.agent` area) +
`tests/test_agent_cpu.py`:
- **Stage 1** — `CompetenceTracker`: online per-agent, per-task-type
  competence estimates (EWMA, α=0.08, conservative 0.5 prior); plus
  `query_competence()`, which conditions the estimate on a cheap
  query-difficulty proxy (the paper's "competence for the current query").
- **Stage 2** — `orchestrate()`: maps the query-conditioned competence to
  a mode (solo / proposer+verifier / 3-role deliberation), activates the
  top-estimated roles, and builds the communication graph
  (`communication_graph()`).
- `simulate_mas()`: seeded simulation with hidden ground-truth
  competence profiles ("high" and "low" backbone regimes), per-query
  difficulty, and a verifier that catches proposer errors at its own
  competence rate. Baselines: fixed solo, fixed full deliberation.

## Results (seeded, deterministic)
| regime | adaptive | solo | full | token ratio ad/full |
|---|---|---|---|---|
| high-competence | **0.925** | 0.837 | 0.937 | **0.44 (−56%)** |
| low-competence | **0.662** | 0.592 | 0.640 | 0.77 (−23%) |

- High backbone: adaptive beats fixed solo by **+8.8pp** (paper: 87.22→89.97),
  within 1.2pp of always-deliberate, at **44% of the token cost**
  (paper: −36.26%).
- Low backbone: adaptive beats solo by **+7.0pp** and fixed full by
  **+2.2pp** (paper: +4.58 over existing methods).
- Stage-1 calibration: mean |estimate − truth| for the top-ranked agent
  per task type = **0.019 (high) / 0.060 (low)** — the decision-relevant
  signal is well-calibrated.
- Mode usage adapts as claimed: high → 87 solo / 425 pair / 88 deliberate;
  low → 3 / 217 / 380.

All 7 CPU tests pass (`python3 tests/test_agent_cpu.py`).

## Honest limits
- **Simulated agents, not an LLM.** Outcomes are Bernoulli draws at
  hidden competence rates; real backbones have correlated errors and
  the difficulty proxy is noisier than modeled here.
- **Selection bias:** rarely-activated agent×task cells stay near the
  prior (standard bandit issue); calibration is reported for the
  top-ranked agents actually used in decisions. A production version
  would add ε-exploration.
- The verifier's catch-rate model (catches errors at its own competence
  rate) is a modeling choice, not measured from a real model.
- Not wired into the deepworld working copy: role activation would cut
  across `OmniTokV4Agent.act()`'s tool loop rather than plug into it;
  noted as a candidate, left for an explicit go-ahead.

## Files
- `src/arxiv_lab/agent/__init__.py`, `src/arxiv_lab/agent/competence.py`
  (graduated to the library)
- `tests/test_agent_cpu.py`
- This README
