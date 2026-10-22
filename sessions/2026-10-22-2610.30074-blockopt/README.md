# Session 2026-10-22 — 2610.30074 — Parallel Agent Collaboration as Graph-Structured Distributed Block Optimization

- **Article:** https://arxiv.org/abs/2610.30074 — "Parallel AI Agent Collaboration as Graph-Structured Distributed Block Optimization"
- **Track:** agent (CPU, deterministic)
- **Why picked:** AGENT=5/TUNE=1. A coordination mechanism for parallel agents with a provable merge guarantee — directly reusable in a local multi-agent harness (arxiv-lab multi-agent line; hermes agent orchestration). Testable end-to-end on CPU within the run budget, unlike heavier harness-evolution papers in the window (HGM-Dual, SSR, VeriEvolve).
- **Verdict: VERIFIED** — core claims reproduced end-to-end with measured numbers.

## What was built

`src/arxiv_lab/multiagent/block_opt_merge.py` (stdlib + numpy, numpy-guarded per repo convention):

- `BlockObjective` — exactly evaluable discrete objective `f(x) = Σ c_i x_i + Σ w_ij [x_i == x_j]` with a known dependency graph G; edits, verified local gains measured against the current shared solution.
- `conflict_graph` — proposals conflict iff their supports share a vertex or are joined by an edge of G.
- Exact fractional chromatic number χ_f(H) via a compact **two-phase simplex** (dense tableau, Bland's rule; artificials purged after Phase I — a zero-cost artificial re-entering the basis would mask infeasibility, caught during testing).
- `fractional_merger` — merges via the max-weight class of an optimal fractional coloring; `collaborate` — repeated collaboration loop.
- `tests/test_multiagent_blockopt_cpu.py` — 6 tests, all pass.

## Results (measured)

| Claim | Test | Result |
|---|---|---|
| Independent sets certify compatible edits; gains add exactly | 12 random graphs × maximal independent set | exact additivity, error < 1e-9 |
| χ_f computed exactly | C5=2.5, C7=2⅓, K4=4.0, empty=1.0, edge=2.0 | all match analytic values |
| Merger preserves ≥ 1/χ_f of total verified gain | 30 random conflict graphs | holds on all 30; worst preserved fraction 0.512 |
| Bound tight among subset mergers | C5, 5 equal-gain proposals | preserved = 2/5 = 1/χ_f exactly; no independent subset beats 2 |
| Naive merging can fail where the merger succeeds | 2-agent conflict demo | naive: f 1→1 (no gain); merger: +11 verified |
| Repeated collaboration contracts gap geometrically | 6 seeds, single-flip oracle, brute-force optimum | gap_{t+1} ≤ (1−ρ̂/χmax)·gap_t on all active rounds; measured rates 0.00–0.68/round; seeds 0/4/5 reached gap 0 |

## Honest limits

- Synthetic exactly-evaluable objectives (Ising-like), not the paper's code-repair / road-network / route-search deployments.
- The single-flip proposal oracle stalls at local optima (seeds 1–3 left residual gaps 0.38–3.24 after 1 active round); geometric convergence is verified conditional on the paper's local-improvement premise, which the test measures (ρ̂) rather than assumes.
- Exact fractional coloring enumerates maximal independent sets (exponential worst case) — fine for small conflict graphs; the paper's large-scale deployments use approximations.
- Verified gains are measured against the same current solution per round, as the paper requires; cross-round staleness is not modeled.
