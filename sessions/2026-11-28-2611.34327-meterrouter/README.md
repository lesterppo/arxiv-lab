# MeterRouter: Budget-Aware Dynamic Routing for Multi-Agent Systems

**Article:** [arXiv:2611.34327](https://arxiv.org/abs/2611.34327) — Xu, Yin, Min, Lai, Johnson, Luan, Shen (submitted 25 Nov 2026).
**Date:** 2026-11-28 (additional run, second run of the day — the first
daily run tested DART 2611.34928 and skill-composition 2611.34398).
**Scores:** AGENT 5 / TUNE 2. Picked as the agent track because it is a
routing mechanism for a local harness, directly testable on CPU with a
deterministic mechanism, and its claims are quantitative and falsifiable.

## Paper in one line

A multi-agent system is built incrementally — one agent at a time,
choosing each agent's role, LLM backbone, and incoming edges —
conditioned on the query and a per-query monetary budget, with a budget
meter tracking the remaining budget after every call. Across five
benchmarks and a three-tier LLM pool, MeterRouter matches the strongest
MAS baseline in accuracy (66.72% vs 66.80%) at 82% lower cost
($2.56e-3 vs $14.15e-3 per query), lies on the accuracy–cost Pareto
frontier, adapts its spending to the budget, and stays in budget on
≥99.98% of queries. Its RL stage is load-bearing: random LLM selection
(w/o RL) drops accuracy 66.72% → 57.34% at the same cost.

## What was built

- `src/arxiv_lab/agent/meter_router.py` — the graduated mechanism:
  Table 6 prices, cost (Eq. 1), remaining budget (Eq. 2), utility
  (Eq. 5), N_max = 5, budgets {1, 4, 5, 50}×10⁻³, Appendix E graph
  templates, and a greedy budget-aware router over the same decision
  space (role, backbone, edges, termination).
- `tests/test_meterrouter_cpu.py` — deterministic CPU test, E1–E8,
  all checks passed (n = 500, seed 20261128).
- `results.json` — canonical run numbers (n = 500, seed 20261128).
- `rust/` — Rust second stage (std-only port): `src/lib.rs`,
  `src/bin/verify.rs`, `tests/meter_router_tests.rs`,
  `tests/parity_cases.txt` (1140 cases from `parity_dump.py`),
  `results.json`. `cargo test --release`: 2/2 passed (parity 0
  mismatches). `cargo run --release --bin verify`: all checks passed.
- `parity_dump.py` — generates `rust/tests/parity_cases.txt`.

## Results (synthetic environment, n = 500, seed 20261128)

The router is a greedy optimiser over the paper's decision space; the
"LLMs" are a synthetic latent-threshold environment (see Honest limits).
Baselines are run on the same queries with paired draws.

| Budget | Router acc | Router cost | Router agents | Strongest MAS acc | Strongest cost | Single high acc |
| --- | --- | --- | --- | --- | --- | --- |
| $1×10⁻³ | 0.560 | $0.89×10⁻³ | 1.00 | 0.798 | $10.74×10⁻³ | 0.714 |
| $4×10⁻³ | 0.714 | $2.04×10⁻³ | 1.00 | 0.798 | $10.74×10⁻³ | 0.714 |
| $5×10⁻³ | 0.714 | $2.04×10⁻³ | 1.00 | 0.798 | $10.74×10⁻³ | 0.714 |
| $50×10⁻³ | **0.782** | **$6.24×10⁻³** | 3.00 | 0.798 | $10.74×10⁻³ | 0.714 |

- **Main prediction (B = $50×10⁻³):** router 0.782 vs strongest 0.798
  (−1.6 pts, within the 5-pt bar) at $6.24×10⁻³ vs $10.74×10⁻³
  (42% lower cost), and +6.8 pts over the best single agent (0.714).
  The paper's numbers are 66.72% vs 66.80% at 82% lower cost vs
  MasRouter; the direction and rough magnitude match, on a synthetic
  environment rather than the paper's benchmarks.
- **Budget compliance:** router overrun rate 0.000 at every budget
  (paper: in-budget rate ≥ 99.98%).
- **Budget adaptation:** mean cost $0.89 → $2.04 → $6.24×10⁻³ and
  accuracy 0.560 → 0.714 → 0.782 as B rises; gains diminish
  (+0.154 from $1→5×10⁻³, +0.068 from $5→50×10⁻³).
- **Ablation (random tier, the paper's w/o RL analogue):** loses
  21.0 pts at $1×10⁻³, 7.8 pts at $4×10⁻³, 6.6 pts at $5×10⁻³,
  8.8 pts at $50×10⁻³, at similar or lower cost (paper: −9.38 pts).
- **Difficulty:** the router's accuracy gain from B = $1×10⁻³ to
  $50×10⁻³ is +0.242 on hard queries vs +0.209 on easy ones
  (paper: the budget matters more on hard problems).

Rust leg (Rust-native SplitMix64 worlds, same properties, different
sampled values): router 0.764 vs strongest 0.780 at B = $50×10⁻³,
cost $6.24×10⁻³ vs $10.74×10⁻³; ablation losses 23.2 / 8.8 / 7.4 /
8.4 pts; hard gain +0.245 vs easy +0.162; determinism identical.

## Verdict

**SUPPORTED** — the mechanism's predicted trade-offs all hold in the
synthetic environment: near-strongest accuracy at much lower cost,
zero budget overruns, spending that adapts to the budget, a large
ablation loss from random tier assignment, and a bigger budget gain
on hard queries. Not VERIFIED: the learned policy (imitation + GRPO)
and the real benchmarks were not tested — the router here is a greedy
optimiser, and the environment is synthetic.

## Honest limits

- The paper's router is a *learned* policy (imitation + GRPO) over
  real LLM calls; this reproduction is a greedy optimiser over the
  same decision space, and the "LLMs" are a synthetic
  latent-threshold environment (per-role/tier solve probabilities,
  hint bonus from predecessor roles, fixed token counts). Absolute
  accuracies and costs are therefore not the paper's benchmark
  numbers; only the relative predictions were tested.
- γ = 0.5 in Eq. (5) is a simulation choice (the paper does not state
  its γ). γ = 1 makes the greedy router decline to build a
  multi-agent team at all, contradicting the paper's observed team
  sizes (1.22 agents at B = $5×10⁻³ rising to 3.13 at B = $50×10⁻³);
  γ = 0.5 gives 1.00 agents at B = $5×10⁻³ and 3.00 at B = $50×10⁻³.
- Token counts (2000 base input + 500 per parent, 700–1500 output
  by role) are simulation choices at the paper's cost scale, not
  measured from the paper's runs.
- No live-LLM leg: this VM has no NVIDIA API key and the NIM client
  from the earlier DART run is gone from /tmp, so no model-backed
  validation was possible for this track today.
- The fixed-topology and single-backbone ablations (Appendix D.2/D.3)
  were not reproduced; only the random-tier (w/o RL) ablation was.
