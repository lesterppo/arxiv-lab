#!/usr/bin/env python3
"""CPU test for MeterRouter (arXiv 2611.34327) -- deterministic, stdlib only.

E1  Unit: cost meter (Eq. 1) with the paper's Table 6 prices; remaining
    budget (Eq. 2); utility (Eq. 5).
E2  Unit: Appendix E graph templates (edge counts per template).
E3  Budget compliance: the router's overrun rate is 0 at every budget
    (paper: in-budget rate >= 99.98%).
E4  Main prediction (B = 5e-2, the paper's main budget): the router is
    within 5 points of the strongest MAS in accuracy, at <= 60% of its
    cost, and beats the best single agent by >= 3 points.
    (Paper: 66.72% vs 66.80% at 82% lower cost vs MasRouter, +3.7 pts
    over the best single agent. Thresholds here are looser than the
    paper's, because this is a synthetic environment, not the paper's
    benchmarks.)
E5  Budget adaptation: mean cost and accuracy rise with B, with
    diminishing returns (the 5e-3 -> 5e-2 accuracy gain is smaller
    than the 1e-3 -> 5e-3 gain).
E6  Ablation (paper's w/o RL: random LLM selection drops accuracy
    66.72% -> 57.34% at the same cost): the random-tier router loses
    >= 3 points to the budget-aware router at every budget, at
    similar or lower cost.
E7  Budget matters more on hard queries than on easy ones (paper
    Fig. 4): the router's accuracy gain from B = 1e-3 to B = 5e-2 is
    larger on hard queries than on easy ones.
E8  Determinism: identical headline numbers across repeated runs.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from arxiv_lab.agent.meter_router import (  # noqa: E402
    Agent,
    BUDGETS,
    call_cost,
    expected_team_success,
    greedy_router,
    make_queries,
    remaining_budget,
    run_experiment,
    template_graph,
    utility,
    _simulate_team,
)
import random  # noqa: E402

FAILURES = []


def check(name, cond, detail=""):
    print(f"{'PASS' if cond else 'FAIL'}  {name}  {detail}")
    if not cond:
        FAILURES.append(name)


# ---------------------------------------------------------------- E1
# Low tier (gpt-oss-20b, 0.06 in / 0.19 out): 2000 in + 1200 out tokens.
c = call_cost(0, 0, "solver")
check("E1 cost low solver", abs(c - (0.06 * 2000 + 0.19 * 1200) / 1e6) < 1e-12,
      f"cost={c:.6f}")
# High tier (MiniMax-M3, 0.30 in / 1.20 out), 2 parents, inspector out 700.
c = call_cost(2, 2, "inspector")
check("E1 cost high inspector 2 parents",
      abs(c - (0.30 * 3000 + 1.20 * 700) / 1e6) < 1e-12, f"cost={c:.6f}")
check("E1 remaining budget",
      abs(remaining_budget(5e-3, [1e-3, 2e-3]) - 2e-3) < 1e-15)
check("E1 utility correct", abs(utility(True, 1e-3, 5e-3) - (1 - 0.5 * 0.2)) < 1e-12)
check("E1 utility wrong", abs(utility(False, 1e-3, 5e-3) - (-0.5 * 0.2)) < 1e-12)
check("E1 utility gamma", abs(utility(True, 1e-3, 5e-3, gamma=1.0) - 0.8) < 1e-12)

# ---------------------------------------------------------------- E2
check("E2 single", template_graph("single", 1) == [()])
check("E2 chain 3", template_graph("chain", 3) == [(), (0,), (1,)])
check("E2 fan-in 3", template_graph("fan-in", 3) == [(), (), (0, 1)])
check("E2 dense 4 edges",
      sum(len(p) for p in template_graph("dense", 4)) == 6)
check("E2 diamond 4", template_graph("diamond", 4) == [(), (0,), (0,), (1, 2)])
check("E2 chain 5 edges",
      sum(len(p) for p in template_graph("chain", 5)) == 4)
check("E2 dense 5 edges",
      sum(len(p) for p in template_graph("dense", 5)) == 10)

# ------------------------------------------------------- headline run
N = 500
SEED = 20261128
results = run_experiment(n_queries=N, seed=SEED)
router = {b: results[b]["meterrouter"] for b in BUDGETS}
strongest = {b: results[b]["strongest"] for b in BUDGETS}
single_high = {b: results[b]["single_high"] for b in BUDGETS}
random_tier = {b: results[b]["random_tier"] for b in BUDGETS}

# ---------------------------------------------------------------- E3
for b in BUDGETS:
    check(f"E3 overrun B={b:g}", router[b].overrun_rate == 0.0,
          f"overrun={router[b].overrun_rate:.3f}")

# ---------------------------------------------------------------- E4
main = 50e-3
check("E4 acc within 5pts of strongest",
      router[main].accuracy >= strongest[main].accuracy - 0.05,
      f"router={router[main].accuracy:.3f} strongest={strongest[main].accuracy:.3f}")
check("E4 cost <= 60% of strongest",
      router[main].mean_cost <= 0.6 * strongest[main].mean_cost,
      f"router={router[main].mean_cost * 1e3:.2f}e-3 "
      f"strongest={strongest[main].mean_cost * 1e3:.2f}e-3")
check("E4 beats best single agent by >= 3pts",
      router[main].accuracy >= single_high[main].accuracy + 0.03,
      f"router={router[main].accuracy:.3f} "
      f"single_high={single_high[main].accuracy:.3f}")

# ---------------------------------------------------------------- E5
costs = [router[b].mean_cost for b in BUDGETS]
accs = [router[b].accuracy for b in BUDGETS]
check("E5 cost rises with budget",
      all(costs[i] <= costs[i + 1] for i in range(len(costs) - 1))
      and costs[-1] > costs[0] and costs[2] > costs[0],
      f"costs={[round(c * 1e3, 2) for c in costs]}")
check("E5 accuracy rises with budget",
      all(accs[i] <= accs[i + 1] for i in range(len(accs) - 1)),
      f"accs={[round(a, 3) for a in accs]}")
gain_lo = accs[2] - accs[0]
gain_hi = accs[3] - accs[2]
check("E5 diminishing returns", gain_hi < gain_lo,
      f"gain 1e-3->5e-3={gain_lo:.3f} 5e-3->5e-2={gain_hi:.3f}")

# ---------------------------------------------------------------- E6
for b in BUDGETS:
    check(f"E6 ablation loss B={b:g}",
          random_tier[b].accuracy <= router[b].accuracy - 0.03,
          f"router={router[b].accuracy:.3f} "
          f"random={random_tier[b].accuracy:.3f}")
    check(f"E6 ablation cost similar B={b:g}",
          random_tier[b].mean_cost <= router[b].mean_cost * 1.2 + 1e-9,
          f"random cost={random_tier[b].mean_cost * 1e3:.2f}e-3 "
          f"router={router[b].mean_cost * 1e3:.2f}e-3")

# ---------------------------------------------------------------- E7
queries = make_queries(N, SEED)
gains = {}
for diff in ("easy", "hard"):
    idx = [i for i, q in enumerate(queries) if q.difficulty == diff]
    acc_by_b = {}
    for b in (1e-3, 50e-3):
        corr = 0
        for qi in idx:
            q = queries[qi]
            rng = random.Random(SEED * 1000003 + qi)
            team, _ = greedy_router(q, b, rng)
            if team and _simulate_team(team, q, rng):
                corr += 1
        acc_by_b[b] = corr / len(idx)
    gains[diff] = acc_by_b[50e-3] - acc_by_b[1e-3]
check("E7 hard gain > easy gain", gains["hard"] > gains["easy"],
      f"hard gain={gains['hard']:.3f} easy gain={gains['easy']:.3f}")

# ---------------------------------------------------------------- E8
results2 = run_experiment(n_queries=N, seed=SEED)
same = all(
    abs(results[b]["meterrouter"].accuracy
        - results2[b]["meterrouter"].accuracy) < 1e-12
    and abs(results[b]["meterrouter"].mean_cost
            - results2[b]["meterrouter"].mean_cost) < 1e-15
    for b in BUDGETS)
check("E8 determinism", same)

# ---------------------------------------------------------------- summary
print(f"\nheadline (n={N}, seed={SEED}):")
for b in BUDGETS:
    r = router[b]
    print(f"  B={b:g}: router acc={r.accuracy:.3f} cost={r.mean_cost * 1e3:.2f}e-3 "
          f"agents={r.mean_agents:.2f} overrun={r.overrun_rate:.3f} stops={r.stops}")
print(f"  strongest B=5e-2: acc={strongest[main].accuracy:.3f} "
      f"cost={strongest[main].mean_cost * 1e3:.2f}e-3")
print(f"  single_high B=5e-2: acc={single_high[main].accuracy:.3f}")

if FAILURES:
    print(f"\n{len(FAILURES)} FAILURES: {FAILURES}")
    sys.exit(1)
print("\nALL CHECKS PASSED")
