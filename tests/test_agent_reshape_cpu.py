"""CPU test for ReSHAPE search-space revision (arXiv:2611.16466), graduated
from the 2026-11-14 daily session. Deterministic; stdlib only.

Re-runs the session's core pre-registered assertions against the graduated
module `arxiv_lab.agent.reshape` (same mechanism, same seed and instance
count), so the library copy cannot drift from the session copy.

Run:  python3 tests/test_agent_reshape_cpu.py
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from arxiv_lab.agent.reshape import (
    RNG, STRATEGIES, gen_instances, plan, revise, revise_oracle,
    run_experiment, space_metrics, _strategy_seed,
)

SEED = 20261114
N_INSTANCES = 300

PASSED = []
FAILED = []


def check(name, cond, detail=""):
    (PASSED if cond else FAILED).append(name)
    print(f"  [{'PASS' if cond else 'FAIL'}] {name} {detail}")


def main():
    results = {}
    instances = {}
    for dom in ("travel", "shopping"):
        instances[dom] = gen_instances(dom, N_INSTANCES, SEED)
        results[dom] = run_experiment(dom, instances[dom], SEED)
    print(f"main experiment ({N_INSTANCES} instances/domain, seed {SEED}):")
    for dom in ("travel", "shopping"):
        for s in STRATEGIES:
            m = results[dom][s]
            print(f"  {dom:9s} {s:7s} adm={m['admission']:.3f} "
                  f"f1={m['f1']:.3f} succ={m['planner_success']:.3f}")

    print("assertions:")
    tr, sh = results["travel"], results["shopping"]
    check("A1 travel F1_SHOW - F1_ASK >= 0.05",
          tr["show"]["f1"] - tr["ask"]["f1"] >= 0.05,
          f"got {tr['show']['f1'] - tr['ask']['f1']:.3f}")
    check("A2 travel planner SHOW > ASK, lift >= 1.15",
          tr["show"]["planner_success"] >= 1.15 * tr["ask"]["planner_success"],
          f"lift {tr['show']['planner_success'] / tr['ask']['planner_success']:.2f}x")
    check("A3 shopping F1_SHOW > F1_ASK",
          sh["show"]["f1"] > sh["ask"]["f1"],
          f"got {sh['show']['f1']:.3f} vs {sh['ask']['f1']:.3f}")
    check("A4 pooled ORACLE admission > 0.52",
          sum(results[d]["oracle"]["admission"] for d in results) / 2 > 0.52)
    check("A5 pooled planner: ASK < NONE < SHOW",
          sum(results[d]["ask"]["planner_success"] for d in results) / 2
          < sum(results[d]["none"]["planner_success"] for d in results) / 2
          < sum(results[d]["show"]["planner_success"] for d in results) / 2)
    inst0 = instances["travel"][0]
    oracle_space = revise_oracle(inst0)
    check("A6 oracle admits every trade-off plan",
          all(i in oracle_space for i, p in enumerate(inst0.plans)
              if inst0.tradeoff_holds(p)))
    m0 = space_metrics(inst0, inst0.spec0)
    inter = len(inst0.spec0 & inst0.acceptable)
    check("A7 space_metrics matches hand computation",
          abs(m0["admission"] - inter / len(inst0.acceptable)) < 1e-12
          and abs(m0["precision"] - inter / len(inst0.spec0)) < 1e-12)
    r1, r2 = RNG(12345), RNG(12345)
    check("A8 RNG streams reproduce exactly",
          [r1.next_u64() for _ in range(5)] == [r2.next_u64() for _ in range(5)])
    m2 = {dom: run_experiment(dom, instances[dom], SEED)
          for dom in ("travel", "shopping")}
    check("A9 same seed -> identical metrics on re-run", m2 == results)

    print(f"\n{len(PASSED)} passed, {len(FAILED)} failed")
    if FAILED:
        print("FAILED:", FAILED)
        sys.exit(1)


if __name__ == "__main__":
    main()
