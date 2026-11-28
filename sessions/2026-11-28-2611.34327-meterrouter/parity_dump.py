#!/usr/bin/env python3
"""Generate rust/tests/parity_cases.txt: router decisions dumped from
Python plus the Python-computed values the Rust port must match.

Line types (one case per line, `#` lines are comments):
  Q <difficulty> <budget> <mode> <tier_draws...> | TEAM <agent;...>
      STOP <stop> EXP <expected success> COST <cost>
      Router case: mode is "aware" (budget-aware) or "random" (random
      tier). tier_draws are the exogenous tier draws in step order
      (empty for aware mode). TEAM lists agents as
      role,tier,parents (parents comma-joined, empty for none),
      agents separated by ";". Rust must reproduce the team and stop
      exactly, the cost exactly, and the expected success within 1e-6
      (Rust uses an erf approximation; see rust/src/lib.rs).
  E <role> <tier> <difficulty> <n_predecessors> | EXP <expected>
      Expected-success case for a single-agent team.
  C <tier> <n_parents> <role> | COST <cost>
      Call-cost case (Eq. 1, Table 6 prices).
Difficulties are strings; budgets and floats are dumped with repr()
so Rust parses the identical f64.
"""
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from arxiv_lab.agent.meter_router import (  # noqa: E402
    BUDGETS,
    ROLES,
    Agent,
    call_cost,
    expected_team_success,
    greedy_router,
    make_queries,
)

OUT = Path(__file__).with_name("rust").joinpath("tests", "parity_cases.txt")


def f(x):
    return repr(float(x))


def team_s(team):
    if not team:
        return "-"
    return ";".join(
        "%s,%d,%s" % (a.role, a.tier, ",".join(str(p) for p in a.parents))
        for a in team)


def main():
    n_queries = int(sys.argv[1]) if len(sys.argv) > 1 else 120
    lines = []
    lines.append("# parity cases for the MeterRouter Rust port (arXiv:2611.34327)")
    lines.append("# Q: router case | E: expected-success case | C: call-cost case")

    # ---- call-cost cases: every tier x parents x role ----
    for tier in range(3):
        for n_parents in (0, 1, 4):
            for role in range(5):
                lines.append("C %d %d %d | COST %s" % (
                    tier, n_parents, role, f(call_cost(tier, n_parents, ROLES[role]))))

    # ---- expected-success cases: single agents ----
    for tier in range(3):
        for role in range(5):
            for diff in ("easy", "medium", "hard"):
                for n_pred in (0, 2, 4):
                    team = [Agent(ROLES[role], tier, tuple(range(n_pred)))]
                    lines.append("E %d %d %s %d | EXP %s" % (
                        role, tier, diff, n_pred,
                        f(expected_team_success(team, diff))))

    # ---- router cases: seeded queries x budgets x modes ----
    queries = make_queries(n_queries, 20261128)
    for qi, q in enumerate(queries):
        for b in BUDGETS:
            for mode in ("aware", "random"):
                if mode == "aware":
                    team, stop = greedy_router(q, b, random.Random(0))
                    draws = []
                else:
                    tier_rng = random.Random(20261128 * 7919 + qi)
                    draws = [tier_rng.randrange(3) for _ in range(5)]
                    team, stop = greedy_router(
                        q, b, random.Random(0), random_tier=True,
                        tier_rng=random.Random(20261128 * 7919 + qi))
                exp = expected_team_success(team, q.difficulty)
                cost = sum(call_cost(a.tier, len(a.parents), a.role)
                           for a in team)
                lines.append("Q %s %s %s %s | TEAM %s STOP %s EXP %s COST %s" % (
                    q.difficulty, f(b), mode, " ".join(str(d) for d in draws),
                    team_s(team), stop, f(exp), f(cost)))

    OUT.write_text("\n".join(lines) + "\n")
    print(f"wrote {OUT} ({len(lines) - 2} cases)")


if __name__ == "__main__":
    main()
