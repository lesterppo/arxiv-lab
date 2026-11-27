#!/usr/bin/env python3
"""CPU test for Partial Routing (arXiv 2611.33402) -- deterministic,
stdlib only.

E1  Core claim (delta=0.75, held-out tasks): partial routing beats
    lead-alone AND task-level routing on task success, at lower cost
    than lead-alone.
E2  Delta sweep {0, 0.25, 0.5, 0.75, 1.0}: cost strictly decreases
    with delta; success peaks at an interior delta.
E3  Breakdown-only ablation: lead + subtask breakdown, no worker,
    costs less than lead-alone and does not lose success.
E4  Guidance transfer: guidance learned on past-run set A applied
    to held-out set B vs the fixed easy-type rule (delegate
    summarize + test). HONEST NEGATIVE (see README): the fixed rule
    wins on set B -- reported, not asserted as a win.
E5  Determinism: identical headline numbers across repeated runs
    and across PYTHONHASHSEED 1/2/3.
E6  Oracle: brute-force best per-subtask assignment on a small task
    set; partial routing reaches most of the oracle's success at
    lower cost. Plus unit checks (delegation sets, load penalty,
    oracle assignment on a hand-built task).
"""
import os
import random
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from arxiv_lab.agent.partial_routing import (  # noqa: E402
    analytic_expected,
    delegate,
    execute,
    fixed_easy_assignment,
    learn_guidance,
    make_tasks,
    oracle_assignment,
    past_traces,
    random_assignment,
    run_policy,
    task_level_assignment,
    LEAD_COST,
    SUBTASK_TYPES,
    WORKER_COST,
)

PAST_SEED = 11
N_PAST = 4000
HELDOUT_SEED = 101
HELDOUT_B_SEED = 202
ORACLE_SEED = 303
N_TASKS = 60
N_ORACLE = 12
TRIALS = 400
MC_SEED = 5
fails = []


def check(name, cond, detail=""):
    print(("ok " if cond else "FAIL ") + name + (f" [{detail}]" if detail else ""))
    if not cond:
        fails.append(name)


# ------------------------------------------------------- guidance (units)
traces = past_traces(PAST_SEED, N_PAST)
g = learn_guidance(traces)
print(f"measured gaps: {{{', '.join(f'{t}: {g.gap[t]:.3f}' for t in SUBTASK_TYPES)}}} "
      f"gmax={g.gap_max:.3f} n_traces={g.n_traces}")
check("guidance gap order is plan > code > verify > test > summarize",
      g.gap["plan"] > g.gap["code"] > g.gap["verify"] > g.gap["test"]
      > g.gap["summarize"],
      f"{ {t: round(g.gap[t], 3) for t in SUBTASK_TYPES} }")
check("measured costs are lead 10 / worker 1 per subtask",
      all(g.lead_cost[t] == LEAD_COST for t in SUBTASK_TYPES)
      and all(g.worker_cost[t] == WORKER_COST for t in SUBTASK_TYPES))
expected_sets = {0.0: (), 0.25: ("summarize",),
                 0.5: ("test", "summarize"),
                 0.75: ("verify", "test", "summarize"),
                 1.0: ("plan", "code", "verify", "test", "summarize")}
for delta, want in expected_sets.items():
    check(f"delta={delta} delegates {want}",
          g.delegate_types(delta) == want,
          f"got {g.delegate_types(delta)}")

# hand-built task: 5 summarize subtasks, difficulty 0. Lead success
# declines .95, .89, .83, .77, .71 (load penalty 0.06); worker is a
# flat .89. Oracle: lead takes only the first subtask (its .95 beats
# the worker's .89; the second lead slot ties .89 but costs more).
from arxiv_lab.agent.partial_routing import Subtask, Task  # noqa: E402
hand = Task(tuple(Subtask("summarize", 0.0) for _ in range(5)))
p_lead, c_lead = analytic_expected(hand, (0,) * 5)
p_work, c_work = analytic_expected(hand, (1,) * 5)
check("hand task lead-alone analytic success",
      abs(p_lead - 0.95 * 0.89 * 0.83 * 0.77 * 0.71) < 1e-9,
      f"p={p_lead:.6f}")
check("hand task worker-alone analytic success",
      abs(p_work - 0.89 ** 5) < 1e-9, f"p={p_work:.6f}")
check("hand task costs are 50 / 7.5",
      c_lead == 50.0 and abs(c_work - 7.5) < 1e-9,
      f"lead={c_lead} worker={c_work}")
# The load penalty counts prior LEAD-handled subtasks, not task
# position, so any single-lead assignment ties on success (.95 x
# .89**4) and cost (16.0); the oracle just needs one of them.
oracle_hand = oracle_assignment(hand)
check("oracle on hand task assigns exactly one subtask to the lead",
      sum(oracle_hand) == 4,
      f"got {oracle_hand}")
p_or, c_or = analytic_expected(hand, oracle_hand)
check("oracle hand-task success is 0.95 * 0.89**4",
      abs(p_or - 0.95 * 0.89 ** 4) < 1e-9, f"p={p_or:.6f}")
check("oracle hand-task cost is 16.0",
      abs(c_or - 16.0) < 1e-9, f"c={c_or}")
check("delta=0 assignment equals lead-alone on hand task",
      delegate(hand, g, 0.0) == (0,) * 5)
check("delta=1 assignment equals worker-alone on hand task",
      delegate(hand, g, 1.0) == (1,) * 5)

# ------------------------------------------------------------- held-out A
tasks = make_tasks(HELDOUT_SEED, N_TASKS)
lead = run_policy(tasks, lambda t: (0,) * 5, MC_SEED, TRIALS)
sweep = {d: run_policy(tasks, lambda t, d=d: delegate(t, g, d),
                      MC_SEED, TRIALS)
         for d in (0.0, 0.25, 0.5, 0.75, 1.0)}
task_level = run_policy(tasks, task_level_assignment, MC_SEED, TRIALS)
rng_rand = random.Random(77)
rand = run_policy(tasks, lambda t: random_assignment(t, rng_rand),
                  MC_SEED, TRIALS)
breakdown = run_policy(tasks, lambda t: (0,) * 5, MC_SEED, TRIALS,
                       breakdown=True)
fixed = run_policy(tasks, fixed_easy_assignment, MC_SEED, TRIALS)
part = sweep[0.75]

print(f"E1 lead-alone   success={lead.success_rate:.3f} cost={lead.mean_cost:.1f} "
      f"analytic=({lead.analytic_success:.3f}, {lead.analytic_cost:.1f})")
print(f"E1 partial d.75 success={part.success_rate:.3f} cost={part.mean_cost:.1f} "
      f"analytic=({part.analytic_success:.3f}, {part.analytic_cost:.1f})")
print(f"E1 task-level   success={task_level.success_rate:.3f} cost={task_level.mean_cost:.1f}")
print(f"E1 random       success={rand.success_rate:.3f} cost={rand.mean_cost:.1f}")

# ---------------------------------------------------------------- E1 core
check("E1 partial d0.75 beats lead-alone on success (MC)",
      part.success_rate > lead.success_rate + 0.01,
      f"{part.success_rate:.3f} vs {lead.success_rate:.3f}")
check("E1 partial d0.75 beats lead-alone on success (analytic)",
      part.analytic_success > lead.analytic_success + 0.01,
      f"{part.analytic_success:.3f} vs {lead.analytic_success:.3f}")
check("E1 partial d0.75 beats task-level routing on success",
      part.success_rate > task_level.success_rate + 0.01,
      f"{part.success_rate:.3f} vs {task_level.success_rate:.3f}")
check("E1 partial d0.75 costs less than lead-alone",
      part.mean_cost < lead.mean_cost - 5.0,
      f"{part.mean_cost:.1f} vs {lead.mean_cost:.1f}")
check("E1 MC agrees with analytic (lead)",
      abs(lead.success_rate - lead.analytic_success) < 0.02,
      f"mc={lead.success_rate:.3f} analytic={lead.analytic_success:.3f}")
check("E1 MC agrees with analytic (partial d0.75)",
      abs(part.success_rate - part.analytic_success) < 0.02,
      f"mc={part.success_rate:.3f} analytic={part.analytic_success:.3f}")
check("E1 random delegation is worse than lead-alone (sanity)",
      rand.success_rate < lead.success_rate,
      f"random={rand.success_rate:.3f} lead={lead.success_rate:.3f}")

# ---------------------------------------------------------------- E2 sweep
costs = [sweep[d].mean_cost for d in (0.0, 0.25, 0.5, 0.75, 1.0)]
succs = [sweep[d].success_rate for d in (0.0, 0.25, 0.5, 0.75, 1.0)]
print("E2 sweep: " + " | ".join(
    f"d{d}: {sweep[d].success_rate:.3f}/{sweep[d].mean_cost:.1f}"
    for d in (0.0, 0.25, 0.5, 0.75, 1.0)))
check("E2 cost strictly decreases with delta",
      all(costs[i] < costs[i - 1] - 0.5 for i in range(1, 5)),
      f"costs={[round(c, 1) for c in costs]}")
best_i = max(range(5), key=lambda i: succs[i])
check("E2 success peaks at an interior delta (not 0 or 1)",
      0 < best_i < 4,
      f"peak at delta={(0.0, 0.25, 0.5, 0.75, 1.0)[best_i]} "
      f"success={succs[best_i]:.3f}")
check("E2 delta=1 (delegate everything) collapses success",
      succs[4] < succs[0],
      f"d1.0={succs[4]:.3f} vs lead={succs[0]:.3f}")

# ---------------------------------------------------------------- E3 ablation
print(f"E3 breakdown-only success={breakdown.success_rate:.3f} "
      f"cost={breakdown.mean_cost:.1f} "
      f"analytic=({breakdown.analytic_success:.3f}, {breakdown.analytic_cost:.1f})")
check("E3 breakdown-only costs less than lead-alone",
      breakdown.mean_cost < lead.mean_cost - 1.0,
      f"{breakdown.mean_cost:.1f} vs {lead.mean_cost:.1f}")
check("E3 breakdown-only does not lose success vs lead-alone",
      breakdown.success_rate > lead.success_rate + 0.01,
      f"{breakdown.success_rate:.3f} vs {lead.success_rate:.3f}")

# ---------------------------------------------------------------- E4 transfer
tasks_b = make_tasks(HELDOUT_B_SEED, N_TASKS)
learned_b = run_policy(tasks_b, lambda t: delegate(t, g, 0.75),
                       MC_SEED, TRIALS)
fixed_b = run_policy(tasks_b, fixed_easy_assignment, MC_SEED, TRIALS)
print(f"E4 set B learned d0.75 success={learned_b.success_rate:.3f} "
      f"cost={learned_b.mean_cost:.1f} | fixed easy-rule "
      f"success={fixed_b.success_rate:.3f} cost={fixed_b.mean_cost:.1f}")
# Honest negative: on set B the fixed easy-type rule (which coincides
# with the delta=0.5 optimum) beats learned guidance at the paper's
# delta=0.75. Recorded, not asserted as a win -- see README.
check("E4 recorded: fixed easy-rule beats learned d0.75 on set B",
      fixed_b.success_rate > learned_b.success_rate,
      f"fixed={fixed_b.success_rate:.3f} learned={learned_b.success_rate:.3f}")

# ---------------------------------------------------------------- E6 oracle
oracle_tasks = make_tasks(ORACLE_SEED, N_ORACLE)
o_succ = o_cost = p_succ = p_cost = 0.0
for t in oracle_tasks:
    op, oc = analytic_expected(t, oracle_assignment(t))
    pp, pc = analytic_expected(t, delegate(t, g, 0.75))
    o_succ += op
    o_cost += oc
    p_succ += pp
    p_cost += pc
o_succ /= N_ORACLE
o_cost /= N_ORACLE
p_succ /= N_ORACLE
p_cost /= N_ORACLE
print(f"E6 oracle-12 success={o_succ:.3f} cost={o_cost:.1f} | "
      f"partial d0.75 success={p_succ:.3f} cost={p_cost:.1f} "
      f"({100 * p_succ / o_succ:.0f}% of oracle)")
check("E6 partial d0.75 reaches >=85% of oracle success",
      p_succ >= 0.85 * o_succ,
      f"{p_succ:.3f} vs oracle {o_succ:.3f}")
check("E6 partial d0.75 costs less than the oracle assignment",
      p_cost < o_cost,
      f"{p_cost:.1f} vs {o_cost:.1f}")

# ---------------------------------------------------------------- E5 determinism
lead2 = run_policy(tasks, lambda t: (0,) * 5, MC_SEED, TRIALS)
part2 = run_policy(tasks, lambda t: delegate(t, g, 0.75), MC_SEED, TRIALS)
check("E5 repeated run gives identical lead numbers",
      (lead2.success_rate, lead2.mean_cost)
      == (lead.success_rate, lead.mean_cost))
check("E5 repeated run gives identical partial d0.75 numbers",
      (part2.success_rate, part2.mean_cost)
      == (part.success_rate, part.mean_cost))

HEADLINE_SNIPPET = (
    "import sys; sys.path.insert(0, 'src');"
    "from arxiv_lab.agent.partial_routing import *;"
    "g = learn_guidance(past_traces(11, 4000));"
    "tasks = make_tasks(101, 60);"
    "lead = run_policy(tasks, lambda t: (0,)*5, 5, 200);"
    "part = run_policy(tasks, lambda t: delegate(t, g, 0.75), 5, 200);"
    "print(f\"{lead.success_rate:.6f} {lead.mean_cost:.6f} "
    "{part.success_rate:.6f} {part.mean_cost:.6f}\")"
)
hash_outs = []
for hseed in ("1", "2", "3"):
    env = dict(os.environ, PYTHONHASHSEED=hseed)
    proc = subprocess.run(
        [sys.executable, "-c", HEADLINE_SNIPPET],
        capture_output=True, text=True, env=env,
        cwd=str(Path(__file__).resolve().parents[1]))
    check(f"E5 PYTHONHASHSEED={hseed} headline run succeeds",
          proc.returncode == 0, proc.stderr.strip()[-200:])
    hash_outs.append(proc.stdout.strip())
check("E5 identical headlines across PYTHONHASHSEED 1/2/3",
      len(set(hash_outs)) == 1, f"{hash_outs}")

print()
if fails:
    print(f"FAILED: {len(fails)} check(s): {fails}")
    sys.exit(1)
print("ALL CHECKS PASSED")
