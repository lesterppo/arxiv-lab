#!/usr/bin/env python3
"""Generate rust/tests/parity_cases.txt: cases dumped from Python plus
the Python-computed values the Rust port must match exactly.

Line types (one case per line, `#` lines are comments):
  G <n> <trace ...> | GAPS <5 floats> LEADQ <5> WORKERQ <5>
      Guidance case: n traces (model,type,difficulty,success,cost)
      in order; Rust relearns guidance and must match the gaps and
      per-type qualities bit-for-bit (same summation order).
  T <n_tasks> <task ...> | GAPS <5 floats> | <per-task expectations>
      Task case: each task is 5 `type,difficulty` pairs joined by
      `;`, tasks joined by `|`. Expectations: for each task, for each
      policy in (lead, partial@0.75, task-level, fixed-easy, oracle):
      `<bits5> <p repr> <cost repr>` joined by `,`. Rust rebuilds the
      guidance from GAPS, recomputes every assignment + analytic
      (p, cost), and must match exactly (0 mismatches required).
Difficulties are exactly representable (0/0.25/0.5/0.75/1.0); all
floats are dumped with repr() so Rust parses the identical f64.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from arxiv_lab.agent.partial_routing import (  # noqa: E402
    analytic_expected, delegate, fixed_easy_assignment, learn_guidance,
    make_tasks, oracle_assignment, past_traces, task_level_assignment,
    SUBTASK_TYPES,
)

OUT = Path(__file__).with_name("rust").joinpath("tests", "parity_cases.txt")


def f(x):
    return repr(float(x))


def main():
    n_task_cases = int(sys.argv[1]) if len(sys.argv) > 1 else 150
    lines = []
    lines.append("# parity cases for the Partial Routing Rust port (arXiv:2611.33402)")
    lines.append("# G: guidance case | T: task case (see parity_dump.py header)")

    # ---- guidance case: 500 past tasks -> 5000 traces ----
    traces = past_traces(11, 500)
    g = learn_guidance(traces)
    enc = []
    for tr in traces:
        enc.append("%d,%d,%s,%d,%s" % (
            0 if tr.model == "lead" else 1,
            SUBTASK_TYPES.index(tr.type), f(tr.difficulty),
            1 if tr.success else 0, f(tr.cost)))
    lines.append("G %d %s || GAPS %s LEADQ %s WORKERQ %s" % (
        len(traces), " ".join(enc),
        " ".join(f(g.gap[t]) for t in SUBTASK_TYPES),
        " ".join(f(g.lead_q[t]) for t in SUBTASK_TYPES),
        " ".join(f(g.worker_q[t]) for t in SUBTASK_TYPES)))

    # ---- task cases: seeded task sets + edge tasks ----
    # Guidance used for the partial-routing expectations: the same
    # guidance learned above (gaps dumped on every line).
    gaps_s = " ".join(f(g.gap[t]) for t in SUBTASK_TYPES)
    policies = [
        ("lead", lambda t: (0,) * 5),
        ("partial075", lambda t: delegate(t, g, 0.75)),
        ("tasklevel", task_level_assignment),
        ("fixedeasy", fixed_easy_assignment),
        ("oracle", oracle_assignment),
    ]

    def dump_task_case(tasks):
        task_s = "|".join(
            ";".join("%d,%s" % (SUBTASK_TYPES.index(st.type), f(st.difficulty))
                     for st in t.subtasks)
            for t in tasks)
        exp_parts = []
        for t in tasks:
            for _name, fn in policies:
                assign = fn(t)
                p, c = analytic_expected(t, assign)
                exp_parts.append("%s %s %s" % (
                    "".join("1" if a else "0" for a in assign), f(p), f(c)))
        lines.append("T %d %s || GAPS %s || %s" % (
            len(tasks), task_s, gaps_s, " , ".join(exp_parts)))

    for seed in range(1000, 1000 + n_task_cases):
        dump_task_case(make_tasks(seed, 8))
    # Edge tasks: all-summarize (hand task), all-plan, all-hardest.
    from arxiv_lab.agent.partial_routing import Subtask, Task  # noqa: E402
    edge = [
        Task(tuple(Subtask("summarize", 0.0) for _ in range(5))),
        Task(tuple(Subtask("plan", 1.0) for _ in range(5))),
        Task(tuple(Subtask(t, d) for t, d in
                   [("plan", 1.0), ("code", 0.75), ("verify", 0.5),
                    ("test", 0.25), ("summarize", 0.0)])),
        Task(tuple(Subtask(t, 1.0) for t in
                   ["summarize", "test", "verify", "code", "plan"])),
    ]
    dump_task_case(edge)

    OUT.write_text("\n".join(lines) + "\n")
    print(f"wrote {OUT} ({len(lines) - 2} cases)")


if __name__ == "__main__":
    main()
