"""CPU tests for arxiv_lab.harness (Mingbird, arXiv:2610.02001).

Ports experiments/2026-10-03/test_mingbird.py: each of the three harness
mechanisms is tested against a scripted policy exhibiting the failure form
it targets (harness varied, policy fixed), with deterministic artifact
scoring. Pure stdlib.

Run:  python3 tests/test_harness_cpu.py
"""

import json
import os
import sys

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from arxiv_lab.harness import PrefillBudget, FinishGate, LoopDetector


# ── Toy environment ──────────────────────────────────────────────────────────
class Env:
    def __init__(self):
        self.fs = {
            "/data/nums.txt": "3\n7\n11\n",
            "/data/flag.txt": "yes\n",
        }

    def ls(self, path):
        return sorted(k for k in self.fs if k.startswith(path))

    def read(self, path):
        return self.fs.get(path)

    def write(self, path, content):
        self.fs[path] = content
        return "ok"


BLOATED_SCHEMAS = [
    {
        "name": "read",
        "description": "Read a file from the virtual filesystem. " * 40,
        "parameters": {"path": {"type": "string",
                                "description": "Absolute path. " * 30}},
        "examples": [{"in": {"path": "/a"}, "out": "x"}] * 25,
    },
    {
        "name": "write",
        "description": "Write content to a file, creating parents. " * 40,
        "parameters": {"path": {"type": "string"},
                       "content": {"type": "string",
                                   "description": "Full text. " * 30}},
        "examples": [{"in": {"path": "/b", "content": "y"}, "out": "ok"}] * 25,
    },
]
SYSTEM = "You are a file-task agent. Use tools.\n"

TASKS = {
    "sum": dict(
        text="Write the sum of the numbers in /data/nums.txt to /out/sum.txt",
        criteria=[("sum_artifact",
                   lambda e: e.fs.get("/out/sum.txt") == "21\n")],
    ),
    "gate": dict(
        text=("If /data/flag.txt exists, append the line 'done' to /out/log.txt; "
              "then write the word 'verified' to /out/status.txt"),
        criteria=[("log_artifact",
                   lambda e: e.fs.get("/out/log.txt") == "done\n"),
                  ("status_artifact",
                   lambda e: e.fs.get("/out/status.txt") == "verified\n")],
    ),
}


# ── Scripted policies (the "small model", harness varied) ────────────────────
def policy_bloat(env):
    """Naive tool use with the FULL bloated schemas in prefill."""
    yield ("read", {"path": "/data/nums.txt"})
    nums = [int(x) for x in env.read("/data/nums.txt").split()]
    yield ("write", {"path": "/out/sum.txt", "content": f"{sum(nums)}\n"})
    yield ("__finish__", {})


def policy_premature(env):
    """Emits 'done' after a partial job (silent abandonment)."""
    yield ("write", {"path": "/out/log.txt", "content": "done\n"})
    yield ("__finish__", {})  # never writes /out/status.txt


def policy_looper(env):
    """Tool-demonstration loop: re-reads the same file forever."""
    for _ in range(60):
        yield ("read", {"path": "/data/nums.txt"})
    yield ("__finish__", {})


# ── Harness runner ───────────────────────────────────────────────────────────
def run(policy_fn, task_key, task, use_budget, use_gate, use_loopdet,
         budget_bytes=600, max_steps=60):
    env = Env()
    budget = PrefillBudget(budget_bytes)
    gate = FinishGate(task["text"], task["criteria"])
    loopdet = LoopDetector(repeat_k=3)
    steps = 0
    finish_accepted = False
    loop_broken = False
    prefill_bytes = None

    for tool, args in policy_fn(env):
        if tool == "__finish__":
            if use_gate:
                ok, failed = gate.review(env)
                finish_accepted = ok
                if not ok:
                    score = sum(1 for _, c in task["criteria"] if c(env)) / len(task["criteria"])
                    return dict(done=False, steps=steps, prefill=0,
                               loop_broken=loop_broken, gate_failed=failed,
                               gate_rejected=True, score=score)
            else:
                finish_accepted = True  # accepted on claim alone
            break
        if use_loopdet:
            hit = loopdet.observe(tool, args)
            if hit:
                loop_broken = True
                break
        if steps == 0:
            if use_budget:
                _, prefill_bytes, _ = budget.build_prefill(SYSTEM, BLOATED_SCHEMAS)
            else:
                prefill_bytes = len(
                    (SYSTEM + "\n".join(
                        json.dumps(s) for s in BLOATED_SCHEMAS)).encode())
        steps += 1
        if steps >= max_steps:
            break
        getattr(env, tool)(**args)

    score = sum(1 for _, c in task["criteria"] if c(env)) / len(task["criteria"])
    return dict(done=finish_accepted, steps=steps, prefill=prefill_bytes or 0,
                loop_broken=loop_broken, score=score, gate_rejected=False)


def main():
    rows = []
    CONTEXT = 4096  # toy context window in bytes

    # Experiment 1 — prefill budget: does the bloated prefill overflow context?
    r_off = run(policy_bloat, "sum", TASKS["sum"], use_budget=False,
                use_gate=False, use_loopdet=False)
    r_on = run(policy_bloat, "sum", TASKS["sum"], use_budget=True,
               use_gate=False, use_loopdet=False, budget_bytes=600)
    rows.append(("prefill OFF: bytes in context", r_off["prefill"],
                 "OVERFLOW" if r_off["prefill"] > CONTEXT else "fits"))
    rows.append(("prefill ON : bytes in context", r_on["prefill"],
                 "fits" if r_on["prefill"] <= 600 else "OVER BUDGET"))

    # Experiment 2 — finish gate: premature 'done' accepted or rejected?
    r_off = run(policy_premature, "gate", TASKS["gate"], use_budget=False,
                use_gate=False, use_loopdet=False)
    r_on = run(policy_premature, "gate", TASKS["gate"], use_budget=False,
               use_gate=True, use_loopdet=False)
    rows.append(("gate OFF: premature finish accepted?", r_off["done"],
                 f'artifact score {r_off["score"]:.2f} (silently incomplete)'))
    rows.append(("gate ON : premature finish accepted?", r_on["done"],
                 f'rejected, missing: {r_on.get("gate_failed")} '
                 f'(score if completed: {r_on["score"]:.2f})'))

    # Experiment 3 — loop detection: how many steps before the loop is broken?
    r_off = run(policy_looper, "sum", TASKS["sum"], use_budget=False,
                use_gate=False, use_loopdet=False)
    r_on = run(policy_looper, "sum", TASKS["sum"], use_budget=False,
               use_gate=False, use_loopdet=True)
    rows.append(("loopdet OFF: steps burned", r_off["steps"], "ran to max_steps"))
    rows.append(("loopdet ON : steps burned", r_on["steps"],
                 "broken" if r_on["loop_broken"] else "NOT broken"))

    print(f"{'condition':42s} {'measured':>12s}  verdict")
    print("-" * 90)
    for cond, val, verdict in rows:
        print(f"{cond:42s} {str(val):>12s}  {verdict}")
    checks = [
        rows[0][1] > CONTEXT,          # bloated prefill really overflows
        rows[1][1] <= 600,             # budget holds
        rows[2][1] is True and rows[2][2].startswith("artifact score 0.50"),
        rows[3][1] is False,           # gate rejects premature finish
        rows[4][1] == 60,              # no detection -> burns full budget
        rows[5][1] <= 4,               # detection breaks loop fast
    ]
    assert all(checks), "MECHANISM CHECK FAILED"
    # library-level regression: build_prefill must not mutate caller schemas
    before = json.dumps(BLOATED_SCHEMAS, sort_keys=True)
    PrefillBudget(600).build_prefill(SYSTEM, BLOATED_SCHEMAS)
    assert json.dumps(BLOATED_SCHEMAS, sort_keys=True) == before, \
        "build_prefill must not mutate the caller's schema dicts"
    print("\nALL HARNESS CHECKS PASSED.")


if __name__ == "__main__":
    main()
