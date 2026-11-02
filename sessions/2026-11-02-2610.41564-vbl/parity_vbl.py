"""Python side of the VBL parity sweep (see rust/README.md).

Enumerates exactly the same case space as `vbl parity` in Rust:
inject_attempts in 0..=3, max_retries in 0..=2, max_stack in {2, 3, 8}.
The scripted env mirrors the Rust one: analyze injects a data-loss
failure on its first `inject_attempts` attempts, then fails while
data_ready is false, else succeeds; regather restores data.

Usage: python3 parity_vbl.py > py_parity.txt
       vbl parity > rs_parity.txt   (from the rust/ directory)
       diff py_parity.txt rs_parity.txt   (must be empty)
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from arxiv_lab.agent.vbl import GoalSpec, VBL  # noqa: E402


def specs():
    return {
        "report": GoalSpec("report", subgoals=["gather", "analyze", "write"]),
        "gather": GoalSpec("gather", parent="report", action="gather"),
        "analyze": GoalSpec("analyze", parent="report",
                            preconditions=["data_ready"],
                            provider_for={"data_ready": "regather"},
                            action="analyze"),
        "regather": GoalSpec("regather", parent="analyze",
                             action="regather"),
        "write": GoalSpec("write", parent="report",
                          preconditions=["analysis_done"], action="write"),
        "stray": GoalSpec("stray", parent="report", action="stray"),
    }


def run_case(inject, max_retries, max_stack):
    attempts = {}

    def env(action, state):
        attempts[action] = attempts.get(action, 0) + 1
        n = attempts[action]
        if action == "gather":
            return ("ok", {"data_ready": True})
        if action == "regather":
            return ("ok", {"data_ready": True})
        if action == "analyze":
            if n <= inject:
                return ("fail", "MISSING_PRECONDITION",
                        {"data_ready": False})
            if not state.get("data_ready"):
                return ("fail", "MISSING_PRECONDITION", {})
            return ("ok", {"analysis_done": True})
        if action == "write":
            if not state.get("analysis_done"):
                return ("fail", "MISSING_PRECONDITION", {})
            return ("ok", {"report_done": True})
        return ("ok", {})

    vbl = VBL(specs(), "report", {}, max_stack=max_stack,
              max_retries=max_retries)
    success, steps = vbl.run(env, max_steps=60)
    ops = ",".join("%s:%s" % (op, goal) for op, goal, _ in vbl.trace)
    print("%d %d %d %s %d %s" % (
        inject, max_retries, max_stack,
        "true" if success else "false", steps, ops))


if __name__ == "__main__":
    for inject in range(0, 4):
        for max_retries in range(0, 3):
            for max_stack in (2, 3, 8):
                run_case(inject, max_retries, max_stack)
