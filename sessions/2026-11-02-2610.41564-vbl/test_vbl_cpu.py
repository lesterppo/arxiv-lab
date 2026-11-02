"""CPU test for VBL (arXiv 2610.41564): structured precondition
recovery beats free-form retry on a scripted long-horizon task, plus
unit cases for each stack operation. Deterministic."""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from arxiv_lab.agent.vbl import (  # noqa: E402
    GoalSpec, GoalState, VBL, react_baseline,
)


def report_specs():
    """Terminal 'report' = gather -> analyze -> write; analyze can
    lose its data precondition mid-run and recover via 'regather'."""
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


class ScriptedEnv(object):
    """Deterministic env: gather ok; analyze fails once (data lost),
    then ok after regather; write ok. Tracks attempts."""

    def __init__(self):
        self.attempts = {}

    def __call__(self, action, state):
        self.attempts[action] = self.attempts.get(action, 0) + 1
        if action == "gather":
            return ("ok", {"data_ready": True})
        if action == "regather":
            return ("ok", {"data_ready": True})
        if action == "analyze":
            if self.attempts[action] == 1:
                # Material state change: the data precondition is lost.
                return ("fail", "MISSING_PRECONDITION",
                        {"data_ready": False})
            if not state.get("data_ready"):
                return ("fail", "MISSING_PRECONDITION", {})
            return ("ok", {"analysis_done": True})
        if action == "write":
            if not state.get("analysis_done"):
                return ("fail", "MISSING_PRECONDITION", {})
            return ("ok", {"report_done": True})
        if action == "stray":
            return ("ok", {})
        raise AssertionError("unknown action %r" % action)


def test_vbl_recovers_where_react_burns_budget():
    env = ScriptedEnv()
    vbl = VBL(report_specs(), "report", {})
    success, steps = vbl.run(env, max_steps=60)
    assert success, vbl.trace
    assert vbl.state.get("report_done") is True
    ops = [t[0] for t in vbl.trace]
    assert "Push" in ops and "BacktrackTo" not in ops
    assert any(t[0] == "Push" and t[1] == "regather" for t in vbl.trace), \
        vbl.trace
    print("VBL: success in %d steps; regather pushed; trace=%s"
          % (steps, vbl.trace))

    env2 = ScriptedEnv()
    ok, steps2 = react_baseline(env2, ["gather", "analyze", "write"],
                                max_steps=60)
    assert not ok, "ReAct baseline should burn its budget on analyze"
    assert steps2 == 60, steps2
    print("ReAct baseline: failed, burned all 60 steps (analyze retries)")


def test_non_contributing_goal_popped():
    specs = report_specs()
    vbl = VBL(specs, "report", {})
    vbl.stack.append(GoalState(specs["stray"]))
    # Stray is not in the terminal closure and not a provider: the
    # first step must pop it, then the task proceeds normally.
    env = ScriptedEnv()
    # Patch: stray would otherwise be "executed" -- popping must come
    # first, so the env must never see it.
    orig = env.__call__

    def guarded(action, state):
        assert action != "stray", "stray goal was executed, not popped"
        return orig(action, state)

    success, _ = vbl.run(guarded, max_steps=60)
    assert success
    assert vbl.trace[0][0] == "Pop" and vbl.trace[0][1] == "stray", vbl.trace
    print("non-contributing goal popped first OK")


def test_decomposition_failure_backtracks():
    specs = {
        "report": GoalSpec("report", subgoals=["analyze"]),
        "analyze": GoalSpec("analyze", parent="report", action="analyze"),
    }

    def env(action, state):
        return ("fail", "DECOMPOSITION_FAILED", {})

    vbl = VBL(specs, "report", {})
    vbl.run(env, max_steps=6)
    assert any(t[0] == "BacktrackTo" for t in vbl.trace), vbl.trace
    print("decomposition failure -> BacktrackTo OK:", vbl.trace)


def test_retry_limit_pops():
    specs = {
        "task": GoalSpec("task", action="flaky"),
    }

    def env(action, state):
        return ("fail", "GENERIC", {})

    vbl = VBL(specs, "task", {}, max_retries=2)
    success, steps = vbl.run(env, max_steps=60)
    assert not success and steps == 3, (success, steps)
    assert [t[0] for t in vbl.trace] == ["retry", "retry", "Pop"], vbl.trace
    print("retry limit (2) then Pop, 3 steps OK")


def test_stack_bound_forces_backtrack():
    specs = report_specs()
    env = ScriptedEnv()
    vbl = VBL(specs, "report", {}, max_stack=2)
    # report > analyze fills the stack; analyze's missing-precondition
    # provider cannot be pushed, so VBL must backtrack instead.
    vbl.run(env, max_steps=8)
    assert any("stack-full" in str(t) for t in vbl.trace), vbl.trace
    print("stack bound -> backtrack instead of push OK:", vbl.trace)


if __name__ == "__main__":
    test_vbl_recovers_where_react_burns_budget()
    test_non_contributing_goal_popped()
    test_decomposition_failure_backtracks()
    test_retry_limit_pops()
    test_stack_bound_forces_backtrack()
    print("ALL VBL CPU TESTS PASSED")
