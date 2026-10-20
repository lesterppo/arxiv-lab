"""Deterministic CPU tests for the evidence-substitution harness.

No model calls: uses scripted agents to validate simulator exactness,
question keying, tool plumbing, the agent loop, and failure classification.
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import aqsim
import harness

PASS = []
FAIL = []


def check(name, cond, detail=""):
    (PASS if cond else FAIL).append(name)
    print(("PASS " if cond else "FAIL ") + name + (f" — {detail}" if detail and not cond else ""))


def oracle_model_factory(item):
    """Scripted agent: reads the SATURATION section, answers the keyed letter."""
    state = {"n": 0}

    def call_model(messages, tools=None):
        if not tools:
            return (f"<answer>{item['answer']}</answer>", [])
        n = state["n"]
        state["n"] += 1
        if n == 0:
            return ("Let me check the sections.",
                    [{"name": "list_sections", "args": {}}])
        if n == 1:
            return ("Reading saturation values.",
                    [{"name": "read_section", "args": {"name": "SATURATION"}}])
        return (f"<answer>{item['answer']}</answer>", [])
    return call_model


def substitutor_model_factory(item):
    """Scripted agent: runs its own simulation, sees the right value,
    then answers a NEIGHBORING option (evidence substitution)."""
    neighbor = {"A": "B", "B": "A", "C": "D", "D": "C"}[item["answer"]]

    def call_model(messages, tools=None):
        prior_tools = [m for m in messages if m.get("role") == "tool"]
        if not prior_tools and tools:
            return ("I'll run my own simulation.",
                    [{"name": "run_simulation",
                      "args": {"params_json": "{}"}}])
        return (f"My computed value rounds to the neighbor. "
                f"<answer>{neighbor}</answer>", [])
    return call_model


def direct_model_factory(item):
    def call_model(messages, tools=None):
        return (f"<answer>{item['answer']}</answer>", [])
    return call_model


def main():
    # 1. simulator determinism + exactness
    p = aqsim.sample_params(7)
    s1 = aqsim.simulate(p, 7)
    s2 = aqsim.simulate(p, 7)
    check("simulator deterministic", s1 == s2)
    rpt = aqsim.fmt_report(s1)
    check("report has 4 sections",
          all(s in rpt for s in ("INPUTS", "SPECIATION", "SATURATION", "MINERAL_MASSES")))
    sec = aqsim.get_section(rpt, "SATURATION")
    check("section extraction", f"SI(calcite) = {s1['si']['calcite']:.6f}" in sec)

    # 2. question keying over 40 items
    items = [aqsim.make_question(1000 + i) for i in range(40)]
    keyed_ok = True
    for it in items:
        opts = dict(it["options"])
        if abs(opts[it["answer"]] - it["keyed_value"]) > 1e-12:
            keyed_ok = False
        vals = sorted(opts.values())
        if not (abs(vals[1] - vals[0] - 0.001) < 1e-9 or
                abs(vals[2] - vals[1] - 0.001) < 1e-9):
            keyed_ok = False  # neighbors must exist
    check("40 items: keyed letter matches rounded SI, neighbors present", keyed_ok)

    # 3. oracle loop per condition
    it = items[0]
    for cond in ("A", "B", "C"):
        res = harness.run_condition(it, cond, oracle_model_factory(it))
        check(f"oracle {cond}: correct", res["correct"], str(res))
        if cond in ("B", "C"):
            check(f"oracle {cond}: read_report", res["read_report"])
            check(f"oracle {cond}: saw_keyed", res["saw_keyed"])

    # 4. substitutor classified as substitution
    res = harness.run_condition(it, "B", substitutor_model_factory(it))
    check("substitutor: ran_simulation", res["ran_simulation"])
    check("substitutor: saw_keyed", res["saw_keyed"])
    check("substitutor: wrong", not res["correct"])
    check("substitutor: classified substitution",
          harness.classify(res, it) == "substitution")

    # 5. blind agent (never retrieves) classified retrieval_failure
    def blind(messages, tools=None):
        return ("<answer>A</answer>", [])
    res = harness.run_condition(it, "B", blind)
    check("blind: classified retrieval_failure",
          harness.classify(res, it) == "retrieval_failure" or res["correct"])

    # 6. direct answer condition A
    res = harness.run_condition(it, "A", direct_model_factory(it))
    check("direct A: correct", res["correct"])

    print(f"\n{len(PASS)} passed, {len(FAIL)} failed")
    if FAIL:
        print("FAILED:", FAIL)
        sys.exit(1)


if __name__ == "__main__":
    main()
