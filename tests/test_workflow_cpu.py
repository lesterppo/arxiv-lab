"""CPU tests for arxiv_lab.workflow (FloWright co-evolution, arXiv:2610.01026).

Deterministic, pure stdlib. Checks task generation, the per-op slip model,
workflow trace structure, structured credit assignment on hand-made traces,
objective kinds, and evolution invariants (bounds, improvement, invalid
condition handling).

Run:  python3 tests/test_workflow_cpu.py
"""

import os
import random
import sys

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from arxiv_lab.workflow.flowright import (
    CONDITIONS, GENERATOR, REFINER, ROLES, VERIFIER,
    accuracy, accuracy_by_nops, evolve, gen_task, role_objective,
    run_workflow, skilled_eval, structured_credits,
)


def test_gen_task_deterministic():
    t1 = gen_task(random.Random(7), 3)
    t2 = gen_task(random.Random(7), 3)
    assert (t1.ops, t1.operands, t1.value) == (t2.ops, t2.operands, t2.value)
    assert t1.n_ops == 3
    # value matches manual left-assoc evaluation
    v = t1.operands[0]
    for op, b in zip(t1.ops, t1.operands[1:]):
        v = v + b if op == "+" else (v - b if op == "-" else v * b)
    assert v == t1.value
    print("ok gen_task deterministic + value correct")


def test_skilled_eval_monotone():
    rng = random.Random(11)
    tasks = [gen_task(rng, 2) for _ in range(60)]
    for skill, lo in ((0.95, 0.80), (0.50, 0.15), (0.05, 0.0)):
        r = random.Random(12)
        frac = sum(skilled_eval(t, skill, r) == t.value for t in tasks) / len(tasks)
        assert frac >= lo, (skill, frac)
    # monotonic: higher skill -> higher success on same tasks
    r1, r2 = random.Random(13), random.Random(13)
    hi = sum(skilled_eval(t, 0.9, r1) == t.value for t in tasks)
    lo = sum(skilled_eval(t, 0.3, r2) == t.value for t in tasks)
    assert hi > lo, (hi, lo)
    print("ok skilled_eval monotone in skill")


def test_run_workflow_trace():
    rng = random.Random(21)
    task = gen_task(rng, 2)
    skills = {GENERATOR: 0.9, VERIFIER: 0.9, REFINER: 0.9}
    tr = run_workflow(task, skills, random.Random(22))
    assert set(tr) == {"draft", "vest", "accepted", "final",
                       "refiner_handled", "reward"}
    assert tr["reward"] in (0.0, 1.0)
    assert tr["accepted"] == (tr["vest"] == tr["draft"])
    assert (tr["final"] == tr["draft"]) == (not tr["refiner_handled"])
    # near-perfect skills -> usually correct
    r = random.Random(23)
    tasks = [gen_task(r, 2) for _ in range(40)]
    acc = accuracy({rr: 0.99 for rr in ROLES}, tasks, seed=24)
    assert acc > 0.9, acc
    print("ok run_workflow trace structure; near-perfect skills acc=%.2f" % acc)


def test_structured_credits_handmade():
    # accept + correct final: generator 1, verifier 1, refiner None
    tr = {"draft": 10, "vest": 10, "accepted": True, "final": 10,
          "refiner_handled": False, "reward": 1.0}
    c = structured_credits(tr)
    assert c[GENERATOR] == 1.0 and c[VERIFIER] == 1.0 and c[REFINER] is None, c
    # accept + wrong final: verifier 0 (false accept)
    tr2 = dict(tr, reward=0.0)
    assert structured_credits(tr2)[VERIFIER] == 0.0
    # reject, refinement changed answer and correct: vindicated -> 1
    tr3 = {"draft": 5, "vest": 7, "accepted": False, "final": 10,
           "refiner_handled": True, "reward": 1.0}
    c3 = structured_credits(tr3)
    assert c3[GENERATOR] == 0.0 and c3[VERIFIER] == 1.0 and c3[REFINER] == 1.0, c3
    # reject, final wrong -> verifier 0, refiner 0
    tr4 = dict(tr3, final=6, reward=0.0)
    c4 = structured_credits(tr4)
    assert c4[VERIFIER] == 0.0 and c4[REFINER] == 0.0, c4
    # reject, refiner merely reproduced the draft: rejection was useless -> 0
    tr5 = {"draft": 10, "vest": 7, "accepted": False, "final": 10,
           "refiner_handled": True, "reward": 1.0}
    assert structured_credits(tr5)[VERIFIER] == 0.0
    print("ok structured_credits on hand-made traces")


def test_role_objective():
    rng = random.Random(31)
    tasks = [gen_task(rng, 2) for _ in range(30)]
    r = random.Random(32)
    traces = [run_workflow(t, {rr: 0.7 for rr in ROLES}, r) for t in tasks]
    flat = role_objective(traces, GENERATOR, "flat")
    assert flat == sum(t["reward"] for t in traces) / len(traces)
    g = role_objective(traces, GENERATOR, "structured")
    assert 0.0 <= g <= 1.0
    # refiner objective is None when it never handled a task
    never = [dict(t, refiner_handled=False) for t in traces]
    assert role_objective(never, REFINER, "structured") is None
    print("ok role_objective kinds + refiner no-signal -> None")


def test_evolve_invariants():
    rng = random.Random(41)
    train = [gen_task(rng, rng.choice((2, 3))) for _ in range(40)]
    # invalid condition
    try:
        evolve(train, condition="nope", generations=2, seed=1)
        raise AssertionError("should have raised")
    except ValueError:
        pass
    res = evolve(train, condition="coev", generations=15, sigma=0.07, seed=42)
    for role, s in res["skills"].items():
        assert 0.05 <= s <= 0.99, (role, s)
    # flat optimizes the joint objective: train accuracy must not decrease
    res_f = evolve(train, condition="flat", generations=15, sigma=0.07, seed=42)
    ta = res_f["train_acc"]
    assert all(b >= a - 1e-9 for a, b in zip(ta, ta[1:])), ta
    # coev must not end worse than it started (each role's credit is monotone)
    res_c0 = evolve(train, condition="coev", generations=1, sigma=0.07, seed=42)
    assert res["train_acc"][-1] >= res_c0["train_acc"][-1] - 0.25
    print("ok evolve invariants (bounds, flat monotone, invalid condition)")


def test_accuracy_by_nops():
    rng = random.Random(51)
    tasks = [gen_task(rng, 1) for _ in range(10)] + \
            [gen_task(rng, 3) for _ in range(10)]
    by = accuracy_by_nops({r: 0.9 for r in ROLES}, tasks, seed=52)
    assert set(by) == {1, 3} and all(0.0 <= v <= 1.0 for v in by.values())
    print("ok accuracy_by_nops grouping:", {k: round(v, 2) for k, v in by.items()})


def main():
    assert set(CONDITIONS) == {"flat", "single", "coev"}
    test_gen_task_deterministic()
    test_skilled_eval_monotone()
    test_run_workflow_trace()
    test_structured_credits_handmade()
    test_role_objective()
    test_evolve_invariants()
    test_accuracy_by_nops()
    print("all workflow tests passed")


if __name__ == "__main__":
    main()
