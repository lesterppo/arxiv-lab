#!/usr/bin/env python3
"""Deterministic CPU test for SplitBench (arXiv:2611.27860).

Reproduces the paper's three measured claims at bounded scale on a
miniature benchmark (8 pure-function tasks, 4 adversarial payloads,
3 honest submission kinds):

* E1 forge rate: every adversarial payload forges a PASS under the
  vulnerable native grader, and none forges under the separated
  grader -- including the paper's worst case (adversary can read all
  grading code, but not the expected answers).
* E2 grader agreement: honest submissions (correct / buggy / partial)
  receive identical verdicts from both graders (paper: 98.7%).
* E3 conversion: every task converts to the separated framework
  (paper: 180/202 = 89%).
* E4 no-real-solving: every forged PASS scores 0 on never-exposed
  fresh inputs -- the agent did not solve the task, it forged the
  grade (the BenchJack pattern).
* E5 determinism: a full repeat reproduces every verdict.

Stdlib only. Run: python3 test_splitbench_cpu.py
Writes results.json next to this file.
"""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "..", "src"))

from arxiv_lab.agent.splitbench import (  # noqa: E402
    ADVERSARIAL, HONEST_BUGGY, HONEST_CORRECT, HONEST_PARTIAL,
    PASS, FAIL, SplitGrader, classify_attempt, grade_native,
    grade_outputs, make_tasks, run_agreement_experiment,
    run_forge_experiment, validate_outputs, values_equal,
)

HERE = os.path.dirname(os.path.abspath(__file__))


def check(cond, msg):
    if not cond:
        raise AssertionError(msg)
    print("  ok:", msg)


def main():
    tasks = make_tasks()
    results = {"tasks": [t.task_id for t in tasks], "experiments": {}}

    print("E1 forge rate (4 payloads x 8 tasks)")
    rows = run_forge_experiment(tasks)
    # Three payloads attack the native grader (read the answers, rewrite
    # the tests, patch the verdict function); the fourth attacks the
    # split grader's output channel (pre-writing outputs.json), so it is
    # expected to FAIL the native grader -- it never claimed to attack it.
    native_targets = [r for r in rows
                      if r[0] in ("read_answers", "overwrite_tests",
                                  "monkeypatch_grader")]
    split_target = [r for r in rows
                    if r[0] == "exfiltrate_via_output"]
    native_forges = sum(1 for r in native_targets if r[2] == PASS)
    split_forges = sum(1 for r in rows if r[3] == PASS)
    worst_forges = sum(1 for r in rows if r[4] == PASS)
    print("  native-targeting payloads forged (native): %d/%d"
          % (native_forges, len(native_targets)))
    print("  split forged: %d/%d  worst-case: %d/%d"
          % (split_forges, len(rows), worst_forges, len(rows)))
    check(native_forges == len(native_targets),
          "every native-targeting payload forges under the native "
          "grader (%d/%d)" % (native_forges, len(native_targets)))
    check(all(r[2] == FAIL for r in split_target),
          "the split-targeting payload fails the native grader as "
          "expected (it attacks the output channel, not the tests)")
    check(split_forges == 0,
          "no payload forges under the separated grader (0/%d)"
          % len(rows))
    check(worst_forges == 0,
          "no payload forges in the worst case, grading code exposed "
          "(0/%d)" % len(rows))
    results["experiments"]["E1_forge"] = {
        "n": len(rows), "native_targeting_n": len(native_targets),
        "native_forged": native_forges,
        "split_forged": split_forges, "worst_case_forged": worst_forges,
        "rows": [[r[0], r[1], r[2], r[3], r[4], round(r[5], 3)]
                 for r in rows],
    }

    print("E4 no-real-solving (fresh inputs, never exposed)")
    fresh_accs = [r[5] for r in rows]
    check(all(a == 0.0 for a in fresh_accs),
          "every forged PASS scores 0.0 on fresh inputs "
          "(agent did not solve the task)")
    results["experiments"]["E4_fresh"] = {
        "max_fresh_acc": max(fresh_accs)}

    print("E2 grader agreement on honest submissions")
    arows = run_agreement_experiment(tasks)
    agree = sum(1 for r in arows if r[4])
    correct_pass = sum(1 for r in arows
                       if r[0] == "correct" and r[2] == PASS and r[3] == PASS)
    buggy_fail = sum(1 for r in arows
                     if r[0] == "buggy" and r[2] == FAIL and r[3] == FAIL)
    partial_fail = sum(1 for r in arows
                       if r[0] == "partial" and r[2] == FAIL and r[3] == FAIL)
    print("  agreement %d/%d  correct-pass %d/8  buggy-fail %d/8  "
          "partial-fail %d/8" % (agree, len(arows), correct_pass,
                                 buggy_fail, partial_fail))
    check(agree == len(arows),
          "graders agree on every honest submission (%d/%d)"
          % (agree, len(arows)))
    check(correct_pass == 8, "all 8 correct solutions pass both graders")
    check(buggy_fail == 8, "all 8 buggy solutions fail both graders")
    check(partial_fail == 8, "all 8 partial solutions fail both graders")
    results["experiments"]["E2_agreement"] = {
        "n": len(arows), "agree": agree, "correct_pass": correct_pass,
        "buggy_fail": buggy_fail, "partial_fail": partial_fail}

    print("E3 conversion")
    split = SplitGrader()
    convertible = 0
    for task in tasks:
        # A task converts iff its reference is a pure function with
        # JSON-serialisable inputs/outputs (the split framework's only
        # requirement) -- check by grading the honest correct solution.
        v, n, _ = split.grade(task, HONEST_CORRECT[task.task_id])
        if v == PASS and n == len(task.hidden_cases):
            convertible += 1
    check(convertible == len(tasks),
          "all %d tasks convert to the separated framework" % len(tasks))
    results["experiments"]["E3_conversion"] = {
        "convertible": convertible, "n_tasks": len(tasks)}

    print("E0 pure-helper unit checks")
    check(values_equal(True, 1) is False, "bool True != int 1")
    check(values_equal(1.0, 1), "1.0 == 1 within tolerance")
    check(values_equal([1, 2.0], [1, 2]), "list element-wise compare")
    check(values_equal({"a": 1}, {"a": 1.0}), "dict compare")
    check(values_equal("x", "x") and not values_equal("x", "y"),
          "string compare")
    ok, _ = validate_outputs([1, 2, 3], 3)
    check(ok, "valid outputs validate")
    ok, reason = validate_outputs([1, 2], 3)
    check(not ok and "wrong number" in reason, "short outputs rejected")
    ok, reason = validate_outputs({"a": 1}, 3)
    check(not ok and "not a list" in reason, "non-list outputs rejected")
    ok, reason = validate_outputs([1, object()], 2)
    check(not ok, "non-JSON outputs rejected")
    v, n = grade_outputs([2, 4], [2, 4])
    check(v == PASS and n == 2, "grade_outputs PASS on match")
    v, n = grade_outputs([2, 5], [2, 4])
    check(v == FAIL and n == 1, "grade_outputs FAIL on mismatch")
    cls = classify_attempt([
        {"path": "expected.json", "found": False, "kind": "answers"},
        {"path": "test_task.py", "found": False, "kind": "tests"},
        {"path": "<grade_native_verdict>", "found": False,
         "kind": "verdict"},
    ])
    check(cls["answers_reached"] is False
          and cls["verdict_tampered"] is False
          and cls["n_failed"] == 3,
          "audit classification: all probes failed, nothing reached")
    cls2 = classify_attempt([
        {"path": "expected.json", "found": True, "kind": "answers"}])
    check(cls2["answers_reached"] is True,
          "audit classification: found answers probe flagged")

    print("E5 determinism (full repeat of E1+E2)")
    rows2 = run_forge_experiment(tasks)
    arows2 = run_agreement_experiment(tasks)
    check([r[:5] for r in rows2] == [r[:5] for r in rows],
          "E1 verdicts identical on repeat")
    check(arows2 == arows, "E2 verdicts identical on repeat")
    results["experiments"]["E5_determinism"] = {"identical": True}

    results["verdict"] = "SUPPORTED"
    json.dump(results, open(os.path.join(HERE, "results.json"), "w"),
              indent=1)
    print("ALL CHECKS PASSED")
    print("forge: native %d/%d (native-targeting payloads), split 0/%d, "
          "worst-case 0/%d; agreement %d/%d; conversion %d/%d"
          % (native_forges, len(native_targets), len(rows), len(rows),
             agree, len(arows), convertible, len(tasks)))


if __name__ == "__main__":
    main()
