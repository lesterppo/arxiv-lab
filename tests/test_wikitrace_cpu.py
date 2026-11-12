#!/usr/bin/env python3
"""CPU test for the graduated wikitrace module (arXiv:2611.14320).

Same assertions as the session test (sessions/2026-11-12-2611.14320-
wikitrace/test_wikitrace_cpu.py), run against the src/ copy. Run from the
repo root: python3 tests/test_wikitrace_cpu.py

Deterministic CPU test for WikiTrace (arXiv:2611.14320).

Pre-registered assertions:
  A1  ordering: WikiTrace success > Insights-Only success by >= 0.10
  A2  WikiTrace success > No-Memory success by >= 0.10
  A3  Insights-Only success > No-Memory success (plain lessons help on
      average, as in the paper: Insights Only 87.0 > No Memory 79.5)
  A4  regression: WikiTrace 0.0 (sound check + default fallback can never
      turn a baseline success into a failure) and Insights-Only > 0.05
  A5  gain: WikiTrace gain > Insights-Only gain
  A6  evidence-check unit cases: corrupted lesson -> CONTRADICTED,
      over-generalized lesson -> UNSUPPORTED, valid narrow lesson ->
      SUPPORTED, unobserved step -> UNKNOWN; a conflict between a
      supported and a contradicted lesson resolves to the supported one;
      a failed-run lesson triggers descent
  A7  descent happens (rate > 0.2) and changes actions (corrections > 0)
  A8  determinism: identical seed -> identical metrics on re-run

Writes results.json next to this file. Stdlib only.
"""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from arxiv_lab.agent.wikitrace import (
    CONTRADICTED, SUPPORTED, UNKNOWN, UNSUPPORTED,
    Insight, Memory, Run, StepRecord, Task,
    compile_memory, descent_triggers, gen_tasks, gold_action,
    run_default, run_experiment, solve, verify_lesson,
)

SEED = 20261112
N_TRAIN = 12
N_TEST = 600


def check(name, cond, detail=""):
    print("%-4s %-60s %s" % ("PASS" if cond else "FAIL", name, detail))
    if not cond:
        raise AssertionError(name + " " + detail)


def main():
    res = run_experiment(SEED, N_TRAIN, N_TEST)
    print(json.dumps(res, indent=2, sort_keys=True))

    wt = res["wikitrace"]
    pl = res["plain"]
    nm = res["none"]
    check("A1 WT > plain by >=0.10", wt["success"] - pl["success"] >= 0.10,
          "%.3f vs %.3f" % (wt["success"], pl["success"]))
    check("A2 WT > none by >=0.10", wt["success"] - nm["success"] >= 0.10,
          "%.3f vs %.3f" % (wt["success"], nm["success"]))
    check("A3 plain > none", pl["success"] > nm["success"],
          "%.3f vs %.3f" % (pl["success"], nm["success"]))
    check("A4a WT regression == 0", wt["regression"] == 0.0,
          "%.4f" % wt["regression"])
    check("A4b plain regression > 0.05", pl["regression"] > 0.05,
          "%.4f" % pl["regression"])
    check("A5 WT gain > plain gain", wt["gain"] > pl["gain"],
          "%.3f vs %.3f" % (wt["gain"], pl["gain"]))
    check("A7a descent rate > 0.2", wt["descent_rate"] > 0.2,
          "%.3f" % wt["descent_rate"])
    check("A7b corrections > 0", wt["correction_rate"] > 0,
          "%.3f" % wt["correction_rate"])

    # A6: evidence-check unit cases on a hand-built memory.
    task = Task(0, 1)  # attr=1 -> verify gold = 1
    # Source run: locate chose 0 (ok), operate chose 1 (ok), verify chose 1 (ok).
    run = Run(0, Task(0, 1), [StepRecord(0, 0, True, True),
                               StepRecord(1, 1, True, True),
                               StepRecord(2, 1, True, True)])
    mem = Memory([], {0: "comp"}, {0: run})
    valid = Insight("I-v", 2, 1, ("attr", 1), 0, True)
    check("A6a valid narrow -> SUPPORTED",
          verify_lesson(valid, 1, mem)[0] == SUPPORTED)
    overgen = Insight("I-o", 2, 1, "any", 0, True)
    check("A6b over-generalized -> UNSUPPORTED",
          verify_lesson(overgen, 2, mem)[0] == UNSUPPORTED)
    corrupt = Insight("I-c", 2, 0, ("attr", 1), 0, True)
    check("A6c corrupted claim -> CONTRADICTED",
          verify_lesson(corrupt, 1, mem)[0] == CONTRADICTED)
    # Failed-run step: verify chose 2, failed.
    run2 = Run(1, Task(1, 1), [StepRecord(0, 0, True, True),
                                StepRecord(1, 1, True, True),
                                StepRecord(2, 2, False, True)])
    mem2 = Memory([], {1: "comp"}, {1: run2})
    failed_claim = Insight("I-f", 2, 2, ("attr", 1), 1, False)
    check("A6d failed-step claim -> CONTRADICTED",
          verify_lesson(failed_claim, 1, mem2)[0] == CONTRADICTED)
    unobs = Run(2, Task(2, 1), [StepRecord(2, 1, True, False)])
    mem3 = Memory([], {2: "comp"}, {2: unobs})
    check("A6e unobserved step -> UNKNOWN",
          verify_lesson(Insight("I-u", 2, 1, ("attr", 1), 2, True), 1, mem3)[0]
          == UNKNOWN)
    # Conflict resolves to the supported lesson (corrupt lesson listed
    # first, so blind-follow would take the wrong action).
    mem4 = Memory([corrupt, valid], {0: "comp"}, {0: run})
    ok, stats = solve(Task(9, 1), mem4, "wikitrace", SEED)
    check("A6f conflict -> supported lesson followed, blind action corrected",
          stats["descents"] == 1 and stats["corrections"] == 1,
          "descents=%d corrections=%d" % (stats["descents"],
                                           stats["corrections"]))
    check("A6g failed-run lesson triggers descent",
          descent_triggers([failed_claim]) is True)
    check("A6h single valid narrow lesson does not trigger descent",
          descent_triggers([valid]) is False)

    # A8: determinism.
    res2 = run_experiment(SEED, N_TRAIN, N_TEST)
    check("A8 determinism", res2 == res)

    # The session copy (sessions/2026-11-12-2611.14320-wikitrace/) writes
    # results.json; this graduated copy only asserts.
    print("ALL ASSERTIONS PASSED")


if __name__ == "__main__":
    main()
