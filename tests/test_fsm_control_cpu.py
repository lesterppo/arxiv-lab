"""CPU test for the graduated FSM-control module (arXiv:2611.10884).

Same assertions as the session test (sessions/2026-11-10-2611.10884-
fsmcontrol/test_fsm_cpu.py), run against the src/ copy. Run from the
repo root: python3 tests/test_fsm_control_cpu.py
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from arxiv_lab.agent import fsm_control as fc

SEED = 20261110
N = 200


def main():
    ok = True
    accounts = fc.make_accounts(N, SEED)
    _, plain = fc.run_batch(accounts, "plain", SEED)
    _, restate = fc.run_batch(accounts, "restate", SEED)
    _, fsm = fc.run_batch(accounts, "fsm", SEED)
    print(f"plain {plain['correct_and_by_procedure']}/200 "
          f"restate {restate['correct_and_by_procedure']}/200 "
          f"fsm {fsm['correct_and_by_procedure']}/200")
    checks = [
        ("FSM >= 170/200", fsm["correct_and_by_procedure"] >= 170),
        ("FSM >= plain + 80",
         fsm["correct_and_by_procedure"] - plain["correct_and_by_procedure"] >= 80),
        ("FSM holds to the end",
         abs(fsm["quartile_rates"][3] - fsm["quartile_rates"][0]) <= 0.10),
        ("plain falls >= 20pp",
         plain["quartile_rates"][0] - plain["quartile_rates"][3] >= 0.20),
        ("restate does not fix drift",
         restate["correct_and_by_procedure"] - plain["correct_and_by_procedure"] <= 10),
        ("FSM zero violations", fsm["total_violations"] == 0),
        ("FSM rejected invalid calls", fsm["total_invalid_attempts"] > 0),
    ]
    for name, cond in checks:
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}")
        ok &= cond
    # Drift off: every arm perfect (no false blocking).
    for arm in ("plain", "restate", "fsm"):
        _, s = fc.run_batch(accounts, arm, SEED, p0=0.0, drift_rate=0.0)
        cond = s["correct_and_by_procedure"] == 200
        print(f"  [{'PASS' if cond else 'FAIL'}] drift-off {arm} 200/200")
        ok &= cond
    print("ALL PASSED" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
