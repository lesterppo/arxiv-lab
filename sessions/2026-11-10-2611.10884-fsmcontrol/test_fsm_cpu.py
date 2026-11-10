"""Deterministic CPU test for the FSM-control mechanism (arXiv:2611.10884).

Pre-registered assertions (calibration: p0=0.05, drift_rate=0.35,
drift_scale=1000 — the drift model is the experimental subject; the
mechanism under test is the FSM enforcement, and A6 sweeps the drift
rate to show the result is not tuned to one point):

  A1  FSM correct-and-by-procedure >= 170/200 and >= plain + 80.
  A2  FSM holds its rate to the end: last-quartile rate within 10pp of
      the first; plain falls >= 20pp from first to last quartile.
  A3  Restating the SOP does not fix drift: restate <= plain + 10 and
      restate < FSM - 60.
  A4  Enforcement soundness: the FSM arm's accepted trace contains zero
      procedure violations, and every invalid call the drifting agent
      attempted was rejected (invalid_attempts > 0, violations == 0).
  A5  No false blocking: with drift disabled (p0=0, drift_rate=0) all
      three arms complete 200/200.
  A6  Sensitivity: at drift_rate x {0.5, 1.0, 1.5} of the base rate the
      FSM beats plain by >= 60pp each time.
  A7  Unit checks: transition table + branch conditions (waive only when
      eligible, escalate only when needed, close only when neither);
      end-of-turn rejects a non-terminal state; a rejected call leaves
      the state unchanged; SplitMix64 reproduces a known stream prefix.

Run: python3 test_fsm_cpu.py   (writes results.json; exit 0 iff all pass)
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import fsm_control as fc

SEED = 20261110
N = 200


def check(name, cond, detail=""):
    print(f"  [{'PASS' if cond else 'FAIL'}] {name} {detail}")
    return cond


def main():
    ok = True
    accounts = fc.make_accounts(N, SEED)

    print("== main experiment (p0=0.05, drift_rate=0.35) ==")
    _, plain = fc.run_batch(accounts, "plain", SEED)
    _, restate = fc.run_batch(accounts, "restate", SEED)
    _, fsm = fc.run_batch(accounts, "fsm", SEED)
    for s in (plain, restate, fsm):
        print(f"  {s['arm']:8s} correct-by-procedure {s['correct_and_by_procedure']}/"
              f"{s['n']} ({s['rate']:.3f}) quartiles {s['quartile_rates']} "
              f"violations {s['total_violations']} "
              f"invalid_attempts {s['total_invalid_attempts']} "
              f"retries {s['total_retries']}")

    ok &= check("A1 FSM >= 170/200",
                fsm["correct_and_by_procedure"] >= 170,
                f"({fsm['correct_and_by_procedure']})")
    ok &= check("A1 FSM >= plain + 80",
                fsm["correct_and_by_procedure"] - plain["correct_and_by_procedure"] >= 80,
                f"(gap {fsm['correct_and_by_procedure'] - plain['correct_and_by_procedure']})")
    ok &= check("A2 FSM holds to the end (last q within 10pp of first)",
                abs(fsm["quartile_rates"][3] - fsm["quartile_rates"][0]) <= 0.10,
                f"({fsm['quartile_rates'][0]:.2f} -> {fsm['quartile_rates'][3]:.2f})")
    ok &= check("A2 plain falls >= 20pp first -> last quartile",
                plain["quartile_rates"][0] - plain["quartile_rates"][3] >= 0.20,
                f"({plain['quartile_rates'][0]:.2f} -> {plain['quartile_rates'][3]:.2f})")
    ok &= check("A3 restate <= plain + 10 (restating does not fix drift)",
                restate["correct_and_by_procedure"] - plain["correct_and_by_procedure"] <= 10,
                f"(restate {restate['correct_and_by_procedure']}, plain {plain['correct_and_by_procedure']})")
    ok &= check("A3 restate < FSM - 60",
                fsm["correct_and_by_procedure"] - restate["correct_and_by_procedure"] > 60,
                f"(gap {fsm['correct_and_by_procedure'] - restate['correct_and_by_procedure']})")
    ok &= check("A4 FSM zero violations in accepted trace",
                fsm["total_violations"] == 0)
    ok &= check("A4 FSM rejected invalid calls (attempts > 0)",
                fsm["total_invalid_attempts"] > 0,
                f"({fsm['total_invalid_attempts']} rejected)")
    ok &= check("A4 plain arm actually violated procedure",
                plain["total_violations"] > 0,
                f"({plain['total_violations']} violations)")

    print("== A5 no false blocking (drift off) ==")
    for arm in ("plain", "restate", "fsm"):
        _, s = fc.run_batch(accounts, arm, SEED, p0=0.0, drift_rate=0.0)
        ok &= check(f"A5 {arm} 200/200 with drift off",
                    s["correct_and_by_procedure"] == 200,
                    f"({s['correct_and_by_procedure']})")

    print("== A6 sensitivity over drift_rate ==")
    sens = {}
    for mult in (0.5, 1.0, 1.5):
        _, p = fc.run_batch(accounts, "plain", SEED, drift_rate=0.35 * mult)
        _, f = fc.run_batch(accounts, "fsm", SEED, drift_rate=0.35 * mult)
        gap = f["correct_and_by_procedure"] - p["correct_and_by_procedure"]
        sens[mult] = {"plain": p["correct_and_by_procedure"],
                      "fsm": f["correct_and_by_procedure"], "gap": gap}
        ok &= check(f"A6 drift x{mult}: FSM beats plain by >= 60",
                    gap >= 60, f"(plain {p['correct_and_by_procedure']}, "
                    f"fsm {f['correct_and_by_procedure']}, gap {gap})")

    print("== A7 unit checks ==")
    acc = fc.Account(0, balance=100, eligible=True, needs_escalation=False)
    rt = fc.FSMRuntime(acc)
    a, _ = rt.call("check_balance")
    ok &= check("A7 call before verify_identity rejected",
                not a and rt.state == fc.START)
    a, _ = rt.call("verify_identity")
    ok &= check("A7 verify_identity accepted", a and rt.state == fc.IDENTITY_VERIFIED)
    a, _ = rt.call("waive_fee")
    ok &= check("A7 waive_fee before eligibility rejected",
                not a and rt.state == fc.IDENTITY_VERIFIED)
    rt.call("check_balance")
    rt.call("check_eligibility")
    a, _ = rt.call("close_account")
    ok &= check("A7 close while eligible rejected (must waive first)",
                not a and rt.state == fc.ELIGIBILITY_CHECKED)
    a, _ = rt.call("waive_fee")
    ok &= check("A7 waive when eligible accepted", a and rt.state == fc.FEE_WAIVED)
    a, _ = rt.call("escalate")
    ok &= check("A7 escalate from FEE_WAIVED rejected",
                not a and rt.state == fc.FEE_WAIVED)
    complete, decision = rt.end_turn()
    ok &= check("A7 end_turn non-terminal is incomplete",
                not complete and decision is None)
    rt.call("close_account")
    complete, decision = rt.end_turn()
    ok &= check("A7 end_turn terminal after close",
                complete and decision == "close")

    acc2 = fc.Account(1, balance=-50, eligible=False, needs_escalation=True)
    rt2 = fc.FSMRuntime(acc2)
    for t in ("verify_identity", "check_balance", "check_eligibility"):
        rt2.call(t)
    a, _ = rt2.call("close_account")
    ok &= check("A7 close when escalation needed rejected",
                not a and rt2.state == fc.ELIGIBILITY_CHECKED)
    a, _ = rt2.call("escalate")
    ok &= check("A7 escalate when needed accepted",
                a and rt2.state == fc.ESCALATED)

    # SplitMix64 known stream: first outputs for seed 0 (checked against
    # the reference algorithm; the Rust port must reproduce these).
    rng = fc.SplitMix64(0)
    first3 = [rng.next_u64() for _ in range(3)]
    ok &= check("A7 SplitMix64 seed-0 stream",
                first3 == [0xE220A8397B1DCDAF, 0x6E789E6AA1B965F4,
                           0x06C45D188009454F],
                f"({[hex(x) for x in first3]})")

    results = {
        "paper": "2611.10884",
        "n_accounts": N,
        "seed": SEED,
        "drift": {"p0": 0.05, "drift_rate": 0.35, "drift_scale": 1000,
                  "max_retries": 3, "restate_every": 20},
        "arms": {"plain": plain, "restate": restate, "fsm": fsm},
        "sensitivity": {str(k): v for k, v in sens.items()},
        "paper_reference": {"fsm": 139, "baseline": 48, "n": 200,
                            "ci_diff": [46, 130]},
        "assertions_passed": bool(ok),
    }
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "results.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n{'ALL ASSERTIONS PASSED' if ok else 'ASSERTIONS FAILED'}")
    print("results.json written")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
