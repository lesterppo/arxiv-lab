#!/usr/bin/env python3
"""Deterministic CPU test for Termination Lag (arXiv:2611.14960).

Pre-registered assertions (paper figures in brackets):
  A1  first-edit solve share within [0.30, 0.85] [paper: 38-82% of
      successful runs solve on the initial edit]
  A2  continue-policy median post-solution share > 0.30 [paper: median
      35-37% Qwen3-Coder, 54-60% Laguna-XS.2]
  A3  continue-policy mean lag > 2 steps (substantial wasted effort)
  A4  stop-on-pass: lag == 1 on every solved run, and its final-correct
      rate >= continue's (stopping early never hurts final correctness)
  A5  continue-policy undo-to-incorrect rate in (0, 0.15] [paper: 1-8%];
      stop-on-pass undo rate == 0
  A6  verify + explore dominate post-solution steps under continue
      (combined share > 0.5) [paper: agents spend most post-solution
      steps verifying and exploring]
  A7  unit cases: verifier passes correct / fails wrong patch; a re-edit
      can undo a correct patch and a later re-edit can re-fix it;
      stop-on-pass never re-edits; fixed-budget runs exactly the budget
  A8  determinism: identical seed -> identical metrics on re-run
  A9  sensitivity: lag grows with mean_extra (1 < 3 < 6) and final-correct
      rate does not improve with more post-solution steps

Writes results.json next to this file. Stdlib only.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from termination_lag import (
    MASK64, P_FIRST, RNG, Task, compare_policies, gen_tasks, run_experiment,
    run_task, verify,
)

SEED = 20261113
N_TASKS = 600

PASSED = []
FAILED = []


def check(name, cond, detail=""):
    (PASSED if cond else FAILED).append(name)
    print(f"  [{'PASS' if cond else 'FAIL'}] {name} {detail}")


def main():
    tasks = gen_tasks(N_TASKS, SEED)
    results = compare_policies(tasks, SEED)
    cont = results["continue"]
    stop = results["stop_on_pass"]
    fixed = results["fixed_budget"]
    print("main experiment (600 tasks, seed 20261113):")
    for p, m in results.items():
        print(f"  {p:13s} solved={m['n_solved']:3d} first_edit={m['first_edit_solve_share']:.3f} "
              f"mean_lag={m['mean_lag']:.2f} med_lag={m['median_lag']:.1f} "
              f"med_post={m['median_post_share']:.3f} final_ok={m['final_correct_rate']:.3f} "
              f"undo_bad={m['undo_to_incorrect_rate']:.3f} steps={m['mean_total_steps']:.2f} "
              f"verify={m['post_verify_share']:.2f} explore={m['post_explore_share']:.2f} "
              f"reedit={m['post_reedit_share']:.2f}")

    print("assertions:")
    check("A1 first-edit solve share in [0.30, 0.85]",
          0.30 <= cont["first_edit_solve_share"] <= 0.85,
          f"got {cont['first_edit_solve_share']:.3f}")
    check("A2 continue median post-solution share > 0.30",
          cont["median_post_share"] > 0.30,
          f"got {cont['median_post_share']:.3f}")
    check("A3 continue mean lag > 2 steps",
          cont["mean_lag"] > 2.0, f"got {cont['mean_lag']:.2f}")
    check("A4a stop-on-pass median lag == 1",
          stop["median_lag"] == 1.0 and stop["mean_lag"] == 1.0,
          f"got mean {stop['mean_lag']:.2f} median {stop['median_lag']:.1f}")
    check("A4b stop-on-pass final-correct >= continue final-correct",
          stop["final_correct_rate"] >= cont["final_correct_rate"],
          f"{stop['final_correct_rate']:.3f} vs {cont['final_correct_rate']:.3f}")
    check("A5a continue undo-to-incorrect in (0, 0.15]",
          0.0 < cont["undo_to_incorrect_rate"] <= 0.15,
          f"got {cont['undo_to_incorrect_rate']:.3f}")
    check("A5b stop-on-pass undo-to-incorrect == 0",
          stop["undo_to_incorrect_rate"] == 0.0,
          f"got {stop['undo_to_incorrect_rate']:.3f}")
    check("A6 verify+explore dominate post-solution steps (> 0.5)",
          cont["post_verify_share"] + cont["post_explore_share"] > 0.5,
          f"got {cont['post_verify_share'] + cont['post_explore_share']:.3f}")
    # A7 unit cases
    check("A7a verifier: correct patch passes, wrong fails",
          verify(True) and not verify(False))
    rng = RNG(7)
    t = Task(0, "easy")
    r = run_task(t, "continue", rng, mean_extra=6.0, p_undo=1.0)
    check("A7b p_undo=1 forces undos whenever re-edits occur",
          r.solved and r.n_undos >= 1, f"got {r.n_undos} undos")
    rng2 = RNG(7)
    r2 = run_task(Task(0, "easy"), "stop_on_pass", rng2)
    check("A7c stop-on-pass never re-edits after solving",
          r2.solved and r2.n_reedit == 0 and r2.n_verify == 1)
    rng3 = RNG(7)
    r3 = run_task(Task(0, "easy"), "fixed_budget", rng3, fixed_budget=10)
    check("A7d fixed-budget solved run ends at exactly 10 steps",
          (not r3.solved) or r3.steps == 10, f"got {r3.steps}")
    # A8 determinism
    m2 = compare_policies(tasks, SEED)
    check("A8 same seed -> identical metrics",
          m2 == results)
    # A9 sensitivity sweep over mean_extra
    lags, finals = [], []
    for me in (1.0, 3.0, 6.0):
        m = run_experiment("continue", tasks, SEED, mean_extra=me)
        lags.append(m["mean_lag"])
        finals.append(m["final_correct_rate"])
    print(f"  sensitivity mean_extra 1/3/6: mean_lag={['%.2f' % x for x in lags]} "
          f"final_ok={['%.3f' % x for x in finals]}")
    check("A9a mean lag grows with mean_extra",
          lags[0] < lags[1] < lags[2], f"got {['%.2f' % x for x in lags]}")
    check("A9b more post-solution steps do not improve final correctness",
          finals[2] <= finals[0] + 1e-9,
          f"got {['%.3f' % x for x in finals]}")

    out = {
        "paper": "2611.14960",
        "seed": SEED,
        "n_tasks": N_TASKS,
        "policies": results,
        "sensitivity_mean_extra": {"1.0": lags[0], "3.0": lags[1], "6.0": lags[2]},
        "assertions_passed": PASSED,
        "assertions_failed": FAILED,
    }
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "results.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n{len(PASSED)} passed, {len(FAILED)} failed")
    if FAILED:
        print("FAILED:", FAILED)
        sys.exit(1)


if __name__ == "__main__":
    main()
