"""CPU test for termination lag (arXiv:2611.14960), graduated from the
2026-11-13 daily session. Deterministic; stdlib only.

Re-runs the session's pre-registered assertions A1-A9 against the graduated
module `arxiv_lab.harness.termination_lag` (same mechanism, same seed and
task count), so the library copy cannot drift from the session copy.

Run:  python3 tests/test_harness_termination_lag_cpu.py
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from arxiv_lab.harness.termination_lag import (
    MASK64, P_UNDO, RNG, Task, compare_policies, gen_tasks, run_experiment,
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
    print("main experiment (600 tasks, seed 20261113):")
    for p, m in results.items():
        print(f"  {p:13s} solved={m['n_solved']:3d} first_edit={m['first_edit_solve_share']:.3f} "
              f"mean_lag={m['mean_lag']:.2f} med_lag={m['median_lag']:.1f} "
              f"med_post={m['median_post_share']:.3f} final_ok={m['final_correct_rate']:.3f} "
              f"undo_bad={m['undo_to_incorrect_rate']:.3f} steps={m['mean_total_steps']:.2f}")

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
    check("A7a verifier: correct patch passes, wrong fails",
          verify(True) and not verify(False))
    rng = RNG(7)
    r = run_task(Task(0, "easy"), "continue", rng, mean_extra=6.0, p_undo=1.0)
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
    m2 = compare_policies(tasks, SEED)
    check("A8 same seed -> identical metrics", m2 == results)
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

    print(f"\n{len(PASSED)} passed, {len(FAILED)} failed")
    if FAILED:
        print("FAILED:", FAILED)
        sys.exit(1)


if __name__ == "__main__":
    main()
