"""CPU test for the graduated student-teaching module (arXiv:2611.13144).

Same assertions as the session test (sessions/2026-11-11-2611.13144-
teaching-harness/test_sat_cpu.py), run against the src/ copy. Run from the
repo root: python3 tests/test_student_teaching_cpu.py
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from arxiv_lab.agent import student_teaching as st

SEED = 20261111
N_DEV = 120
N_TEST = 200


def main():
    dev_tasks = st.gen_tasks(SEED, N_DEV)
    test_tasks = st.gen_tasks(SEED + 5000, N_TEST)
    teacher = st.Teacher(seed=SEED)

    student_a = st.StudentAgent({"L1", "L3", "L5"})
    student_b = st.StudentAgent({"L2", "L4"})

    acc_a0, _ = st.run_batch(student_a, test_tasks)
    acc_b0, _ = st.run_batch(student_b, test_tasks)

    lib_a, rep_a = teacher.teach(student_a, dev_tasks)
    acc_a_sat, _ = st.run_batch(st.StudentAgent(student_a.limitations, set(lib_a)), test_tasks)

    lib_pf = st.profile_free_teach()
    acc_a_pf, _ = st.run_batch(st.StudentAgent(student_a.limitations, set(lib_pf)), test_tasks)

    lib_po, _ = st.profile_only_teach(student_a, dev_tasks)
    acc_a_po, _ = st.run_batch(st.StudentAgent(student_a.limitations, set(lib_po)), test_tasks)

    wrong = [l for l in st.LIMITATIONS if l not in student_a.limitations]
    lib_mis, _ = st.misleading_teach(student_a, dev_tasks, wrong)

    lib_b, _ = teacher.teach(student_b, dev_tasks)
    acc_b_sat, _ = st.run_batch(st.StudentAgent(student_b.limitations, set(lib_b)), test_tasks)
    acc_a_with_b, _ = st.run_batch(st.StudentAgent(student_a.limitations, set(lib_b)), test_tasks)
    acc_b_with_a, _ = st.run_batch(st.StudentAgent(student_b.limitations, set(lib_a)), test_tasks)

    probe_ok = probe_tot = 0
    for lim, pr in rep_a["probes"].items():
        probe_tot += 1
        probe_ok += int(pr["confirmed"] == (lim in student_a.limitations))
    probe_acc = probe_ok / probe_tot if probe_tot else 0.0

    dev2 = st.gen_tasks(SEED, N_DEV)
    test2 = st.gen_tasks(SEED + 5000, N_TEST)
    lib_a2, _ = st.Teacher(seed=SEED).teach(st.StudentAgent({"L1", "L3", "L5"}), dev2)
    acc_a_sat2, _ = st.run_batch(st.StudentAgent({"L1", "L3", "L5"}, set(lib_a2)), test2)

    checks = [
        ("A1 sat>profile_free>untaught (A)", acc_a_sat > acc_a_pf > acc_a0),
        ("A2 misleading admits 0", len(lib_mis) == 0),
        ("A3 probe accuracy>=0.80", probe_acc >= 0.80),
        ("A4 exchange: A own>B lib", acc_a_sat > acc_a_with_b),
        ("A4 exchange: B own>A lib", acc_b_sat > acc_b_with_a),
        ("A5 profile-only < full SAT", acc_a_po < acc_a_sat),
        ("A6 admitted deltas>=0.15",
         all(r["delta"] >= 0.15 for r in rep_a["admissions"].values() if r["admitted"])
         and sum(1 for r in rep_a["admissions"].values() if r["admitted"]) == len(lib_a)),
        ("A7 deterministic rerun", lib_a2 == lib_a and acc_a_sat2 == acc_a_sat),
    ]
    ok = True
    for name, cond in checks:
        print(("PASS " if cond else "FAIL ") + name)
        ok = ok and cond
    print(f"untaught_A={acc_a0:.3f} profile_free_A={acc_a_pf:.3f} "
          f"profile_only_A={acc_a_po:.3f} sat_A={acc_a_sat:.3f} sat_B={acc_b_sat:.3f} "
          f"A_with_B={acc_a_with_b:.3f} B_with_A={acc_b_with_a:.3f}")
    print(f"libraries: SAT-A={lib_a} SAT-B={lib_b} misleading admitted={lib_mis}")
    if not ok:
        print("ASSERTIONS FAILED")
        sys.exit(1)
    print("ALL ASSERTIONS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
