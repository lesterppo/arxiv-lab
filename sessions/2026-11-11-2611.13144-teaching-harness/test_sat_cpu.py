#!/usr/bin/env python3
"""Deterministic CPU test for Student-Aware Teaching (arXiv:2611.13144).

Pre-registered assertions:
  A1  ordering: SAT success > profile-free > untaught on held-out tasks
  A2  misleading-profile control admits 0 skills
  A3  probes resolve hypotheses with >= 0.80 accuracy
  A4  library exchange: each student does better with its own library
  A5  ablation (profile-only, no probes) < full SAT
  A6  every admitted skill individually raises matched-dev success
      by >= the admission threshold (admission gate sanity check)
  A7  determinism: identical seed -> identical success rates on re-run

Writes results.json next to this file. Stdlib only.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from student_teaching import (
    LIMITATIONS, SKILL_TEXT, StudentAgent, Teacher, gen_tasks,
    profile_free_teach, profile_only_teach, misleading_teach, run_batch,
)

SEED = 20261111
N_DEV = 120
N_TEST = 200


def main():
    dev_tasks = gen_tasks(SEED, N_DEV)
    test_tasks = gen_tasks(SEED + 5000, N_TEST)
    teacher = Teacher(seed=SEED)

    student_a = StudentAgent({"L1", "L3", "L5"})   # missing / stale / math
    student_b = StudentAgent({"L2", "L4"})          # search / retry

    results = {"seed": SEED, "n_dev": N_DEV, "n_test": N_TEST, "arms": {}}

    # ---- untaught baselines
    acc_a0, _ = run_batch(student_a, test_tasks)
    acc_b0, _ = run_batch(student_b, test_tasks)
    results["arms"]["untaught_A"] = acc_a0
    results["arms"]["untaught_B"] = acc_b0

    # ---- SAT (full pipeline) for student A
    lib_a, rep_a = teacher.teach(student_a, dev_tasks)
    sat_a = StudentAgent(student_a.limitations, skills=lib_a)
    acc_a_sat, _ = run_batch(sat_a, test_tasks)
    results["arms"]["sat_A"] = acc_a_sat
    results["sat_A"] = {"library": lib_a, "report": rep_a}

    # ---- profile-free teacher for student A (all skills, no gate)
    lib_pf = profile_free_teach()
    pf_a = StudentAgent(student_a.limitations, skills=lib_pf)
    acc_a_pf, _ = run_batch(pf_a, test_tasks)
    results["arms"]["profile_free_A"] = acc_a_pf
    results["profile_free_A"] = {"library": lib_pf}

    # ---- ablation: profile-only (no probes) for student A
    lib_po, rep_po = profile_only_teach(student_a, dev_tasks)
    po_a = StudentAgent(student_a.limitations, skills=lib_po)
    acc_a_po, _ = run_batch(po_a, test_tasks)
    results["arms"]["profile_only_A"] = acc_a_po
    results["profile_only_A"] = {"library": lib_po, "report": rep_po}

    # ---- misleading-profile control for student A
    wrong = [l for l in LIMITATIONS if l not in student_a.limitations]
    lib_mis, rep_mis = misleading_teach(student_a, dev_tasks, wrong)
    results["misleading_A"] = {"candidate_skills": wrong,
                               "admitted": lib_mis, "report": rep_mis}

    # ---- SAT for student B + library exchange
    lib_b, rep_b = teacher.teach(student_b, dev_tasks)
    sat_b = StudentAgent(student_b.limitations, skills=lib_b)
    acc_b_sat, _ = run_batch(sat_b, test_tasks)
    results["arms"]["sat_B"] = acc_b_sat
    results["sat_B"] = {"library": lib_b, "report": rep_b}
    a_with_b = StudentAgent(student_a.limitations, skills=lib_b)
    b_with_a = StudentAgent(student_b.limitations, skills=lib_a)
    acc_a_with_b, _ = run_batch(a_with_b, test_tasks)
    acc_b_with_a, _ = run_batch(b_with_a, test_tasks)
    results["arms"]["A_with_B_library"] = acc_a_with_b
    results["arms"]["B_with_A_library"] = acc_b_with_a

    # ---- A3 probe accuracy: hypotheses resolved correctly
    true_lims = student_a.limitations
    probe_ok = probe_tot = 0
    for lim, pr in rep_a["probes"].items():
        probe_tot += 1
        # a hypothesis is resolved correctly if the probe verdict matches
        # whether the student actually has the limitation
        probe_ok += int(pr["confirmed"] == (lim in true_lims))
    results["probe_accuracy_A"] = probe_ok / probe_tot if probe_tot else None
    results["probe_detail_A"] = {"correct": probe_ok, "total": probe_tot}

    # ---- A6: admitted-skill individual deltas (from the admission report)
    results["admission_deltas_A"] = {
        lim: r["delta"] for lim, r in rep_a["admissions"].items() if r["admitted"]
    }

    # ---- A7 determinism: re-run the whole SAT arm from scratch
    dev2 = gen_tasks(SEED, N_DEV)
    test2 = gen_tasks(SEED + 5000, N_TEST)
    lib_a2, _ = Teacher(seed=SEED).teach(StudentAgent({"L1", "L3", "L5"}), dev2)
    acc_a_sat2, _ = run_batch(StudentAgent({"L1", "L3", "L5"}, skills=lib_a2), test2)
    results["determinism"] = {"library_rerun": lib_a2, "sat_A_rerun": acc_a_sat2}

    # ---- assertions
    failures = []

    def check(name, cond, detail=""):
        print(("PASS " if cond else "FAIL ") + name + (" | " + str(detail) if detail else ""))
        if not cond:
            failures.append(name)

    check("A1 sat>profile_free>untaught (A)",
          acc_a_sat > acc_a_pf > acc_a0,
          f"sat={acc_a_sat:.3f} pf={acc_a_pf:.3f} untaught={acc_a0:.3f}")
    check("A2 misleading admits 0", len(lib_mis) == 0,
          f"admitted={lib_mis}")
    check("A3 probe accuracy>=0.80",
          results["probe_accuracy_A"] is not None and results["probe_accuracy_A"] >= 0.80,
          f"{probe_ok}/{probe_tot}")
    check("A4 exchange: A own>B lib", acc_a_sat > acc_a_with_b,
          f"own={acc_a_sat:.3f} other={acc_a_with_b:.3f}")
    check("A4 exchange: B own>A lib", acc_b_sat > acc_b_with_a,
          f"own={acc_b_sat:.3f} other={acc_b_with_a:.3f}")
    check("A5 profile-only < full SAT", acc_a_po < acc_a_sat,
          f"po={acc_a_po:.3f} sat={acc_a_sat:.3f}")
    check("A6 admitted deltas>=0.15",
          all(d >= 0.15 for d in results["admission_deltas_A"].values())
          and len(results["admission_deltas_A"]) == len(lib_a),
          str(results["admission_deltas_A"]))
    check("A7 deterministic rerun",
          lib_a2 == lib_a and acc_a_sat2 == acc_a_sat,
          f"lib {lib_a} vs {lib_a2}; acc {acc_a_sat} vs {acc_a_sat2}")

    print("\nARM SUCCESS RATES (n=%d held-out tasks):" % N_TEST)
    for k, v in sorted(results["arms"].items()):
        print("  %-18s %.3f" % (k, v))
    print("SAT-A library:", lib_a, "| SAT-B library:", lib_b)
    print("profile rates A:", {k: round(v, 3) for k, v in rep_a["profile_rates"].items()})
    print("probes A:", rep_a["probes"])
    print("admissions A:", rep_a["admissions"])

    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=1, default=str)
    print("wrote", out)

    if failures:
        print("ASSERTIONS FAILED:", failures)
        sys.exit(1)
    print("ALL ASSERTIONS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
