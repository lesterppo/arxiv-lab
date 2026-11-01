"""CPU tests for SCARL (arXiv 2610.40407).

Estimator properties: reshaping never re-signs, preserves each class's
advantage budget, reduces exactly to the base estimator when confidences
agree, misaligned samples absorb more force, and a paired policy step
preserves entropy better while moving low-confidence successes up more
and correcting confident errors harder.

Requires numpy (optional dependency); skipped gracefully without it.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

from arxiv_lab.training.scarl import run_experiment


def check(name, cond):
    print(("PASS " if cond else "FAIL ") + name)
    if not cond:
        raise SystemExit(1)


def main():
    if not HAS_NUMPY:
        print("SKIP: numpy not installed")
        return
    r = run_experiment()
    print("results:", {k: v for k, v in r.items() if not isinstance(v, list)})
    check("never re-signs", r["s1_sign_error"] == 0.0)
    check("preserves class budgets", r["s1_budget_error"] < 1e-9)
    check("reduces to base when confidences agree", r["s1_reduction_error"] < 1e-9)
    check("misaligned absorbs more force", r["misaligned_to_aligned_ratio"] > 1.0)
    check("preserves entropy better than base", r["s2_entropy_scarl_minus_base"] > 0.0)
    check("equalizes same-advantage confidences", r["s2_positive_spread_scarl_minus_base"] < 0.0)
    check("low-confidence successes gain more", r["s2_low_conf_success_extra_gain"] > 0.0)
    check("high-confidence successes gain less", r["s2_high_conf_success_extra_gain"] < 0.0)
    check("confident errors corrected harder", r["s2_confident_error_extra_change"] < 0.0)
    print("SCARL_GRADUATED_ALL_PASS")


if __name__ == "__main__":
    main()
