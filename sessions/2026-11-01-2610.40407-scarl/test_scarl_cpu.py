"""Deterministic CPU test for SCARL (2610.40407). Prints measured numbers
and asserts the paper's estimator properties at bounded scale:
  - reshaping never re-signs an advantage, preserves each class's budget,
    and reduces exactly to the base estimator when confidences agree;
  - a misaligned (low-confidence) success gets a larger |advantage| than
    an aligned (high-confidence) success with the same base advantage;
  - in a paired softmax-policy step, SCARL preserves entropy better than
    the base estimator, moves low-confidence successes up more and
    high-confidence successes up less, and corrects confident errors
    harder than exploratory ones.
"""
import json, sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scarl import run_experiment

if __name__ == "__main__":
    r = run_experiment()
    print(json.dumps(r, indent=1))
    assert r["s1_sign_error"] == 0.0, "SCARL must never re-sign"
    assert r["s1_budget_error"] < 1e-9, "SCARL must preserve class budgets"
    assert r["s1_reduction_error"] < 1e-9, "SCARL must reduce to base when confidences agree"
    assert r["misaligned_to_aligned_ratio"] > 1.0, "misaligned samples must absorb more force"
    assert r["s2_entropy_scarl_minus_base"] > 0.0, "SCARL must preserve entropy better than base"
    assert r["s2_positive_spread_scarl_minus_base"] < 0.0, "SCARL must equalize same-advantage confidences"
    assert r["s2_low_conf_success_extra_gain"] > 0.0, "low-confidence successes must gain more under SCARL"
    assert r["s2_high_conf_success_extra_gain"] < 0.0, "high-confidence successes must gain less under SCARL"
    assert r["s2_confident_error_extra_change"] < 0.0, "confident errors must be corrected harder under SCARL"
    print("SCARL_CPU_ALL_PASS")
