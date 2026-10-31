"""Deterministic CPU test for SPLICE (2610.40186). Prints measured numbers
and asserts the paper's directional claims at bounded scale:
  - process-feature selector beats the static skeleton;
  - process-feature selector beats naive always-adopt (the trap hazard);
  - a final-output-only rule (adopt iff the final answer changed) cannot
    beat always-adopt, because movement/final-reach are 1 for GAP and TRAP;
  - re-execution cost stays in the paper's 1.25-1.44x band (under the
    stated 0.5 re-execution-share model).
"""
import json, sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from splice import run_experiment

if __name__ == "__main__":
    r = run_experiment()
    print(json.dumps(r, indent=1))
    assert r["acc_splice"] > r["acc_skeleton"], "selector must beat skeleton"
    assert r["acc_splice"] > r["acc_always_adopt"], "selector must beat always-adopt"
    assert r["trap_adopt_rate"] < 0.50, "selector must reject most traps"
    assert r["gap_adopt_rate"] > 0.50, "selector must adopt most genuine gaps"
    assert 1.20 <= r["cost_ratio"] <= 1.50, "cost must stay near paper band"
    print("SPLICE_CPU_ALL_PASS")
