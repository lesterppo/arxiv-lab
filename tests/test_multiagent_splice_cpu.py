"""CPU tests for SPLICE (arXiv 2610.40186).

Directional checks across seeds: the process-feature selector beats the
static skeleton and naive always-adopt, rejects most traps, adopts most
genuine gaps, and keeps re-execution cost near the paper's band.

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

from arxiv_lab.multiagent.splice import run_experiment


def check(name, cond):
    print(("PASS " if cond else "FAIL ") + name)
    if not cond:
        raise SystemExit(1)


def main():
    if not HAS_NUMPY:
        print("SKIP test_multiagent_splice_cpu (numpy not installed)")
        return
    r = run_experiment()
    print("canonical seed: splice=%.4f skeleton=%.4f always=%.4f cost=%.3f"
          % (r["acc_splice"], r["acc_skeleton"], r["acc_always_adopt"],
             r["cost_ratio"]))
    check("selector beats skeleton", r["acc_splice"] > r["acc_skeleton"])
    check("selector beats always-adopt",
          r["acc_splice"] > r["acc_always_adopt"])
    check("trap adoption < 50%", r["trap_adopt_rate"] < 0.50)
    check("gap adoption > 50%", r["gap_adopt_rate"] > 0.50)
    check("cost in band", 1.20 <= r["cost_ratio"] <= 1.50)
    for seed in (1, 7, 42):
        rr = run_experiment(seed=seed)
        check("seed %d selector beats skeleton" % seed,
              rr["acc_splice"] > rr["acc_skeleton"])
        check("seed %d selector beats always-adopt" % seed,
              rr["acc_splice"] > rr["acc_always_adopt"])
    print("ALL PASS test_multiagent_splice_cpu")


if __name__ == "__main__":
    main()
