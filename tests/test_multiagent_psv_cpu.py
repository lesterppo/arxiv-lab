"""CPU tests for PSV (arXiv 2610.40649).

Logit-PSV (frozen model, validation selection, majority vote) beats the
frozen single model and full-weight PSV at bounded scale, selection beats
vote-all, and voting does not lose to the best single member.

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

from arxiv_lab.multiagent.psv import run_experiment


def check(name, cond):
    print(("PASS " if cond else "FAIL ") + name)
    if not cond:
        raise SystemExit(1)


def main():
    if not HAS_NUMPY:
        print("SKIP: numpy not installed")
        return
    r = run_experiment()
    m = r["means"]
    print("means:", {k: round(v, 4) for k, v in m.items()})
    check("logit-PSV beats single model", m["logit_selvote"] > m["base"] + 0.01)
    check("logit-PSV beats weight-PSV", m["logit_selvote"] > m["weight_selvote"] + 0.01)
    check("selection beats vote-all", m["logit_selvote"] > m["logit_voteall"] + 0.005)
    check("voting keeps up with best member", m["logit_selvote"] >= m["logit_best"] - 0.005)
    check("logit wins most runs vs weight", r["logit_beats_weight_runs"] >= 7)
    check("logit wins most runs vs base", r["logit_beats_base_runs"] >= 7)
    print("PSV_GRADUATED_ALL_PASS")


if __name__ == "__main__":
    main()
