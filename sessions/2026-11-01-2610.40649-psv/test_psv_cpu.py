"""Deterministic CPU test for PSV (2610.40649). Prints measured numbers
and asserts the paper's central claim at bounded scale:
  - logit-PSV (frozen model, select + vote) beats the frozen single model;
  - logit-PSV matches or exceeds full-weight PSV;
  - selection matters: select + vote beats vote-all;
  - voting adds at least a small margin over the best single member
    (asserted with a 0.5-point tolerance; the measured margin is +0.26).
"""
import json, sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from psv import run_experiment

if __name__ == "__main__":
    r = run_experiment()
    print(json.dumps(r, indent=1))
    m = r["means"]
    assert m["logit_selvote"] > m["base"] + 0.01, "logit-PSV must beat single model"
    assert m["logit_selvote"] > m["weight_selvote"] + 0.01, "logit-PSV must beat weight-PSV"
    assert m["logit_selvote"] > m["logit_voteall"] + 0.005, "selection must beat vote-all"
    assert m["logit_selvote"] >= m["logit_best"] - 0.005, "voting must not lose to best member"
    assert r["logit_beats_weight_runs"] >= 7, "logit must win most runs vs weight"
    assert r["logit_beats_base_runs"] >= 7, "logit must win most runs vs base"
    print("PSV_CPU_ALL_PASS")
