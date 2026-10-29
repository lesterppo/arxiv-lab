"""Session experiment for CASEL (arXiv:2610.36570): regenerate results.json.

Run from the repo root: python3 sessions/2026-10-29-2610.36570-casel/experiment_casel.py
"""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from arxiv_lab.harness.casel import (CANONICAL_SEED, run_casel_experiment,
                                     risk_ledger_check,
                                     threshold_validity_check)

if __name__ == "__main__":
    out = {
        "paper": "2610.36570",
        "canonical_seed": CANONICAL_SEED,
        "policies": run_casel_experiment(CANONICAL_SEED),
        "threshold_check": threshold_validity_check(),
        "ledger_check": risk_ledger_check(),
    }
    here = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(here, "results.json"), "w") as f:
        json.dump(out, f, indent=1, sort_keys=True)
    print(json.dumps(out, indent=1, sort_keys=True))
