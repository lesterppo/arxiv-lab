"""Session runner for arXiv:2610.36114 (Context Is a Pipeline, Not a
Window). Imports the graduated mechanism from src/arxiv_lab/harness/
context_attrition.py, runs the full 3-policy x 3-position experiment on
the canonical seed plus robustness seeds, and writes results.json next
to this script.

Run:  python3 run_experiment.py   (from this session directory, with the
      repo root's src/ on the path via the insertion below)
"""

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from arxiv_lab.harness.context_attrition import (  # noqa: E402
    CANONICAL_SEED,
    ROBUSTNESS_SEEDS,
    format_table,
    run_experiment,
)

N_TASKS = 300


def main():
    out = {"n_tasks_per_cell": N_TASKS, "seeds": {}}
    for seed in (CANONICAL_SEED,) + ROBUSTNESS_SEEDS:
        rows = run_experiment(n_tasks=N_TASKS, seed=seed)
        out["seeds"][str(seed)] = rows
        print("seed %d" % seed)
        print(format_table(rows))
        print()
    here = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(here, "results.json"), "w") as f:
        json.dump(out, f, indent=1)
    print("wrote results.json")


if __name__ == "__main__":
    main()
