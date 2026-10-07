#!/usr/bin/env python3
"""Session snapshot 2026-10-07 -- arXiv 2610.08048 DAEDALUS (AGENT track).

Verbatim snapshot of the experiment that produced results_daedalus.json.
Run:  cd ~/workspace/arxiv-lab-build && python3 sessions/2026-10-07-2610.08048-daedalus/test_session_daedalus.py
"""
import sys
import os
import json
import random
from collections import Counter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from arxiv_lab.daedalus import run_daedalus, make_task, evaluate, CONVENTIONS

SEEDS = (7, 123)
N_SESSIONS = 24
N_HELDOUT_PER_CONV = 16
N_ROLLOUTS = 5


def main():
    out = {}
    for seed in SEEDS:
        bank, log = run_daedalus(n_sessions=N_SESSIONS, seed=seed)
        rng = random.Random(99)
        heldout = []
        for conv in CONVENTIONS:
            heldout += [make_task("h-%s-%d" % (conv, i), conv, 7, rng)
                        for i in range(N_HELDOUT_PER_CONV)]
        base = evaluate(heldout, solver_seed=seed, heuristics=(),
                        n_rollouts=N_ROLLOUTS)
        mem = evaluate(heldout, solver_seed=seed,
                       heuristics=bank.consolidated, n_rollouts=N_ROLLOUTS)
        out[str(seed)] = {
            "bank": [h.convention for h in bank.consolidated],
            "outcomes": dict(Counter(e["outcome"] for e in log)),
            "base_success": round(base["success_rate"], 4),
            "daedalus_success": round(mem["success_rate"], 4),
            "base_pass5": round(base["pass_5"], 4),
            "daedalus_pass5": round(mem["pass_5"], 4),
        }
        print("seed %d: bank=%s base=%.4f daedalus=%.4f pass^5 %.4f -> %.4f"
              % (seed, out[str(seed)]["bank"], base["success_rate"],
                 mem["success_rate"], base["pass_5"], mem["pass_5"]))
    here = os.path.dirname(os.path.abspath(__file__))
    json.dump(out, open(os.path.join(here, "results_daedalus.json"), "w"),
              indent=2)
    print("wrote results_daedalus.json")


if __name__ == "__main__":
    main()
