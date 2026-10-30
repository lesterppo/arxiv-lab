#!/usr/bin/env python3
"""Regenerate results.json for the Masked Erosion session (arXiv:2610.38392).

Canonical run: 20 seeds, both environments, all gates, batch sweep,
high-friction sensitivity arm, and the frozen analytic prediction.
Deterministic (seed 20261030). Writes results.json next to this file.
"""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from arxiv_lab.harness.masked_erosion import run_experiment

if __name__ == "__main__":
    results = run_experiment()
    # JSON keys must be strings: stringify the analytic (safeguard, N) keys.
    results["analytic"] = {"%d,%d" % k: v
                           for k, v in results["analytic"].items()}
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "results.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=1)
    print("wrote", out)
    for row in results["gates"]:
        print("%-14s N=%d %-6s %-7s acc=%.2f depthlost=%.2f "
              "harmStrong=%+.4f harmWeak=%+.4f" % (
                  row["gate"], row["batch_size"], row["env"],
                  "genuine" if row["genuine"] else "chance",
                  row["mean_accepted"], row["frac_depth_lost"],
                  row["mean_harm_strong"], row["mean_harm_weak"]))
    for row in results["batch_sweep"]:
        print("sweep paired N=%-3d %-6s acc=%.2f depthlost=%.2f "
              "exposed=%.2f harmWeak=%+.4f bySaf=%s" % (
                  row["batch_size"], row["env"], row["mean_accepted"],
                  row["frac_depth_lost"], row["frac_fully_exposed"],
                  row["mean_harm_weak"],
                  [round(x, 2) for x in row["accepted_by_safeguard"]]))
    for row in results.get("sensitivity", []):
        print("sensitivity  %-7s %-7s acc=%.2f depthlost=%.2f harmWeak=%+.4f"
              % (row["gate"], "genuine" if row["genuine"] else "chance",
                 row["mean_accepted"], row["frac_depth_lost"],
                 row["mean_harm_weak"]))
    print("deletion:", json.dumps(results["deletion"]))
    print("analytic:", json.dumps(results["analytic"]))
