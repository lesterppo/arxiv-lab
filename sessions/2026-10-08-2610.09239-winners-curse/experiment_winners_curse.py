"""Experiment snapshot for arXiv 2610.09239 — the Winner's Curse in LLM
self-improvement loops (session 2026-10-08-2610.09239-winners-curse).

Runnable from the repo root::

    python3 sessions/2026-10-08-2610.09239-winners-curse/experiment_winners_curse.py

Reproduces the paper's selection-noise mechanism in a scripted statistical
model (see src/arxiv_lab/loops/winners_curse.py for the model description
and the simulation-honesty note). Writes results_winners_curse.json next
to this script. Deterministic (seeded); stdlib + numpy only; no network.
"""

import json
import os
import sys

SESSION_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(SESSION_DIR))
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))

from arxiv_lab.loops import winners_curse as wc  # noqa: E402

N_SEEDS = 300
SEED = 20261008
N_ITEMS_GRID = (16, 64, 256)
RULES = ("greedy", "fresh64", "conservative")


def mean(xs):
    xs = list(xs)
    return sum(xs) / len(xs)


def main():
    results = {
        "paper": "arXiv 2610.09239",
        "title": "The Winner's Curse in LLM Self-Improvement Loops: "
                 "Selection Noise, Lock-in, and Acceptance Rules",
        "n_seeds": N_SEEDS,
        "seed": SEED,
        "model": "scripted statistical stand-in (item-level Bernoulli + "
                 "shared set difficulty); NOT LLMs — see module docstring",
        "defaults": {"n_gens": 8, "n_candidates": 6, "delta": 2.0,
                     "sigma_prop": 3.0, "q0": wc.Q0},
    }

    # Claim 1: correlated candidate errors within a decision.
    results["error_correlation_n16"] = wc.error_correlation(16, seed=7)

    # One-shot winner's curse (equal-quality candidates, pure selection).
    results["one_shot_gap"] = {
        str(n): wc.winners_curse_gap(n, n_candidates=8,
                                     n_seeds=N_SEEDS, seed=SEED)
        for n in N_ITEMS_GRID
    }

    # Claims 2-4 + acceptance-rule comparison: full loops.
    table = {}
    for rule in RULES:
        for n in N_ITEMS_GRID:
            runs = [wc.run_loop_with_rule(rule, n_items=n, seed=SEED + s)
                    for s in range(N_SEEDS)]
            key = f"{rule}/n={n}"
            table[key] = {
                # paper claim 3 headline: final selection-set score minus
                # held-out accuracy
                "mean_selection_bias": mean(r["selection_bias"] for r in runs),
                "mean_reported_gain": mean(r["reported_gain"] for r in runs),
                "mean_true_gain": mean(r["true_gain"] for r in runs),
                "mean_reported_vs_true_bias": mean(
                    r["reported_gain"] - r["true_gain"] for r in runs),
                "mean_harmful_proposal_frac": mean(
                    r["harmful_proposal_frac"] for r in runs),
                "mean_harmful_commit_frac": mean(
                    r["harmful_commit_frac"] for r in runs),
                "mean_n_commits": mean(r["n_commits"] for r in runs),
                # paper claim 4 diagnostic (fresh-64 probe of start/current)
                "mean_fresh64_gain_bias": mean(
                    r["fresh64_gain_bias"] for r in runs),
                "mean_fresh64_gain_abs_err": mean(
                    r["fresh64_gain_abs_err"] for r in runs),
            }
            b = table[key]
            print(f"{key:22s} sel_bias={b['mean_selection_bias']:6.2f} "
                  f"reported={b['mean_reported_gain']:6.2f} "
                  f"true={b['mean_true_gain']:6.2f} "
                  f"harm_commit={b['mean_harmful_commit_frac']:.3f} "
                  f"f64_bias={b['mean_fresh64_gain_bias']:6.2f} "
                  f"f64_|err|={b['mean_fresh64_gain_abs_err']:5.2f}")
    results["loop_table"] = table

    out = os.path.join(SESSION_DIR, "results_winners_curse.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
