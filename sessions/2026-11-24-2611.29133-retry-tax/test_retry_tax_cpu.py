"""Deterministic CPU test for arXiv:2611.29133 "The Retry Tax".

Reproduces the paper's core claims on a synthetic task population:

  E0  Closed-form inequalities: cost/completion >= one-shot cost, and
      marginal cost per additional completion nondecreasing in the cap.
  E1  Monte-Carlo simulation of the static policy: same inequalities
      measured from simulated attempts (seeded).
  E2  Replay procedure (the paper's method): per-task attempt-outcome
      sequences, reordered, stopped at verified success or cap,
      averaged — completion rises 59.7%-like -> ~70%, CPC rises.
  E3  Homogeneous population: CPC exactly constant in K (equality case).
  E4  Survivor selection: mean p of unresolved tasks falls with attempts.
  E5  Determinism: a full repeat reproduces every number bit-for-bit.
  E6  Edge cases: p = 0 never completes (marginal cost undefined when
      the increment is zero); p = 1 completes on attempt 1.

Run:  python3 test_retry_tax_cpu.py   (from this session dir)
Writes results.json next to this file. Exits non-zero on any failure.
"""

from __future__ import annotations

import json
import math
import os
import random
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from arxiv_lab.agent import retry_tax as rt

SEED = 20261124
N_TASKS = 20000
CAPS = [1, 2, 3, 4, 5, 6, 8]
COST_USD = 0.418  # calibrated so CPC(1) == paper's $0.70 at C(1) = 0.597


def check(cond, msg):
    if not cond:
        raise AssertionError(msg)
    print(f"  ok: {msg}")


def main():
    t0 = time.time()
    results = {"seed": SEED, "n_tasks": N_TASKS, "caps": CAPS,
               "cost_usd_per_attempt": COST_USD, "experiments": {}}
    ps = rt.calibrated_mixture(N_TASKS, SEED)

    # ---- E0: closed-form inequalities -------------------------------
    print("E0 closed-form inequalities (heterogeneous mixture)")
    cpcs, mcs, comps = [], [], []
    for k in CAPS:
        m = rt.population_metrics(ps, k, COST_USD)
        comps.append(m.completion)
        cpcs.append(m.cost_per_completion)
        if k >= 2:
            mcs.append(rt.marginal_cost(ps, k, COST_USD))
    for k, c, cpc in zip(CAPS, comps, cpcs):
        print(f"  cap {k}: completion {c:.4f}  CPC ${cpc:.4f}")
    print("  marginal costs:", [f"${m:.4f}" for m in mcs])
    check(all(c >= cpcs[0] - 1e-12 for c in cpcs),
          "CPC(K) >= CPC(1) for every cap")
    check(all(cpcs[i + 1] >= cpcs[i] - 1e-12 for i in range(len(cpcs) - 1)),
          "CPC nondecreasing in cap")
    check(all(mcs[i + 1] >= mcs[i] - 1e-12 for i in range(len(mcs) - 1)),
          "marginal cost nondecreasing in cap")
    check(abs(comps[0] - rt.PAPER_COMPLETION_K1) < 0.005,
          f"calibrated C(1) {comps[0]:.4f} ~= paper 0.597")
    check(abs(comps[3] - rt.PAPER_COMPLETION_K4) < 0.02,
          f"calibrated C(4) {comps[3]:.4f} ~= paper 0.700")
    results["experiments"]["E0_closed_form"] = {
        "completion": [round(c, 4) for c in comps],
        "cpc_usd": [round(c, 4) for c in cpcs],
        "marginal_cost_usd": [round(m, 4) for m in mcs],
        "cpc_k1_vs_paper": [round(cpcs[0], 4), rt.PAPER_CPC_K1_USD],
        "cpc_k4_vs_paper": [round(cpcs[3], 4), rt.PAPER_CPC_K4_USD],
    }

    # ---- E1: Monte-Carlo simulation ---------------------------------
    print("E1 Monte-Carlo simulation (seeded)")
    # NOTE: the simulation RNG seed must differ from the mixture seed
    # (SEED): calibrated_mixture consumes Random(SEED) uniforms to assign
    # easy/hard, and reusing the same stream for attempt draws correlates
    # each task's p with its first attempt outcome.
    rng = random.Random(SEED + 100)
    sim_cpcs, sim_comps = [], []
    for k in CAPS:
        c, s, cpc = rt.simulate_population(ps, k, COST_USD, rng)
        sim_comps.append(c)
        sim_cpcs.append(cpc)
        print(f"  cap {k}: completion {c:.4f}  CPC ${cpc:.4f}")
    check(all(c >= sim_cpcs[0] - 0.02 for c in sim_cpcs),
          "simulated CPC(K) >= CPC(1) (within MC noise)")
    check(all(sim_cpcs[i + 1] >= sim_cpcs[i] - 0.02
              for i in range(len(sim_cpcs) - 1)),
          "simulated CPC nondecreasing in cap (within MC noise)")
    # analytic vs simulated agreement at K=4
    check(abs(sim_comps[3] - comps[3]) < 0.01,
          "simulated C(4) agrees with closed form")
    results["experiments"]["E1_monte_carlo"] = {
        "completion": [round(c, 4) for c in sim_comps],
        "cpc_usd": [round(c, 4) for c in sim_cpcs],
    }

    # ---- E2: replay procedure (the paper's method) ------------------
    print("E2 replay of per-task attempt sequences (paper's method)")
    rng = random.Random(SEED + 1)
    # Build recorded run data: 8 attempts per task from the same mixture.
    seqs = []
    for p in ps:
        seqs.append([rng.random() < p for _ in range(8)])
    replay = {}
    for k in [1, 2, 3, 4]:
        c, s, cpc = rt.replay_runs(seqs, k, COST_USD,
                                   n_reorderings=40,
                                   rng=random.Random(SEED + 2))
        replay[k] = (c, s, cpc)
        print(f"  cap {k}: completion {c:.4f}  spend/task ${s:.4f}  "
              f"CPC ${cpc:.4f}")
    check(replay[4][0] > replay[1][0],
          "replay: completion rises with cap (paper 0.597 -> 0.700)")
    check(replay[4][2] > replay[1][2],
          "replay: CPC rises with cap (paper $0.70 -> $1.32)")
    check(abs(replay[1][0] - rt.PAPER_COMPLETION_K1) < 0.01,
          f"replay C(1) {replay[1][0]:.4f} ~= paper 0.597")
    check(abs(replay[4][0] - rt.PAPER_COMPLETION_K4) < 0.02,
          f"replay C(4) {replay[4][0]:.4f} ~= paper 0.700")
    results["experiments"]["E2_replay"] = {
        str(k): {"completion": round(v[0], 4),
                 "spend_per_task_usd": round(v[1], 4),
                 "cpc_usd": round(v[2], 4)} for k, v in replay.items()
    }

    # ---- E3: homogeneous population = equality case -----------------
    print("E3 homogeneous population (equality case)")
    homo = [0.4] * 5000
    homo_cpcs = [rt.population_metrics(homo, k, COST_USD).cost_per_completion
                 for k in CAPS]
    print("  CPCs:", [f"${c:.6f}" for c in homo_cpcs])
    check(max(homo_cpcs) - min(homo_cpcs) < 1e-9,
          "homogeneous CPC constant in K (== cost/p)")
    check(abs(homo_cpcs[0] - COST_USD / 0.4) < 1e-9,
          "homogeneous CPC == cost / p")
    results["experiments"]["E3_homogeneous"] = {
        "cpc_usd": [round(c, 6) for c in homo_cpcs],
        "expected_cost_over_p": round(COST_USD / 0.4, 6),
    }

    # ---- E4: survivor selection -------------------------------------
    print("E4 survivor selection (mean p of unresolved tasks)")
    rng = random.Random(SEED + 3)
    means = [rt.unresolved_mean_p(ps, k, rng) for k in [0, 1, 2, 4]]
    # k=0: whole population mean (no attempts yet) — compute directly.
    means[0] = sum(ps) / len(ps)
    print("  mean p after k attempts:", [f"{m:.4f}" for m in means])
    check(all(means[i + 1] < means[i] for i in range(len(means) - 1)),
          "mean p of unresolved tasks strictly falls with attempts")
    results["experiments"]["E4_survivor_selection"] = {
        "mean_p_unresolved_after_k": [round(m, 4) for m in means],
        "k": [0, 1, 2, 4],
    }

    # ---- E5: determinism --------------------------------------------
    print("E5 determinism")
    again = rt.simulate_population(ps, 4, COST_USD, random.Random(SEED + 100))
    first = rt.simulate_population(ps, 4, COST_USD, random.Random(SEED + 100))
    check(again == first, "repeat simulation reproduces identical numbers")
    again_seq = rt.replay_runs(seqs, 4, COST_USD, n_reorderings=40,
                               rng=random.Random(SEED + 2))
    check(again_seq == replay[4], "repeat replay reproduces identical numbers")
    results["experiments"]["E5_determinism"] = {"repeat_identical": True}

    # ---- E6: edge cases ---------------------------------------------
    print("E6 edge cases")
    check(rt.completion_prob(0.0, 4) == 0.0, "p=0 never completes")
    check(rt.completion_prob(1.0, 4) == 1.0, "p=1 always completes")
    check(rt.expected_attempts(0.0, 4) == 4.0, "p=0 spends the full cap")
    check(rt.expected_attempts(1.0, 4) == 1.0, "p=1 spends one attempt")
    check(rt.marginal_cost([0.0] * 100, 2) is None,
          "marginal cost undefined when cap adds no completions")
    check(rt.population_metrics([0.0] * 10, 2).cost_per_completion == math.inf,
          "CPC infinite when nothing completes")
    results["experiments"]["E6_edge_cases"] = {"all_passed": True}

    results["elapsed_s"] = round(time.time() - t0, 2)
    results["verdict"] = "SUPPORTED"
    out = os.path.join(os.path.dirname(__file__), "results.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=1)
    print(f"\nALL CHECKS PASSED in {results['elapsed_s']} s — "
          f"results in {out}")


if __name__ == "__main__":
    main()
