#!/usr/bin/env python3
"""CPU test for CAPE (arXiv:2611.31938, src/arxiv_lab/agent/cape.py).

Reproduces the paper's core claims at bounded scale, against the
independently computed exact optimum (full-knowledge DP):
  E1  main suite: 12-config worlds, cap 3, budget 1000, seeds 0..7.
      CAPE's value equals the exact optimum; R0 preserved (Prop. 1);
      certification rate; query saving vs eager exact (which acquires
      the whole effect table); completion gain vs the error-feedback
      reviser proxy.
  E2  small worlds: 8 configs, cap 3, budget 400, seeds 0..3.
  E3  scale spot check: 24 configs, cap 3, budget 2000, seed 0.
  E4  consistency on every compiled result: value == evaluated
      (completions, -calls); completed set contains R0; queries within
      budget; certified implies value == root upper value.
  E5  budget-bound behaviour (report only): 12 configs, cap 3,
      budget 200, seeds 0..1 -- the paper's 502/504 optimal rate is at
      64-request scale with the paper's budget, not at this scale.

Paper headline numbers (abstract): optimum reached in 502/504
compilations; 22.0% less compilation time than eager exact search;
completion +15.38 to +34.66pp over the LLM error-feedback reviser.
Compilation time is measured here as calibration queries (each query
is one candidate-call evaluation): in this harness a query is a dict
lookup while CAPE's bound recomputation dominates wall-clock, so a
wall-clock comparison would measure the implementation, not the
paper's calibration-cost model. The error-feedback baseline is a
deterministic proxy, not an LLM (see module docstring).

Deterministic given a seed (module sorts set iterations; verified
across PYTHONHASHSEED values 2026-11-25). Runtime ~6 min, dominated
by E1/E3 bound recomputation; progress is printed per world.
"""
import json, os, sys, time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from arxiv_lab.agent.cape import (
    compile_policy, exact_optimum, make_world,
    program_only_policy, error_feedback_policy,
    PAPER_OPTIMAL_RATE, PAPER_TIME_SAVING, PAPER_COMPLETION_GAIN_PP,
)

def run_world(n_configs, cap, budget, seed):
    w = make_world(n_configs=n_configs, n_aspects=4, seed=seed)
    t0 = time.time()
    r = compile_policy(w, cap=cap, budget=budget)
    dt = time.time() - t0
    opt, _ = exact_optimum(w, cap=cap)
    ef_c, _ef_calls = error_feedback_policy(w, cap)
    po, _ = program_only_policy(w, cap)
    table = len(w.actions()) * len(w.configs)
    row = {
        "seed": seed, "n_configs": n_configs, "cap": cap, "budget": budget,
        "cape_comp": len(r.completed), "cape_calls": r.total_calls,
        "opt_comp": opt[0], "opt_calls": -opt[1],
        "optimal_value": r.value == opt,
        "optimal_comp": len(r.completed) == opt[0],
        "certified": r.certified,
        "queries": r.queries, "table": table,
        "query_saving": round(1 - r.queries / table, 4),
        "r0": len(r.r0), "r0_kept": len(r.r0 & r.completed),
        "ef_comp": len(ef_c), "po_comp": po[0],
        "ef_gain_pp": round((len(r.completed) - len(ef_c)) / n_configs * 100, 2),
        "time_s": round(dt, 1),
    }
    print("  world n=%d cap=%d seed=%d: cape=(%d,%d) exact=(%d,%d) "
          "cert=%s queries=%d/%d ef=%d gain=%.1fpp %.1fs"
          % (n_configs, cap, seed, row["cape_comp"], -row["cape_calls"],
             row["opt_comp"], row["opt_calls"], row["certified"],
             row["queries"], table, row["ef_comp"], row["ef_gain_pp"], dt),
          flush=True)
    # E4 consistency
    assert r.value == (len(r.completed), -r.total_calls), "value mismatch"
    assert r.r0 <= r.completed, "R0 not preserved"
    assert r.queries <= budget, "budget exceeded"
    if r.certified:
        assert r.value == r.upper, "certified but value != upper"
    return row

def main():
    all_rows = []
    print("E1 main suite: 12 configs, cap 3, budget 1000, seeds 0..7")
    e1 = [run_world(12, 3, 1000, s) for s in range(8)]
    all_rows += e1
    opt_rate = sum(r["optimal_value"] for r in e1) / len(e1)
    cert_rate = sum(r["certified"] for r in e1) / len(e1)
    saving = sum(r["queries"] for r in e1) / sum(r["table"] for r in e1)
    saving = 1 - saving
    ef_gain = sum(r["ef_gain_pp"] for r in e1) / len(e1)
    print("E1: optimal value %d/8, certified %d/8, mean query saving %.1f%% "
          "(paper %.1f%%), mean EF gain %.2fpp (paper %.2f-%.2fpp)"
          % (sum(r["optimal_value"] for r in e1),
             sum(r["certified"] for r in e1), saving * 100,
             PAPER_TIME_SAVING * 100, ef_gain,
             PAPER_COMPLETION_GAIN_PP[0], PAPER_COMPLETION_GAIN_PP[1]))
    assert opt_rate == 1.0, "E1: CAPE must reach the exact optimum value"
    assert all(r["r0_kept"] == r["r0"] for r in e1), "E1: R0 must be kept"
    assert cert_rate >= 0.75, "E1: certification rate below 6/8"
    assert ef_gain > PAPER_COMPLETION_GAIN_PP[0], \
        "E1: EF gain below the paper's range"

    print("E2 small worlds: 8 configs, cap 3, budget 400, seeds 0..3")
    e2 = [run_world(8, 3, 400, s) for s in range(4)]
    all_rows += e2
    print("E2: optimal value %d/4, certified %d/4"
          % (sum(r["optimal_value"] for r in e2),
             sum(r["certified"] for r in e2)))
    assert sum(r["optimal_value"] for r in e2) == 4, "E2: must reach optimum"
    assert sum(r["certified"] for r in e2) >= 2, "E2: certification too low"

    print("E3 scale spot check: 24 configs, cap 3, budget 2000, seed 0")
    e3 = [run_world(24, 3, 2000, 0)]
    all_rows += e3
    assert e3[0]["optimal_value"], "E3: must reach the optimum value"
    print("E3: optimal value reached, certified=%s, queries=%d/%d"
          % (e3[0]["certified"], e3[0]["queries"], e3[0]["table"]))

    print("E5 budget-bound (report only): 12 configs, cap 3, budget 200")
    e5 = [run_world(12, 3, 200, s) for s in range(2)]
    all_rows += e5
    print("E5: optimal value %d/2 at budget 200 (paper rate %.4f is at "
          "64-request scale; not asserted here)"
          % (sum(r["optimal_value"] for r in e5), PAPER_OPTIMAL_RATE))

    summary = {
        "paper": "arXiv:2611.31938",
        "e1_optimal_value": "%d/8" % sum(r["optimal_value"] for r in e1),
        "e1_certified": "%d/8" % sum(r["certified"] for r in e1),
        "e1_mean_query_saving": round(saving, 4),
        "e1_mean_ef_gain_pp": round(ef_gain, 2),
        "e2_optimal_value": "%d/4" % sum(r["optimal_value"] for r in e2),
        "e2_certified": "%d/4" % sum(r["certified"] for r in e2),
        "e3_optimal_value": bool(e3[0]["optimal_value"]),
        "e3_certified": bool(e3[0]["certified"]),
        "e5_optimal_value_budget200": "%d/2"
            % sum(r["optimal_value"] for r in e5),
        "rows": all_rows,
    }
    print("RESULTS_JSON " + json.dumps(summary))
    print("ALL CHECKS PASSED")

if __name__ == "__main__":
    main()
