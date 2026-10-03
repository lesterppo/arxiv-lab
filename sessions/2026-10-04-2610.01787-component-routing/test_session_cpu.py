"""Session-level CPU test for 2610.01787 component routing.

Re-runs the session experiment at reduced size and asserts the paper's
three claims hold directionally. Deterministic, pure stdlib.

Run:  python3 sessions/2026-10-04-2610.01787-component-routing/test_session_cpu.py
"""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import experiment
from arxiv_lab.routing import WEIGHTS


def main():
    experiment.N_PER_CELL = 6
    pool = experiment.gen_pool()
    rows_bal = experiment.observe(pool, "balanced", experiment.MASTER_SEED + 3)
    rows_mem = experiment.observe(pool, "memorizer", experiment.MASTER_SEED + 1)
    rows_ic = experiment.observe(pool, "in-context", experiment.MASTER_SEED + 2)

    # (i) locators/lessons favor weights; procedures/state facts favor context
    for kind, lo, hi in (("locator", 0.7, 1.01), ("lesson", 0.7, 1.01),
                         ("procedure", -0.01, 0.3),
                         ("state_fact", -0.01, 0.3)):
        items = [r for r in rows_bal if r["type"] == kind]
        wf = sum(1 for r in items if r["winner"] == WEIGHTS) / len(items)
        assert lo <= wf <= hi, (kind, wf)
    print("ok claim-i type pattern")

    # (ii) rule fit on two families, held-out 24-cell recovery
    rule = experiment.fit_rule(rows_mem, rows_ic)
    assert rule.a_r > 0 and rule.a_s < 0, rule.as_dict()
    score, total, _ = experiment.heldout_cells(rule, rows_bal)
    assert score >= 22, (score, total)
    print("ok claim-ii held-out recovery %d/%d" % (score, total))

    # interventions move toward the boundary
    proc = experiment.Component("procedure", 0.30, 0.72)
    iv = experiment.intervention_check(rule, proc, "recurrence", +0.35)
    assert iv["dist_to_boundary_after"] < iv["dist_to_boundary_before"], iv
    print("ok claim-ii intervention toward boundary")

    # routing-by-rule beats whole-trajectory baselines
    base = experiment.baselines(rule, rows_bal)
    assert base["rule"] >= max(base["all_weights"], base["all_context"]), base
    print("ok claim-ii rule beats baselines (margin %.4f)"
          % (base["rule"] - max(base["all_weights"], base["all_context"])))

    # (iii) readout degrades after weights-write, most for high recurrence
    qs = [experiment.blend_readout_quality(r, trials=120)
          for r in (0.05, 0.4, 0.95)]
    assert qs[0] == 1.0 and qs[0] > qs[1] > qs[2], qs
    print("ok claim-iii readout degradation", [round(q, 3) for q in qs])

    # (iii) context gain rises with info gap; weights gain falls w/ policy gap
    ig = experiment.info_gap_curve()
    assert all(b > a for (_, a), (_, b) in zip(ig, ig[1:])), ig
    pg = experiment.policy_gap_curve()
    assert all(b < a for (_, _, a), (_, _, b) in zip(pg, pg[1:])), pg
    print("ok claim-iii gap curves monotone")
    print("session test passed")


if __name__ == "__main__":
    main()
