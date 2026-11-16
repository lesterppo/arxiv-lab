"""CPU test for LeBet (arXiv:2611.17662).

Runs the full experiment (generate -> learn OR-of-AND monitor by
stochastic local search -> calibrate the fixed betting statistic ->
evaluate on held-out trajectories) and asserts the paper's core claims
in miniature:
  * false-positive control: held-out FPR near the prescribed alpha=0.15
  * detection power beats the turn-limit and error-filter baselines
  * token savings are positive and precision improves over no monitor
  * the betting statistic beats fire-immediately on FPR
  * the learned monitor is interpretable (<=3 conjunctions, <=3 literals)
  * the label-permuted negative control shows no separation
  * determinism: same seed reproduces the same monitor and metrics

Stdlib only. Usage: python3 tests/test_lebet_cpu.py [--results PATH]
"""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from arxiv_lab.agent import lebet


def main():
    results_path = None
    if "--results" in sys.argv:
        results_path = sys.argv[sys.argv.index("--results") + 1]

    r = lebet.run_experiment()
    b = r["betting"]
    print("LeBet experiment (seed=%s, alpha=%s)" % (r["seed"], r["alpha"]))
    print("  monitor: %s" % r["monitor"])
    print("  train TPR=%.3f FPR=%.3f score=%.3f  q=%.4f "
          "betting_power=%.3f betting_FPR=%.3f"
          % (r["train_tpr"], r["train_fpr"], r["train_score"], r["q"],
             r["train_betting_power"], r["train_betting_fpr"]))
    print("  held-out betting: FPR=%.3f power=%.3f token_savings=%.3f "
          "precision=%.3f (base %.3f) median_intervene_step=%s"
          % (b["fpr"], b["power"], b["token_savings"],
             b["precision_completed"], r["base_precision_no_monitor"],
             b["median_intervene_step_fail"]))
    print("  immediate: FPR=%.3f power=%.3f savings=%.3f"
          % (r["immediate"]["fpr"], r["immediate"]["power"],
             r["immediate"]["token_savings"]))
    print("  turn_limit_tuned (K*=%s): FPR=%.3f power=%.3f savings=%.3f"
          % (r["turn_limit_tuned"]["k_star"], r["turn_limit_tuned"]["fpr"],
             r["turn_limit_tuned"]["power"], r["turn_limit_tuned"]["token_savings"]))
    print("  error_filter_tuned (%s): FPR=%.3f power=%.3f savings=%.3f"
          % (r["error_filter_tuned"]["predicate"],
             r["error_filter_tuned"]["fpr"], r["error_filter_tuned"]["power"],
             r["error_filter_tuned"]["token_savings"]))
    print("  negative control (label-permuted): TPR-FPR=%.3f monitor=%s"
          % (r["negative_control"]["train_tpr_minus_fpr"],
             r["negative_control"]["monitor"]))

    # ---- assertions: the paper's core claims, in miniature ----
    alpha = r["alpha"]
    # 1. FPR control: held-out FPR within alpha + estimation slack
    #    (n=200 successes; binomial sd at 0.15 is ~0.025; 0.20 is ~2 sd).
    assert b["fpr"] <= alpha + 0.05, f"FPR {b['fpr']:.3f} exceeds alpha+slack"
    # 2. Detection power: catches most failing runs, beats both baselines.
    assert b["power"] >= 0.60, f"power {b['power']:.3f} < 0.60"
    assert b["power"] > r["turn_limit_tuned"]["power"], "power <= turn-limit baseline"
    assert b["power"] > r["error_filter_tuned"]["power"], "power <= error-filter baseline"
    # 3. Token savings positive and better than the turn-limit baseline.
    assert b["token_savings"] > 0.10, f"savings {b['token_savings']:.3f} <= 0.10"
    assert b["token_savings"] > r["turn_limit_tuned"]["token_savings"], \
        "savings <= turn-limit baseline"
    # 4. Precision of completed runs improves over running everything.
    assert b["precision_completed"] > r["base_precision_no_monitor"], \
        "precision did not improve over no-monitor"
    # 5. The betting statistic matters: fire-immediately has higher FPR.
    assert r["immediate"]["fpr"] > b["fpr"], \
        "betting did not reduce FPR vs fire-immediately"
    # 6. Interpretability: small rule set.
    n_conj = len(r["monitor_raw"])
    assert 1 <= n_conj <= 3, f"{n_conj} conjunctions"
    assert all(1 <= len(c) <= 3 for c in r["monitor_raw"]), "conjunction too large"
    # 7. Negative control: no separation on permuted labels.
    assert r["negative_control"]["train_tpr_minus_fpr"] < 0.15, \
        "negative control shows separation (leakage?)"
    # 8. Determinism: rerun reproduces the monitor and headline metrics.
    r2 = lebet.run_experiment()
    assert r2["monitor"] == r["monitor"], "monitor not deterministic"
    assert abs(r2["betting"]["fpr"] - b["fpr"]) < 1e-9, "FPR not deterministic"
    assert abs(r2["betting"]["power"] - b["power"]) < 1e-9, "power not deterministic"

    print("ALL LEBET CHECKS PASSED")
    if results_path:
        with open(results_path, "w") as f:
            json.dump(r, f, indent=2)
        print("results written to %s" % results_path)
    return r


if __name__ == "__main__":
    main()
