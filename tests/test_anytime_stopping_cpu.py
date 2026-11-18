"""
Anytime evaluation of when coding agents should stop (arXiv:2611.22901)
-- deterministic CPU test.

Reproduces the paper's core claims directionally on the CALIBRATED
synthetic trajectory model (the paper's 6,000 recorded SWE-bench
Verified trajectories are not public; see the module docstring and the
session README for every calibrated parameter and its source):
  1. the double-pass rule (stop after the SAME patch passes self-tests
     twice) saves substantial cost -- paper: 23% in the preregistered
     replication; calibrated regime: ~26%
  2. the hindsight saving available at passing self-tests lands inside
     the paper's <1%-57% range (all-runs reading ~18%; conditional
     reading ~57%)
  3. the oracle (first hidden pass) bounds the quality-preserving
     saving, and the rule out-saves it -- because ~40% of its stops
     are on false self-passes, which is the quality risk
  4. the paper's caution -- quality non-inferiority is NOT
     established: quality-loss cases exist (the rule stops on a broken
     patch that would have repaired), and across the sensitivity sweep
     the sign of the paired quality delta flips with the regime
     (p_break / post-pass edit rate), so whether the rule preserves
     quality depends on the model/scaffold regime, exactly as the
     paper found for savings (<1%-57%)
  5. the paper's explanation mechanism: after the rule fires, the
     patch almost always changes (85%), and those changes both repair
     (209+ cases) and break (416+ cases) it

Run: python3 tests/test_anytime_stopping_cpu.py  (~10 s)
"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "src"))

from arxiv_lab.agent.anytime_stopping import (
    CALIBRATED,
    POLICIES,
    SplitMix64,
    Step,
    evaluate_all,
    generate_trajectory,
    run_experiment,
    sensitivity_sweep,
    stop_double_pass,
    stop_double_pass_cumulative,
    stop_first_self_pass,
    stop_oracle,
    stop_run_to_budget,
    trajectory_digest,
)

N_RUNS = 2000
SEEDS = (20261118, 7, 123, 2026, 99)


def check(cond, msg):
    if not cond:
        raise AssertionError(msg)
    print("  ok:", msg)


def hand_trajectory():
    """Hand-built trajectories (no RNG) pinning each policy's logic."""
    # Patch 0 passes self twice in a row -> double-pass fires at step 2.
    t1 = [Step(1, 0, True, True, True), Step(2, 0, False, True, True),
          Step(3, 0, False, True, True)]
    check(stop_double_pass(t1) == 2, "double-pass fires at 2nd same-patch pass")
    check(stop_first_self_pass(t1) == 1, "first-self-pass fires at step 1")
    check(stop_oracle(t1) == 1, "oracle fires at first hidden pass")
    # Patch changes between the two self-passes -> counter resets.
    t2 = [Step(1, 0, True, False, True), Step(2, 1, True, False, True),
          Step(3, 1, False, False, True)]
    check(stop_double_pass(t2) == 3, "double-pass waits for same patch twice")
    check(stop_double_pass_cumulative(t2) == 3,
          "cumulative variant agrees on patch-change reset")
    # Never passes anything -> every policy runs to budget.
    t3 = [Step(i, i - 1, True, False, False) for i in range(1, 6)]
    for name, fn in POLICIES.items():
        check(fn(t3) == 5, "%s runs to budget when nothing passes" % name)
    # Hidden pass arrives late; oracle stops there, self-tests lag it.
    t4 = [Step(1, 0, True, False, False), Step(2, 1, True, False, False),
          Step(3, 2, True, True, False), Step(4, 2, False, True, False),
          Step(5, 3, True, True, True), Step(6, 3, False, True, True)]
    check(stop_oracle(t4) == 3, "oracle stops at first hidden pass (step 3)")
    check(stop_first_self_pass(t4) == 5, "first self-pass at step 5")
    check(stop_double_pass(t4) == 6, "double-pass at step 6")


def test_rng():
    r = SplitMix64(0)
    check(r.next_u64() == 0xE220A8397B1DCDAF,
          "SplitMix64 seed 0 first output matches the reference vector")
    a = SplitMix64(20261118)
    b = SplitMix64(20261118)
    check([a.next_u64() for _ in range(100)] ==
          [b.next_u64() for _ in range(100)],
          "same seed -> identical stream")
    check(all(0.0 <= SplitMix64(s).uniform() < 1.0 for s in range(50)),
          "uniform() in [0, 1)")
    # Deterministic trajectories: same seed, same digest; new seed differs.
    d1 = [trajectory_digest(generate_trajectory(SplitMix64(7)))
          for _ in range(20)]
    d2 = [trajectory_digest(generate_trajectory(SplitMix64(7)))
          for _ in range(20)]
    check(d1 == d2, "same seed -> identical trajectories")
    d3 = [trajectory_digest(generate_trajectory(SplitMix64(8)))
          for _ in range(20)]
    check(d1 != d3, "different seed -> different trajectories")


def test_policies_on_generated():
    # Sticky self-tests => cumulative and consecutive double-pass rules
    # coincide on every trajectory (a self-fail can never intervene
    # between two passes of the same patch).
    rng = SplitMix64(20261118)
    mismatch = 0
    for _ in range(2000):
        t = generate_trajectory(rng)
        if stop_double_pass(t) != stop_double_pass_cumulative(t):
            mismatch += 1
    check(mismatch == 0,
          "cumulative == consecutive double-pass on 2000 trajectories")
    # Every policy's stop step is within the trajectory.
    rng = SplitMix64(99)
    for _ in range(500):
        t = generate_trajectory(rng)
        for name, fn in POLICIES.items():
            s = fn(t)
            assert 1 <= s <= len(t), (name, s)
    print("  ok: all policy stop steps in range on 500 trajectories")


def main():
    t0 = time.time()
    print("== hand-built policy checks ==")
    hand_trajectory()
    print("== RNG checks ==")
    test_rng()
    print("== generated-trajectory policy checks ==")
    test_policies_on_generated()

    print("== main experiment: %d seeds x %d runs = %d runs =="
          % (len(SEEDS), N_RUNS, len(SEEDS) * N_RUNS))
    res = run_experiment(n_runs=N_RUNS, seeds=SEEDS)
    pol = res["policies"]
    for name in ("baseline", "first_self_pass", "double_pass", "oracle"):
        p = pol[name]
        print("  %-16s saving=%6.2f%% pass=%.4f fire=%.3f mean_steps=%.2f"
              % (name, p["saving_pct"], p["final_hidden_pass_rate"],
                 p["fire_rate"], p["mean_steps"]))
    pf = res["double_pass_post_fire"]
    print("  post-fire: n_fired=%d changed_after=%d (%.1f%%) "
          "repairs=%d breaks=%d"
          % (pf["n_fired"], pf["changed_after_fire"],
             100.0 * pf["changed_after_fire_rate"],
             pf["repairs_after_fire"], pf["breaks_after_fire"]))
    hs = res["hindsight_at_first_self_pass"]
    print("  hindsight: conditional=%.2f%% (n=%d) all-runs=%.2f%%"
          % (hs["mean_saving_pct"], hs["n_true_first_pass"],
             hs["mean_saving_pct_all_runs"]))
    qd = res["quality_delta_double_pass_vs_baseline"]
    print("  quality delta double-pass vs baseline: %+.2fpp "
          "(wins=%d losses=%d ties=%d) 95%% CI (%+.2f, %+.2f)pp"
          % (qd["delta_pp"], qd["wins"], qd["losses"], qd["ties"],
             qd["ci_95_pp"][0], qd["ci_95_pp"][1]))
    seeds = [p["double_pass_saving_pct"] for p in res["per_seed"]]
    print("  per-seed double-pass saving: %s"
          % ", ".join("%.2f%%" % s for s in seeds))

    print("== assertions: paper's core claims, directionally ==")
    check(0.28 <= pol["baseline"]["final_hidden_pass_rate"] <= 0.50,
          "baseline final pass rate in open-model SWE-bench band "
          "(~0.30-0.50; 100k-run check gives 0.297): %.3f"
          % pol["baseline"]["final_hidden_pass_rate"])
    check(15.0 <= pol["double_pass"]["saving_pct"] <= 35.0,
          "double-pass rule saves substantially, near paper's 23%%: %.2f%%"
          % pol["double_pass"]["saving_pct"])
    check(5.0 <= pol["oracle"]["saving_pct"] <= 57.0,
          "oracle (hindsight) saving in paper's <1%%-57%% range: %.2f%%"
          % pol["oracle"]["saving_pct"])
    check(1.0 <= hs["mean_saving_pct_all_runs"] <= 57.0,
          "hindsight saving at passing self-tests (all runs) in "
          "paper's range: %.2f%%" % hs["mean_saving_pct_all_runs"])
    check(1.0 <= hs["mean_saving_pct"] <= 60.0,
          "hindsight saving, conditional reading, near paper's range: "
          "%.2f%%" % hs["mean_saving_pct"])
    check(0.30 <= pol["double_pass"]["fire_rate"] <= 0.90,
          "double-pass rule fires on a substantial share of runs: %.3f"
          % pol["double_pass"]["fire_rate"])
    check(pf["repairs_after_fire"] > 0 and pf["breaks_after_fire"] > 0,
          "post-fire changes both repair and break the patch "
          "(repairs=%d breaks=%d)"
          % (pf["repairs_after_fire"], pf["breaks_after_fire"]))
    check(pf["changed_after_fire_rate"] > 0.5,
          "the patch usually changes after the rule fires: %.1f%%"
          % (100.0 * pf["changed_after_fire_rate"]))
    check(qd["losses"] > 0,
          "quality-loss cases exist: the rule stops on %d patches that "
          "would have repaired" % qd["losses"])
    check(abs(qd["delta_pp"]) <= 4.0,
          "quality delta is small (rule neither clearly wins nor "
          "clearly loses): %+.2fpp" % qd["delta_pp"])
    check(pol["first_self_pass"]["saving_pct"] >
          pol["double_pass"]["saving_pct"],
          "first-self-pass rule saves more than double-pass "
          "(earlier stop, worse quality: pass %.4f vs %.4f)"
          % (pol["first_self_pass"]["final_hidden_pass_rate"],
             pol["double_pass"]["final_hidden_pass_rate"]))
    check(max(seeds) - min(seeds) < 5.0,
          "per-seed double-pass saving stable (spread %.2fpp)"
          % (max(seeds) - min(seeds)))

    print("== sensitivity sweep: does the quality delta change sign? ==")
    cells = sensitivity_sweep(n_runs=N_RUNS, seeds=(20261118, 7))
    deltas = [c["quality_delta_pp"] for c in cells]
    for c in cells:
        print("  p_break=%.2f p_change_after_self_pass=%.2f "
              "p_self_false_pass=%.2f: saving=%.1f%% delta=%+.2fpp "
              "(base pass %.3f)"
              % (c["p_break"], c["p_change_after_self_pass"],
                 c["p_self_false_pass"], c["double_pass_saving_pct"],
                 c["quality_delta_pp"], c["baseline_pass_rate"]))
    check(min(deltas) < 0.0 < max(deltas),
          "quality delta changes sign across regimes "
          "(min %+.2fpp, max %+.2fpp) -- quality preservation is "
          "regime-dependent, non-inferiority NOT established"
          % (min(deltas), max(deltas)))
    check(all(10.0 <= c["double_pass_saving_pct"] <= 65.0
              for c in cells),
          "double-pass saving stays substantial in every sweep cell")
    res["sensitivity_sweep"] = cells

    print("ALL CHECKS PASSED in %.1fs" % (time.time() - t0))
    print("RESULTS_JSON " + json.dumps(res, sort_keys=True))
    return res


if __name__ == "__main__":
    main()
