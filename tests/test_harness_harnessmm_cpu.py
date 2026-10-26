"""CPU tests for arxiv_lab.harness.harnessmm (arXiv:2610.32813).

Deterministic, model-free: a synthetic task-stream simulation reproduces
the paper's core claims as checkable directional statements —

  C1. Guarded decision reuse preserves actor success (1.000) while
      cutting cost/task by >40% (canonical seed: 708.5 -> 328.5, -53.6%).
  C2. Naive surface reuse (action + identifier, ignoring payload values
      and dependencies) collapses success (canonical 0.420) with hundreds
      of false reuses, while guarded reuse has zero false reuses and a
      1.000 trap-fallback rate (arg-change, dep-change, and no-receipt
      traps all fall back to the actor).
  C3. Persistent graphs beat single-use graphs: guarded cost is >25%
      below single-use cost (canonical -53.4%; single-use ~= actor cost
      because tasks sample routine decisions without replacement, so no
      within-task repeat ever exists to reuse).
  C4. RSI validation selection picks the guarded config over the naive
      config on every seed (validation success 1.000 vs ~0.46-0.52).

Canonical seed 20261026, 400 held-out tasks. Robustness seeds 1, 7, 42.
Assertions use margins around these values, not exact equality. Limits:
simulated decisions/actor, call-equivalent token costs, no live LLM — so
the verdict is SUPPORTED, not VERIFIED. One honest divergence from the
paper's numbers: the persistent-vs-single-use saving here (53%) exceeds
the paper's MBPP ablation (26.91%) because this simulation's tasks never
repeat a routine decision within a task; the direction matches, the
magnitude is a property of the task mix, not a reproduction of MBPP.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from arxiv_lab.harness.harnessmm import run_experiment

SEED = 20261026


def test_c1_guarded_preserves_success_cuts_cost():
    r = run_experiment(seed=SEED)
    t = r["test"]
    assert t["guarded"]["success"] == t["actor"]["success"] == 1.0
    assert t["guarded"]["cost_per_task"] <= 0.60 * t["actor"]["cost_per_task"]


def test_c2_guards_block_traps_naive_does_not():
    r = run_experiment(seed=SEED)
    t = r["test"]
    assert t["guarded"]["trap_fallback_rate"] == 1.0
    assert t["guarded"]["false_reuse"] == 0
    assert t["naive"]["success"] <= 0.60
    assert t["naive"]["false_reuse"] > 100
    assert t["naive"]["trap_fallback_rate"] < 0.50


def test_c3_persistent_beats_single_use():
    r = run_experiment(seed=SEED)
    t = r["test"]
    assert t["guarded"]["success"] == t["single_use"]["success"] == 1.0
    assert t["guarded"]["cost_per_task"] <= 0.75 * t["single_use"]["cost_per_task"]


def test_c4_validation_selects_guarded():
    for seed in (SEED, 1, 7, 42):
        r = run_experiment(seed=seed)
        assert r["selected_config"] == "guarded", seed
        assert r["validation"]["guarded"]["success"] > r["validation"]["naive"]["success"]


def test_robustness_seeds_hold_direction():
    for seed in (1, 7, 42):
        r = run_experiment(seed=seed)
        t = r["test"]
        assert t["guarded"]["success"] == 1.0
        assert t["guarded"]["trap_fallback_rate"] == 1.0
        assert t["guarded"]["false_reuse"] == 0
        assert t["naive"]["success"] <= 0.60
        assert t["guarded"]["cost_per_task"] <= 0.60 * t["actor"]["cost_per_task"]


if __name__ == "__main__":
    test_c1_guarded_preserves_success_cuts_cost()
    test_c2_guards_block_traps_naive_does_not()
    test_c3_persistent_beats_single_use()
    test_c4_validation_selects_guarded()
    test_robustness_seeds_hold_direction()
    print("test_harness_harnessmm_cpu: 5/5 pass")
