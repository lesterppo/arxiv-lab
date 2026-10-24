"""CPU tests for arxiv_lab.skills.skill_granularity (arXiv:2610.31461).

Deterministic, model-free: the simulated coding-agent environment reproduces
the paper's core claims as checkable statements —

  C1a. fixed macro is efficient when applicable (cost < fixed micro, 100% success)
  C1b. fixed macro is brittle under mismatch (near-zero success)
  C2a. adaptive routing improves robustness (success >= fixed micro, cost below it)
  C2b. the recovery executor recovers controlled macro failures (high rate)
  C2c. recovery is never entered without a macro failure; all FSM transitions
       are legal and terminate (transition correctness)

Seeds are fixed; numbers below are the measured values of this harness.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from arxiv_lab.skills.skill_granularity import (
    GranularityController,
    generate_tasks,
    oracle_route,
    run_protocol,
)

SEED = 20261024
N = 400

TASKS = generate_tasks(N, seed=SEED, mismatch_frac=0.5, latent_fail_frac=0.25)


def _classes(tasks):
    applicable = [t for t in tasks if t.latent_failure is None and _full_match(t)]
    mismatch = [t for t in tasks if not _full_match(t)]
    latent = [t for t in tasks if t.latent_failure is not None]
    return applicable, mismatch, latent


def _full_match(task):
    from arxiv_lab.skills.skill_granularity import _best_macro

    _, score = _best_macro(task.features)
    return score == 1.0


def test_task_suite_structure():
    applicable, mismatch, latent = _classes(TASKS)
    # both brittleness classes are non-trivially populated
    assert len(mismatch) >= 100, len(mismatch)
    assert len(latent) >= 40, len(latent)
    assert len(applicable) >= 100, len(applicable)
    print(
        f"suite: applicable={len(applicable)} mismatch={len(mismatch)} "
        f"latent_fail={len(latent)}"
    )


def test_c1a_fixed_macro_efficient_when_applicable():
    applicable, _, _ = _classes(TASKS)
    fm = run_protocol(applicable, "fixed_macro", seed=SEED + 1)
    fu = run_protocol(applicable, "fixed_micro", seed=SEED + 2)
    assert fm.success_rate == 1.0, fm.success_rate
    assert fm.mean_cost < fu.mean_cost, (fm.mean_cost, fu.mean_cost)
    print(
        f"C1a: fixed-macro applicable success={fm.success_rate:.3f} "
        f"cost={fm.mean_cost:.2f} vs fixed-micro cost={fu.mean_cost:.2f}"
    )


def test_c1b_fixed_macro_brittle_under_mismatch():
    _, mismatch, _ = _classes(TASKS)
    fm = run_protocol(mismatch, "fixed_macro", seed=SEED + 3)
    assert fm.success_rate <= 0.05, fm.success_rate
    print(f"C1b: fixed-macro mismatch success={fm.success_rate:.3f}")


def test_c2a_adaptive_robust_and_cheaper():
    fm_all = run_protocol(TASKS, "fixed_macro", seed=SEED + 4)
    fu_all = run_protocol(TASKS, "fixed_micro", seed=SEED + 5)
    ad = run_protocol(TASKS, "adaptive", seed=SEED + 6)
    assert ad.success_rate >= fu_all.success_rate, (ad.success_rate, fu_all.success_rate)
    assert ad.success_rate >= 0.99, ad.success_rate
    assert ad.mean_cost < fu_all.mean_cost, (ad.mean_cost, fu_all.mean_cost)
    assert ad.success_rate > fm_all.success_rate + 0.2, (
        ad.success_rate,
        fm_all.success_rate,
    )
    print(
        f"C2a: adaptive success={ad.success_rate:.3f} cost={ad.mean_cost:.2f} | "
        f"fixed-micro success={fu_all.success_rate:.3f} cost={fu_all.mean_cost:.2f} | "
        f"fixed-macro success={fm_all.success_rate:.3f}"
    )


def test_c2b_recovery_of_controlled_failures():
    _, _, latent = _classes(TASKS)
    fm = run_protocol(latent, "fixed_macro", seed=SEED + 7)
    ad = run_protocol(latent, "adaptive", seed=SEED + 8)
    # fixed-macro leaves every controlled failure unresolved
    assert fm.success_rate == 0.0, fm.success_rate
    # adaptive recovers a large majority via macro-to-micro recovery
    assert ad.recovery_rate >= 0.80, ad.recovery_rate
    assert ad.success_rate >= 0.80, ad.success_rate
    print(
        f"C2b: latent-failure class — fixed-macro success={fm.success_rate:.3f}, "
        f"adaptive recovery_rate={ad.recovery_rate:.3f} "
        f"success={ad.success_rate:.3f}"
    )


def test_c2c_route_match_and_transition_correctness():
    ad = run_protocol(TASKS, "adaptive", seed=SEED + 9)
    assert ad.route_match >= 0.95, ad.route_match
    assert ad.transition_correct == 1.0, ad.transition_correct
    # recovery is only ever entered after a macro failure
    for t in ad.traces:
        if "MICRO_RECOVERY" in t.transitions:
            assert "MACRO_FAILED" in t.transitions, t.task_id
    # the terminal state is always reached
    for t in ad.traces:
        assert t.transitions[-1] in ("DONE", "FAIL"), t.transitions
    print(
        f"C2c: route_match={ad.route_match:.3f} "
        f"transition_correct={ad.transition_correct:.3f} "
        f"macro_adoption={ad.macro_adoption:.3f}"
    )


def test_controller_learns_from_failure_statistics():
    # Low-noise regime: the controller keeps the macro (rare failures are
    # cheaply recovered) — high adoption.
    low = generate_tasks(200, seed=7, mismatch_frac=0.2, latent_fail_frac=0.1)
    ad_low = run_protocol(low, "adaptive", seed=SEED + 10)
    # High-noise regime: failures are frequent, so expected macro cost beats
    # the veto — adoption collapses while success is preserved via micro.
    high = generate_tasks(200, seed=7, mismatch_frac=0.2, latent_fail_frac=0.9)
    ad_high = run_protocol(high, "adaptive", seed=SEED + 11)
    assert ad_low.macro_adoption > 0.5, ad_low.macro_adoption
    assert ad_high.macro_adoption < ad_low.macro_adoption - 0.2, (
        ad_high.macro_adoption,
        ad_low.macro_adoption,
    )
    assert ad_high.success_rate >= 0.99, ad_high.success_rate
    print(
        f"failure-conditioning: low-noise adoption={ad_low.macro_adoption:.3f} "
        f"high-noise adoption={ad_high.macro_adoption:.3f} "
        f"high-noise success={ad_high.success_rate:.3f}"
    )


def test_oracle_route_sanity():
    applicable, mismatch, _ = _classes(TASKS)
    assert all(oracle_route(t)[0] == "macro" for t in applicable)
    assert all(oracle_route(t)[0] == "micro" for t in mismatch)
    print("oracle route sanity OK")


if __name__ == "__main__":
    test_task_suite_structure()
    test_c1a_fixed_macro_efficient_when_applicable()
    test_c1b_fixed_macro_brittle_under_mismatch()
    test_c2a_adaptive_robust_and_cheaper()
    test_c2b_recovery_of_controlled_failures()
    test_c2c_route_match_and_transition_correctness()
    test_controller_learns_from_failure_statistics()
    test_oracle_route_sanity()
    print("ALL SKILL-GRANULARITY CPU TESTS PASSED")
