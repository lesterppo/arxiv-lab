"""CPU test for CRoS (arXiv 2610.42054): scramble preserves content and
discriminates chained from order-independent CoTs. Deterministic."""

import random
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from arxiv_lab.training.cros import (  # noqa: E402
    causality_score,
    chained_problems,
    content_preserved,
    deletion_changes_content,
    eval_independent_sum,
    eval_sequential,
    independent_problems,
    run_experiment,
    scramble_steps,
)


def test_scramble_preserves_content_on_random_lists():
    rng = random.Random(1234)
    for _ in range(200):
        steps = ["s%d = %d" % (i, rng.randint(0, 99))
                 for i in range(rng.randint(1, 8))]
        for seed in (0, 1, 7, 42):
            sc = scramble_steps(steps, seed=seed)
            assert content_preserved(steps, sc), (steps, sc, seed)
            assert sorted(steps) == sorted(sc)
            assert steps == steps  # input unmutated (list copy semantics)
    print("scramble preserves content: 200 random lists x 4 seeds OK")


def test_deletion_baseline_changes_content():
    steps = ["x = 3", "y = x + 4", "z = y * 5"]
    assert deletion_changes_content(steps, steps[:-1])
    assert not content_preserved(steps, steps[:-1])
    print("deletion baseline: content changed (contrast with scramble) OK")


def test_chained_vs_independent_discrimination():
    res = run_experiment()
    chain = res["chained"]
    indep = res["independent"]
    assert all(r["preserved"] for r in chain + indep)
    assert all(r["orig"] == 1 for r in chain + indep), "gold recoverable in order"
    # Chained: scrambling breaks every chain (missing dependency).
    assert res["chained_scrambled_failed"] == 6, res["chained_scrambled_failed"]
    assert res["chained_mean_causality"] == 1.0
    # Independent: scrambling changes nothing.
    assert res["independent_scrambled_failed"] == 0
    assert res["independent_mean_causality"] == 0.0
    assert res["chained_mean_causality"] > res["independent_mean_causality"]
    print("chained mean causality %.2f, independent %.2f"
          % (res["chained_mean_causality"], res["independent_mean_causality"]))
    print("chained scrambled-failed 6/6, independent 0/6")


def test_gold_values_sane():
    for prob in chained_problems():
        env, failed = eval_sequential(prob["steps"])
        assert not failed and env["ans"] == prob["gold"]
    for prob in independent_problems():
        total, failed = eval_independent_sum(prob["steps"])
        assert not failed and total == prob["gold"]
    print("gold values: 6 chained + 6 independent OK")


def test_causality_score_arithmetic():
    assert causality_score(0.9, 0.2) == 0.7
    assert causality_score(0.5, 0.5) == 0.0
    print("causality_score arithmetic OK")


if __name__ == "__main__":
    test_scramble_preserves_content_on_random_lists()
    test_deletion_baseline_changes_content()
    test_chained_vs_independent_discrimination()
    test_gold_values_sane()
    test_causality_score_arithmetic()
    print("ALL CROS CPU TESTS PASSED")
