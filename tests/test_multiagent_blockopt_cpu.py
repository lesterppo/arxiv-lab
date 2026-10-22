"""CPU tests for arxiv 2610.30074: parallel agent collaboration as
graph-structured distributed block optimization.

Reproduces the paper's core claims on synthetic exactly-evaluable objectives:
(1) independent-set proposals' verified gains add exactly;
(2) the fractional-coloring merger preserves >= 1/chi_f of total verified gain;
(3) the bound is tight among subset mergers;
(4) naive merging can fail where the merger succeeds;
(5) repeated collaboration contracts the objective gap geometrically.

Requires numpy (optional dependency); skipped gracefully without it.

Run:  python3 tests/test_multiagent_blockopt_cpu.py
"""
import itertools
import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

from arxiv_lab.multiagent.block_opt_merge import (
    BlockObjective,
    conflict_graph,
    fractional_chromatic_number,
    fractional_merger,
    maximal_independent_sets,
    naive_merge,
    brute_force_optimum,
    collaborate,
)


def _skip():
    print("SKIP: numpy not available")
    sys.exit(0)


def random_objective(n, p, rng):
    unary = [rng.uniform(-2, 2) for _ in range(n)]
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if rng.random() < p:
                edges.append((i, j, rng.uniform(-3, 3)))
    return BlockObjective(unary, edges)


def test_independent_set_gains_add_exactly():
    """Independent sets certify compatible edits whose verified gains, measured
    against the same current solution, add exactly."""
    for seed in range(12):
        rng = random.Random(1000 + seed)
        obj = random_objective(8, 0.3, rng)
        x = np.array([rng.randint(0, 1) for _ in range(8)])
        proposals = [{v: 1 - x[v]} for v in range(8)]  # flip each block
        gains = [obj.gain(x, e) for e in proposals]
        hadj = conflict_graph(proposals, obj.adj)
        ind = max(maximal_independent_sets(hadj), key=len)
        merged = {}
        for i in ind:
            merged.update(proposals[i])
        realized = obj.f(obj.apply(x, merged)) - obj.f(x)
        expect = sum(gains[i] for i in ind)
        assert abs(realized - expect) < 1e-9, (seed, realized, expect)
    print("ok: independent-set gains add exactly (12 seeds)")


def test_chi_f_known_graphs():
    """Exact fractional chromatic number on graphs with analytic values."""
    def cycle(n):
        return [{(i - 1) % n, (i + 1) % n} for i in range(n)]

    def clique(n):
        return [set(range(n)) - {i} for i in range(n)]

    assert abs(fractional_chromatic_number(cycle(5)) - 2.5) < 1e-6
    assert abs(fractional_chromatic_number(cycle(7)) - (2 + 1 / 3)) < 1e-6
    assert abs(fractional_chromatic_number(clique(4)) - 4.0) < 1e-6
    assert abs(fractional_chromatic_number([set(), set(), set()]) - 1.0) < 1e-6
    assert abs(fractional_chromatic_number([{1}, {0}]) - 2.0) < 1e-6
    print("ok: chi_f exact on C5/C7/K4/empty/edge")


def test_merger_preserves_one_over_chi_f():
    """Fractional-coloring merger preserves >= 1/chi_f(H) of the total
    verified local improvement."""
    worst = 1.0
    for seed in range(30):
        rng = random.Random(2000 + seed)
        m = rng.randint(2, 6)
        obj = random_objective(6, 0.35, rng)
        x = np.array([rng.randint(0, 1) for _ in range(6)])
        proposals = [{v: 1 - x[v]} for v in rng.sample(range(6), m)]
        gains = [abs(obj.gain(x, e)) + 0.5 for e in proposals]
        hadj = conflict_graph(proposals, obj.adj)
        chosen, preserved, chi_f = fractional_merger(proposals, gains, hadj)
        total = float(np.sum(gains))
        assert preserved >= total / chi_f - 1e-6, (seed, preserved, total / chi_f)
        worst = min(worst, preserved / total)
    print(f"ok: merger >= 1/chi_f on 30 random instances "
          f"(worst preserved fraction {worst:.3f})")


def test_bound_tight_on_odd_cycle():
    """The 1/chi_f bound is tight among subset mergers: C5 with equal gains,
    chi_f = 2.5, the merger preserves exactly 2/5 = 1/chi_f, and no
    independent subset does better."""
    hadj = [{(i - 1) % 5, (i + 1) % 5} for i in range(5)]
    proposals = [{i: 1} for i in range(5)]
    gains = [1.0] * 5
    chosen, preserved, chi_f = fractional_merger(proposals, gains, hadj)
    assert abs(chi_f - 2.5) < 1e-6
    assert preserved == 2.0
    assert abs(preserved / 5.0 - 1.0 / chi_f) < 1e-9
    best = 0.0
    for r in range(6):
        for combo in itertools.combinations(range(5), r):
            ok = all(j not in hadj[i] for a, i in enumerate(combo)
                     for j in combo[a + 1:])
            if ok:
                best = max(best, float(r))
    assert best == 2.0
    print("ok: bound tight on C5 (preserved = 2/5 = 1/chi_f)")


def test_naive_merge_fails_where_merger_succeeds():
    """Individually useful edits whose naive combination degrades the shared
    solution; the fractional merger keeps a verified improvement."""
    obj = BlockObjective([1.0, 1.0], [(0, 1, 10.0)])  # equality rewarded
    x = np.array([0, 1])                              # f = 1
    a = {0: 1}                                        # -> (1,1): f = 12
    b = {1: 0}                                        # -> (0,0): f = 10
    ga, gb = obj.gain(x, a), obj.gain(x, b)
    assert ga > 0 and gb > 0
    assert obj.f(obj.apply(x, naive_merge([a, b]))) <= obj.f(x)
    chosen, preserved, chi_f = fractional_merger(
        [a, b], [ga, gb], conflict_graph([a, b], obj.adj))
    assert preserved == max(ga, gb)
    assert preserved >= (ga + gb) / chi_f - 1e-9
    print(f"ok: naive merge degrades (f 1 -> 1), merger preserves +{preserved:.0f}")


def test_repeated_collaboration_converges_geometrically():
    """Under conditional local-improvement guarantees, repeated collaboration
    converges geometrically in the objective gap:
    gap_{t+1} <= (1 - rho_hat/chi_max) * gap_t."""
    for seed in range(6):
        rng = random.Random(3000 + seed)
        obj = random_objective(7, 0.35, rng)
        fstar, _ = brute_force_optimum(obj)
        x0 = np.array([rng.randint(0, 1) for _ in range(7)])

        def oracle(x, _obj=obj):
            return [{v: 1 - x[v]} for v in range(_obj.n)
                    if _obj.gain(x, {v: 1 - x[v]}) > 1e-9]

        hist = collaborate(obj, x0, oracle, rounds=40, fstar=fstar)
        gaps_after = [h["gap"] for h in hist]
        gaps_before = [fstar - obj.f(x0)] + gaps_after[:-1]
        assert gaps_after[-1] < gaps_before[0], "gap must shrink"
        for h in hist:
            if h["chi_f"] > 0:  # rounds with no proposals have chi_f = 0
                assert h["preserved"] >= h["total_verified"] / h["chi_f"] - 1e-6
        rhos = [h["total_verified"] / gaps_before[i]
                for i, h in enumerate(hist)
                if gaps_before[i] > 1e-9 and h["total_verified"] > 0]
        assert rhos, "oracle must propose improvements on some round"
        rho_hat = min(rhos)
        chi_max = max(h["chi_f"] for h in hist if h["chi_f"] > 0)
        rate = 1.0 - rho_hat / chi_max
        assert rate < 1.0
        # geometric contraction, asserted only on rounds where the oracle
        # actually proposed improvements (the paper's conditional
        # local-improvement premise; a stalled round keeps the gap flat)
        active = [i for i, h in enumerate(hist) if h["total_verified"] > 0]
        assert active, "oracle must propose improvements on some round"
        for i in active:
            assert gaps_after[i] <= rate * gaps_before[i] + 1e-6, \
                (seed, i, gaps_after[i], rate * gaps_before[i])
        print(f"  seed {seed}: gap {gaps_before[0]:.3f} -> {gaps_after[-1]:.6f}, "
              f"measured contraction <= {rate:.4f}/round "
              f"({len(active)} active, {len(hist)-len(active)} stalled)")
    print("ok: geometric convergence on 6 seeds")


if __name__ == "__main__":
    if not HAS_NUMPY:
        _skip()
    test_independent_set_gains_add_exactly()
    test_chi_f_known_graphs()
    test_merger_preserves_one_over_chi_f()
    test_bound_tight_on_odd_cycle()
    test_naive_merge_fails_where_merger_succeeds()
    test_repeated_collaboration_converges_geometrically()
    print("ALL BLOCK-OPT-MERGE CPU TESTS PASSED")
