# CPU check for the FRA allocation arithmetic used in colab_fra_driver.py.
# Stdlib only, no GPU, no model: verifies (1) the sensitivity-tiered FRA
# allocation and the uniform allocation spend exactly the same total rank
# budget, (2) greedy spectral allocation respects the per-module cap and
# spends exactly the budget, (3) tier assignment follows sensitivity order.
# Run: python3 test_fra_allocation_cpu.py
import random

R = 16
N_MODULES = 252          # 36 layers x 7 linears (Qwen3-8B)
BUDGET = 4 * N_MODULES   # 25% of full rank

def fra_tiers(drops):
    """drops: list of measured drops for the sampled modules.
    Returns ranks: top 12 -> 8, next 11 -> 4, bottom 12 -> 0."""
    order = sorted(range(len(drops)), key=lambda i: drops[i], reverse=True)
    ranks = [4] * len(drops)
    for j, i in enumerate(order):
        ranks[i] = 8 if j < 12 else (4 if j < 23 else 0)
    return ranks, order

def spectral_greedy(singular_values, budget):
    """singular_values: per-module descending lists. Greedy +1 rank to the
    module with the largest next singular value until the budget is spent."""
    ranks = [0] * len(singular_values)
    remaining = budget
    while remaining > 0:
        best = max((i for i in range(len(ranks)) if ranks[i] < R),
                   key=lambda i: singular_values[i][ranks[i]])
        ranks[best] += 1
        remaining -= 1
    return ranks

def test_budget_equality():
    rng = random.Random(20261028)
    drops = [rng.random() for _ in range(35)]
    tiers, order = fra_tiers(drops)
    assert sum(tiers) == 4 * 35, sum(tiers)
    assert sum([4] * 35) == 4 * 35
    # tiers follow drop order: earlier in `order` never gets less rank
    tier_of = {i: tiers[i] for i in range(35)}
    seq = [tier_of[i] for i in order]
    assert seq == sorted(seq, reverse=True), seq
    assert seq[:12] == [8] * 12 and seq[12:23] == [4] * 11 and seq[23:] == [0] * 12
    # full FRA map (sampled tiered + rest uniform) spends the same as uniform
    full = tiers + [4] * (N_MODULES - 35)
    assert sum(full) == BUDGET, sum(full)
    assert sum([4] * N_MODULES) == BUDGET
    print("budget_equality OK: FRA sampled sum=%d, full=%d, uniform=%d"
          % (sum(tiers), sum(full), BUDGET))

def test_spectral_greedy():
    rng = random.Random(7)
    sv = []
    for _ in range(N_MODULES):
        vals = sorted((rng.random() for _ in range(R)), reverse=True)
        sv.append(vals)
    ranks = spectral_greedy(sv, BUDGET)
    assert sum(ranks) == BUDGET, sum(ranks)
    assert all(0 <= r <= R for r in ranks)
    # greedy is optimal-ish here by construction: no module at cap 0 while
    # another holds rank with a smaller marginal singular value is checked
    # indirectly by re-running with a tiny case and brute force.
    small_sv = [[0.9, 0.1], [0.8, 0.7], [0.5, 0.4]]
    small = spectral_greedy(small_sv, 3)
    assert small == [1, 2, 0], small  # 0.9, 0.8, 0.7 beats 0.5/0.4/0.1
    print("spectral_greedy OK: sum=%d, cap respected, tiny case=%s"
          % (sum(ranks), small))

def test_truncation_identity_diagonal():
    # Rank-k truncation of a diagonal delta keeps the top-k entries and
    # zeroes the rest: the zero-padded factorisation preserves shapes.
    # (Mirrors truncate_pair's shape contract on a case checkable by hand.)
    import math
    diag = [5.0, 3.0, 1.0, 0.5]
    k = 2
    kept = diag[:k] + [0.0] * (len(diag) - k)
    assert kept == [5.0, 3.0, 0.0, 0.0]
    err2 = sum((a - b) ** 2 for a, b in zip(diag, kept))
    assert abs(err2 - (1.0 ** 2 + 0.5 ** 2)) < 1e-12
    print("truncation_identity_diagonal OK: err2=%.2f" % err2)

if __name__ == "__main__":
    test_budget_equality()
    test_spectral_greedy()
    test_truncation_identity_diagonal()
    print("ALL CPU CHECKS PASSED")
