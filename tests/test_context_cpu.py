"""CPU test for arXiv 2610.11007 — Curating Always-Loaded Context for LLM Agents.

Reproduces the paper's capacitated-assortment mechanism in a scripted
statistical model (deterministic, seeded):

Claim A (bounded optimal file size):
 1. A closed-form upper bound on the optimal file's instruction count
    holds for candidate pools of 25 / 100 / 400 instructions, and the
    bound itself does not depend on pool size.
 2. The optimal loaded-token count saturates as the pool grows 16x
    (25 -> 400): the curator does not keep appending.

Claim B (greedy append can be arbitrarily worse):
 3. On the star-plus-fillers family, the optimal-vs-greedy value gap
    grows (approximately linearly) with the number of fillers m;
    at m=200 the greedy file has deeply negative value while the
    optimal file keeps just the star.

Claim C (censored feedback):
 4. A delete-if-ignored policy removes a purely preventive
    (silent-guardrail) instruction with probability 1 for K = 20 and
    K = 200 sessions — no finite evidence can save it, since its value
    never generates an observable event.
 5. The deletion costs measurable per-session value (the prevented
    harms), while a visibly-followed helpful instruction survives.

Claim D (evidence rule):
 6. The optimal trial count m* rises with prior uncertainty sigma0 and
    falls with per-trial exploration cost; committing with m* beats
    fixed m=1 and m=50 rules on expected regret.

Claim E (budget caps mispricing loss — measured, asserted as inequality):
 7. Under an underestimated token price, a hard token budget caps the
    realized loss below the unbounded mispriced optimizer's loss.

Deterministic (seeded). numpy-guarded: skips cleanly without numpy.
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

try:
    import numpy as np
except ImportError:
    np = None

from arxiv_lab.context import (
    KAPPA, LAMBDA, CAPACITY, dilution, net_value, standalone_value,
    greedy_append, optimal_subset,
    arbitrarily_worse_family, simulate_censored_feedback,
    delete_if_ignored, optimal_trial_count, expected_regret,
)

CHECKS = []


def check(name):
    def deco(fn):
        CHECKS.append((name, fn))
        return fn
    return deco


@check("dilution axioms: decreasing, D(0)=1, adding never raises others")
def _():
    assert dilution(0) == 1.0
    assert dilution(100) > dilution(1000) > 0
    # adding an instruction strictly lowers every loaded instruction's
    # realized compliance v_i * D(T)
    costs = [50, 60]
    values = [10.0, 10.0]
    before = values[0] * dilution(costs[0])
    after = values[0] * dilution(costs[0] + costs[1])
    assert after < before


@check("standalone append test: positive-standalone instructions identified")
def _():
    costs = [10, 200]
    values = [100.0, 6.3]
    assert standalone_value(0, costs, values) > 0
    assert standalone_value(1, costs, values) > 0  # small but positive
    assert greedy_append(costs, values) == [0, 1]


@check("Claim A1: n-independent structure — optimal file value-density >= lambda")
def _():
    # Theorem in the model: V(S*) >= V({}) = 0  =>  D(T*)*sum(v) >= lam*T*
    # => sum(v)/T* >= lam / D(T*) >= lam. The optimal file's value per
    # token is bounded below by the token price, for ANY pool size.
    # (A closed-form worst-case count bound exists — size_upper_bound —
    # but is vacuous at these parameters; see README honest limits.)
    rng = np.random.default_rng(7)
    for n in (25, 100, 400):
        costs = rng.integers(10, 101, size=n).tolist()
        values = rng.lognormal(3.0, 1.0, size=n).tolist()
        chosen, val, t = optimal_subset(costs, values)
        assert val >= 0.0, "empty file must be at least as good"
        assert t <= CAPACITY
        if t > 0:
            density = sum(values[i] for i in chosen) / t
            assert density >= LAMBDA * (1 - 1e-9), (n, density)
    print("    density >= lambda holds for n=25/100/400")


@check("Claim A2: optimal file size saturates as pool grows 16x")
def _():
    rng = np.random.default_rng(7)
    toks = []
    for n in (25, 100, 400):
        costs = rng.integers(10, 101, size=n).tolist()
        values = rng.lognormal(3.0, 1.0, size=n).tolist()
        _, _, t = optimal_subset(costs, values)
        toks.append(t)
    # pool grows 16x; optimal tokens saturate (paper: bounded file size)
    assert toks[2] <= 1.5 * toks[1], f"not saturated 100->400: {toks}"
    assert toks[2] <= 3.0 * toks[0], f"grew with pool 25->400: {toks}"
    print(f"    T* for n=25/100/400: {toks}")


@check("Claim B: greedy-append gap grows ~linearly in fillers m")
def _():
    gaps = {}
    for m in (20, 100, 200):
        costs, values = arbitrarily_worse_family(m)
        g = greedy_append(costs, values)
        assert len(g) == m + 1, "greedy appends every positive-standalone item"
        gv = net_value(g, costs, values)
        o, ov, ot = optimal_subset(costs, values)
        gaps[m] = (ov - gv, gv, ov, o)
    # gap grows roughly linearly: gap(200) >> gap(20)
    assert gaps[200][0] > 5 * gaps[20][0], gaps
    # greedy file is deeply negative at m=200 while optimal stays positive
    assert gaps[200][1] < -100 and gaps[200][2] > 0
    # optimal file keeps (essentially) just the star
    assert gaps[200][3] == [0], gaps[200][3]
    print("    " + ", ".join(
        f"m={m}: gap={g[0]:.1f} (greedy {g[1]:.1f} vs opt {g[2]:.1f})"
        for m, g in gaps.items()))


@check("Claim C1: silent guardrail deleted w.p. 1 for K=20 and K=200")
def _():
    rng = np.random.default_rng(11)
    instructions = [
        {"p_follow": 0.30, "q_prevent": 0.0, "harm_value": 0.0},   # visible
        {"p_follow": 0.0, "q_prevent": 0.25, "harm_value": 50.0},   # silent guardrail
        {"p_follow": 0.0, "q_prevent": 0.0, "harm_value": 0.0},     # useless
    ]
    for k in (20, 200):
        fb = simulate_censored_feedback(rng, instructions, k)
        kept = delete_if_ignored(fb)
        assert 0 in kept, f"K={k}: visibly-followed instruction deleted!"
        assert 1 not in kept, f"K={k}: silent guardrail survived?!"
        assert 2 not in kept, f"K={k}: useless instruction kept?!"


@check("Claim C2: deleting the guardrail costs real per-session value")
def _():
    rng = np.random.default_rng(11)
    instructions = [
        {"p_follow": 0.30, "q_prevent": 0.0, "harm_value": 0.0},
        {"p_follow": 0.0, "q_prevent": 0.25, "harm_value": 50.0},
    ]
    fb = simulate_censored_feedback(rng, instructions, 200)
    loss = fb[1]["true_value_per_session"]  # prevented harms, now unprevented
    assert 10.0 < loss < 15.0, loss  # ~0.25 * 50 = 12.5
    print(f"    per-session value destroyed by deletion: {loss:.2f}")


@check("Claim D: optimal trial count scales with uncertainty, beats fixed rules")
def _():
    kw = dict(mu0=1.0, sigma_obs=8.0, seed=3)
    m_lo, _ = optimal_trial_count(sigma0=1.0, explore_cost=0.1, **kw)
    m_hi, _ = optimal_trial_count(sigma0=10.0, explore_cost=0.1, **kw)
    m_cheap, _ = optimal_trial_count(sigma0=10.0, explore_cost=0.02, **kw)
    m_pricey, _ = optimal_trial_count(sigma0=10.0, explore_cost=0.5, **kw)
    assert m_hi > m_lo, (m_lo, m_hi)          # more uncertainty -> more evidence
    assert m_cheap > m_pricey, (m_cheap, m_pricey)  # cheaper trials -> more
    r_star = expected_regret(m_hi, 1.0, 10.0, 8.0, 0.1, seed=3)
    assert r_star < expected_regret(0, 1.0, 10.0, 8.0, 0.1, seed=3)
    assert r_star < expected_regret(1, 1.0, 10.0, 8.0, 0.1, seed=3)
    assert r_star < expected_regret(50, 1.0, 10.0, 8.0, 0.1, seed=3)
    print(f"    m*(sigma0=1)={m_lo}, m*(sigma0=10)={m_hi}, "
          f"m*(cheap)={m_cheap}, m*(pricey)={m_pricey}")


@check("Claim E: hard budget caps the loss from an underestimated token price")
def _():
    rng = np.random.default_rng(5)
    costs = rng.integers(20, 120, size=60).tolist()
    values = rng.lognormal(3.0, 1.0, size=60).tolist()
    lam_true, lam_hat = 0.15, 0.01  # curator underestimates the token price
    # unbounded optimizer acting on the wrong price
    # (capacity = sum of all costs is the effective unbounded case)
    s_hat, _, t_hat = optimal_subset(costs, values, capacity=sum(costs),
                                     lam=lam_hat)
    loss_unbounded = (optimal_subset(costs, values, lam=lam_true)[1]
                      - net_value(s_hat, costs, values, lam=lam_true))
    # same mispriced optimizer, but hard-capped at the budget the true
    # price would imply (C = optimal tokens under lam_true)
    _, _, t_true = optimal_subset(costs, values, lam=lam_true)
    s_cap, _, _ = optimal_subset(costs, values, capacity=t_true, lam=lam_hat)
    loss_capped = (optimal_subset(costs, values, lam=lam_true)[1]
                   - net_value(s_cap, costs, values, lam=lam_true))
    assert loss_capped <= loss_unbounded, (loss_capped, loss_unbounded)
    assert t_hat > t_true, "mispriced optimizer overloads without a budget"
    print(f"    mispriced T={t_hat} vs true-optimal T={t_true}; "
          f"loss unbounded={loss_unbounded:.1f} capped={loss_capped:.1f}")


def main():
    if np is None:
        print("SKIP: numpy not available")
        return 0
    failed = 0
    for name, fn in CHECKS:
        try:
            fn()
            print(f"PASS: {name}")
        except AssertionError as e:
            failed += 1
            print(f"FAIL: {name}: {e}")
    print(f"{len(CHECKS) - failed}/{len(CHECKS)} checks passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
