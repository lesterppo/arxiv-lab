"""CPU tests for arxiv_lab.routing (component routing, arXiv:2610.01787).

Deterministic, pure stdlib. Checks the fitted rule recovers a known
linear boundary in (recurrence, state-conditionality) space on held-out
points, that interventions move predictions toward the boundary, and
that routing groups components correctly.

Run:  python3 tests/test_routing_cpu.py
"""

import os
import random
import sys

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from arxiv_lab.routing import (
    COMPONENT_TYPES, CONTEXT, WEIGHTS, Component,
    RoutingRule, fit_routing_rule, route_components,
)

try:
    import numpy as _np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

if HAS_NUMPY:
    from arxiv_lab.routing.leash import (
        LeashPolicy, TaskSim, UncertaintyEstimator, auc, evaluate,
    )


def make_labeled(n=400, seed=5):
    """Points labeled by the true rule: weights iff r - s > 0.1."""
    rng = random.Random(seed)
    rs, ss, ys = [], [], []
    for _ in range(n):
        r = rng.random()
        s = rng.random()
        rs.append(r)
        ss.append(s)
        ys.append(1 if r - s > 0.1 else 0)
    return rs, ss, ys


def test_fit_recovers_boundary():
    rs, ss, ys = make_labeled()
    rule = fit_routing_rule(rs, ss, ys)
    assert rule.a_r > 0, rule.as_dict()
    assert rule.a_s < 0, rule.as_dict()
    # held-out grid accuracy
    rng = random.Random(999)
    correct = total = 0
    for _ in range(500):
        r = rng.random()
        s = rng.random()
        truth = WEIGHTS if r - s > 0.1 else CONTEXT
        if rule.destination((r, s)) == truth:
            correct += 1
        total += 1
    assert correct / total > 0.97, correct / total
    print("ok fit_recovers_boundary acc=%.4f a_r=%.2f a_s=%.2f b=%.2f"
          % (correct / total, rule.a_r, rule.a_s, rule.b))


def test_monotone_properties():
    rule = RoutingRule(a_r=5.0, a_s=-5.0, b=0.0)
    # higher recurrence -> more likely weights
    ps = [rule.p_weights(r, 0.5) for r in (0.1, 0.5, 0.9)]
    assert ps[0] < ps[1] < ps[2], ps
    # higher state-conditionality -> less likely weights
    ps = [rule.p_weights(0.5, s) for s in (0.1, 0.5, 0.9)]
    assert ps[0] > ps[1] > ps[2], ps
    print("ok monotone_properties")


def test_intervention_toward_boundary():
    rule = RoutingRule(a_r=5.0, a_s=-5.0, b=0.0)
    proc = Component("procedure", 0.2, 0.8)
    p0 = rule.p_weights(*proc.features())
    boosted = Component("procedure", 0.7, 0.8)  # recurrence intervention
    p1 = rule.p_weights(*boosted.features())
    assert abs(p1 - 0.5) < abs(p0 - 0.5), (p0, p1)
    print("ok intervention_toward_boundary %.3f -> %.3f" % (p0, p1))


def test_route_components():
    rule = RoutingRule(a_r=5.0, a_s=-5.0, b=0.0)
    comps = [Component("locator", 0.9, 0.1),
             Component("lesson", 0.8, 0.2),
             Component("procedure", 0.2, 0.8),
             Component("state_fact", 0.1, 0.9)]
    out = route_components(rule, comps)
    assert len(out[WEIGHTS]) == 2 and len(out[CONTEXT]) == 2, out
    assert all(c.kind in ("locator", "lesson") for c in out[WEIGHTS])
    print("ok route_components")


def test_validation():
    try:
        Component("nonsense", 0.5, 0.5)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for bad kind")
    try:
        fit_routing_rule([], [], [])
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for empty fit")
    assert len(COMPONENT_TYPES) == 4
    print("ok validation")


# ------------------------------------------------------------------ LEASH
# CPU tests for arxiv_lab.routing.leash (arXiv:2610.22280).
# Claim: latent-uncertainty escalation (L1 task routing + L2 step-level
# escalation + L3 shadow observer) beats random escalation at matched rates.
# Seeded and deterministic. Requires numpy; skipped gracefully without it.

_LEASH_SEED = 20261018


def _leash_fit(seed=_LEASH_SEED, n_bank=300):
    sim = TaskSim(seed=seed)
    bank = []
    for _ in range(n_bank):
        task = sim.sample_task()
        states, ok = sim.weak_rollout(task)
        bank.append(([h for h, _ in states], ok))
    est = UncertaintyEstimator().fit(bank)
    return sim, est


def test_leash_uncertainty_predicts_failure():
    sim, est = _leash_fit()
    maxs, labels = [], []
    for _ in range(800):
        task = sim.sample_task()
        states, ok = sim.weak_rollout(task)
        maxs.append(max(est(h) for h, _ in states))
        labels.append(not ok)
    a = auc(maxs, labels)
    print("ok leash_uncertainty_auc=%.3f" % a)
    assert a > 0.65, a


def test_leash_beats_random_at_matched_rate():
    sim, est = _leash_fit()
    rng = _np.random.default_rng(_LEASH_SEED + 1)
    val = [sim.sample_task() for _ in range(800)]
    best = None
    for t1 in (0.65, 0.80, 0.90):
        for t2 in (0.85, 0.90, 0.95):
            m = evaluate(sim, LeashPolicy(est, t1, t2, l3=True), val, rng)
            if best is None or abs(m["escalation_rate"] - 0.30) < abs(best[0] - 0.30):
                best = (m["escalation_rate"], t1, t2)
    rate, t1, t2 = best
    n = 1500
    pol = LeashPolicy(est, t1, t2, l3=True)
    r2 = _np.random.default_rng(_LEASH_SEED + 2)
    tasks = [TaskSim(seed=10_000_000 + i).sample_task() for i in range(n)]
    leash_ok, esc_flags = [], []
    for i, task in enumerate(tasks):
        ok, e, _ = pol.run_task(TaskSim(seed=_LEASH_SEED + i), task)
        leash_ok.append(ok); esc_flags.append(e)
    k = int(sum(esc_flags))
    rand_idx = set(r2.choice(n, size=k, replace=False).tolist())
    b = c = rs = 0
    for i, task in enumerate(tasks):
        s = TaskSim(seed=_LEASH_SEED + i)
        if i in rand_idx:
            ok = s.strong_full(task)
        else:
            _, ok = s.weak_rollout(task)
        rs += ok
        if leash_ok[i] and not ok:
            b += 1
        elif ok and not leash_ok[i]:
            c += 1
    delta = sum(leash_ok) / n - rs / n
    z = (b - c) / max((b + c) ** 0.5, 1e-9)
    print("ok leash_vs_random delta=%+.3f z=%.2f (b=%d c=%d) esc=%.3f"
          % (delta, z, b, c, k / n))
    assert delta > 0.0 and z > 2.58, (delta, z)


def test_leash_step_escalation_adds_value():
    sim, est = _leash_fit()
    rng = _np.random.default_rng(_LEASH_SEED + 1)
    val = [sim.sample_task() for _ in range(800)]
    best = None
    for t1 in (0.65, 0.80, 0.90):
        for t2 in (0.85, 0.90, 0.95):
            m = evaluate(sim, LeashPolicy(est, t1, t2, l3=True), val, rng)
            if best is None or abs(m["escalation_rate"] - 0.30) < abs(best[0] - 0.30):
                best = (m["escalation_rate"], t1, t2)
    _, t1, t2 = best
    n = 1500
    tasks = [TaskSim(seed=10_000_000 + i).sample_task() for i in range(n)]
    full = LeashPolicy(est, t1, t2, l3=True)
    l1only = LeashPolicy(est, t1, 2.0, l3=False)
    sf = sn = 0
    for i, task in enumerate(tasks):
        sf += full.run_task(TaskSim(seed=_LEASH_SEED + i), task)[0]
        sn += l1only.run_task(TaskSim(seed=_LEASH_SEED + i), task)[0]
    d = sf / n - sn / n
    print("ok leash_L2_adds=%+.3f" % d)
    assert d > 0.01, d


def test_leash_shadow_observer_trigger():
    sim, est = _leash_fit()
    pol = LeashPolicy(est, 0.8, 0.85, l3=True)
    sim_b = TaskSim(seed=_LEASH_SEED + 31337)
    scored = []
    for _ in range(300):
        task = sim_b.sample_task()
        states, _ = sim_b.weak_rollout(task)
        for h, _ in states:
            scored.append((est(h), h))
    scored.sort(key=lambda x: x[0])
    lo = [h for _, h in scored[:3]]
    hi = [h for _, h in scored[-3:]]
    assert all(est(h) > 0.85 for h in hi) and all(est(h) < 0.85 for h in lo)
    s = TaskSim(seed=_LEASH_SEED + 777)
    task = s.sample_task()
    assert pol._shadow_observe(s, task, 0, hi, False) is True
    assert pol._shadow_observe(s, task, 0, lo, False) is False
    assert pol._shadow_observe(s, task, 0, hi, True) is True
    print("ok leash_shadow_observer trigger/rescue/passthrough")


if __name__ == "__main__":
    test_fit_recovers_boundary()
    test_monotone_properties()
    test_intervention_toward_boundary()
    test_route_components()
    test_validation()
    if HAS_NUMPY:
        test_leash_uncertainty_predicts_failure()
        test_leash_beats_random_at_matched_rate()
        test_leash_step_escalation_adds_value()
        test_leash_shadow_observer_trigger()
    else:
        print("skipped leash tests (numpy not installed)")
    print("all routing tests passed")
