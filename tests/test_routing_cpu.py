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


if __name__ == "__main__":
    test_fit_recovers_boundary()
    test_monotone_properties()
    test_intervention_toward_boundary()
    test_route_components()
    test_validation()
    print("all routing tests passed")
