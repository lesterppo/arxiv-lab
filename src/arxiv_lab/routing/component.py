"""Component routing for self-improving agents (arXiv:2610.01787).

One-line claim: experience splits into components (locators, procedures,
state facts, lessons), and a rule fit on two pre-training properties —
recurrence and state-conditionality — recovers each component's winning
destination (weights vs context).

Pure stdlib; no numpy needed. The rule is a logistic regression on
``(recurrence, state_conditionality) -> P(destination == weights)``,
fit by deterministic batch gradient descent.
"""

import math

WEIGHTS = "weights"
CONTEXT = "context"

COMPONENT_TYPES = ("locator", "procedure", "state_fact", "lesson")


class Component:
    """One experience component with its two pre-training properties."""

    __slots__ = ("kind", "recurrence", "state_conditionality")

    def __init__(self, kind, recurrence, state_conditionality):
        if kind not in COMPONENT_TYPES:
            raise ValueError("unknown component kind: %r" % (kind,))
        self.kind = kind
        self.recurrence = float(recurrence)
        self.state_conditionality = float(state_conditionality)

    def features(self):
        return (self.recurrence, self.state_conditionality)

    def __repr__(self):
        return "Component(%r, r=%.3f, s=%.3f)" % (
            self.kind, self.recurrence, self.state_conditionality)


def _sigmoid(z):
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    e = math.exp(z)
    return e / (1.0 + e)


class RoutingRule:
    """Fitted rule: P(weights | recurrence, state-conditionality)."""

    __slots__ = ("a_r", "a_s", "b")

    def __init__(self, a_r, a_s, b):
        self.a_r = float(a_r)
        self.a_s = float(a_s)
        self.b = float(b)

    def p_weights(self, recurrence, state_conditionality):
        """Probability the component's winning destination is weights."""
        return _sigmoid(self.a_r * recurrence
                        + self.a_s * state_conditionality + self.b)

    def destination(self, component, threshold=0.5):
        """Route one component; ties (p == threshold) go to weights."""
        if isinstance(component, Component):
            r, s = component.features()
        else:  # allow a plain (recurrence, state_conditionality) pair
            r, s = component
        return WEIGHTS if self.p_weights(r, s) >= threshold else CONTEXT

    def as_dict(self):
        return {"a_r": self.a_r, "a_s": self.a_s, "b": self.b}


def fit_routing_rule(recurrences, state_conds, winners,
                     lr=0.5, iters=4000):
    """Fit a :class:`RoutingRule` by batch gradient descent on log-loss.

    ``winners``: 1 where weights was the winning destination, 0 for context.
    Deterministic: zero init, fixed learning rate and iteration count.
    """
    r = [float(x) for x in recurrences]
    s = [float(x) for x in state_conds]
    y = [float(x) for x in winners]
    n = len(y)
    if n == 0 or not (len(r) == len(s) == n):
        raise ValueError("need non-empty, equal-length inputs")
    a_r = a_s = b = 0.0
    for _ in range(iters):
        g_r = g_s = g_b = 0.0
        for ri, si, yi in zip(r, s, y):
            p = _sigmoid(a_r * ri + a_s * si + b)
            err = p - yi
            g_r += err * ri
            g_s += err * si
            g_b += err
        a_r -= lr * g_r / n
        a_s -= lr * g_s / n
        b -= lr * g_b / n
    return RoutingRule(a_r, a_s, b)


def route_components(rule, components, threshold=0.5):
    """Route every component; returns {WEIGHTS: [...], CONTEXT: [...]}."""
    out = {WEIGHTS: [], CONTEXT: []}
    for c in components:
        out[rule.destination(c, threshold)].append(c)
    return out
