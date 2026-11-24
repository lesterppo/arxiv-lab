"""Group-Balanced SFT loss weights — arXiv:2611.29127 "Group-Balanced
Supervised Fine-Tuning for Knowledge-Intensive Domains" (Heimann &
Jacob, Ivo AI).

Standard SFT weights every supervised token equally, so the loss
budget follows token count: on the paper's legal QA benchmark
citation strings absorb 81% of the budget while the answers being
taught absorb 5%. GB-SFT equalises the loss budget across the
semantic roles of tokens within a target sequence.

Unified formula (paper Eqs. 4-5). A target sequence is split into
spans; each span s belongs to one class c and has n_s tokens.
u_s = n_s**beta is the span mass, m_c = sum_{s in c} u_s the class
mass, M = sum_c m_c**alpha. The weight of token t in span s of
class c is

    w_t = (m_c**alpha / M) * (u_s / m_c) * (1 / n_s)
        = (1 / M) * m_c**(alpha - 1) * n_s**(beta - 1),

and sum_t w_t = 1 for any (alpha, beta). Loss L = sum_t w_t CE_t.

Corners: (alpha, beta) = (1, 1) is standard SFT; (0, 1) is
GB-SFT_class (each class gets 1/K of the budget); (1, 0) is
GB-SFT_span (each span gets 1/P of the budget). alpha is the tunable
dial (class budget by mass vs by count); beta is the structural
switch (span budget by length vs by count, endpoints only).

Pure stdlib. The Colab driver vendors a copy of `gb_weights`
(self-contained driver rule); the two copies are diff-checked in
the session CPU test.
"""

from __future__ import annotations

__all__ = ["gb_weights", "class_budgets", "span_budgets"]


def gb_weights(spans, alpha: float = 0.0, beta: float = 1.0):
    """Per-token GB-SFT weights for one target sequence.

    `spans` is a list of (class_label, n_tokens) pairs, one per span,
    in sequence order. Spans of the same class need not be contiguous.
    Returns a list of per-token weights, in token order, summing to 1.

    Raises ValueError on an empty span list or a non-positive span
    length.
    """
    if not spans:
        raise ValueError("empty span list")
    for _c, n in spans:
        if n <= 0:
            raise ValueError("span lengths must be positive")
    # Span masses u_s = n_s**beta; class masses m_c = sum u_s.
    u = [float(n) ** beta for _c, n in spans]
    classes = []
    m = {}
    for (c, _n), u_s in zip(spans, u):
        if c not in m:
            m[c] = 0.0
            classes.append(c)
        m[c] += u_s
    big_m = sum(m[c] ** alpha for c in classes)
    if big_m <= 0.0:
        raise ValueError("degenerate class masses")
    weights = []
    for (c, n), u_s in zip(spans, u):
        w_t = (m[c] ** alpha / big_m) * (u_s / m[c]) * (1.0 / n)
        weights.extend([w_t] * n)
    return weights


def class_budgets(spans, alpha: float = 0.0, beta: float = 1.0):
    """Total weight budget per class: {class: sum of its token weights}."""
    w = gb_weights(spans, alpha, beta)
    budgets = {}
    i = 0
    for c, n in spans:
        budgets[c] = budgets.get(c, 0.0) + sum(w[i:i + n])
        i += n
    return budgets


def span_budgets(spans, alpha: float = 0.0, beta: float = 1.0):
    """Total weight budget per span (in order): [sum of its token weights]."""
    w = gb_weights(spans, alpha, beta)
    budgets = []
    i = 0
    for _c, n in spans:
        budgets.append(sum(w[i:i + n]))
        i += n
    return budgets
