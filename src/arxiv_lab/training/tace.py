"""TACE: Two-stage Advantage and Credit Estimation for GRPO.

Paper: arXiv 2610.40250 "Distributional Reward Calibration and
Budget-Neutral Credit Assignment for Multimodal RLVR" (Chen et al.,
submitted 29 Oct 2026).

One-line claim: any token-level GRPO advantage factorizes into a
sample-level baselining step and a token-level credit-assignment step,
and TACE fixes both — S1 replaces affine standardization with a
one-dimensional optimal-transport map to a standard normal (a common
population reward law across tasks, with a deterministic finite-sample
tail bound), and S2 rank-calibrates a stop-gradient evidence contrast
within each trajectory, clips the token weights, and renormalizes them
to mean exactly one (credit redistributed with total mass and sign
conserved). Stdlib only.

Finite-sample notes (verified in the 2026-10-31 session): the 1-D OT
map is rank-Gaussianization (ranks -> normal quantiles), so its tail
cap is exactly Phi^{-1}(n / (n + 1)); an affine z-score also saturates
in finite samples (at sqrt(n - 1) for a lone outlier) but at a much
higher cap. Rank-Gaussianization does not fully equalize laws under
heavy ties (a Bernoulli-cliff reward law keeps a point mass), so S1's
"common law" holds for continuous reward laws, not discrete ones. S2 is
implemented for the abstract's stated steps; the paper's exact evidence
contrast formula is not in the abstract, so S2 is validated on its
claimed properties (conservation / boundedness / sign anchoring /
scale invariance) rather than bit-parity with that formula.
"""
import math
from statistics import NormalDist

_ND = NormalDist()


def _ranks_average(x):
    order = sorted(range(len(x)), key=lambda i: x[i])
    ranks = [0.0] * len(x)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and x[order[j + 1]] == x[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0  # 1-based average rank
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return ranks


def ot_gaussianize(x):
    """Finite-sample 1-D OT map to N(0,1): ranks -> normal quantiles."""
    n = len(x)
    return [_ND.inv_cdf(r / (n + 1.0)) for r in _ranks_average(x)]


def ot_tail_cap(n):
    """Deterministic finite-sample tail bound: max |OT score| for n."""
    return _ND.inv_cdf(n / (n + 1.0))


def affine_standardize(x):
    """Baseline: per-group z-score."""
    n = len(x)
    mu = sum(x) / n
    sd = math.sqrt(sum((v - mu) ** 2 for v in x) / n) or 1.0
    return [(v - mu) / sd for v in x]


def s2_weights(contrasts, cap=2.0):
    """Rank-calibrate contrasts, clip, renormalize to mean exactly one.

    Rank-calibration discards the raw contrast scale by design; the
    weight is the rank quantile scaled to (0, 2), clipped at `cap`,
    then divided by the clipped mean so mean == 1 exactly.
    """
    n = len(contrasts)
    ranks = _ranks_average(contrasts)
    raw = [r / (n + 1.0) * 2.0 for r in ranks]
    clipped = [min(max(v, 0.0), cap) for v in raw]
    mean = sum(clipped) / n
    return [v / mean for v in clipped]


def token_advantages(seq_advantage, contrasts, cap=2.0):
    """Token advantages = seq signal * S2 weights. Total = seq * L
    exactly, with the seq signal's sign (sign anchoring)."""
    w = s2_weights(contrasts, cap=cap)
    return [seq_advantage * wi for wi in w], w


def skew(x):
    n = len(x)
    mu = sum(x) / n
    sd = math.sqrt(sum((v - mu) ** 2 for v in x) / n) or 1.0
    return sum(((v - mu) / sd) ** 3 for v in x) / n


def kurtosis_excess(x):
    n = len(x)
    mu = sum(x) / n
    sd = math.sqrt(sum((v - mu) ** 2 for v in x) / n) or 1.0
    return sum(((v - mu) / sd) ** 4 for v in x) / n - 3.0


def ks_to_normal(x):
    """Max |empirical CDF - Phi| over sorted values."""
    xs = sorted(x)
    n = len(xs)
    worst = 0.0
    for i, v in enumerate(xs):
        f = _ND.cdf(v)
        worst = max(worst, abs((i + 1) / n - f), abs(i / n - f))
    return worst
