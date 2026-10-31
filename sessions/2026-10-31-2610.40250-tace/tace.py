"""TACE (2610.40250) — Two-stage Advantage and Credit Estimation for GRPO,
reproduced as deterministic CPU property tests + a synthetic GRPO step.

S1 (sample-level baselining): replace affine standardization (z-score)
with a one-dimensional optimal-transport map to a standard normal. In
finite samples the 1-D OT map to N(0,1) is rank-Gaussianization:
u_i = rank_i / (n + 1), score_i = Phi^{-1}(u_i). Claimed properties:
  * a common population reward law across tasks (shape equity: every
    task's transformed law is ~N(0,1), whatever its raw shape);
  * a deterministic finite-sample tail bound: max |score| =
    Phi^{-1}(n / (n + 1)), independent of outlier magnitude, while an
    affine z-score grows without bound as an outlier grows.
S2 (token-level credit assignment): rank-calibrate a stop-gradient
evidence contrast within each trajectory, clip the token weights, and
renormalize them to mean exactly one. Claimed properties:
  * mass conservation: sum of weights = trajectory length L, exactly;
  * boundedness: no token weight exceeds cap / mean(clipped weights);
  * sign anchoring: token advantages = A_seq * w_i, so the total credit
    sum has the sign of A_seq and magnitude |A_seq| * L, independent of
    the raw contrast scale (unnormalized weights fail this: rescaling
    the contrasts rescales the total credit).
Operationalization note (honest limit): the abstract names S2's steps
but not the paper's exact "evidence contrast" formula, so S2 is tested
for the CLAIMED PROPERTIES (conservation / boundedness / sign anchoring
/ scale invariance) with rank-calibrated contrasts as the input, not for
bit-parity with the paper's unpublished contrast definition. S1's OT map
is exact in finite samples by construction (ranks -> normal quantiles).
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
    """Finite-sample 1-D OT map to N(0,1)."""
    n = len(x)
    ranks = _ranks_average(x)
    return [ _ND.inv_cdf(r / (n + 1.0)) for r in ranks ]


def affine_standardize(x):
    n = len(x)
    mu = sum(x) / n
    var = sum((v - mu) ** 2 for v in x) / n
    sd = math.sqrt(var) or 1.0
    return [(v - mu) / sd for v in x]


def s2_weights(contrasts, cap=2.0):
    """Rank-calibrate contrasts, clip, renormalize to mean exactly one.

    Rank-calibration maps each contrast to its within-trajectory rank
    quantile; the weight is that quantile scaled to [0, 1] (the raw
    contrast scale is discarded by design). Clipping then bounds the
    weights; renormalization restores mean one exactly.
    """
    n = len(contrasts)
    ranks = _ranks_average(contrasts)
    raw = [r / (n + 1.0) * 2.0 for r in ranks]  # in (0, 2)
    clipped = [min(max(v, 0.0), cap) for v in raw]
    mean = sum(clipped) / n
    return [v / mean for v in clipped]


def token_advantages(seq_advantage, contrasts, cap=2.0):
    w = s2_weights(contrasts, cap=cap)
    return [seq_advantage * wi for wi in w], w


def _skew(x):
    n = len(x); mu = sum(x) / n
    sd = math.sqrt(sum((v - mu) ** 2 for v in x) / n) or 1.0
    return sum(((v - mu) / sd) ** 3 for v in x) / n


def _kurtosis_excess(x):
    n = len(x); mu = sum(x) / n
    sd = math.sqrt(sum((v - mu) ** 2 for v in x) / n) or 1.0
    return sum(((v - mu) / sd) ** 4 for v in x) / n - 3.0


def _ks_to_normal(x):
    """Max |empirical CDF - Phi| over sorted standardized values."""
    xs = sorted(x)
    n = len(xs)
    worst = 0.0
    for i, v in enumerate(xs):
        f = _ND.cdf(v)
        worst = max(worst, abs((i + 1) / n - f), abs(i / n - f))
    return worst
