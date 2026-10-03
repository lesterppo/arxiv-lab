"""
CARM: Cancellation-Aware Response Masking for LLM Reinforcement Learning.

Paper: arXiv:2610.02039.

Problem: in RL post-training, sampled responses can be off-policy (policy
updated between rollout and training). Sequence-level masking decides
whether a whole response contributes to the update. The common rule uses
the length-normalized geometric mean of token probability ratios
r_i = pi_new(t_i)/pi_old(t_i), i.e. it keeps the response iff
|mean(log r_i)| <= log(1+eps). Signed log-ratios CANCEL across positions:
half the tokens at +2d and half at -2d gives mean 0 -> accepted, hiding
substantial bidirectional drift.

CARM: take the absolute value of each token log-ratio BEFORE averaging:
keep iff mean(|log r_i|) <= log(1+eps). Opposing changes can no longer
cancel. Accepted responses then satisfy a joint bound on (a) the fraction
of ratios outside the [1/(1+eps), 1+eps] band and (b) their mean
log-distance beyond the band edges.

This module implements both masks; the Colab script (colab_grpo_carm.py)
runs the real A/B inside a GRPO loop.
"""
import math


def standard_mask(log_ratios, eps=0.2):
    """Keep iff |mean(log r_i)| <= log(1+eps)  (geometric-mean rule)."""
    if not log_ratios:
        return False
    return abs(sum(log_ratios) / len(log_ratios)) <= math.log1p(eps)


def carm_mask(log_ratios, eps=0.2):
    """Keep iff mean(|log r_i|) <= log(1+eps)  (cancellation-aware)."""
    if not log_ratios:
        return False
    return sum(abs(x) for x in log_ratios) / len(log_ratios) <= math.log1p(eps)


def drift_stats(log_ratios, eps=0.2):
    """For bound checking: fraction of ratios outside the band and their
    mean log-distance beyond the band edges."""
    band = math.log1p(eps)
    outside = [abs(x) - band for x in log_ratios if abs(x) > band]
    return {
        "frac_outside": len(outside) / len(log_ratios),
        "mean_excess": sum(outside) / len(outside) if outside else 0.0,
        "max_abs": max((abs(x) for x in log_ratios), default=0.0),
    }
