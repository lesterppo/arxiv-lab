"""CIG-SFT: Conditional Information Gain for generalizable SFT.

Paper: arXiv 2610.12995 "What Should SFT Learn? Conditional Information
Gain for Generalizable Post-Training".

One-line claim: reweighting SFT token supervision by the conditional
information gain (how much the task input reduces uncertainty about each
demonstrated token, measured under a frozen reference model) improves
generalization to challenging eval distributions, with no extra data and
no online rollouts.

This module implements the paper's closed-loop weighting math in
miniature (the mechanism, not the LLM harness):
  CIG_j = log p_ref(t_j | task, prefix) - log p_ref(t_j | prefix)
  w_j   = exp(clip(CIG_j) / tau), mean-1 normalized per sequence.

numpy-guarded per repo conventions.
"""

try:
    import numpy as np
except ImportError:  # pragma: no cover
    np = None


def _need_numpy():
    if np is None:
        raise RuntimeError("numpy is required for arxiv_lab.training.cig_sft")


def cig_weights(lp_full, lp_naked, tau=1.0, clip=5.0):
    """Token weights from conditional information gain.

    lp_full:  (n,) log p_ref(t_j | task_input, prefix_<j})
    lp_naked: (n,) log p_ref(t_j | prefix_<j})  (no task input)
    Returns w with mean(w) == 1: w_j = exp(clip(CIG_j)/tau), normalized
    by the sequence mean (documented simplification of the paper's
    KL-regularized projection; preserves upweight-informative /
    downweight-boilerplate).
    """
    _need_numpy()
    lp_full = np.asarray(lp_full, dtype=float)
    lp_naked = np.asarray(lp_naked, dtype=float)
    cig = np.clip(lp_full - lp_naked, -clip, clip)
    ex = np.exp(cig / tau)
    return ex / ex.mean()


def weighted_nll(nlls, weights):
    """Scalar-weighted SFT loss: sum(nll * w) / sum(w)."""
    _need_numpy()
    nlls = np.asarray(nlls, dtype=float)
    weights = np.asarray(weights, dtype=float)
    return float((nlls * weights).sum() / weights.sum())


def simulate_admission(n_tokens=60, n_boiler=20, seed=0):
    """Miniature: boilerplate tokens predictable with/without the task
    (low CIG), substance tokens only predictable with it (high CIG).

    Returns dict with mean weights on the two groups and the enrichment
    ratio. Deterministic given seed.
    """
    _need_numpy()
    rng = np.random.default_rng(seed)
    # boilerplate: high logprob both ways -> CIG ~ 0
    lp_b_full = -0.2 + 0.05 * rng.standard_normal(n_boiler)
    lp_b_naked = -0.25 + 0.05 * rng.standard_normal(n_boiler)
    # substance: predictable only with the task input
    n_sub = n_tokens - n_boiler
    lp_s_full = -0.5 + 0.2 * rng.standard_normal(n_sub)
    lp_s_naked = -6.0 + 1.0 * rng.standard_normal(n_sub)
    lp_full = np.concatenate([lp_b_full, lp_s_full])
    lp_naked = np.concatenate([lp_b_naked, lp_s_naked])
    w = cig_weights(lp_full, lp_naked)
    return {
        "boiler_w": float(w[:n_boiler].mean()),
        "substance_w": float(w[n_boiler:].mean()),
        "enrichment": float(w[n_boiler:].mean() / w[:n_boiler].mean()),
    }
