"""SCARL: Sequence Confidence-Aligned Reinforcement Learning.

Paper: arXiv 2610.40407 "SCARL: Sequence Confidence-Aligned
Reinforcement Learning" (Ke et al., submitted 29 Oct 2026).

One-line claim: in RL with verifiable rewards, nothing ties a policy's
confidence in a response to its reward, so a barely-found success earns
the same update as a near-certain one and a confident error is corrected
no harder than an exploratory one. SCARL reshapes — but never re-signs —
each advantage with the alignment gap between the response's
group-relative confidence and the outcome its advantage certifies,
normalized into shares of each class's advantage budget. Misaligned
samples absorb the force, responses with the same advantage are driven
toward equal confidence, the reshaping reduces to the base estimator
when a class's confidences agree, and repairing the misalignment doubles
as a remedy for the entropy collapse that erodes exploration. It adds
no extra passes and one hyperparameter.

Operationalization note (see the 2026-11-01 session): the abstract does
not print SCARL's exact formula, so this module implements the abstract's
stated steps literally — gap = |confidence - outcome-certified
confidence| (1 for the positive-advantage class, 0 for the negative
class), raw magnitude |A| * (1 + lam * gap), renormalized within each
class so the class's total advantage budget is preserved exactly — and
validates the claimed PROPERTIES (sign, budget, reduction, misaligned
amplification, paired-step entropy/confidence effects) rather than
bit-parity with the paper's code. Stdlib + numpy per repo conventions
(the reshaping itself is stdlib; numpy is used only for the paired
softmax-policy step).
"""

try:
    import numpy as np
except ImportError:  # pragma: no cover
    np = None


def _need_numpy():
    if np is None:
        raise RuntimeError("numpy is required for scarl")


def group_advantages(rewards):
    """Group-relative advantages: reward minus the group mean."""
    _need_numpy()
    r = np.asarray(rewards, dtype=float)
    return r - r.mean()


def alignment_gaps(advantages, confidences):
    """|confidence - outcome-certified confidence| per response.

    Positive-advantage responses certify confidence 1, negative-advantage
    responses certify 0, zero-advantage responses have gap 0.
    """
    _need_numpy()
    a = np.asarray(advantages, dtype=float)
    c = np.asarray(confidences, dtype=float)
    target = np.where(a > 1e-12, 1.0, np.where(a < -1e-12, 0.0, c))
    gap = np.abs(c - target)
    gap = np.where(np.abs(a) <= 1e-12, 0.0, gap)
    return gap


def scarl_advantages(rewards, confidences, lam=1.0):
    """SCARL-reshaped advantages (one hyperparameter, ``lam``).

    Never re-signs: the sign of each advantage is preserved by
    construction. Preserves each class's total |advantage| budget
    exactly via within-class renormalization. Reduces exactly to the
    base estimator when confidences agree within a class (then all gaps
    in the class are equal, so the raw magnitudes stay proportional to
    |A| and renormalization returns |A| itself).
    """
    _need_numpy()
    A = group_advantages(rewards)
    gap = alignment_gaps(A, confidences)
    out = np.zeros_like(A)
    for mask in (A > 1e-12, A < -1e-12):
        idx = np.where(mask)[0]
        if len(idx) == 0:
            continue
        raw = np.abs(A[idx]) * (1.0 + lam * gap[idx])
        budget = np.abs(A[idx]).sum()
        out[idx] = np.sign(A[idx]) * budget * (raw / raw.sum())
    return out


def softmax_policy_step(theta, advantages, lr=0.5):
    """One softmax policy-gradient step: theta += lr * sum_i A_i (e_i - p)."""
    _need_numpy()
    theta = np.asarray(theta, dtype=float)
    p = np.exp(theta - theta.max())
    p /= p.sum()
    grad = np.zeros_like(theta)
    for i in range(len(theta)):
        e = np.zeros_like(theta)
        e[i] = 1.0
        grad += advantages[i] * (e - p)
    return theta + lr * grad


def policy_probs(theta):
    """Softmax probabilities for a logit vector."""
    _need_numpy()
    theta = np.asarray(theta, dtype=float)
    p = np.exp(theta - theta.max())
    return p / p.sum()


def entropy(probs):
    """Shannon entropy (nats) of a probability vector."""
    _need_numpy()
    p = np.asarray(probs, dtype=float)
    return float(-(p * np.log(p + 1e-12)).sum())


def run_experiment(n_groups=200, seeds=(1, 7, 42), lr=0.5):
    """S1 property checks (500 random groups) + S2 paired policy steps.

    S2: per group of 8 responses (3 rewarded), the SAME initial logits
    take one base-advantage step and one SCARL-advantage step (confidence
    = current softmax probability). Reported deltas are SCARL minus base
    where noted, averaged over groups and seeds.
    """
    _need_numpy()
    rng = np.random.default_rng(0)
    sign_err = 0.0
    budget_err = 0.0
    reduction_err = 0.0
    for _ in range(500):
        n = int(rng.integers(4, 13))
        rewards = rng.choice([0.0, 1.0], size=n, p=[0.4, 0.6])
        if len(set(rewards)) < 2:
            continue
        conf = rng.random(n)
        A = group_advantages(rewards)
        S = scarl_advantages(rewards, conf)
        sign_err = max(sign_err, float(np.max(np.abs(np.sign(S) - np.sign(A)))))
        for mask in (A > 1e-12, A < -1e-12):
            if mask.sum():
                budget_err = max(budget_err,
                                 abs(float(np.abs(S[mask]).sum() - np.abs(A[mask]).sum())))
        conf_agree = np.where(A > 0, 0.7, np.where(A < 0, 0.3, 0.5))
        S_agree = scarl_advantages(rewards, conf_agree)
        reduction_err = max(reduction_err, float(np.max(np.abs(S_agree - A))))
    # Canonical misaligned pair: two equal-advantage successes, conf 0.1 vs 0.9.
    pair = scarl_advantages([1.0, 1.0, 0.0], [0.1, 0.9, 0.5])
    misaligned_ratio = float(pair[0] / pair[1])
    # S2 paired steps.
    d_entropy, d_spread, low_gain, high_gain, conf_err, expl_err = [], [], [], [], [], []
    for seed in seeds:
        r = np.random.default_rng(seed)
        for _ in range(n_groups):
            theta = r.normal(size=8)
            pos = r.choice(8, size=3, replace=False)
            rew = np.zeros(8)
            rew[pos] = 1.0
            p = policy_probs(theta)
            A = group_advantages(rew)
            S = scarl_advantages(rew, p)
            th_b = softmax_policy_step(theta, A, lr=lr)
            th_s = softmax_policy_step(theta, S, lr=lr)
            pb, ps = policy_probs(th_b), policy_probs(th_s)
            d_entropy.append(entropy(ps) - entropy(pb))
            d_spread.append(float(ps[pos].std()) - float(pb[pos].std()))
            lo, hi = pos[np.argmin(p[pos])], pos[np.argmax(p[pos])]
            low_gain.append(float((ps[lo] - p[lo]) - (pb[lo] - p[lo])))
            high_gain.append(float((ps[hi] - p[hi]) - (pb[hi] - p[hi])))
            neg = np.array([i for i in range(8) if i not in pos])
            hi_neg, lo_neg = neg[np.argmax(p[neg])], neg[np.argmin(p[neg])]
            conf_err.append(float((ps[hi_neg] - p[hi_neg]) - (pb[hi_neg] - p[hi_neg])))
            expl_err.append(float((ps[lo_neg] - p[lo_neg]) - (pb[lo_neg] - p[lo_neg])))
    return {
        "s1_sign_error": sign_err,
        "s1_budget_error": budget_err,
        "s1_reduction_error": reduction_err,
        "misaligned_to_aligned_ratio": misaligned_ratio,
        "s2_entropy_scarl_minus_base": float(np.mean(d_entropy)),
        "s2_positive_spread_scarl_minus_base": float(np.mean(d_spread)),
        "s2_low_conf_success_extra_gain": float(np.mean(low_gain)),
        "s2_high_conf_success_extra_gain": float(np.mean(high_gain)),
        "s2_confident_error_extra_change": float(np.mean(conf_err)),
        "s2_exploratory_error_extra_change": float(np.mean(expl_err)),
        "n_property_groups": 500,
        "n_policy_groups_per_seed": n_groups,
        "seeds": list(seeds),
    }
