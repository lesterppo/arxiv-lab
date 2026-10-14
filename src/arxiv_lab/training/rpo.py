"""Root-token Policy Optimization, RPO (arXiv:2610.17767).

Paper: "Efficiently Learning To Reason or Not to Reason: Root-token
Policy Optimization for Adaptive Thinking".

One-line claim: training *only the root token* (the decision whether to
invoke <think>) via group-relative rewards and group-wise advantages
learns difficulty-aware adaptive thinking at ~2% of the training compute
of prior adaptive-reasoning methods.

This module implements the paper's learning rule in miniature, model-free
form:

- ``group_advantages(rewards)``: per-group baseline ``A_i = r_i -
  mean(r)`` (the paper's group-relative reward / group-wise advantage).
- ``root_token_losses(logp, advantages, logp_ref, beta)``: the root-token
  policy-gradient loss ``-mean(A * logp)`` plus a KL-to-reference penalty
  (Schulman unbiased estimator), i.e. the only parameters that ever see
  gradient are those feeding the first decision token.
- ``simulate_adaptive_thinking(...)``: a seeded, deterministic toy world
  with easy/hard items where thinking helps only on hard items and costs
  a fixed penalty. Group rollouts with forced THINK/DIRECT arms drive a
  logistic root policy; the test asserts the paper's signature emerges:
  think-rate(hard) >> think-rate(easy) with accuracy preserved.

numpy-guarded per repo conventions.
"""

from __future__ import annotations

import math
import random

try:
    import numpy as np
except Exception:  # pragma: no cover
    np = None


def _need_numpy():
    if np is None:
        raise RuntimeError("numpy is required for arxiv_lab.training.rpo")


def group_advantages(rewards):
    """Group-relative advantages: A_i = r_i - mean(r).

    ``rewards``: 1-D array of one rollout group's rewards.
    Returns an array of the same shape summing to ~0.
    """
    _need_numpy()
    r = np.asarray(rewards, dtype=float)
    return r - r.mean()


def kl_penalty(logp, logp_ref):
    """Unbiased KL(pi || pi_ref) estimator (Schulman) per sample, given
    log-probs of the *sampled* root token under the current and reference
    policies."""
    _need_numpy()
    logp = np.asarray(logp, dtype=float)
    logp_ref = np.asarray(logp_ref, dtype=float)
    ratio = logp_ref - logp
    return np.exp(ratio) - ratio - 1.0


def root_token_losses(logp, advantages, logp_ref=None, beta=0.0):
    """Root-token objective pieces.

    loss = -mean(A * logp) + beta * mean(KL(pi || pi_ref)).
    Returns (policy_loss, kl_mean, total).
    """
    _need_numpy()
    logp = np.asarray(logp, dtype=float)
    adv = np.asarray(advantages, dtype=float)
    policy_loss = -float(np.mean(adv * logp))
    kl_mean = 0.0
    if logp_ref is not None and beta > 0.0:
        kl_mean = float(np.mean(kl_penalty(logp, logp_ref)))
    return policy_loss, kl_mean, policy_loss + beta * kl_mean


def _sigmoid(x):
    return 1.0 / (1.0 + math.exp(-x))


# Toy-world correctness model: (difficulty, thought) -> P(correct).
# Thinking helps only on hard items; it always costs LAMBDA.
CORRECT_PROB = {
    ("easy", False): 0.95,
    ("easy", True): 0.97,
    ("hard", False): 0.25,
    ("hard", True): 0.80,
}
THINK_COST = 0.15


def simulate_adaptive_thinking(n_steps=4000, lr=0.5, seed=0):
    """Seeded toy RPO: one logistic root policy per difficulty,
    ``P(think|d) = sigmoid(theta_d)``.

    Each step picks an alternating-difficulty item, rolls out a forced
    group [THINK, THINK, DIRECT, DIRECT], scores
    ``reward = correct - THINK_COST * thought``, forms group-relative
    advantages, and takes one SGD step on the item's theta with the
    root-token policy gradient (the only gradient in the system).

    Returns dict with final think rates per difficulty, the adaptive gap,
    and mean accuracy. Fully deterministic given ``seed``.
    """
    rng = random.Random(seed)
    theta = {"easy": 0.0, "hard": 0.0}
    correct_total = 0
    n_rollouts = 0
    for step in range(n_steps):
        d = "easy" if step % 2 == 0 else "hard"
        arms = [True, True, False, False]  # forced THINK x2, DIRECT x2
        rewards = []
        logps = []
        p = _sigmoid(theta[d])
        p = min(max(p, 1e-9), 1.0 - 1e-9)
        for think in arms:
            correct = 1.0 if rng.random() < CORRECT_PROB[(d, think)] else 0.0
            rewards.append(correct - THINK_COST * think)
            logps.append(math.log(p if think else 1.0 - p))
            correct_total += correct
            n_rollouts += 1
        adv = [r - sum(rewards) / len(rewards) for r in rewards]
        # d/dtheta [-mean(A * logp)]: logp = think*log p + (1-think)*log(1-p)
        grad = 0.0
        for a, think in zip(adv, arms):
            dlogp = (1.0 - p) if think else (-p)
            grad += -a * dlogp / len(arms)
        theta[d] -= lr * grad
    return {
        "think_rate_easy": _sigmoid(theta["easy"]),
        "think_rate_hard": _sigmoid(theta["hard"]),
        "adaptive_gap": _sigmoid(theta["hard"]) - _sigmoid(theta["easy"]),
        "mean_accuracy": correct_total / n_rollouts,
        "theta": dict(theta),
    }
