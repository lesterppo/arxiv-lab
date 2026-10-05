"""Prospective Hindsight (PH): self-calibrating RL via prediction-reality gaps.

Paper: arXiv 2610.02740 "Prospective Hindsight: Self-Calibrating
Reinforcement Learning via Prediction-Reality Gaps".

One-line claim: augmenting any retrospective base RL method with a signal
derived from the gap between the agent's *prospective* prediction (belief at
action time) and the *retrospective* evaluation (outcome after feedback) —
amplifying the gradient on high-surprise samples through a stop-gradient
surprise-weighted advantage — provides a descent pathway on the agent's
miscalibration rate. Because the prospective predictor shares parameters
with the policy, the two co-evolve and calibration emerges as a byproduct of
optimization rather than from an added objective.

This module implements the PH weighting primitive (pure functions; the
stop-gradient is structural — weights are computed from detached values) plus
small calibration metrics. numpy-guarded per repo conventions.
"""
from __future__ import annotations

try:
    import numpy as np
except ImportError:  # pragma: no cover
    np = None


def _need_numpy():
    if np is None:
        raise RuntimeError("numpy is required for arxiv_lab.training")


def surprise_weights(prospective, retrospective, lam: float = 1.0):
    """Per-sample PH weights: w_i = 1 + lam * |p_i - r_i|.

    prospective: the agent's pre-feedback prediction of success (shares
        parameters with the policy; passed in detached).
    retrospective: the observed outcome after feedback (0/1 or return).
    lam: surprise amplification strength (paper's ablations center lam ~ 1).
    """
    _need_numpy()
    p = np.asarray(prospective, dtype=float)
    r = np.asarray(retrospective, dtype=float)
    surprise = np.abs(p - r)
    return 1.0 + lam * surprise, surprise


def ph_advantages(advantages, prospective, retrospective, lam: float = 1.0):
    """Surprise-weighted advantages: A_PH = A * (1 + lam * |p - r|).

    Amplifies the gradient contribution of rollouts where the agent's
    self-model was most inaccurate, shifting focus to remaining blind spots
    as predictor and policy co-evolve.
    """
    _need_numpy()
    a = np.asarray(advantages, dtype=float)
    w, surprise = surprise_weights(prospective, retrospective, lam)
    return a * w, surprise


def miscalibration_rate(prospective, retrospective) -> float:
    """Mean absolute prediction-reality gap E|p - r|."""
    _need_numpy()
    p = np.asarray(prospective, dtype=float)
    r = np.asarray(retrospective, dtype=float)
    return float(np.mean(np.abs(p - r)))


def expected_calibration_error(confidences, outcomes, n_bins: int = 10) -> float:
    """ECE between predicted success probabilities and empirical outcomes."""
    _need_numpy()
    c = np.asarray(confidences, dtype=float)
    y = np.asarray(outcomes, dtype=float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for b in range(n_bins):
        lo, hi = edges[b], edges[b + 1]
        mask = (c > lo) & (c <= hi) if b else (c >= lo) & (c <= hi)
        if mask.sum() == 0:
            continue
        ece += (mask.sum() / len(c)) * abs(c[mask].mean() - y[mask].mean())
    return float(ece)
