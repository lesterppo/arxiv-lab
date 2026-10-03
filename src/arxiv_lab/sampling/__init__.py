"""FERPO forward-KL mode-covering sampler (arXiv:2610.02198).

Self-normalized importance sampling from a KL-regularized improvement
target, with forward-KL (mode-covering) and reverse-KL (mode-seeking) actor
fits for comparison.

Requires numpy (optional dependency): ``pip install numpy``.
"""

from arxiv_lab.sampling.ferpo import (
    TGT_MEANS,
    TGT_SIGS,
    TGT_WTS,
    ROLL_MU,
    ROLL_SIG,
    log_target,
    log_rollout,
    snis_weights,
    forward_kl_fit,
    reverse_kl_fit,
    mode_coverage,
    expected_log_target,
)

__all__ = [
    "TGT_MEANS", "TGT_SIGS", "TGT_WTS", "ROLL_MU", "ROLL_SIG",
    "log_target", "log_rollout", "snis_weights",
    "forward_kl_fit", "reverse_kl_fit", "mode_coverage",
    "expected_log_target",
]
