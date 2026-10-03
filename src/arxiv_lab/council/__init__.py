"""Bayesian Dialectical Argumentation council (arXiv:2610.02005).

Calibrated multi-LLM council aggregation: typed deliberation moves are fit to
a per-agent reliability (Dawid-Skene style) annotator model via EM, so
adversarial agents are inverted rather than merely outvoted.

Requires numpy (optional dependency): ``pip install numpy``.
"""

from arxiv_lab.council.bda import (
    simulate_council,
    bda_fit,
    majority_vote,
    brier_score,
)

__all__ = ["simulate_council", "bda_fit", "majority_vote", "brier_score"]
