"""Context-curation mechanisms (arXiv 2610.11007)."""
from .curated_context import (
    KAPPA, LAMBDA, CAPACITY,
    dilution, net_value, standalone_value,
    greedy_append, optimal_subset, size_upper_bound,
    arbitrarily_worse_family,
    simulate_censored_feedback, delete_if_ignored,
    expected_regret, optimal_trial_count,
)

__all__ = [
    "KAPPA", "LAMBDA", "CAPACITY",
    "dilution", "net_value", "standalone_value",
    "greedy_append", "optimal_subset", "size_upper_bound",
    "arbitrarily_worse_family",
    "simulate_censored_feedback", "delete_if_ignored",
    "expected_regret", "optimal_trial_count",
]
