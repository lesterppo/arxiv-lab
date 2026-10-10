"""Context-curation mechanisms (arXiv 2610.11007, 2610.10590)."""
from .curated_context import (
    KAPPA, LAMBDA, CAPACITY,
    dilution, net_value, standalone_value,
    greedy_append, optimal_subset, size_upper_bound,
    arbitrarily_worse_family,
    simulate_censored_feedback, delete_if_ignored,
    expected_regret, optimal_trial_count,
)
from .reversible_forgetting import (
    ForgettingError, ForgettingHarness, IrreversibleHarness,
    ToolResult, Message, stub_text, estimate_tokens,
    make_noisy_observation, make_dense_observation, extract_note,
    TrajectoryReplay,
)

__all__ = [
    "KAPPA", "LAMBDA", "CAPACITY",
    "dilution", "net_value", "standalone_value",
    "greedy_append", "optimal_subset", "size_upper_bound",
    "arbitrarily_worse_family",
    "simulate_censored_feedback", "delete_if_ignored",
    "expected_regret", "optimal_trial_count",
    "ForgettingError", "ForgettingHarness", "IrreversibleHarness",
    "ToolResult", "Message", "stub_text", "estimate_tokens",
    "make_noisy_observation", "make_dense_observation", "extract_note",
    "TrajectoryReplay",
]
