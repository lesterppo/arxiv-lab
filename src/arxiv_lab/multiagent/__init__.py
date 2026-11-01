"""Multi-agent collaboration mechanisms."""

from arxiv_lab.multiagent.block_opt_merge import (
    BlockObjective,
    collaborate,
    conflict_graph,
    fractional_chromatic_number,
    fractional_coloring,
    fractional_merger,
)
from arxiv_lab.multiagent.psv import (
    generate_prior_shift_data,
    train_softmax_regression,
    accuracy,
    majority_vote_accuracy,
    make_perturbed_members,
    select_and_vote,
    run_experiment as psv_run_experiment,
)
from arxiv_lab.multiagent.splice import (
    make_queries,
    run_skeleton,
    run_candidate,
    locate_gap,
    process_features,
    train_selector,
    selector_predict,
    run_experiment,
)

__all__ = [
    "generate_prior_shift_data",
    "train_softmax_regression",
    "accuracy",
    "majority_vote_accuracy",
    "make_perturbed_members",
    "select_and_vote",
    "psv_run_experiment",
    "BlockObjective",
    "collaborate",
    "conflict_graph",
    "fractional_chromatic_number",
    "fractional_coloring",
    "fractional_merger",
    "make_queries",
    "run_skeleton",
    "run_candidate",
    "locate_gap",
    "process_features",
    "train_selector",
    "selector_predict",
    "run_experiment",
]
