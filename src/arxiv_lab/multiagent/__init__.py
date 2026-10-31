"""Multi-agent collaboration mechanisms."""

from arxiv_lab.multiagent.block_opt_merge import (
    BlockObjective,
    collaborate,
    conflict_graph,
    fractional_chromatic_number,
    fractional_coloring,
    fractional_merger,
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
