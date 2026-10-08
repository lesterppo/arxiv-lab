"""Self-improvement loop mechanisms (arXiv read-and-test).

- ``winners_curse`` — The Winner's Curse in LLM self-improvement loops:
  keep-if-better on a small selection set as selection under measurement
  noise; correlated candidate errors, lock-in from mostly-harmful
  proposals, and acceptance-rule comparison (arXiv:2610.09239).
"""

from arxiv_lab.loops.winners_curse import (
    make_selection_set,
    score_candidates,
    heldout_score,
    generate_candidates,
    error_correlation,
    winners_curse_gap,
    run_greedy_loop,
    run_loop_with_rule,
    selection_bias,
)

__all__ = [
    "make_selection_set",
    "score_candidates",
    "heldout_score",
    "generate_candidates",
    "error_correlation",
    "winners_curse_gap",
    "run_greedy_loop",
    "run_loop_with_rule",
    "selection_bias",
]
