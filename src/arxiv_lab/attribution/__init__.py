"""Failure-attribution mechanisms (arXiv read-and-test).

- ``emfa`` — Error-Propagation Modeling for Failure Attribution: decisive
  (agent, step) identification in failed multi-agent trajectories via
  dependency-traced error propagation, persistent-loop extraction, and
  counterfactual verification (arXiv:2610.11600).
"""

from arxiv_lab.attribution.emfa import (
    Step,
    Trajectory,
    find_final_failure,
    build_dependencies,
    backward_trace,
    annotate_propagation,
    extract_loops,
    screen_candidates,
    counterfactual_verify,
    emfa_attribute,
    blame_final,
    blame_latest_wrong,
    blame_earliest_wrong,
)
