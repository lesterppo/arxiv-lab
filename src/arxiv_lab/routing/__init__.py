"""arxiv-lab.routing — component routing for self-improving agents,
and LEASH uncertainty-aware weak-to-strong escalation.

See :mod:`arxiv_lab.routing.component` for the mechanism
(arXiv:2610.01787) and :mod:`arxiv_lab.routing.leash` for LEASH
(arXiv:2610.22280).
"""

from arxiv_lab.routing.component import (
    COMPONENT_TYPES,
    CONTEXT,
    WEIGHTS,
    Component,
    RoutingRule,
    fit_routing_rule,
    route_components,
)
from arxiv_lab.routing.leash import (
    LeashPolicy,
    TaskSim,
    UncertaintyEstimator,
    auc,
    evaluate,
    evaluate_random,
)

__all__ = [
    "COMPONENT_TYPES",
    "CONTEXT",
    "WEIGHTS",
    "Component",
    "RoutingRule",
    "fit_routing_rule",
    "route_components",
    "LeashPolicy",
    "TaskSim",
    "UncertaintyEstimator",
    "auc",
    "evaluate",
    "evaluate_random",
]
