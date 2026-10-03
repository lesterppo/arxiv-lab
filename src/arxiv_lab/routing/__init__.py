"""arxiv-lab.routing — component routing for self-improving agents.

See :mod:`arxiv_lab.routing.component` for the mechanism
(arXiv:2610.01787).
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

__all__ = [
    "COMPONENT_TYPES",
    "CONTEXT",
    "WEIGHTS",
    "Component",
    "RoutingRule",
    "fit_routing_rule",
    "route_components",
]
