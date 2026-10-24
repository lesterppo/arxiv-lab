"""arxiv-lab.skills — skill-library routing protocols for agents.

See :mod:`arxiv_lab.skills.skill_routing` for the routing-collapse mechanism
(arXiv:2610.18522) and :mod:`arxiv_lab.skills.skill_granularity` for
adaptive skill granularity (arXiv:2610.31461).
"""

from arxiv_lab.skills.skill_granularity import (
    MACRO_LIBRARY,
    ExecutionTrace,
    GranularityController,
    Macro,
    RecoveryExecutor,
    StrategyResult,
    Task,
    generate_tasks,
    oracle_route,
    run_protocol,
)
from arxiv_lab.skills.skill_routing import (
    Skill,
    Task as RoutingTask,
    aggregate,
    build_library,
    collapse_gap,
    parse_loaded,
    preselect_prompt,
    render_library,
    route_prompt,
    routing_scores,
)

__all__ = [
    "MACRO_LIBRARY",
    "ExecutionTrace",
    "GranularityController",
    "Macro",
    "RecoveryExecutor",
    "StrategyResult",
    "Task",
    "generate_tasks",
    "oracle_route",
    "run_protocol",
    "Skill",
    "RoutingTask",
    "aggregate",
    "build_library",
    "collapse_gap",
    "parse_loaded",
    "preselect_prompt",
    "render_library",
    "route_prompt",
    "routing_scores",
]
