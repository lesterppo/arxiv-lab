"""arxiv-lab.skills — skill-library routing protocols for agents.

See :mod:`arxiv_lab.skills.skill_routing` for the mechanism
(arXiv:2610.18522).
"""

from arxiv_lab.skills.skill_routing import (
    Skill,
    Task,
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
    "Skill",
    "Task",
    "aggregate",
    "build_library",
    "collapse_gap",
    "parse_loaded",
    "preselect_prompt",
    "render_library",
    "route_prompt",
    "routing_scores",
]
