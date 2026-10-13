"""Agent-track mechanisms (multi-agent orchestration, harness state)."""

from arxiv_lab.agent.competence import (
    CompetenceTracker,
    orchestrate,
    select_mode,
    query_competence,
    communication_graph,
    simulate_mas,
    calibration_spearman,
    query_calibration_spearman,
    make_backbone,
    SOLO,
    PAIR,
    DELIBERATE,
    MODE_TOKENS,
    TASK_TYPES,
)

__all__ = [
    "CompetenceTracker",
    "orchestrate",
    "select_mode",
    "query_competence",
    "communication_graph",
    "simulate_mas",
    "calibration_spearman",
    "query_calibration_spearman",
    "make_backbone",
    "SOLO",
    "PAIR",
    "DELIBERATE",
    "MODE_TOKENS",
    "TASK_TYPES",
]
