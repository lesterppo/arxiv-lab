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

from arxiv_lab.agent.vbl import (
    GoalSpec,
    GoalState,
    VBL,
    MAX_RETRIES,
    react_baseline,
)

from arxiv_lab.agent.spec_agent import (
    SpecAgent,
    Step,
    StrongPolicy,
    ToolWorld,
    WeakPolicy,
    evaluate,
    run_strong_only,
    run_weak_only,
    TOKEN_COSTS,
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
    # vbl (arXiv:2610.41564)
    "GoalSpec",
    "GoalState",
    "VBL",
    "MAX_RETRIES",
    "react_baseline",
    # spec_agent (arXiv:2610.21463)
    "SpecAgent",
    "Step",
    "StrongPolicy",
    "ToolWorld",
    "WeakPolicy",
    "evaluate",
    "run_strong_only",
    "run_weak_only",
    "TOKEN_COSTS",
]
