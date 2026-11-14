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

from arxiv_lab.agent.receipt import (
    RunResult,
    extract_waiting_prompt,
    run_closed,
    run_open_unattended,
    run_receipt,
    simulate_outcome,
)

from arxiv_lab.agent.runtime_tickets import (
    Obligation,
    Send,
    Ticket,
    coarse_allowlist_allows,
    decode,
    decode_gate_trace,
    gate_decision,
    oracle_allows,
    run_trace,
    scanner_flags,
)

from arxiv_lab.agent.reshape import (
    DOMAINS,
    Instance,
    RNG,
    STRATEGIES,
    gen_instances,
    plan,
    revise,
    revise_ask,
    revise_oracle,
    revise_show,
    run_experiment,
    space_metrics,
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
    # receipt (arXiv:2611.01930)
    "RunResult",
    "extract_waiting_prompt",
    "run_closed",
    "run_open_unattended",
    "run_receipt",
    "simulate_outcome",
    # runtime_tickets (arXiv:2611.03772)
    "Obligation",
    "Send",
    "Ticket",
    "coarse_allowlist_allows",
    "decode",
    "decode_gate_trace",
    "gate_decision",
    "oracle_allows",
    "run_trace",
    "scanner_flags",
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
    # reshape (arXiv:2611.16466)
    "DOMAINS",
    "Instance",
    "RNG",
    "STRATEGIES",
    "gen_instances",
    "plan",
    "revise",
    "revise_ask",
    "revise_oracle",
    "revise_show",
    "run_experiment",
    "space_metrics",
]
