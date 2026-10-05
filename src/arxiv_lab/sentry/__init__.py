"""Sentry: conditional failure-knowledge layer for LLM agents
(arXiv:2610.02994).

Failure lessons are conditional knowledge and must be conditionally exposed:
on a detected failure, matching lessons are retrieved from an external
playbook to guide recovery, recovery is verified WITHOUT task rewards, and a
new lesson is stored only when recovery is verified. The full playbook never
enters the agent's context.

Pure stdlib. No numpy needed.
"""

from arxiv_lab.sentry.sentry import (
    Action,
    FailureDetector,
    FailureEvent,
    Lesson,
    Playbook,
    RecoveryVerifier,
    Sentry,
    INVALID_TOOL_CALL,
    REPEATED_ACTION,
    POOR_GROUNDING,
    PREMATURE_FINISH,
)

__all__ = [
    "Action",
    "FailureDetector",
    "FailureEvent",
    "Lesson",
    "Playbook",
    "RecoveryVerifier",
    "Sentry",
    "INVALID_TOOL_CALL",
    "REPEATED_ACTION",
    "POOR_GROUNDING",
    "PREMATURE_FINISH",
]
