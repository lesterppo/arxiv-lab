"""Finish gate that re-reads the task (Mingbird, arXiv:2610.02001).

Failure form addressed: tasks silently abandoned — the model emits "task
complete" while artifacts are incomplete, and the harness accepts the claim.

The mechanism: before accepting completion, re-read the original task and
require explicit evidence for *every* success criterion, checked against the
current environment rather than the model's claim. Pure stdlib.
"""


class FinishGate:
    """Gate completion claims against re-checked task criteria."""

    def __init__(self, task_text: str, criteria: list):
        """``criteria``: list of (name, check_fn(env) -> bool)."""
        self.task_text = task_text
        self.criteria = criteria

    def review(self, env):
        """Re-read the task; check each criterion against the CURRENT
        environment (not the model's claim).

        Returns ``(accepted, failed_criteria)``.
        """
        failed = [name for name, check in self.criteria if not check(env)]
        return (len(failed) == 0), failed
