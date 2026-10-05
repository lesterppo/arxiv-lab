"""Sentry: a failure-management layer for LLM agents.

Paper: arXiv 2610.02994 "Sentry: Learning to Recover from LLM Agent Failures
at Test Time".

One-line claim: failure knowledge is *conditional* knowledge and must be
conditionally exposed — on a detected failure, matching lessons are retrieved
from an external playbook to guide recovery, recovery is verified WITHOUT
task rewards, and a new lesson is stored only if recovery verified. The full
playbook never enters the agent's context (exposing it lowers performance).

stdlib only.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# Failure taxonomy
# ---------------------------------------------------------------------------

INVALID_TOOL_CALL = "invalid_tool_call"
REPEATED_ACTION = "repeated_action"
POOR_GROUNDING = "poor_grounding"
PREMATURE_FINISH = "premature_finish"

ERROR_MARKERS = ("error", "invalid", "failed", "unknown tool", "bad args",
                 "exception", "traceback")


@dataclass
class Action:
    """One agent step: what it did and what came back."""
    tool: str
    args: dict
    observation: str = ""
    thought: str = ""


@dataclass
class FailureEvent:
    ftype: str
    step: int
    detail: str


class FailureDetector:
    """Detect agent failures from an action trace (no task labels needed)."""

    def __init__(self, allowed_tools=None, repeat_k: int = 3):
        self.allowed_tools = set(allowed_tools) if allowed_tools else None
        self.repeat_k = repeat_k

    def _is_error(self, obs: str) -> bool:
        o = (obs or "").lower()
        return any(m in o for m in ERROR_MARKERS)

    def detect(self, trace: list[Action]) -> list[FailureEvent]:
        events: list[FailureEvent] = []
        for i, a in enumerate(trace):
            # invalid tool call: unknown tool or erroring observation
            if self.allowed_tools is not None and a.tool not in self.allowed_tools \
                    and a.tool != "finish":
                events.append(FailureEvent(INVALID_TOOL_CALL, i,
                                           f"unknown tool {a.tool!r}"))
            elif self._is_error(a.observation):
                # only flag once per contiguous error run (avoid double count)
                if not (events and events[-1].ftype == INVALID_TOOL_CALL
                        and events[-1].step == i - 1):
                    events.append(FailureEvent(INVALID_TOOL_CALL, i,
                                               f"error obs: {a.observation[:60]!r}"))
            # premature finish: 'finish' before any evidence-gathering action
            if a.tool == "finish":
                evidence = any(t.tool not in ("finish",) and not self._is_error(t.observation)
                               for t in trace[:i])
                if not evidence:
                    events.append(FailureEvent(PREMATURE_FINISH, i,
                                               "finished with no valid evidence"))
            # repeated action: same (tool, args) repeat_k times in a row
            if i + 1 >= self.repeat_k:
                window = trace[i + 1 - self.repeat_k:i + 1]
                sig = [(t.tool, tuple(sorted(t.args.items()))) for t in window]
                if len(set(sig)) == 1 and not self._is_error(window[-1].observation):
                    # error-free repeats are still a loop; erroring repeats are
                    # already covered by INVALID_TOOL_CALL above
                    events.append(FailureEvent(REPEATED_ACTION, i,
                                               f"repeated {a.tool} x{self.repeat_k}"))
            # poor grounding: thought references an identifier never observed
            if a.thought:
                mentioned = set()
                for tok in re.findall(r"[A-Za-z_][A-Za-z0-9_]{3,}", a.thought):
                    mentioned.add(tok)
                seen_text = " ".join(t.observation or "" for t in trace[:i + 1])
                tools_seen = {t.tool for t in trace[:i + 1]}
                novel = [t for t in mentioned
                         if t not in seen_text and t not in tools_seen
                         and t.lower() not in ("the", "with", "then", "this",
                                               "that", "from", "will")]
                if novel:
                    events.append(FailureEvent(POOR_GROUNDING, i,
                                               f"ungrounded refs: {novel[:3]}"))
        return events


# ---------------------------------------------------------------------------
# Playbook: external lesson store. The agent NEVER sees all of it.
# ---------------------------------------------------------------------------

@dataclass
class Lesson:
    id: str
    failure_type: str
    trigger: str      # when this lesson applies (human-readable condition)
    guidance: str     # the recovery instruction shown to the agent
    source_task: str = ""
    verified_recoveries: int = 0


class Playbook:
    """External lesson store. Conditional exposure: agents only ever receive
    the output of `retrieve()`, never the whole store."""

    def __init__(self):
        self._lessons: list[Lesson] = []

    def add(self, lesson: Lesson) -> None:
        self._lessons.append(lesson)

    def retrieve(self, failure: FailureEvent, k: int = 2) -> list[Lesson]:
        """Return at most k lessons matching the failure type, most-verified
        first. Returns [] when nothing matches — the agent then gets nothing,
        which is the point of conditional exposure."""
        matches = [l for l in self._lessons if l.failure_type == failure.ftype]
        matches.sort(key=lambda l: -l.verified_recoveries)
        return matches[:k]

    def get(self, lesson_id: str) -> Lesson | None:
        for l in self._lessons:
            if l.id == lesson_id:
                return l
        return None

    def __len__(self) -> int:
        return len(self._lessons)


# ---------------------------------------------------------------------------
# Recovery verification — no task rewards, no success labels.
# ---------------------------------------------------------------------------

class RecoveryVerifier:
    """Decide whether an agent recovered from a failure using only the traces.

    Deliberately takes NO task-success label and NO reward: verification is
    structural — (1) no new failure of the same type after the lesson,
    (2) the agent issued at least one valid (non-error, non-repeated) action
    afterwards, (3) the post-lesson action pattern differs from the failed one.
    """

    def __init__(self, detector: FailureDetector | None = None):
        self.detector = detector or FailureDetector()

    def verify(self, trace_before: list[Action], trace_after: list[Action],
               failure: FailureEvent) -> bool:
        if not trace_after:
            return False
        # (1) no recurrence of the same failure type after the lesson
        for ev in self.detector.detect(trace_after):
            if ev.ftype == failure.ftype:
                return False
        # (2) at least one valid action after the lesson
        valid = [a for a in trace_after
                 if a.tool != "finish" and not self.detector._is_error(a.observation)]
        if not valid:
            return False
        # (3) the recovered pattern differs from the failed pattern
        failed_sig = (trace_before[failure.step].tool,
                      tuple(sorted(trace_before[failure.step].args.items()))) \
            if failure.step < len(trace_before) else None
        after_sigs = {(a.tool, tuple(sorted(a.args.items()))) for a in valid}
        if failed_sig is not None and after_sigs == {failed_sig}:
            return False
        return True


# ---------------------------------------------------------------------------
# Sentry: detect -> retrieve -> recover -> verify -> maybe store
# ---------------------------------------------------------------------------

class Sentry:
    """Failure-management layer that runs alongside the agent."""

    def __init__(self, playbook: Playbook,
                 detector: FailureDetector | None = None,
                 verifier: RecoveryVerifier | None = None):
        self.playbook = playbook
        self.detector = detector or FailureDetector()
        self.verifier = verifier or RecoveryVerifier(self.detector)
        self.stats = {"failures": 0, "retrievals": 0, "recovered": 0,
                      "stored": 0, "rejected": 0}

    def on_failure(self, failure: FailureEvent) -> list[Lesson]:
        """Conditionally expose: only lessons matching THIS failure."""
        self.stats["failures"] += 1
        lessons = self.playbook.retrieve(failure)
        self.stats["retrievals"] += 1
        return lessons

    def maybe_store(self, candidate: Lesson, recovered: bool) -> bool:
        """The store gate: keep the lesson only if recovery verified."""
        if recovered:
            self.playbook.add(candidate)
            self.stats["stored"] += 1
            return True
        self.stats["rejected"] += 1
        return False

    def record_recovery(self, lesson: Lesson) -> None:
        lesson.verified_recoveries += 1
        self.stats["recovered"] += 1
