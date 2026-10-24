"""Adaptive skill granularity for coding agents (arXiv:2610.31461).

Paper: "Adaptive Skill Granularity for Coding Agents: Runtime Routing and
Macro-to-Micro Recovery".

One-line claim: treating skill granularity as an action-abstraction policy
(conditioned on task structure, execution state, and macro-skill failures)
beats fixed-granularity execution — fixed macros are cheap when applicable
but brittle under mismatch, while runtime routing plus a finite-state
macro-to-micro recovery executor recovers controlled macro failures that
fixed execution leaves unresolved.

This module is the LLM-agnostic harness: a deterministic simulated
coding-agent environment (primitive tool catalog, macro skill library with
preconditions, generated tasks with verifier-defined success), a
feature-based granularity controller, a finite-state recovery executor,
the fixed-macro / fixed-micro baselines, and the evaluation protocol
(success rate, total cost, macro adoption, failure recovery, route match,
transition correctness). Stdlib only; deterministic given a seed.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

__all__ = [
    "PRIMITIVES",
    "Macro",
    "Task",
    "MACRO_LIBRARY",
    "GranularityController",
    "RecoveryExecutor",
    "ExecutionTrace",
    "StrategyResult",
    "run_protocol",
    "generate_tasks",
    "oracle_route",
    "ORACLE_PLANS",
]

# ---------------------------------------------------------------------------
# Environment: primitives, macros, tasks
# ---------------------------------------------------------------------------

PRIMITIVES: Tuple[str, ...] = (
    "read_spec",
    "search_code",
    "write_impl",
    "run_tests",
    "commit",
    "install_dep",
    "configure",
    "lint",
    "migrate_schema",
    "write_docs",
)


@dataclass(frozen=True)
class Macro:
    """A reusable macro skill: named procedure with preconditions.

    Executing a macro as one unit costs ``overhead`` tool-call equivalents
    (the efficiency gain over issuing every primitive). If a hidden
    precondition is violated the macro fails deterministically at
    ``fail_step`` (index into ``plan``); a latent execution error fails at
    ``flaky_step`` instead.
    """

    name: str
    requires: Dict[str, object]
    plan: Tuple[str, ...]
    overhead: int
    fail_step: int = -1  # -1: no precondition-mismatch failure
    flaky_step: int = -1  # -1: no latent execution failure


MACRO_LIBRARY: Tuple[Macro, ...] = (
    Macro(
        name="add_feature",
        requires={"kind": "add_feature", "has_spec": True},
        plan=("read_spec", "write_impl", "run_tests", "commit"),
        overhead=2,
    ),
    Macro(
        name="fix_bug",
        requires={"kind": "fix_bug"},
        plan=("search_code", "write_impl", "run_tests", "commit"),
        overhead=2,
    ),
    Macro(
        name="refactor",
        requires={"kind": "refactor", "test_suite": True},
        plan=("read_spec", "write_impl", "lint", "run_tests", "commit"),
        overhead=3,
    ),
    Macro(
        name="migrate",
        requires={"kind": "migrate_config", "migration": True},
        plan=("read_spec", "migrate_schema", "configure", "run_tests"),
        overhead=2,
    ),
    Macro(
        name="document",
        requires={"kind": "write_docs"},
        plan=("search_code", "write_docs", "commit"),
        overhead=2,
    ),
)

# Oracle primitive plans per task kind (the verifier's ground truth).
ORACLE_PLANS: Dict[str, Tuple[str, ...]] = {
    "add_feature": ("read_spec", "write_impl", "run_tests", "commit"),
    "fix_bug": ("search_code", "write_impl", "run_tests", "commit"),
    "refactor": ("read_spec", "write_impl", "lint", "run_tests", "commit"),
    "migrate_config": ("read_spec", "migrate_schema", "configure", "run_tests"),
    "write_docs": ("search_code", "write_docs", "commit"),
}


@dataclass
class Task:
    """A deterministic coding task.

    ``features`` is what the controller sees. ``oracle`` is the verifier's
    required primitive sequence. ``latent_failure`` names a macro step index
    where a *controlled* execution failure occurs even though all
    preconditions hold (None = no latent failure).
    """

    tid: int
    kind: str
    features: Dict[str, object]
    oracle: Tuple[str, ...]
    latent_failure: Optional[Tuple[str, int]] = None  # (macro_name, step_idx)

    @property
    def complexity(self) -> int:
        return len(self.oracle)


def generate_tasks(
    n: int,
    seed: int = 0,
    mismatch_frac: float = 0.35,
    latent_fail_frac: float = 0.25,
) -> List[Task]:
    """Generate a deterministic task suite.

    ``mismatch_frac`` of tasks violate some macro precondition (fixed-macro
    brittle class). ``latent_fail_frac`` of the remaining applicable tasks
    carry a controlled latent execution failure at a random macro step (the
    recovery class).
    """
    rng = random.Random(seed)
    kinds = list(ORACLE_PLANS)
    tasks: List[Task] = []
    for tid in range(n):
        kind = kinds[tid % len(kinds)]
        features: Dict[str, object] = {"kind": kind}
        mismatch = rng.random() < mismatch_frac
        if kind in ("add_feature",):
            features["has_spec"] = not mismatch
        if kind in ("refactor",):
            features["test_suite"] = not mismatch
        if kind in ("migrate_config",):
            features["migration"] = not mismatch
        oracle = ORACLE_PLANS[kind]
        latent_failure = None
        if not mismatch and rng.random() < latent_fail_frac:
            macro, match = _best_macro({"kind": kind, **features})
            # latent failures only make sense on fully applicable macros
            if macro is not None and match == 1.0:
                latent_failure = (macro.name, rng.randrange(len(macro.plan)))
        tasks.append(
            Task(
                tid=tid,
                kind=kind,
                features=features,
                oracle=oracle,
                latent_failure=latent_failure,
            )
        )
    return tasks


def _match_score(macro: Macro, features: Dict[str, object]) -> float:
    if not macro.requires:
        return 1.0
    hits = sum(1 for k, v in macro.requires.items() if features.get(k) == v)
    return hits / len(macro.requires)


def _best_macro(
    features: Dict[str, object],
) -> Tuple[Optional[Macro], float]:
    best, best_score = None, -1.0
    for macro in MACRO_LIBRARY:
        s = _match_score(macro, features)
        if s > best_score:
            best, best_score = macro, s
    return best, best_score


# ---------------------------------------------------------------------------
# Feature-based granularity controller
# ---------------------------------------------------------------------------


@dataclass
class GranularityController:
    """Routes each task to macro / micro execution from features.

    Policy features: best macro's precondition match score (task structure),
    task complexity, and the macro's observed failure statistics (execution
    state). The controller keeps a per-macro Laplace-smoothed failure rate
    and a running mean of recovery extra-cost, and routes "macro" iff its
    expected cost (overhead + P(fail) * mean recovery extra-cost) beats the
    micro cost — i.e. the abstraction policy is explicitly conditioned on
    macro-skill failures. Rare, cheaply-recovered failures do not veto the
    macro; frequent or expensive ones do.
    """

    complexity_threshold: int = 8
    prior_mean_extra_cost: float = 2.0
    min_attempts_before_veto: int = 8
    attempts: Dict[str, int] = field(default_factory=dict)
    failures: Dict[str, int] = field(default_factory=dict)
    recovery_extra_cost: Dict[str, float] = field(default_factory=dict)

    def expected_macro_cost(self, macro: Macro) -> float:
        a = self.attempts.get(macro.name, 0)
        f = self.failures.get(macro.name, 0)
        p_fail = (f + 0.5) / (a + 1.0)  # Laplace-smoothed
        mean_extra = self.recovery_extra_cost.get(macro.name, self.prior_mean_extra_cost)
        return macro.overhead + p_fail * mean_extra

    def observe(self, macro_name: str, failed: bool, extra_cost: float) -> None:
        self.attempts[macro_name] = self.attempts.get(macro_name, 0) + 1
        if failed:
            n = self.failures.get(macro_name, 0) + 1
            self.failures[macro_name] = n
            prev = self.recovery_extra_cost.get(macro_name, self.prior_mean_extra_cost)
            self.recovery_extra_cost[macro_name] = prev + (extra_cost - prev) / n

    def decide(self, task: Task) -> Tuple[str, Optional[Macro]]:
        macro, match = _best_macro(task.features)
        if (
            macro is None
            or match < 1.0
            or task.complexity > self.complexity_threshold
        ):
            return "micro", None
        if self.expected_macro_cost(macro) <= float(task.complexity):
            return "macro", macro
        # no veto on thin evidence: keep exploring until min attempts
        if self.attempts.get(macro.name, 0) < self.min_attempts_before_veto:
            return "macro", macro
        return "micro", None


def oracle_route(task: Task) -> Tuple[str, Optional[Macro]]:
    """Cheapest successful route (knows latent failures): macro only when it
    is both fully applicable and failure-free; otherwise micro."""
    macro, match = _best_macro(task.features)
    if macro is None or match < 1.0:
        return "micro", None
    if task.latent_failure is not None and task.latent_failure[0] == macro.name:
        return "micro", None
    return "macro", macro


# ---------------------------------------------------------------------------
# Finite-state recovery executor
# ---------------------------------------------------------------------------


@dataclass
class ExecutionTrace:
    task_id: int
    route: str  # "macro" | "micro"
    transitions: List[str] = field(default_factory=list)
    primitives: List[str] = field(default_factory=list)
    cost: int = 0
    success: bool = False
    recovered: bool = False
    recovery_entered_without_failure: bool = False

    def step(self, state: str) -> None:
        self.transitions.append(state)


# Legal FSM transitions (transition correctness is measured against these).
_LEGAL: Dict[str, Tuple[str, ...]] = {
    "START": ("MACRO_RUNNING", "MICRO_RUNNING"),
    "MACRO_RUNNING": ("VERIFY", "MACRO_FAILED"),
    "MICRO_RUNNING": ("VERIFY",),
    "MACRO_FAILED": ("DIAGNOSE",),
    "DIAGNOSE": ("MICRO_RECOVERY", "FAIL"),
    "MICRO_RECOVERY": ("VERIFY", "FAIL"),
    "VERIFY": ("DONE", "FAIL"),
    "DONE": (),
    "FAIL": (),
}


@dataclass
class RecoveryExecutor:
    """Finite-state executor with macro-to-micro recovery.

    On a macro failure the FSM diagnoses the failure type: precondition
    mismatches are unrecoverable (the route was wrong), latent execution
    failures are recoverable by replaying the remaining steps as primitives,
    subject to ``recovery_budget`` extra primitive calls.
    """

    recovery_budget: int = 8

    def run(
        self,
        task: Task,
        route: str,
        macro: Optional[Macro],
        controller: Optional[GranularityController] = None,
    ) -> ExecutionTrace:
        trace = ExecutionTrace(task_id=task.tid, route=route)
        trace.step("START")

        if route == "micro" or macro is None:
            trace.step("MICRO_RUNNING")
            for prim in task.oracle:
                trace.primitives.append(prim)
                trace.cost += 1
            trace.step("VERIFY")
            trace.success = _verify(trace.primitives, task.oracle)
            trace.step("DONE" if trace.success else "FAIL")
            return trace

        # --- macro route ---
        trace.step("MACRO_RUNNING")
        trace.cost += macro.overhead
        fail_at = self._failure_step(task, macro)
        if fail_at is None:
            trace.primitives.extend(macro.plan)
            trace.step("VERIFY")
            trace.success = _verify(trace.primitives, task.oracle)
            trace.step("DONE" if trace.success else "FAIL")
            if controller is not None:
                controller.observe(macro.name, failed=False, extra_cost=0.0)
            return trace

        # --- macro failed: diagnose + recover ---
        trace.step("MACRO_FAILED")
        trace.step("DIAGNOSE")
        failure_type = self._failure_type(task, macro)
        if failure_type == "precondition_mismatch":
            # Wrong route, nothing to salvage: unrecoverable by design.
            trace.step("FAIL")
            if controller is not None:
                controller.observe(
                    macro.name, failed=True, extra_cost=1e6
                )
            return trace
        trace.step("MICRO_RECOVERY")
        remaining = list(macro.plan[fail_at:])
        if len(remaining) > self.recovery_budget:
            trace.step("FAIL")
            if controller is not None:
                controller.observe(
                    macro.name, failed=True, extra_cost=1e6
                )
            return trace
        trace.primitives.extend(macro.plan[:fail_at])
        for prim in remaining:
            trace.primitives.append(prim)
            trace.cost += 1
        trace.recovered = True
        trace.step("VERIFY")
        trace.success = _verify(trace.primitives, task.oracle)
        trace.step("DONE" if trace.success else "FAIL")
        if controller is not None:
            controller.observe(
                macro.name,
                failed=True,
                extra_cost=float(max(0, trace.cost - macro.overhead)),
            )
        return trace

    @staticmethod
    def _failure_step(task: Task, macro: Macro) -> Optional[int]:
        # Precondition mismatch: fail at the first step (controlled, hidden).
        _, match = _best_macro(task.features)
        if match < 1.0:
            return 0
        # Latent execution failure at the injected step.
        if task.latent_failure is not None and task.latent_failure[0] == macro.name:
            return task.latent_failure[1]
        return None

    @staticmethod
    def _failure_type(task: Task, macro: Macro) -> str:
        _, match = _best_macro(task.features)
        if match < 1.0:
            return "precondition_mismatch"
        return "execution_error"


def _verify(executed: Sequence[str], oracle: Sequence[str]) -> bool:
    """Verifier-defined success: oracle primitives appear in order."""
    it = iter(executed)
    return all(any(p == want for p in it) for want in oracle)


# ---------------------------------------------------------------------------
# Evaluation protocol
# ---------------------------------------------------------------------------


@dataclass
class StrategyResult:
    name: str
    traces: List[ExecutionTrace]
    success_rate: float = 0.0
    mean_cost: float = 0.0
    mean_cost_solved: float = 0.0
    recovery_rate: float = 0.0
    route_match: float = 0.0
    transition_correct: float = 0.0
    macro_adoption: float = 0.0

    def finalize(self, tasks: List[Task]) -> None:
        n = len(self.traces)
        self.success_rate = sum(t.success for t in self.traces) / n
        self.mean_cost = sum(t.cost for t in self.traces) / n
        solved = [t.cost for t in self.traces if t.success]
        self.mean_cost_solved = sum(solved) / len(solved) if solved else float("inf")
        failed_macro = [t for t in self.traces if t.route == "macro" and self._macro_failed(t)]
        self.recovery_rate = (
            sum(t.recovered and t.success for t in failed_macro) / len(failed_macro)
            if failed_macro
            else 1.0
        )
        self.macro_adoption = sum(t.route == "macro" for t in self.traces) / n
        # route match is measured on tasks with an unambiguous best route
        # (latent-failure tasks admit a genuine macro-with-recovery vs micro
        # trade-off, so the oracle's foreknowledge is not a fair target).
        unambiguous = [
            (t, o) for t, o in zip(self.traces, tasks) if o.latent_failure is None
        ]
        oracle_routes = [oracle_route(o)[0] for _, o in unambiguous]
        self.route_match = (
            sum(t.route == r for (t, _), r in zip(unambiguous, oracle_routes))
            / len(unambiguous)
            if unambiguous
            else 1.0
        )
        self.transition_correct = sum(
            self._transitions_legal(t) for t in self.traces
        ) / n

    @staticmethod
    def _macro_failed(trace: ExecutionTrace) -> bool:
        return "MACRO_FAILED" in trace.transitions

    @staticmethod
    def _transitions_legal(trace: ExecutionTrace) -> bool:
        tr = trace.transitions
        for a, b in zip(tr, tr[1:]):
            if b not in _LEGAL.get(a, ()):
                return False
        # terminal states must be last
        return tr[-1] in ("DONE", "FAIL")


def run_protocol(
    tasks: Sequence[Task],
    strategy: str,
    seed: int = 0,
    recovery_budget: int = 8,
) -> StrategyResult:
    """Run the deterministic coding-agent protocol under one strategy.

    Strategies: "fixed_macro" (always the best-partial macro, no recovery),
    "fixed_micro" (always primitives), "adaptive" (feature-based controller +
    finite-state recovery executor).
    """
    executor = RecoveryExecutor(recovery_budget=recovery_budget)
    traces: List[ExecutionTrace] = []
    for task in tasks:
        if strategy == "fixed_macro":
            macro, _ = _best_macro(task.features)
            route, m = ("macro", macro) if macro else ("micro", None)
            # fixed-macro has no recovery: cap budget at 0 via plain executor
            # with a zero-budget twin, then strip the recovery attempt.
            trace = RecoveryExecutor(recovery_budget=0).run(task, route, m)
            traces.append(trace)
        elif strategy == "fixed_micro":
            traces.append(executor.run(task, "micro", None))
        elif strategy == "adaptive":
            controller = _ADAPTIVE_CONTROLLERS.setdefault(seed, GranularityController())
            route, macro = controller.decide(task)
            traces.append(executor.run(task, route, macro, controller))
        else:
            raise ValueError(f"unknown strategy {strategy!r}")
    result = StrategyResult(name=strategy, traces=traces)
    result.finalize(list(tasks))
    return result


_ADAPTIVE_CONTROLLERS: Dict[int, GranularityController] = {}
