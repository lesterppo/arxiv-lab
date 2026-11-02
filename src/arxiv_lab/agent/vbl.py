"""VBL: Verify--Backtrack Loop for long-horizon language agents.

Paper: arXiv 2610.41564 "Verify--Backtrack Loop: Structured
Precondition Recovery for Long-Horizon Language Agents" (submitted
30 Oct 2026).

One-line claim: long-horizon failure is mostly a recovery problem, not
a trajectory-length problem -- when a prerequisite is missing or a
decomposition fails, free-form retry of another plausible action rarely
helps. VBL, a training-free extension of ReAct, turns precondition
discovery and failure diagnosis into structured planning-state updates:
it keeps the task decomposition as a bounded goal stack and, on goal
activation, execution failure, or material state change, jointly
verifies two horizons -- whether the active goal contributes to the
terminal goal, and whether it is feasible in the current state. Those
judgments, combined with typed failure attribution, trigger explicit
operations: Push, Pop, retry, or BacktrackTo the parent.

This module implements that control state literally (stdlib only, no
model, no network): a goal stack, dual-horizon verification,
typed failure attribution, and the four operations. The "agent" and
"environment" are scripted callables supplied by the test, so the
recovery logic itself is what is measured.
"""

MAX_RETRIES = 2


class GoalSpec(object):
    """Static spec for one goal: subgoals, preconditions, providers."""

    def __init__(self, name, parent=None, subgoals=(), preconditions=(),
                 provider_for=None, action=None):
        self.name = name
        self.parent = parent
        self.subgoals = list(subgoals)
        self.preconditions = list(preconditions)
        # state key -> goal name whose completion supplies that key
        self.provider_for = dict(provider_for or {})
        self.action = action  # leaf action name, or None for composite


class GoalState(object):
    """Runtime state of one stacked goal."""

    def __init__(self, spec):
        self.spec = spec
        self.retries = 0
        self.done_subgoals = set()
        self.done = False


class VBL(object):
    """Verify--Backtrack Loop controller over a bounded goal stack."""

    def __init__(self, specs, terminal, state, max_stack=8,
                 max_retries=MAX_RETRIES):
        self.specs = specs  # name -> GoalSpec
        self.terminal = terminal
        self.state = dict(state)
        self.max_stack = max_stack
        self.max_retries = max_retries
        self.stack = [GoalState(specs[terminal])]
        self.trace = []  # (operation, goal, detail) records
        self.steps = 0

    # -- dual-horizon verification ---------------------------------------

    def _contributes(self, name):
        """True iff ``name`` is the terminal goal or inside its closure.

        Provider goals also contribute: a goal that supplies a
        precondition of a contributing goal contributes transitively,
        which is what lets a pushed recovery goal (e.g. "regather")
        survive verification even though it is not in the static
        decomposition.
        """
        if name == self.terminal:
            return True
        for spec in self.specs.values():
            if name in spec.provider_for.values():
                return True
        seen, frontier = set(), [self.terminal]
        while frontier:
            cur = frontier.pop()
            if cur in seen:
                continue
            seen.add(cur)
            frontier.extend(self.specs[cur].subgoals)
        return name in seen

    def verify(self, goal_state):
        """(contributes, feasible, missing) for the active goal."""
        spec = goal_state.spec
        missing = [k for k in spec.preconditions if not self.state.get(k)]
        return self._contributes(spec.name), not missing, missing

    # -- stack operations --------------------------------------------------

    def _push(self, name, reason):
        if len(self.stack) >= self.max_stack:
            return False
        self.stack.append(GoalState(self.specs[name]))
        self.trace.append(("Push", name, reason))
        return True

    def _pop(self, reason):
        top = self.stack.pop()
        self.trace.append(("Pop", top.spec.name, reason))
        return top

    def _backtrack_to_parent(self, reason):
        """Pop until the active goal's parent is on top (or stack empty)."""
        top = self.stack[-1]
        parent = top.spec.parent
        while self.stack and self.stack[-1].spec.name != parent:
            self._pop("BacktrackTo:%s:%s" % (parent, reason))
        if self.stack:
            self.trace.append(("BacktrackTo", parent, reason))
            return True
        return False

    # -- typed failure diagnosis -------------------------------------------

    def diagnose(self, goal_state, failure_type, missing):
        """Map a typed failure to an explicit stack operation."""
        spec = goal_state.spec
        if failure_type == "MISSING_PRECONDITION" and missing:
            key = missing[0]
            provider = spec.provider_for.get(key)
            if provider and provider in self.specs:
                if self._push(provider, "missing:%s" % key):
                    return "pushed-provider"
                return "backtracked-stack-full"
            if spec.parent and self._backtrack_to_parent("missing:%s" % key):
                return "backtracked"
        elif failure_type == "DECOMPOSITION_FAILED":
            if spec.parent and self._backtrack_to_parent("decomposition"):
                return "backtracked"
        # Generic failure: retry up to the limit, then give up on it.
        goal_state.retries += 1
        if goal_state.retries <= self.max_retries:
            self.trace.append(("retry", spec.name, failure_type))
            return "retried"
        self._pop("retry-limit:%s" % failure_type)
        return "popped-retry-limit"

    # -- main loop -----------------------------------------------------------

    def run(self, env, max_steps=60):
        """Run until the stack empties or ``max_steps`` is hit.

        ``env(action, state)`` returns ("ok", effects) or
        ("fail", failure_type, effects). Returns (success, steps).
        The terminal goal counts as success only if it is marked done
        (all its subgoals done) before the stack empties.
        """
        success = False
        while self.stack and self.steps < max_steps:
            self.steps += 1
            top = self.stack[-1]
            spec = top.spec
            contributes, feasible, missing = self.verify(top)
            if not contributes:
                self._pop("non-contributing")
                continue
            if not feasible:
                key = missing[0]
                provider = spec.provider_for.get(key)
                if provider and provider in self.specs:
                    if not self._push(provider, "verify-missing:%s" % key):
                        self._backtrack_to_parent("verify-stack-full")
                    continue
                if spec.parent:
                    self._backtrack_to_parent("verify-missing:%s" % key)
                    continue
                self._pop("verify-missing-no-provider")
                continue
            pending = [s for s in spec.subgoals
                       if s not in top.done_subgoals]
            if pending:
                self._push(pending[0], "decompose")
                continue
            if spec.action is None:
                # Composite goal with every subgoal done.
                top.done = True
                if spec.name == self.terminal:
                    success = True
                self._pop("completed")
                if self.stack:
                    self.stack[-1].done_subgoals.add(spec.name)
                continue
            # Leaf goal: execute.
            outcome = env(spec.action, self.state)
            if outcome[0] == "ok":
                self.state.update(outcome[1])
                top.done = True
                self._pop("completed")
                if self.stack:
                    self.stack[-1].done_subgoals.add(spec.name)
            else:
                _, failure_type, effects = outcome
                self.state.update(effects)
                # Re-diagnose missing preconditions from current state.
                _, _, missing_now = self.verify(top)
                result = self.diagnose(top, failure_type, missing_now)
                if result == "backtracked-stack-full":
                    self._backtrack_to_parent("stack-full")
        return success, self.steps


def react_baseline(env, actions, max_steps=60):
    """Free-form retry baseline: same actions in order, no stack revision.

    Retries the current action until it succeeds, then moves on; a
    persistently failing action burns the whole budget. Returns
    (success, steps).
    """
    steps = 0
    state = {}
    for action in actions:
        while steps < max_steps:
            steps += 1
            outcome = env(action, state)
            if outcome[0] == "ok":
                state.update(outcome[1])
                break
            # Material state changes happen even on failure (e.g. the
            # data precondition is lost) -- the baseline just cannot
            # act on them structurally.
            state.update(outcome[2])
        else:
            return False, steps
        if steps >= max_steps:
            return False, steps
    return True, steps
