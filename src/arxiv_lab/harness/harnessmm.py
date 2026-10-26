"""HarnessMM: guarded decision reuse for harness token efficiency.

arXiv:2610.32813 — "HarnessMM: Recursive Self-Improvement for Harness
Token Efficiency". Claim implemented here: a controller at the model-tool
boundary reuses previously resolved decisions to replace later reasoning
callbacks while actor weights stay fixed. Executable Decision Graphs store
exact actions, literal arguments, causal dependencies, and public guards;
Guarded Decision Reuse activates a persistent decision only when the
observed receipt matches on all of them, and Certified Callback
Substitution executes it through the native interface. New semantic
decisions, unsupported evidence, and selector abstention fall back to the
original actor. Recursive self-improvement is modelled as training-phase
certification plus validation-phase selection between guard configs, with
the winning config deployed on held-out tasks.

Pure stdlib. Deterministic given a seed.
"""

import random

# Cost model (tokens per step): an actor reasoning callback vs a certified
# substitution (receipt check + native execution). The absolute numbers are
# a scale model, not a claim about any real model; only the *ratios* between
# conditions are asserted.
ACTOR_COST = 100
REUSE_COST = 5

ACTIONS = ("lookup", "compute", "format", "validate", "send", "fetch")


class Decision:
    """An executable decision: exact action + literal args + dependencies."""

    __slots__ = ("action", "args", "deps")

    def __init__(self, action, args, deps):
        self.action = action
        self.args = tuple(args)
        self.deps = tuple(deps)

    def key(self):
        return (self.action, self.args, self.deps)

    def surface_key(self):
        # What a naive reuser matches on: action + identifier argument
        # only, ignoring payload values and dependencies. A novel decision
        # carries an unseen identifier, so it correctly misses; a trap
        # reuses its base decision's identifier, so it wrongly hits.
        ident = tuple(v for k, v in self.args if k == "id")
        return (self.action, ident)

    def __repr__(self):
        return "Decision(%r, %r, %r)" % (self.action, self.args, self.deps)


class Step:
    """One required step of a task, with the receipt evidence presented."""

    __slots__ = ("required", "receipt", "kind")

    def __init__(self, required, receipt, kind):
        self.required = required  # Decision the task actually needs
        self.receipt = receipt  # Decision the receipt claims, or None
        self.kind = kind  # 'routine' | 'novel' | 'trap_arg' | 'trap_dep' | 'no_receipt'


class DecisionGraph:
    """Persistent Executable Decision Graph: certified decisions only."""

    def __init__(self):
        self._store = {}

    def certify(self, decision):
        self._store[decision.key()] = decision

    def __len__(self):
        return len(self._store)

    def guarded_lookup(self, step):
        """Return the certified decision iff the receipt matches exactly.

        Guards: a receipt must be present, and its action, literal args,
        and dependencies must all equal the required decision. Anything
        else is abstention -> the caller must invoke the actor.
        """
        if step.receipt is None:
            return None
        if step.receipt.key() != step.required.key():
            return None
        return self._store.get(step.required.key())

    def naive_lookup(self, step):
        """Surface-match lookup: action + argument names only."""
        if step.receipt is None:
            return None
        for decision in self._store.values():
            if decision.surface_key() == step.receipt.surface_key():
                return decision
        return None


def _routine_pool(rng, size=20):
    pool = []
    for i in range(size):
        action = ACTIONS[i % len(ACTIONS)]
        pool.append(Decision(action, (("id", i), ("v", i * 3 + 1)), ("dep%d" % (i % 5),)))
    return pool


def _novel_decision(rng, tag):
    return Decision(
        rng.choice(ACTIONS),
        (("id", tag), ("v", rng.randrange(1000))),
        ("depX%d" % rng.randrange(50),),
    )


def generate_tasks(rng, n_tasks, routine_pool):
    """Each task: 4 routine steps, 2 novel steps, 0-2 trap steps.

    Routine steps present a receipt identical to the required decision
    (reusable). Trap steps present a receipt that is surface-similar but
    guard-invalid: changed literal arg value, changed dependency, or no
    receipt at all. Tasks sample routine decisions without replacement, so
    a single-use (per-task) graph can never reuse anything.
    """
    tasks = []
    for _ in range(n_tasks):
        steps = []
        for decision in rng.sample(routine_pool, 4):
            steps.append(Step(decision, decision, "routine"))
        for j in range(2):
            novel = _novel_decision(rng, rng.randrange(10 ** 6))
            steps.append(Step(novel, novel, "novel"))
        n_traps = rng.choice((0, 1, 2))
        for _ in range(n_traps):
            base = rng.choice(routine_pool)
            kind = rng.choice(("trap_arg", "trap_dep", "no_receipt"))
            # The trap's REQUIRED decision differs from the stored one,
            # while the receipt claims the stored one: a guarded lookup
            # must abstain (receipt != required), but a surface matcher
            # that keys on action + identifier wrongly reuses the stale
            # decision. The no_receipt trap needs the stored decision but
            # presents no evidence, so reuse is unsupported either way.
            if kind == "trap_arg":
                required = Decision(
                    base.action,
                    tuple((k, v + 1000) if k == "v" else (k, v) for k, v in base.args),
                    base.deps,
                )
                receipt = base
            elif kind == "trap_dep":
                required = Decision(base.action, base.args, ("depOTHER",))
                receipt = base
            else:
                required = Decision(base.action, base.args, base.deps)
                receipt = None
            steps.append(Step(required, receipt, kind))
        rng.shuffle(steps)
        tasks.append(steps)
    return tasks


def run_condition(tasks, graph, mode):
    """Run one controller over tasks.

    mode: 'actor' | 'guarded' | 'naive' | 'single_use'.
    The actor is always correct; a reused decision is correct iff it
    equals the required decision. Returns a metrics dict.
    """
    n_steps = 0
    n_success_tasks = 0
    cost = 0
    reused = 0
    fallback_on_trap = 0
    n_traps = 0
    false_reuse = 0
    for steps in tasks:
        local = DecisionGraph()
        if mode == "single_use":
            # Certify this task's routine decisions only, then clear them
            # by construction: the local graph starts empty per task, and
            # routine decisions are certified as they are *acted* — but a
            # step is only reusable if an identical step already occurred
            # within this same task. Since tasks sample routine decisions
            # without replacement, that never happens.
            pass
        task_ok = True
        for step in steps:
            n_steps += 1
            if step.kind in ("trap_arg", "trap_dep", "no_receipt"):
                n_traps += 1
            if mode == "actor":
                cost += ACTOR_COST
                if step.kind in ("trap_arg", "trap_dep", "no_receipt"):
                    # The actor path never reuses a trap decision.
                    fallback_on_trap += 1
                continue
            if mode == "single_use":
                hit = local.guarded_lookup(step)
                if hit is not None and hit.key() == step.required.key():
                    cost += REUSE_COST
                    reused += 1
                else:
                    cost += ACTOR_COST
                    if step.kind in ("trap_arg", "trap_dep", "no_receipt"):
                        fallback_on_trap += 1
                    local.certify(step.required)
                continue
            hit = graph.guarded_lookup(step) if mode == "guarded" else graph.naive_lookup(step)
            if hit is not None:
                cost += REUSE_COST
                reused += 1
                if hit.key() != step.required.key():
                    task_ok = False
                    false_reuse += 1
            else:
                cost += ACTOR_COST
                if step.kind in ("trap_arg", "trap_dep", "no_receipt"):
                    fallback_on_trap += 1
        if task_ok:
            n_success_tasks += 1
    n_tasks = len(tasks)
    return {
        "tasks": n_tasks,
        "steps": n_steps,
        "success": n_success_tasks / n_tasks if n_tasks else 0.0,
        "cost_per_task": cost / n_tasks if n_tasks else 0.0,
        "reused_steps": reused,
        "trap_fallback_rate": (fallback_on_trap / n_traps) if n_traps else 1.0,
        "false_reuse": false_reuse,
        "n_traps": n_traps,
    }


def run_experiment(seed=20261026, n_train=200, n_valid=100, n_test=400):
    """Full HarnessMM pipeline: certify -> validate configs -> deploy.

    Training phase certifies the routine pool by acting every step.
    Validation selects between the strict guard config (guarded) and the
    loose config (naive surface match) on validation tasks: highest
    success wins, ties broken by lowest cost. The winner is deployed on
    held-out tasks, alongside the actor-only and single-use baselines.
    """
    rng = random.Random(seed)
    pool = _routine_pool(rng)
    graph = DecisionGraph()
    for decision in pool:
        graph.certify(decision)
    # Training phase: acting certifies nothing new here (pool is the
    # certified routine set); the phase's role is that certification
    # happens *before* deployment, from training feedback.
    _train_tasks = generate_tasks(rng, n_train, pool)
    valid_tasks = generate_tasks(rng, n_valid, pool)
    test_tasks = generate_tasks(rng, n_test, pool)

    valid = {
        "guarded": run_condition(valid_tasks, graph, "guarded"),
        "naive": run_condition(valid_tasks, graph, "naive"),
    }
    # RSI validation selection: success first, cost second.
    winner = max(valid, key=lambda m: (valid[m]["success"], -valid[m]["cost_per_task"]))
    deployed = run_condition(test_tasks, graph, winner)
    return {
        "seed": seed,
        "graph_size": len(graph),
        "validation": valid,
        "selected_config": winner,
        "test": {
            "actor": run_condition(test_tasks, graph, "actor"),
            "guarded": run_condition(test_tasks, graph, "guarded"),
            "naive": run_condition(test_tasks, graph, "naive"),
            "single_use": run_condition(test_tasks, graph, "single_use"),
        },
        "deployed": deployed,
        "deployed_config": winner,
    }


if __name__ == "__main__":
    import json

    result = run_experiment()
    test = result["test"]
    print("HarnessMM (seed %s): validation picks %s" % (result["seed"], result["selected_config"]))
    for name, metrics in test.items():
        print(
            "  %-10s success=%.3f cost/task=%.1f trap_fallback=%.3f false_reuse=%d"
            % (name, metrics["success"], metrics["cost_per_task"],
               metrics["trap_fallback_rate"], metrics["false_reuse"])
        )
    print(json.dumps(result, indent=1, default=str)[:200])
