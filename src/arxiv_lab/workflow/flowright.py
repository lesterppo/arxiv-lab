"""FloWright-style hierarchical credit assignment and role co-evolution
(arXiv:2610.01026).

One-line claim: in a multi-role workflow judged by a single sparse outcome,
attributing hierarchical, structure-aware credit per role — computed from
workflow artifacts alone (no extra models, labels, or executions) — and
co-evolving two or more roles beats evolving a single role, which beats a
flat shared-reward baseline with no credit assignment.

Toy world (CPU, no LLM): tasks are small arithmetic expressions. Three roles
with scalar skills in [0, 1]:

- generator: drafts an answer; each operation is evaluated correctly with
  probability ``g`` (a slip adds a small nonzero offset);
- verifier: independently re-evaluates with per-op accuracy ``v`` and accepts
  the draft iff its own estimate matches it;
- refiner: on rejection, re-evaluates with per-op accuracy ``r``; its answer
  is final.

The sparse outcome is 1 iff the final answer equals the true value.

Structured per-role credit (artifacts + sparse outcome only):

- generator: downstream acceptance rate (did the verifier accept the draft?);
- verifier: 1 on accept iff the final answer is correct; on reject, 1 iff the
  refinement changed the answer *and* the final answer is correct
  (vindicated rejection), else 0;
- refiner: correctness rate on the tasks it actually handled (None when the
  verifier accepted everything — no signal, keep incumbent).

Evolution is (1+1)-ES per evolving role, round-robin: propose
``skill + N(0, sigma)`` (clipped), keep iff the role's own objective does not
decrease. ``condition="flat"`` evolves all roles on the shared sparse reward
(no credit assignment); ``"single"`` evolves only the generator on its
structured credit; ``"coev"`` evolves all roles on their structured credits.

Pure stdlib. Deterministic given explicit ``random.Random`` instances /
integer seeds.
"""

import random

GENERATOR = "generator"
VERIFIER = "verifier"
REFINER = "refiner"
ROLES = (GENERATOR, VERIFIER, REFINER)

CONDITIONS = ("flat", "single", "coev")

OPS = ("+", "-", "*")
SKILL_LO = 0.05
SKILL_HI = 0.99


class Task:
    """A small arithmetic expression: left-associative chain of binary ops."""

    __slots__ = ("ops", "operands", "value", "n_ops")

    def __init__(self, ops, operands):
        self.ops = tuple(ops)
        self.operands = tuple(operands)
        self.n_ops = len(self.ops)
        v = self.operands[0]
        for op, b in zip(self.ops, self.operands[1:]):
            v = _apply(v, op, b)
        self.value = v

    def __repr__(self):
        return "Task(%r)" % (self.describe(),)

    def describe(self):
        s = str(self.operands[0])
        for op, b in zip(self.ops, self.operands[1:]):
            s = "(%s %s %s)" % (s, op, b)
        return "%s = %d" % (s, self.value)


def _apply(a, op, b):
    if op == "+":
        return a + b
    if op == "-":
        return a - b
    return a * b


def gen_task(rng, n_ops):
    """Generate a random task with ``n_ops`` binary operations."""
    operands = [rng.randint(2, 9) for _ in range(n_ops + 1)]
    ops = [rng.choice(OPS) for _ in range(n_ops)]
    return Task(ops, operands)


def skilled_eval(task, skill, rng):
    """Evaluate the task's expression, slipping per-op with prob 1-skill.

    A slip replaces the correct intermediate value with a nearby wrong one
    (nonzero offset in [-5, 5]).
    """
    v = task.operands[0]
    for op, b in zip(task.ops, task.operands[1:]):
        correct = _apply(v, op, b)
        if rng.random() < skill:
            v = correct
        else:
            offset = 0
            while offset == 0:
                offset = rng.randint(-5, 5)
            v = correct + offset
    return v


def run_workflow(task, skills, rng):
    """Run the generator -> verifier -> (refiner) workflow on one task.

    Returns a trace dict of workflow artifacts plus the sparse reward.
    """
    draft = skilled_eval(task, skills[GENERATOR], rng)
    vest = skilled_eval(task, skills[VERIFIER], rng)
    accepted = vest == draft
    if accepted:
        final = draft
        refiner_handled = False
    else:
        final = skilled_eval(task, skills[REFINER], rng)
        refiner_handled = True
    return {
        "draft": draft,
        "vest": vest,
        "accepted": accepted,
        "final": final,
        "refiner_handled": refiner_handled,
        "reward": 1.0 if final == task.value else 0.0,
    }


def structured_credits(trace):
    """Per-role credit from workflow artifacts + sparse outcome (no labels).

    Returns {role: credit in [0,1] or None}. The refiner credit is None when
    it handled nothing (no signal).
    """
    credits = {}
    credits[GENERATOR] = 1.0 if trace["accepted"] else 0.0
    if trace["accepted"]:
        credits[VERIFIER] = 1.0 if trace["reward"] == 1.0 else 0.0
    else:
        vindicated = trace["reward"] == 1.0 and trace["final"] != trace["draft"]
        credits[VERIFIER] = 1.0 if vindicated else 0.0
    if trace["refiner_handled"]:
        credits[REFINER] = 1.0 if trace["reward"] == 1.0 else 0.0
    else:
        credits[REFINER] = None
    return credits


def role_objective(traces, role, kind):
    """Mean objective for ``role`` over traces; ``kind`` is 'flat'/'structured'.

    Returns None when the role saw no signal (refiner never handled a task).
    """
    if kind == "flat":
        return sum(t["reward"] for t in traces) / len(traces)
    vals = [c for c in (structured_credits(t)[role] for t in traces)
            if c is not None]
    if not vals:
        return None
    return sum(vals) / len(vals)


def _clip(skill):
    return max(SKILL_LO, min(SKILL_HI, skill))


def evolve(train_tasks, condition="coev", generations=60, sigma=0.07,
           seed=0, init_skill=0.5):
    """Co-evolve role skills with (1+1)-ES, round-robin over evolving roles.

    ``condition``: 'flat' (all roles, shared sparse reward), 'single'
    (generator only, structured credit), 'coev' (all roles, structured
    credits). Returns {"skills": {role: skill}, "train_acc": [per-gen joint
    train accuracy]}.
    """
    if condition not in CONDITIONS:
        raise ValueError("unknown condition: %r" % (condition,))
    rng = random.Random(seed)
    kind = "flat" if condition == "flat" else "structured"
    evolving = ROLES if condition in ("flat", "coev") else (GENERATOR,)
    skills = {role: init_skill for role in ROLES}
    train_acc = []

    # Common random numbers: task i is always evaluated with the same noise
    # realization, so incumbent-vs-mutant comparisons are fair (no selection
    # on lucky slips) and the recorded train objective is monotone for the
    # flat condition. Generalization is still measured honestly: the held-out
    # test set uses fresh tasks and fresh noise via accuracy().
    task_seeds = [seed * 100003 + 17 + i for i in range(len(train_tasks))]

    def evaluate(sk):
        return [run_workflow(t, sk, random.Random(ts))
                for t, ts in zip(train_tasks, task_seeds)]

    for _ in range(generations):
        for role in evolving:
            cur_traces = evaluate(skills)
            cur_obj = role_objective(cur_traces, role, kind)
            prop = dict(skills)
            prop[role] = _clip(skills[role] + rng.gauss(0.0, sigma))
            prop_traces = evaluate(prop)
            prop_obj = role_objective(prop_traces, role, kind)
            if prop_obj is not None and cur_obj is not None \
                    and prop_obj >= cur_obj:
                skills = prop
        train_acc.append(
            sum(t["reward"] for t in evaluate(skills)) / len(train_tasks))
    return {"skills": skills, "train_acc": train_acc}


def accuracy(skills, tasks, seed=0):
    """Mean sparse reward of the workflow on ``tasks``."""
    rng = random.Random(seed)
    return sum(run_workflow(t, skills, rng)["reward"] for t in tasks) / len(tasks)


def accuracy_by_nops(skills, tasks, seed=0):
    """Mean sparse reward grouped by task chain length."""
    rng = random.Random(seed)
    groups = {}
    for t in tasks:
        groups.setdefault(t.n_ops, []).append(
            run_workflow(t, skills, rng)["reward"])
    return {k: sum(v) / len(v) for k, v in sorted(groups.items())}
