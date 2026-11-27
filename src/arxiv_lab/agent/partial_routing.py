"""Partial Routing: Trace-Tuned Delegation for Resource-Efficient
Agentic Workflows.

arXiv:2611.33402 "Partial Routing: Trace-Tuned Delegation for
Resource-Efficient Agentic Workflows".

Core claim implemented here: an agentic workflow can be split into
subtasks and routed between two models -- an expensive high-quality
LEAD and a cheap lower-quality WORKER -- at subtask granularity. From
both models' past runs, recurring subtasks are measured per subtask
type (quality and cost); those measurements produce delegation
guidance, and the lead decides during execution when and what to
delegate, WITHOUT training either model. A parameter delta controls
how much is delegated. On held-out ViBench tasks the paper reports
partial routing raising the lead's score 70.2 -> 78.5 at 19% lower
cost (delta=0.75), and on Terminal-Bench a pass rate 54% -> 61% at
similar cost; it beats the lead alone AND task-level routing (one
model for the whole task). An ablation finds that giving the lead
only the subtask breakdown, without a worker, also reduces cost and
raises the Terminal-Bench pass rate.

Bounded CPU reproduction (no model, no network; deterministic given
a seed):

* World: a task is 5 subtasks of known types (plan, code, verify,
  test, summarize), each with a difficulty in [0, 1]. Each model has
  a per-type base success probability (lead high everywhere; worker
  cheaper and nearly as good on routine types, clearly worse on hard
  types) and a per-subtask cost (lead 10, worker 1, plus a 0.5 lead
  coordination cost per delegated subtask).
* Lead load penalty (the mechanism by which delegation can RAISE
  success, not just cut cost): the lead's success probability on a
  subtask falls by LOAD_PENALTY for every subtask it has already
  handled in the task -- long in-task context dilutes the lead.
  The worker is stateless per call and takes no load penalty.
  Delegating routine subtasks keeps the lead fresh for hard ones.
* Past-run corpus: both models' past runs are per-subtask attempts
  (each attempt is a fresh model call, as in the paper's per-type
  measurement), so the measured per-type quality is the model's
  quality on that subtask type. The lead load penalty is an
  IN-TASK execution effect -- it appears only in live runs, where
  the lead handles several subtasks in one session -- and does not
  contaminate the past-run measurements.
* Delegation rule (delta): delegate a subtask of type t iff
  gap[t] <= delta * gap_max. delta=0 delegates (almost) nothing;
  delta=1 delegates everything. With the measured gaps in this
  world (plan 0.30, code 0.25, verify 0.19, test 0.12, summarize
  0.06), delta=0.25 delegates summarize only, 0.5 adds test, 0.75
  adds verify, 1.0 adds code and plan.
* Task success (strict, Terminal-Bench style): ALL subtasks must
  succeed. Mean subtask score (fraction succeeded) is reported
  alongside.
* Baselines: lead-alone, task-level routing (whole task to the
  worker if its hard-subtask fraction is <= 0.4, else to the lead),
  random delegation (seeded), and a fixed easy-type rule (delegate
  summarize + test always, no trace guidance).
* Breakdown-only ablation: the lead executes everything but with
  the subtask breakdown available. MODELLED ASSUMPTION (stated,
  not from the paper): structure halves the load penalty and cuts
  the lead's per-subtask cost 10 -> 9 (less wasted context).
* Oracle: brute-force best per-subtask assignment by analytic
  expected success (tie-break: lowest cost) over the 2**5
  assignments. The oracle uses TRUE qualities, so it is an upper
  bound the trace-tuned rule cannot be expected to reach.
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from itertools import product

__all__ = [
    "Subtask",
    "Task",
    "Trace",
    "Guidance",
    "RunResult",
    "make_tasks",
    "past_traces",
    "learn_guidance",
    "delegate",
    "fixed_easy_assignment",
    "task_level_assignment",
    "random_assignment",
    "oracle_assignment",
    "execute",
    "analytic_expected",
    "run_policy",
    "SUBTASK_TYPES",
    "LEAD_Q",
    "WORKER_Q",
    "LEAD_COST",
    "WORKER_COST",
    "DELEGATION_OVERHEAD",
    "LOAD_PENALTY",
    "DIFF_EFFECT",
    "N_SUBTASKS",
]

# ------------------------------------------------------------------- world
SUBTASK_TYPES = ("plan", "code", "verify", "test", "summarize")
HARD_TYPES = ("plan", "code")
EASY_TYPES = ("verify", "test", "summarize")
# Base success probabilities per subtask type (before difficulty and
# lead-load adjustments). Gaps lead-worker: plan .30, code .25,
# verify .19, test .12, summarize .06.
LEAD_Q = {"plan": 0.88, "code": 0.90, "verify": 0.92, "test": 0.93,
          "summarize": 0.95}
WORKER_Q = {"plan": 0.58, "code": 0.65, "verify": 0.73, "test": 0.81,
            "summarize": 0.89}
LEAD_COST = 10.0
WORKER_COST = 1.0
DELEGATION_OVERHEAD = 0.5  # lead coordination cost per delegated subtask
LOAD_PENALTY = 0.06         # lead success lost per prior lead subtask
DIFF_EFFECT = 0.05         # success lost per unit subtask difficulty
N_SUBTASKS = 5
BREAKDOWN_COST = 9.0       # lead cost/subtask with a subtask breakdown
BREAKDOWN_LOAD_FACTOR = 0.5  # breakdown halves the load penalty
TASK_LEVEL_HARD_FRAC = 0.4  # route whole task to worker iff hard
                            # fraction <= this
DIFFICULTIES = (0.0, 0.25, 0.5, 0.75, 1.0)
HARD_PROFILES = (0.2, 0.4, 0.6, 0.8)  # per-task P(hard subtask)


@dataclass(frozen=True)
class Subtask:
    type: str  # one of SUBTASK_TYPES
    difficulty: float  # in [0, 1]


@dataclass(frozen=True)
class Task:
    subtasks: tuple  # tuple[Subtask, ...], execution order


@dataclass(frozen=True)
class Trace:
    """One subtask attempt in a past run: which model, what it
    attempted, whether it succeeded, what it cost."""
    model: str  # "lead" | "worker"
    type: str
    difficulty: float
    success: bool
    cost: float


@dataclass
class Guidance:
    """Delegation guidance measured from past-run traces."""
    lead_q: dict    # per-type measured lead success rate
    worker_q: dict  # per-type measured worker success rate
    lead_cost: dict
    worker_cost: dict
    gap: dict       # lead_q - worker_q per type
    gap_max: float
    n_traces: int = 0

    def delegate_types(self, delta: float) -> tuple:
        """Types delegated at this delta, in SUBTASK_TYPES order."""
        thr = delta * self.gap_max
        return tuple(t for t in SUBTASK_TYPES if self.gap[t] <= thr)


def make_tasks(seed: int, n_tasks: int = 60) -> list:
    """Seeded held-out tasks. Each task draws a hard-profile, then
    5 subtasks: hard types (plan/code) with P=profile, easy types
    (verify/test/summarize) otherwise; difficulty uniform over
    DIFFICULTIES."""
    rng = random.Random(seed)
    tasks = []
    for _ in range(n_tasks):
        profile = rng.choice(HARD_PROFILES)
        subs = []
        for _ in range(N_SUBTASKS):
            pool = HARD_TYPES if rng.random() < profile else EASY_TYPES
            subs.append(Subtask(rng.choice(pool),
                                rng.choice(DIFFICULTIES)))
        tasks.append(Task(tuple(subs)))
    return tasks


# -------------------------------------------------------------- past runs
def _success_prob(model: str, subtask: Subtask, lead_done: int,
                  breakdown: bool = False) -> float:
    """True success probability of one subtask attempt."""
    if model == "lead":
        load = LOAD_PENALTY * lead_done
        if breakdown:
            load *= BREAKDOWN_LOAD_FACTOR
        q = LEAD_Q[subtask.type] - DIFF_EFFECT * subtask.difficulty - load
    else:
        q = WORKER_Q[subtask.type] - DIFF_EFFECT * subtask.difficulty
    return min(0.99, max(0.05, q))


def past_traces(seed: int, n_tasks: int = 4000) -> list:
    """Past-run corpus: both models' past runs, recorded per
    subtask attempt. Each attempt is a FRESH model call (no in-task
    load: the load penalty is an execution effect of a live run, not
    a property of the model on the type), so the measured per-type
    quality is the model's quality on that subtask type -- the
    quantity the paper's guidance is built from."""
    rng = random.Random(seed)
    tasks = make_tasks(seed, n_tasks)
    traces = []
    for task in tasks:
        for model in ("lead", "worker"):
            for st in task.subtasks:
                q = _success_prob(model, st, 0)
                ok = rng.random() < q
                cost = LEAD_COST if model == "lead" else WORKER_COST
                traces.append(Trace(model, st.type, st.difficulty, ok,
                                    cost))
    return traces


def learn_guidance(traces: list) -> Guidance:
    """Measure per-type quality and cost from past-run traces."""
    lead_q, worker_q, lead_c, worker_c = {}, {}, {}, {}
    for t in SUBTASK_TYPES:
        for model, qd, cd in (("lead", lead_q, lead_c),
                              ("worker", worker_q, worker_c)):
            sel = [tr for tr in traces if tr.model == model
                   and tr.type == t]
            qd[t] = sum(1 for tr in sel if tr.success) / len(sel)
            cd[t] = sum(tr.cost for tr in sel) / len(sel)
    gap = {t: lead_q[t] - worker_q[t] for t in SUBTASK_TYPES}
    return Guidance(lead_q, worker_q, lead_c, worker_c, gap,
                    max(gap.values()), len(traces))


# --------------------------------------------------------------- policies
def delegate(task: Task, guidance: Guidance, delta: float) -> tuple:
    """Partial-routing assignment: 0 = lead, 1 = worker, per
    subtask, from the trace-tuned guidance at this delta."""
    allowed = set(guidance.delegate_types(delta))
    return tuple(1 if st.type in allowed else 0 for st in task.subtasks)


def fixed_easy_assignment(task: Task) -> tuple:
    """No-guidance heuristic: delegate summarize + test always."""
    return tuple(1 if st.type in ("summarize", "test") else 0
                 for st in task.subtasks)


def task_level_assignment(task: Task) -> tuple:
    """Task-level routing baseline: the WHOLE task goes to one
    model -- worker if its hard-subtask fraction is low, else lead."""
    hard = sum(1 for st in task.subtasks if st.type in HARD_TYPES)
    whole = 1 if hard / len(task.subtasks) <= TASK_LEVEL_HARD_FRAC else 0
    return tuple(whole for _ in task.subtasks)


def random_assignment(task: Task, rng: random.Random) -> tuple:
    """Random delegation baseline: each subtask to the worker w.p. 1/2."""
    return tuple(1 if rng.random() < 0.5 else 0 for _ in task.subtasks)


def oracle_assignment(task: Task, breakdown: bool = False) -> tuple:
    """Brute-force best assignment by analytic expected success
    (tie-break: lowest cost). Uses TRUE qualities -- an upper bound,
    not a fair competitor for trace-tuned guidance."""
    best = None
    for assign in product((0, 1), repeat=len(task.subtasks)):
        p, c = analytic_expected(task, assign, breakdown)
        key = (p, -c)
        if best is None or key > best[0]:
            best = (key, assign)
    return best[1]


# -------------------------------------------------------------- execution
def execute(task: Task, assignment: tuple, rng: random.Random,
            breakdown: bool = False) -> tuple:
    """One stochastic run of a task under an assignment.
    Returns (task_success, cost, subtask_score). Task success is
    strict: every subtask must succeed."""
    lead_done = 0
    cost = 0.0
    ok_n = 0
    for st, a in zip(task.subtasks, assignment):
        if a == 0:
            q = _success_prob("lead", st, lead_done, breakdown)
            cost += BREAKDOWN_COST if breakdown else LEAD_COST
            lead_done += 1
        else:
            q = _success_prob("worker", st, lead_done)
            cost += WORKER_COST + DELEGATION_OVERHEAD
        ok = rng.random() < q
        ok_n += 1 if ok else 0
    n = len(task.subtasks)
    return (ok_n == n, cost, ok_n / n)


def analytic_expected(task: Task, assignment: tuple,
                      breakdown: bool = False) -> tuple:
    """Exact (success probability, cost) of an assignment under the
    world's true qualities -- no sampling noise."""
    lead_done = 0
    p, cost = 1.0, 0.0
    for st, a in zip(task.subtasks, assignment):
        if a == 0:
            q = _success_prob("lead", st, lead_done, breakdown)
            cost += BREAKDOWN_COST if breakdown else LEAD_COST
            lead_done += 1
        else:
            q = _success_prob("worker", st, lead_done)
            cost += WORKER_COST + DELEGATION_OVERHEAD
        p *= q
    return p, cost


@dataclass
class RunResult:
    success_rate: float
    mean_cost: float
    mean_score: float
    n: int
    trials: int
    # analytic (exact-expectation) counterparts, same policies/tasks
    analytic_success: float = 0.0
    analytic_cost: float = 0.0


def run_policy(tasks: list, assignment_fn, seed: int, trials: int = 400,
               breakdown: bool = False) -> RunResult:
    """Monte Carlo evaluation of one routing policy over tasks:
    `trials` seeded repetitions per task. `assignment_fn` takes a
    task and returns a 0/1 tuple (or, for random, takes (task, rng)).
    The same seed is used for every policy in an experiment (common
    random numbers), so policy comparisons are paired."""
    rng = random.Random(seed)
    succ = cost = score = 0
    n_runs = 0
    a_succ = a_cost = 0.0
    for task in tasks:
        assign = assignment_fn(task)
        ap, ac = analytic_expected(task, assign, breakdown)
        a_succ += ap
        a_cost += ac
        for _ in range(trials):
            ok, c, s = execute(task, assign, rng, breakdown)
            succ += 1 if ok else 0
            cost += c
            score += s
            n_runs += 1
    return RunResult(succ / n_runs, cost / n_runs, score / n_runs,
                     len(tasks), trials, a_succ / len(tasks),
                     a_cost / len(tasks))
