"""Semantic Context Handoff (arXiv:2611.24357, \"Semantic Context Handoff
with On-Demand Evidence for Long-Horizon Agents\").

Paper claim implemented: long-horizon agents that hand off context as
full history pay growing token cost; methods that keep only compact
conclusions discard evidence that later becomes relevant. SCH separates
state from evidence: a compact semantic state carries the agent's current
interpretation, while supporting action-observation evidence stays
separately addressable and re-enters context when later execution makes
it relevant. The paper's headline numbers (controlled 150-task
SWE-bench study): evidence recovery improves Pass@1 from 80.0% to
86.9% at essentially unchanged token use; the complete system reaches
86.9% Pass@1 while using 43% fewer tokens and 41% less wall-clock
time than the strongest baseline.

This module implements the mechanism as a calibrated synthetic
long-horizon task model (see TaskParams): every parameter is
documented with the paper aggregate it is calibrated to, or marked as
an explicit assumption. Tasks are generated ONCE, policy-independently
-- exactly the paper's replay setup -- so policies are compared on the
same tasks. The verdict ceiling is therefore SUPPORTED, never
VERIFIED -- see the session README.

Task model. A task is a sequence of T steps. Each step produces m
evidence facts (key, token size, relevance delay). A fact with delay
d is needed at step t+d: d=0 means needed immediately (always in
context -- it is the current observation); d>0 means delayed
relevance, the paper's crux: the fact must survive summarisation or
be recovered later. A step succeeds iff every fact it needs is in
context; a task passes iff every step succeeds.

Policies, all evaluated on the same generated tasks:
- FULL: context = every fact produced so far, at full token size.
  Quality ceiling (always passes), worst cost.
- SUMMARY: context = the compact semantic state only. At production
  time each fact is retained into the state with probability
  ``state_retention``; retained facts cost ``compression`` of their
  tokens (they are represented, not quoted). No recovery.
- SCH: same state as SUMMARY, plus an evidence store holding every
  fact. When a step needs a fact that is not in the state, the agent
  issues a recovery lookup: with probability ``recall`` the fact
  re-enters context at full size plus a lookup overhead; with
  probability ``fp_per_lookup`` a distractor fact is also pulled
  (cost only). A failed lookup fails the step.
- ORACLE: SCH with recall=1 and no false positives -- the recovery
  upper bound.

A step succeeds iff every fact it needs is in context. A fact needed
at the step that produced it (delay 0) is the current observation and
is always in context -- only delayed facts can be lost to the state.
A task passes iff at least ``pass_step_fraction`` of its steps
succeed: a missed fact costs a retry, and the task fails only when
more than a fifth of its steps fail (an explicit assumption -- the
strict all-steps pass rate is also reported, and it amplifies
per-step loss; see the session README).

Token accounting is the cost proxy (the paper's wall-clock saving
tracks its token saving); wall-clock is reported as proportional to
tokens, not measured.

Determinism: all randomness flows through SplitMix64, implemented
identically in the Rust second stage so Python-vs-Rust parity is
bit-exact. Draw order (fixed, do not reorder):
  task rng, seeded ``seed + task_index``: for each step t, for each
  of the m facts in order: size draw (uniform in [min_fact_tokens,
  max_fact_tokens]), delay decision (bernoulli(delayed_frac)), if
  delayed a delay draw (1 + floor(uniform * max_delay)); then, for
  each fact in production order, a retention draw
  (bernoulli(state_retention)).
  recovery rng, seeded ``seed * 1_000_003 + task_index * 131 + 7``:
  during SCH evaluation, for each missing needed fact in need order:
  a recall draw (bernoulli(recall)), then a false-positive draw
  (bernoulli(fp_per_lookup)); a pulled distractor costs one size
  draw (uniform in [min_fact_tokens, max_fact_tokens]).

Stdlib only. No model calls, no network at import time.
"""

import json
import math

__all__ = [
    "SplitMix64",
    "TaskParams",
    "CALIBRATED",
    "Fact",
    "Task",
    "generate_task",
    "POLICIES",
    "evaluate_policy",
    "evaluate_all",
    "sensitivity_sweep",
    "run_experiment",
    "task_digest",
]

_MASK64 = (1 << 64) - 1
_GOLDEN = 0x9E3779B97F4A7C15


class SplitMix64:
    """SplitMix64 PRNG (Stafford). Identical arithmetic in the Rust port.

    ``uniform()`` returns a float in [0, 1) with 53 bits of randomness:
    ``(next_u64() >> 11) * 2**-53``. Seeded identically, Python and Rust
    produce the same stream bit-for-bit.
    """

    def __init__(self, seed):
        self.state = int(seed) & _MASK64

    def next_u64(self):
        self.state = (self.state + _GOLDEN) & _MASK64
        z = self.state
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & _MASK64
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & _MASK64
        return z ^ (z >> 31)

    def uniform(self):
        return (self.next_u64() >> 11) * (1.0 / 9007199254740992.0)

    def bernoulli(self, p):
        return self.uniform() < p

    def randint(self, lo, hi):
        """Uniform integer in [lo, hi] inclusive."""
        return lo + int(self.uniform() * (hi - lo + 1))


class TaskParams:
    """Parameters of the calibrated synthetic long-horizon task model.

    Each field documents its calibration source. Fields marked [paper]
    come from an aggregate reported in the abstract; fields marked
    [assumption] are explicit modelling choices, not paper numbers --
    changing them is a sensitivity check, not a refutation.
    """

    def __init__(
        self,
        steps=12,
        facts_per_step=3,
        delayed_frac=0.25,
        max_delay=4,
        min_fact_tokens=10,
        max_fact_tokens=30,
        state_retention=0.85,
        compression=0.3,
        state_overhead=40,
        recall=0.9,
        lookup_overhead=5,
        fp_per_lookup=0.2,
        pass_step_fraction=0.8,
    ):
        self.steps = int(steps)
        self.facts_per_step = int(facts_per_step)
        self.delayed_frac = float(delayed_frac)
        self.max_delay = int(max_delay)
        self.min_fact_tokens = int(min_fact_tokens)
        self.max_fact_tokens = int(max_fact_tokens)
        self.state_retention = float(state_retention)
        self.compression = float(compression)
        self.state_overhead = int(state_overhead)
        self.recall = float(recall)
        self.lookup_overhead = int(lookup_overhead)
        self.fp_per_lookup = float(fp_per_lookup)
        self.pass_step_fraction = float(pass_step_fraction)

    def to_dict(self):
        return {
            "steps": self.steps,
            "facts_per_step": self.facts_per_step,
            "delayed_frac": self.delayed_frac,
            "max_delay": self.max_delay,
            "min_fact_tokens": self.min_fact_tokens,
            "max_fact_tokens": self.max_fact_tokens,
            "state_retention": self.state_retention,
            "compression": self.compression,
            "state_overhead": self.state_overhead,
            "recall": self.recall,
            "lookup_overhead": self.lookup_overhead,
            "fp_per_lookup": self.fp_per_lookup,
            "pass_step_fraction": self.pass_step_fraction,
        }

    @classmethod
    def from_dict(cls, d):
        return cls(**d)

    def with_overrides(self, **kw):
        d = self.to_dict()
        d.update(kw)
        return TaskParams.from_dict(d)


# Calibrated regime. Sources:
# - steps=12, facts_per_step=3 [assumption]: a long-horizon task is a
#   dozen steps with a few observations per step; 36 facts per task.
# - delayed_frac=0.25, max_delay=4 [assumption]: a quarter of
#   observations only become relevant later (the paper's crux -- a
#   summary that discards them loses exactly these); delay up to 4
#   steps, so evidence must survive several summarisation rounds.
# - min/max_fact_tokens=10/30 [assumption]: an observation (a file
#   path + snippet, a test line) is tens of tokens.
# - state_retention=0.85 [assumption]: summarisation keeps most facts
#   in compressed form; the loss is concentrated in the delayed ones
#   the summary did not know would matter.
# - pass_step_fraction=0.8 [assumption]: a missed fact costs a retry;
#   the task fails only when more than a fifth of its steps fail.
# - compression=0.3 [assumption]: a retained fact costs ~30% of its
#   tokens inside the compact state.
# - state_overhead=40 [assumption]: the interpretation itself -- the
#   current goal, plan, and conclusions -- is a fixed ~40 tokens.
# - recall=0.9, lookup_overhead=5, fp_per_lookup=0.2 [assumption]:
#   keyed recovery usually works (the evidence is addressable by
#   key); a lookup costs a few tokens and sometimes pulls a
#   distractor.
CALIBRATED = TaskParams()


class Fact:
    """One evidence item produced at a step."""

    __slots__ = ("key", "produced_step", "needed_step", "tokens", "retained")

    def __init__(self, key, produced_step, needed_step, tokens, retained):
        self.key = key
        self.produced_step = produced_step
        self.needed_step = needed_step
        self.tokens = tokens
        self.retained = retained

    def to_dict(self):
        return {
            "key": self.key,
            "produced_step": self.produced_step,
            "needed_step": self.needed_step,
            "tokens": self.tokens,
            "retained": self.retained,
        }


class Task:
    """A generated long-horizon task: facts per step + needs per step."""

    def __init__(self, facts, needs):
        self.facts = facts  # list of Fact, production order
        self.needs = needs  # list per step: list of fact keys needed
        self.steps = len(needs)

    def to_dict(self):
        return {
            "steps": self.steps,
            "facts": [f.to_dict() for f in self.facts],
            "needs": self.needs,
        }


def generate_task(rng, params):
    """Generate one task. Draw order is fixed -- see module docstring."""
    facts = []
    needs = [[] for _ in range(params.steps)]
    key = 0
    for t in range(params.steps):
        for _ in range(params.facts_per_step):
            tokens = rng.randint(params.min_fact_tokens, params.max_fact_tokens)
            if rng.bernoulli(params.delayed_frac):
                delay = rng.randint(1, params.max_delay)
            else:
                delay = 0
            needed_step = t + delay
            facts.append(Fact(key, t, needed_step, tokens, False))
            if needed_step < params.steps:
                needs[needed_step].append(key)
            key += 1
    # Retention draws: production order, after all facts are generated.
    for f in facts:
        f.retained = rng.bernoulli(params.state_retention)
    return Task(facts, needs)


def _recovery_rng(seed, task_index):
    return SplitMix64((int(seed) * 1_000_003 + int(task_index) * 131 + 7) & _MASK64)


def evaluate_policy(task, policy, params, seed=0, task_index=0):
    """Evaluate one policy on one task.

    Returns a dict with per-step success, task pass, token totals and
    (for SCH) recovery counts. ``seed``/``task_index`` seed the
    recovery stream; they must match the values used at generation
    for the parity sweep to be meaningful.
    """
    by_key = {f.key: f for f in task.facts}
    steps_ok = 0
    total_tokens = 0
    recoveries = 0
    recovery_hits = 0
    rng = None
    if policy in ("sch", "oracle"):
        rng = _recovery_rng(seed, task_index)

    for t in range(task.steps):
        needed = [by_key[k] for k in task.needs[t]]
        if policy == "full":
            # Context = every fact produced so far, at full size.
            produced = [f for f in task.facts if f.produced_step <= t]
            total_tokens += sum(f.tokens for f in produced)
            steps_ok += 1
            continue
        # State tokens: retained facts so far at compressed size, plus
        # the fixed interpretation overhead.
        retained_so_far = [f for f in task.facts
                           if f.produced_step <= t and f.retained]
        state_tokens = params.state_overhead + sum(
            int(f.tokens * params.compression) for f in retained_so_far)
        step_tokens = state_tokens
        step_ok = True
        # A fact needed at its own production step is the current
        # observation: always in context. Only delayed facts can be
        # lost to the state.
        delayed_needed = [f for f in needed if f.produced_step < t]
        if policy == "summary":
            for f in delayed_needed:
                if not f.retained:
                    step_ok = False
                    break
        elif policy in ("sch", "oracle"):
            recall = 1.0 if policy == "oracle" else params.recall
            fp = 0.0 if policy == "oracle" else params.fp_per_lookup
            for f in delayed_needed:
                if f.retained:
                    continue
                recoveries += 1
                if rng.bernoulli(recall):
                    recovery_hits += 1
                    step_tokens += f.tokens + params.lookup_overhead
                    if rng.bernoulli(fp):
                        step_tokens += rng.randint(
                            params.min_fact_tokens, params.max_fact_tokens)
                else:
                    step_ok = False
                    # Still pay the lookup overhead on a miss.
                    step_tokens += params.lookup_overhead
                    if rng.bernoulli(fp):
                        step_tokens += rng.randint(
                            params.min_fact_tokens, params.max_fact_tokens)
        else:
            raise ValueError("unknown policy: %r" % policy)
        total_tokens += step_tokens
        if step_ok:
            steps_ok += 1

    threshold = math.ceil(params.pass_step_fraction * task.steps)
    return {
        "policy": policy,
        "steps_ok": steps_ok,
        "steps": task.steps,
        "task_pass": steps_ok >= threshold,
        "strict_task_pass": steps_ok == task.steps,
        "total_tokens": total_tokens,
        "recoveries": recoveries,
        "recovery_hits": recovery_hits,
    }


POLICIES = ("full", "summary", "sch", "oracle")


def evaluate_all(tasks, params, seed=0):
    """Evaluate every policy on the same list of generated tasks."""
    agg = {p: {"tasks": 0, "passes": 0, "strict_passes": 0, "steps_ok": 0,
               "steps": 0, "tokens": 0, "recoveries": 0, "recovery_hits": 0}
           for p in POLICIES}
    for i, task in enumerate(tasks):
        for p in POLICIES:
            r = evaluate_policy(task, p, params, seed=seed, task_index=i)
            a = agg[p]
            a["tasks"] += 1
            a["passes"] += 1 if r["task_pass"] else 0
            a["strict_passes"] += 1 if r["strict_task_pass"] else 0
            a["steps_ok"] += r["steps_ok"]
            a["steps"] += r["steps"]
            a["tokens"] += r["total_tokens"]
            a["recoveries"] += r["recoveries"]
            a["recovery_hits"] += r["recovery_hits"]
    out = {}
    for p, a in agg.items():
        out[p] = {
            "task_pass_rate": a["passes"] / a["tasks"],
            "strict_task_pass_rate": a["strict_passes"] / a["tasks"],
            "step_success_rate": a["steps_ok"] / a["steps"],
            "mean_tokens_per_task": a["tokens"] / a["tasks"],
            "mean_tokens_per_step": a["tokens"] / a["steps"],
            "recoveries_per_task": a["recoveries"] / a["tasks"],
            "recovery_hit_rate": (a["recovery_hits"] / a["recoveries"]
                                  if a["recoveries"] else None),
            "tasks": a["tasks"],
        }
    return out


def sensitivity_sweep(seed=0, n_tasks=400, grid=None):
    """Sweep the three load-bearing parameters; report SCH vs SUMMARY
    task-pass gain and SCH token saving vs FULL per cell."""
    if grid is None:
        grid = {
            "delayed_frac": [0.15, 0.35, 0.55],
            "state_retention": [0.55, 0.75, 0.9],
            "recall": [0.7, 0.9, 1.0],
        }
    cells = []
    for df in grid["delayed_frac"]:
        for sr in grid["state_retention"]:
            for rc in grid["recall"]:
                params = CALIBRATED.with_overrides(
                    delayed_frac=df, state_retention=sr, recall=rc)
                tasks = [generate_task(SplitMix64(seed + i), params)
                         for i in range(n_tasks)]
                res = evaluate_all(tasks, params, seed=seed)
                cells.append({
                    "delayed_frac": df,
                    "state_retention": sr,
                    "recall": rc,
                    "summary_task_pass": res["summary"]["task_pass_rate"],
                    "sch_task_pass": res["sch"]["task_pass_rate"],
                    "sch_gain_pp": 100 * (res["sch"]["task_pass_rate"]
                                          - res["summary"]["task_pass_rate"]),
                    "sch_token_saving_vs_full_pct": 100 * (
                        1 - res["sch"]["mean_tokens_per_task"]
                        / res["full"]["mean_tokens_per_task"]),
                    "sch_within_full_pp": 100 * (
                        res["full"]["task_pass_rate"]
                        - res["sch"]["task_pass_rate"]),
                })
    return cells


def run_experiment(seed=0, n_tasks=2000, n_seeds=5, params=None):
    """Main experiment: n_seeds x n_tasks tasks, all policies, plus the
    sensitivity sweep and the no-delay sanity check."""
    params = params or CALIBRATED
    per_seed = []
    for s in range(n_seeds):
        tasks = [generate_task(SplitMix64(seed + s * 100_003 + i), params)
                 for i in range(n_tasks)]
        per_seed.append(evaluate_all(tasks, params, seed=seed + s * 100_003))
    agg = {}
    for p in POLICIES:
        agg[p] = {
            k: sum(ps[p][k] for ps in per_seed) / n_seeds
            for k in ("task_pass_rate", "strict_task_pass_rate",
                      "step_success_rate",
                      "mean_tokens_per_task", "mean_tokens_per_step",
                      "recoveries_per_task")
        }
        agg[p]["tasks"] = n_tasks * n_seeds
        hits = [ps[p]["recovery_hit_rate"] for ps in per_seed
                if ps[p]["recovery_hit_rate"] is not None]
        agg[p]["recovery_hit_rate"] = (
            sum(hits) / len(hits) if hits else None)
    # Per-seed spread for the headline numbers.
    spread = {
        p: [round(100 * ps[p]["task_pass_rate"], 2) for ps in per_seed]
        for p in POLICIES
    }
    sweep = sensitivity_sweep(seed=seed, n_tasks=400)
    # Sanity: with no delayed relevance, SUMMARY == SCH == FULL.
    nd_params = params.with_overrides(delayed_frac=0.0)
    nd_tasks = [generate_task(SplitMix64(seed + 999_983 + i), nd_params)
                for i in range(400)]
    no_delay = evaluate_all(nd_tasks, nd_params, seed=seed + 999_983)
    return {
        "params": params.to_dict(),
        "seeds": n_seeds,
        "tasks_per_seed": n_tasks,
        "policies": agg,
        "per_seed_task_pass_pct": spread,
        "sensitivity_sweep": sweep,
        "no_delay_sanity": {p: no_delay[p]["task_pass_rate"]
                            for p in POLICIES},
    }


def task_digest(task):
    """Stable short digest of a generated task (parity checks)."""
    h = 0
    for f in task.facts:
        h = (h * 31 + f.key) & _MASK64
        h = (h * 31 + f.tokens) & _MASK64
        h = (h * 31 + f.needed_step) & _MASK64
        h = (h * 31 + (1 if f.retained else 0)) & _MASK64
    return h
