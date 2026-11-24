"""Static same-task retry economics — arXiv:2611.29133 "The Retry Tax:
Why Static Same-Task Retries Can Cost More per Completion".

Core claim implemented here: a static retry policy that stops at verified
success or an attempt cap, resets the original prompt and workspace, and
passes no feedback between attempts. Under stationary per-task success
probabilities, attempts independent conditional on the task, and a common
attempt cost:

  * cost per completion at cap K is at least the one-shot cost (K = 1);
  * marginal cost per additional completion is nondecreasing in K wherever
    the completion increments are positive;
  * easier tasks succeed earlier, so the unresolved mix shifts toward
    harder tasks (survivor selection).

With a homogeneous population (all tasks share one p) cost per completion
is exactly constant in K — the equality case of the first claim.

Pure stdlib. No model, no network, deterministic given a seed.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass

__all__ = [
    "completion_prob",
    "expected_attempts",
    "population_metrics",
    "marginal_cost",
    "simulate_population",
    "replay_runs",
    "unresolved_mean_p",
    "calibrated_mixture",
    "PAPER_COMPLETION_K1",
    "PAPER_COMPLETION_K4",
    "PAPER_CPC_K1_USD",
    "PAPER_CPC_K4_USD",
]

# Paper's replay estimates for one OpenHands configuration (abstract).
PAPER_COMPLETION_K1 = 0.597
PAPER_COMPLETION_K4 = 0.700
PAPER_CPC_K1_USD = 0.70
PAPER_CPC_K4_USD = 1.32


def completion_prob(p: float, cap: int) -> float:
    """P(task succeeds within `cap` attempts) = 1 - (1 - p)**cap.

    Computed as -expm1(cap * log1p(-p)): for p < ~2**-53 the naive
    (1 - p) rounds to 1.0 in float64 and the naive form returns 0,
    while the log1p form stays exact.
    """
    if not 0.0 <= p <= 1.0:
        raise ValueError("p must be in [0, 1]")
    if cap < 1:
        raise ValueError("cap must be >= 1")
    if p == 0.0:
        return 0.0
    if p == 1.0:
        return 1.0
    return -math.expm1(cap * math.log1p(-p))


def expected_attempts(p: float, cap: int) -> float:
    """Expected attempts spent on one task under the static policy.

    Attempts stop at first success or at the cap, so the count is
    sum_{j=0}^{cap-1} (1-p)**j = (1 - (1-p)**cap) / p for p > 0,
    and exactly `cap` for p == 0 (never succeeds).
    """
    if not 0.0 <= p <= 1.0:
        raise ValueError("p must be in [0, 1]")
    if cap < 1:
        raise ValueError("cap must be >= 1")
    if p == 0.0:
        return float(cap)
    return completion_prob(p, cap) / p


@dataclass(frozen=True)
class PopulationMetrics:
    cap: int
    completion: float          # fraction of tasks completed within cap
    spend_per_task: float      # expected cost per task (all tasks)
    cost_per_completion: float  # spend_per_task / completion


def population_metrics(ps, cap: int, cost: float = 1.0) -> PopulationMetrics:
    """Closed-form expectations over a task population with per-task p.

    `ps` is any iterable of per-task success probabilities.
    """
    ps = list(ps)
    if not ps:
        raise ValueError("empty population")
    if cost <= 0:
        raise ValueError("cost must be positive")
    n = len(ps)
    completion = sum(completion_prob(p, cap) for p in ps) / n
    spend = cost * sum(expected_attempts(p, cap) for p in ps) / n
    cpc = spend / completion if completion > 0 else math.inf
    return PopulationMetrics(cap=cap, completion=completion,
                             spend_per_task=spend, cost_per_completion=cpc)


def marginal_cost(ps, cap: int, cost: float = 1.0):
    """Marginal cost of the completions added by raising the cap to `cap`.

    (S(K) - S(K-1)) / (C(K) - C(K-1)); None when the cap adds no
    completions (increment zero), which is the paper's "wherever the
    completion increments are positive" qualifier.

    The increments are computed directly per task — ΔS = E[(1-p)^(K-1)]
    and ΔC = E[p (1-p)^(K-1)] — not by differencing the population
    means: for p near 1 both C(K) and C(K-1) are ~1 and their
    difference loses ~6 digits to cancellation.
    """
    if cap < 2:
        raise ValueError("marginal cost needs cap >= 2")
    ps = list(ps)
    if not ps:
        raise ValueError("empty population")
    if cost <= 0:
        raise ValueError("cost must be positive")
    n = len(ps)
    d_spend = cost * sum(_survival(p, cap - 1) for p in ps) / n
    d_comp = sum(p * _survival(p, cap - 1) for p in ps) / n
    if d_comp <= 0.0:
        return None
    return d_spend / d_comp


def _survival(p: float, k: int) -> float:
    """(1 - p)**k, accurate for p near 0 and near 1."""
    if p == 0.0:
        return 1.0
    if p == 1.0:
        return 0.0
    return math.exp(k * math.log1p(-p))


def simulate_population(ps, cap: int, cost: float = 1.0,
                        rng: random.Random | None = None):
    """Monte-Carlo run of the static policy: one attempt sequence per task.

    Returns (completion, spend_per_task, cost_per_completion) measured
    from simulated Bernoulli attempts (independent conditional on p).
    """
    ps = list(ps)
    rng = rng or random.Random()
    done = 0
    attempts = 0
    for p in ps:
        for _ in range(cap):
            attempts += 1
            if rng.random() < p:
                done += 1
                break
    n = len(ps)
    completion = done / n
    spend = cost * attempts / n
    cpc = spend / completion if done else math.inf
    return completion, spend, cpc


def replay_runs(sequences, cap: int, cost: float = 1.0,
                n_reorderings: int = 50,
                rng: random.Random | None = None):
    """The paper's replay procedure.

    `sequences` is a list of per-task attempt-outcome lists (True =
    verified success on that attempt), as recorded from existing
    benchmark runs. Each task's sequence is arranged in different
    orders (random permutations), each arrangement is stopped at
    verified success or the attempt cap, and completion and spending
    are averaged over arrangements and tasks.

    Returns (completion, spend_per_task, cost_per_completion).
    """
    rng = rng or random.Random()
    seqs = [list(s) for s in sequences]
    if not seqs:
        raise ValueError("empty run set")
    if cap < 1:
        raise ValueError("cap must be >= 1")
    total_done = 0
    total_attempts = 0
    total_tasks = 0
    for _ in range(n_reorderings):
        for seq in seqs:
            order = list(seq)
            rng.shuffle(order)
            total_tasks += 1
            for j, ok in enumerate(order[:cap]):
                total_attempts += 1
                if ok:
                    total_done += 1
                    break
            else:
                # Ran out of recorded attempts before the cap without
                # success: count the remaining attempts as spent.
                # (Sequences here are always at least `cap` long, so
                # this branch only fires for shorter recordings.)
                total_attempts += max(0, cap - len(order))
    completion = total_done / total_tasks
    spend = cost * total_attempts / total_tasks
    cpc = spend / completion if total_done else math.inf
    return completion, spend, cpc


def unresolved_mean_p(ps, k: int, rng: random.Random | None = None) -> float:
    """Mean per-task p among tasks still unresolved after k attempts.

    Survivor selection: easier tasks succeed earlier, so this mean
    falls as k grows (for a heterogeneous population).
    """
    ps = list(ps)
    rng = rng or random.Random()
    remaining = []
    for p in ps:
        survived = True
        for _ in range(k):
            if rng.random() < p:
                survived = False
                break
        if survived:
            remaining.append(p)
    if not remaining:
        return float("nan")
    return sum(remaining) / len(remaining)


def calibrated_mixture(n: int = 20000, seed: int = 20261124):
    """Two-point task population calibrated to the paper's replay.

    Mixture: 64.3% of tasks with p = 0.90 (easy), 35.7% with p = 0.05
    (hard). Closed form: C(1) = 0.597 (paper 0.597), C(4) = 0.709
    (paper 0.700). The mixture is a stand-in for the paper's
    OpenHands run data, which is not public — it matches the paper's
    completion endpoints approximately, not exactly, and the test
    asserts the inequalities, not the paper's exact CPC numbers.
    """
    rng = random.Random(seed)
    return [0.90 if rng.random() < 0.643 else 0.05 for _ in range(n)]
