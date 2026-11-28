"""MeterRouter: Budget-Aware Dynamic Routing for Multi-Agent Systems.

arXiv:2611.34327 "MeterRouter: Budget-Aware Dynamic Routing for
Multi-Agent Systems" (Xu et al.).

Core claim implemented here: a multi-agent system can be built
incrementally -- one agent at a time, choosing each agent's role,
LLM backbone, and incoming edges -- conditioned on the query and a
per-query monetary budget, with a budget meter tracking the remaining
budget after every call. The paper reports that across five benchmarks
and a three-tier LLM pool, MeterRouter matches the strongest MAS
baseline in accuracy (66.72% vs 66.80%) at 82% lower cost
($2.56e-3 vs $14.15e-3 per query), lies on the accuracy-cost Pareto
frontier, adapts its spending to the budget, and stays in budget on
>=99.98% of queries. Its RL stage is load-bearing: random LLM
selection (w/o RL) drops accuracy 66.72% -> 57.34% at the same cost.

Bounded CPU reproduction (no model, no network; deterministic given
a seed). What is exact from the paper:

* Three-tier pool prices (Table 6, $/1M tokens): low gpt-oss-20b
  (0.06 in / 0.19 out), mid gpt-oss-120b (0.15 / 0.595), high
  MiniMax-M3 (0.30 / 1.20).
* Cost (Eq. 1): C_i = c_in * T_in + c_out * T_out.
  Remaining budget (Eq. 2): b_i = B - sum_{j<i} C_j.
* Utility (Eq. 5): U = S - gamma * C / B, S in {0, 1} correct.
  gamma is the paper's tuning parameter (value not stated in the
  paper); gamma = 0.5 here, chosen because gamma = 1 makes the
  greedy router decline to build a multi-agent team at all, which
  contradicts the paper's observed team sizes (1.22 agents at
  B = 5e-3 rising to 3.13 at B = 5e-2 on Omni-MATH-2).
* N_max = 5 agents; budgets B in {1, 4, 5, 50} x 10^-3 USD.
* Termination: learned stop / agent-cap stop / budget stop; an
  overrun (realised cost > B) is a violation.
* Graph templates (Appendix E): single, chain, fan-in, dense,
  diamond, pruned -- used here for the fixed-topology baselines.

SIMULATION CHOICES (stated, not from the paper): the paper's router
is a learned policy (imitation + GRPO) over real LLM calls. Here the
router is a greedy optimiser over the same decision space (role,
backbone, edges, termination) and the "LLMs" are a synthetic
environment: each query has a latent difficulty threshold, each
(role, tier) has a base solve probability, and an agent succeeds
when its effective skill (base + role adjustment + hint bonus from
distinct predecessor roles) plus small Gaussian noise exceeds the
threshold. Token counts are fixed per role and per incoming edge.
The test therefore validates the budget-metered incremental
construction MECHANISM and its predicted trade-offs (near-strongest
accuracy at much lower cost, budget compliance, budget adaptation,
ablation loss), not the learned policy and not the paper's absolute
benchmark numbers.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from functools import lru_cache

__all__ = [
    "POOL",
    "ROLES",
    "N_MAX",
    "BUDGETS",
    "GAMMA",
    "Agent",
    "Query",
    "RunResult",
    "call_cost",
    "remaining_budget",
    "utility",
    "template_graph",
    "expected_team_success",
    "greedy_router",
    "strongest_mas",
    "single_agent",
    "fixed_router",
    "random_tier_router",
    "run_experiment",
    "pareto_points",
]

# (name, $/1M input tokens, $/1M output tokens) -- Table 6, exact.
POOL = (
    ("gpt-oss-20b", 0.06, 0.19),
    ("gpt-oss-120b", 0.15, 0.595),
    ("MiniMax-M3", 0.30, 1.20),
)

# Roles (Appendix F) with output-token counts (simulation choice).
ROLES = ("solver", "analyst", "programmer", "inspector", "tester")
ROLE_OUT_TOKENS = {
    "solver": 1200,
    "analyst": 1000,
    "programmer": 1500,
    "inspector": 700,
    "tester": 800,
}
# Role adjustment to base solve probability (simulation choice).
ROLE_ADJ = {
    "solver": 0.04,
    "analyst": 0.02,
    "programmer": 0.00,
    "inspector": -0.12,
    "tester": -0.06,
}
# Base solve probability by tier at medium difficulty (simulation).
TIER_SKILL = (0.38, 0.53, 0.65)
# Difficulty shift on solve probability (simulation).
DIFF_SHIFT = {"easy": 0.13, "medium": 0.0, "hard": -0.13}
# Hint bonus per distinct predecessor role (simulation, capped).
HINT_BONUS = 0.03
NOISE_SD = 0.08
# Token counts (simulation choice, at the paper's scale: a high-tier
# call costs ~2e-3 USD, so the tightest budget B=1e-3 admits only
# low-tier calls, as in the paper).
BASE_IN_TOKENS = 2000
IN_TOKENS_PER_PARENT = 500
N_MAX = 5
BUDGETS = (1e-3, 4e-3, 5e-3, 50e-3)
GAMMA = 0.5
# Minimum expected success gain for the greedy router to add an agent.
MIN_GAIN = 0.01


@dataclass(frozen=True)
class Agent:
    """One agent in a built MAS: role, tier index into POOL, parents."""

    role: str
    tier: int
    parents: tuple[int, ...] = ()


@dataclass(frozen=True)
class Query:
    """A synthetic query: difficulty band and latent threshold."""

    difficulty: str
    threshold: float


@dataclass
class RunResult:
    """Outcome of one method on one query set at one budget."""

    method: str
    budget: float
    n: int = 0
    correct: int = 0
    total_cost: float = 0.0
    overruns: int = 0
    total_agents: int = 0
    stops: dict = field(default_factory=dict)

    @property
    def accuracy(self) -> float:
        return self.correct / self.n if self.n else 0.0

    @property
    def mean_cost(self) -> float:
        return self.total_cost / self.n if self.n else 0.0

    @property
    def overrun_rate(self) -> float:
        return self.overruns / self.n if self.n else 0.0

    @property
    def mean_agents(self) -> float:
        return self.total_agents / self.n if self.n else 0.0


def call_cost(tier: int, n_parents: int, role: str) -> float:
    """Eq. (1): C_i = c_in * T_in + c_out * T_out, in USD."""
    _, c_in, c_out = POOL[tier]
    t_in = BASE_IN_TOKENS + IN_TOKENS_PER_PARENT * n_parents
    t_out = ROLE_OUT_TOKENS[role]
    return (c_in * t_in + c_out * t_out) / 1e6


def remaining_budget(budget: float, costs: list[float]) -> float:
    """Eq. (2): b_i = B - sum_{j<i} C_j."""
    return budget - sum(costs)


def utility(correct: bool, cost: float, budget: float,
            gamma: float = GAMMA) -> float:
    """Eq. (5): U = S - gamma * C / B."""
    return (1.0 if correct else 0.0) - gamma * cost / budget


def template_graph(name: str, n: int) -> list[tuple[int, ...]]:
    """Parents per node for the Appendix E templates (n = 1..5)."""
    if n == 1:
        return [()]
    if name == "single":
        return [() for _ in range(n)]
    if name == "chain":
        return [()] + [(i - 1,) for i in range(1, n)]
    if name == "fan-in":
        return [() for _ in range(n - 1)] + [tuple(range(n - 1))]
    if name == "dense":
        return [tuple(range(i)) for i in range(n)]
    if name == "diamond" and n == 4:
        return [(), (0,), (0,), (1, 2)]
    if name == "pruned":
        rng = random.Random(1000 + n)
        parents = [()]
        for i in range(1, n):
            k = rng.randint(1, i)
            parents.append(tuple(sorted(rng.sample(range(i), k))))
        return parents
    raise ValueError(f"unknown template {name!r} for n={n}")


def _phi(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _effective_skill(role: str, tier: int, difficulty: str,
                     n_distinct_predecessors: int) -> float:
    skill = (TIER_SKILL[tier] + ROLE_ADJ[role]
             + DIFF_SHIFT[difficulty]
             + HINT_BONUS * n_distinct_predecessors)
    return min(max(skill, 0.01), 0.95)


def _team_skills(team: list[Agent], difficulty: str) -> list[float]:
    skills = []
    seen_roles: set[str] = set()
    for agent in team:
        skills.append(_effective_skill(
            agent.role, agent.tier, difficulty, len(seen_roles)))
        seen_roles.add(agent.role)
    return skills


def expected_team_success(team: list[Agent], difficulty: str) -> float:
    """Analytic P(team solves) under the latent-threshold model.

    Query threshold theta ~ Uniform(0, 1) (the difficulty band shifts
    skills, not theta). Given theta, agent i fails with probability
    Phi((theta - skill_i) / NOISE_SD); the team succeeds unless every
    agent fails. Integrate over theta on a fixed grid.
    """
    if not team:
        return 0.0
    skills = _team_skills(team, difficulty)
    return _expected_from_skills(tuple(round(s, 3) for s in skills))


@lru_cache(maxsize=None)
def _expected_from_skills(skills: tuple[float, ...]) -> float:
    """E_theta[1 - prod Phi((theta - s_i)/sd)] on a fixed grid.

    Cached: skills come from a small discrete set (role x tier x
    difficulty x hint count), so the same team shapes recur.
    """
    total = 0.0
    steps = 100
    for k in range(steps):
        theta = (k + 0.5) / steps
        p_fail = 1.0
        for s in skills:
            p_fail *= _phi((theta - s) / NOISE_SD)
        total += 1.0 - p_fail
    return total / steps


def _simulate_team(team: list[Agent], query: Query,
                   rng: random.Random) -> bool:
    """One seeded live run of a team on a query (latent-threshold)."""
    skills = _team_skills(team, query.difficulty)
    for s in skills:
        if s + rng.gauss(0.0, NOISE_SD) > query.threshold:
            return True
    return False


def _team_cost(team: list[Agent]) -> float:
    return sum(call_cost(a.tier, len(a.parents), a.role) for a in team)


def greedy_router(query: Query, budget: float, rng: random.Random,
                  random_tier: bool = False,
                  tier_rng: random.Random | None = None) -> tuple[list[Agent], str]:
    """Incremental budget-aware construction (the mechanism).

    At each step, score every candidate (role, tier, parents) by its
    expected utility gain (Eq. 5): dU = dE[success] - gamma * cost / B,
    and add the best affordable candidate whose dU is positive and
    whose success gain clears MIN_GAIN. Terminates by learned stop
    (no candidate clears the bar), agent-cap stop (N_MAX reached),
    or budget stop (nothing affordable). Returns (team, stop_reason).
    """
    team: list[Agent] = []
    spent = 0.0
    used_roles: set[str] = set()
    while len(team) < N_MAX:
        remaining = budget - spent
        base = expected_team_success(team, query.difficulty)
        # In random-tier mode the tier is exogenous: one draw per
        # step, before any role/parents choice, so the router cannot
        # cherry-pick a high tier. This is the paper's w/o RL
        # ablation (random LLM selection) as opposed to the learned
        # allocation. Tier draws come from a separate stream so the
        # simulation draws stay paired across methods.
        tier_stream = tier_rng if tier_rng is not None else rng
        step_tier = tier_stream.randrange(len(POOL)) if random_tier else None
        best = None  # (score, gain, cost, agent)
        for role in ROLES:
            if role in used_roles:
                continue
            parent_options: list[tuple[int, ...]] = [()]
            if team:
                parent_options.append((len(team) - 1,))
                parent_options.append(tuple(range(len(team))))
            for parents in parent_options:
                tiers = (step_tier,) if random_tier else range(len(POOL))
                for tier in tiers:
                    cost = call_cost(tier, len(parents), role)
                    if cost > remaining:
                        continue
                    cand = team + [Agent(role, tier, parents)]
                    gain = expected_team_success(cand, query.difficulty) - base
                    if gain < MIN_GAIN:
                        continue
                    # Eq. (5): after the first agent, add only if the
                    # expected success gain outweighs the cost penalty
                    # at this budget. The first agent is mandatory: a
                    # query is always attempted with the best
                    # affordable agent (the paper grades "no agent
                    # affordable" as incorrect, and a router that
                    # declines to answer is not a router).
                    if team:
                        d_utility = gain - GAMMA * cost / budget
                        if d_utility <= 0:
                            continue
                        score = d_utility
                    else:
                        score = gain
                    if best is None or score > best[0]:
                        best = (score, gain, cost,
                                Agent(role, tier, parents))
        if best is None:
            cheapest = min(call_cost(t, 0, r) for t in range(3)
                           for r in ROLES)
            if random_tier:
                cheapest = min(call_cost(step_tier, 0, r) for r in ROLES)
            if remaining < cheapest:
                return team, "budget"
            return team, "learned"
        team.append(best[3])
        spent += best[2]
        used_roles.add(best[3].role)
    return team, "cap"


def strongest_mas(query: Query, budget: float) -> list[Agent]:
    """Strongest MAS baseline: all 5 roles, high tier, dense edges."""
    team = []
    for i, role in enumerate(ROLES):
        team.append(Agent(role, 2, tuple(range(i))))
    return team


def single_agent(query: Query, budget: float, tier: int) -> list[Agent]:
    return [Agent("solver", tier, ())]


def fixed_router(query: Query, budget: float) -> list[Agent]:
    """Fixed 3-agent mid-tier chain, ignoring the budget (baseline)."""
    parents = template_graph("chain", 3)
    return [Agent(role, 1, parents[i])
            for i, role in enumerate(("solver", "analyst", "inspector"))]


def random_tier_router(query: Query, budget: float,
                       rng: random.Random,
                       tier_rng: random.Random | None = None) -> tuple[list[Agent], str]:
    """Greedy construction with the tier chosen uniformly at random
    (the paper's w/o RL ablation analogue: random LLM selection)."""
    return greedy_router(query, budget, rng, random_tier=True,
                         tier_rng=tier_rng)


def make_queries(n: int, seed: int) -> list[Query]:
    """Synthetic query set: difficulty mix easy/medium/hard 25/50/25,
    latent threshold uniform within each band."""
    rng = random.Random(seed)
    queries = []
    for _ in range(n):
        r = rng.random()
        if r < 0.25:
            difficulty = "easy"
        elif r < 0.75:
            difficulty = "medium"
        else:
            difficulty = "hard"
        queries.append(Query(difficulty, rng.random()))
    return queries


def _run_method(method: str, queries: list[Query], budget: float,
                seed: int) -> RunResult:
    rng = random.Random(seed)
    result = RunResult(method=method, budget=budget)
    for qi, query in enumerate(queries):
        # Paired draws: every method sees the same noise draws for a
        # given query, so method differences are the mechanism's, not
        # sampling noise. (seed is the experiment seed, the same for
        # every method; tier draws use a separate stream.)
        rng = random.Random(seed * 1000003 + qi)
        tier_rng = random.Random(seed * 7919 + qi)
        if method == "meterrouter":
            team, stop = greedy_router(query, budget, rng)
        elif method == "random_tier":
            team, stop = random_tier_router(query, budget, rng, tier_rng)
        elif method == "strongest":
            team, stop = strongest_mas(query, budget), "fixed"
        elif method == "single_high":
            team, stop = single_agent(query, budget, 2), "fixed"
        elif method == "single_low":
            team, stop = single_agent(query, budget, 0), "fixed"
        elif method == "fixed":
            team, stop = fixed_router(query, budget), "fixed"
        else:
            raise ValueError(f"unknown method {method!r}")
        cost = _team_cost(team)
        result.n += 1
        result.total_cost += cost
        result.total_agents += len(team)
        result.stops[stop] = result.stops.get(stop, 0) + 1
        if cost > budget + 1e-12:
            result.overruns += 1
        if team and _simulate_team(team, query, rng):
            result.correct += 1
    return result


def run_experiment(n_queries: int = 500, seed: int = 20261128,
                   budgets: tuple[float, ...] = BUDGETS) -> dict:
    """Run the router and all baselines on the same query set.

    Returns {budget: {method: RunResult}}; the same queries (same
    thresholds) are used for every method, so comparisons are paired.
    """
    queries = make_queries(n_queries, seed)
    methods = ("meterrouter", "random_tier", "strongest",
               "single_high", "single_low", "fixed")
    out: dict = {}
    for budget in budgets:
        out[budget] = {}
        for method in methods:
            out[budget][method] = _run_method(method, queries, budget, seed)
    return out


def pareto_points(results: dict) -> list[tuple[float, float, str]]:
    """(mean_cost, accuracy, label) points for the frontier plot."""
    points = []
    for budget, methods in results.items():
        r = methods["meterrouter"]
        points.append((r.mean_cost, r.accuracy,
                       f"meterrouter B={budget:g}"))
    for label, method in (("strongest", "strongest"),
                          ("single_high", "single_high"),
                          ("single_low", "single_low"),
                          ("fixed", "fixed")):
        r = results[max(results)][method]
        points.append((r.mean_cost, r.accuracy, label))
    return points
