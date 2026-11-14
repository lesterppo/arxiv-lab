#!/usr/bin/env python3
"""ReSHAPE: revising a planning agent's search space (arXiv:2611.16466).

Paper claim: planning agents treat a user's initial specification as fixed,
yet requests exclude plans users would gladly take. The ReSHAPE task judges
a revision by whether its search space matches the acceptable plans and a
planner finds one there. On 5,800 agent-revised specifications (travel and
shopping, ReSHAPE Planter with a human-validated simulated user): one
correct trade-off makes the same planner reach an acceptable plan up to
4.6x as often; search-space quality predicts what the planner finds;
current agents admit at most 32% of the acceptable plans, where that
trade-off alone admits over 52%, because they ask about preferences instead
of showing plans; in travel, showing plans raises F1 by 9.7 points over
asking, with two model families. Lesson: treat a request as a hypothesis to
test against the user's reactions to concrete plans.

Miniature: synthetic travel / shopping domains. Each instance has a finite
plan universe (108 plans over 5 attributes), a hidden user utility with ONE
trade-off (a sacrificed dimension compensated by other attributes at their
best), an acceptable set = the top 22 plans by that utility, and an initial
spec S0 that constrains 3 attributes to non-worst levels (so it excludes
the trade-off plans). Three revision strategies are compared on the SAME
instances with per-(instance, strategy) RNG streams:
  - "ask"    : the agent asks about each constrained attribute in the
               abstract. The user answers from *stated* (marginal)
               preferences: flexibility falls with the attribute's weight,
               so the dimension the user cares most about -- the trade-off's
               sacrifice dimension -- is the least likely to be relaxed.
               The agent relaxes the single most-flexible dimension,
               unconditionally.
  - "show"   : the agent shows concrete candidate plans drawn from the
               single-relaxation neighbourhood of S0; the user accepts /
               rejects each by the hidden utility. The agent fits the
               revision that best explains those labels: for each candidate
               (relax dimension d, optionally requiring a condition on
               other attributes) it scores F1 on the shown labels and picks
               the best. Concrete reactions can reveal the interaction that
               abstract questions cannot.
  - "oracle" : the one correct trade-off (paper reference arm): S0 plus
               exactly the plans satisfying the hidden trade-off condition.
  - "none"   : no revision (S0 itself), the unrevised baseline.
A sampling planner draws K plans uniformly without replacement from the
revised space and succeeds iff any drawn plan is acceptable.
Stdlib only, deterministic (SplitMix64, same construction as the
2026-11-13 termination-lag session so the Rust stage reproduces the
exact streams).
"""

import itertools
import math

# ---------------------------------------------------------------- RNG
MASK64 = (1 << 64) - 1


class RNG:
    """SplitMix64. `next_u64` returns the next 64-bit draw."""

    def __init__(self, seed):
        self.state = seed & MASK64

    def next_u64(self):
        self.state = (self.state + 0x9E3779B97F4A7C15) & MASK64
        z = self.state
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & MASK64
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & MASK64
        return z ^ (z >> 31)

    def uniform(self):
        # 53-bit precision float in [0, 1)
        return (self.next_u64() >> 11) * (1.0 / (1 << 53))

    def bernoulli(self, p):
        return self.uniform() < p

    def randint(self, lo, hi):
        # inclusive lo, exclusive hi
        return lo + (self.next_u64() % (hi - lo))

    def shuffle(self, xs):
        # Fisher-Yates, in place; same draw order as the Rust port.
        xs = list(xs)
        for i in range(len(xs) - 1, 0, -1):
            j = self.randint(0, i + 1)
            xs[i], xs[j] = xs[j], xs[i]
        return xs

    def sample_without_replacement(self, xs, k):
        # Partial Fisher-Yates: the first k entries of a shuffled copy.
        xs = list(xs)
        k = min(k, len(xs))
        for i in range(k):
            j = i + self.randint(0, len(xs) - i)
            xs[i], xs[j] = xs[j], xs[i]
        return xs[:k]


# ---------------------------------------------------------------- domains
# Levels are ordered best-first; `values` give the per-level utility value.
# A stated constraint is "value >= 1" (exclude the worst level).
DOMAINS = {
    "travel": {
        "attrs": ["price", "duration", "stops", "pace", "lodging"],
        "levels": [3, 3, 2, 2, 3],
        "values": [[2, 1, 0], [2, 1, 0], [1, 0], [1, 0], [2, 1, 0]],
    },
    "shopping": {
        "attrs": ["price", "brand", "delivery", "quality", "warranty"],
        "levels": [3, 3, 2, 3, 2],
        "values": [[2, 1, 0], [2, 1, 0], [1, 0], [2, 1, 0], [1, 0]],
    },
}

N_CONSTRAINED = 3   # attributes the initial spec constrains
N_ACCEPTABLE = 22   # |A| per instance (top ~20% of the 108-plan universe)
BONUS_FRAC = 0.25   # trade-off bonus as a fraction of max base utility
                    # (calibrated so S0 admits ~1/4 of A and the oracle
                    # trade-off admits ~2/3; see results.json)
PLANNER_K = 2       # plans the planner samples from the revised space
                    # (K=2: budgeted planner; K=4 saturates, see test A10)
SHOW_K = 8          # concrete plans shown to the user per instance


def universe(domain):
    """All plans as tuples of level indices, lexicographic order."""
    spec = DOMAINS[domain]
    return list(itertools.product(*(range(n) for n in spec["levels"])))


def _value(domain, attr, level):
    return DOMAINS[domain]["values"][attr][level]


def _max_value(domain, attr):
    return DOMAINS[domain]["values"][attr][0]


# ---------------------------------------------------------------- instances
class Instance:
    __slots__ = ("idx", "domain", "plans", "weights", "constrained",
                 "sacrifice", "compensating", "bonus", "acceptable",
                 "spec0")

    def __init__(self, idx, domain, plans, weights, constrained,
                 sacrifice, compensating, bonus, acceptable, spec0):
        self.idx = idx
        self.domain = domain
        self.plans = plans
        self.weights = weights          # normalised, sum 1, per attribute
        self.constrained = constrained  # sorted attr indices in S0
        self.sacrifice = sacrifice      # trade-off's sacrificed attribute
        self.compensating = compensating  # sorted attr indices at best
        self.bonus = bonus
        self.acceptable = acceptable    # frozenset of plan indices
        self.spec0 = spec0              # frozenset of plan indices

    def tradeoff_holds(self, plan):
        """The hidden trade-off: worst level on the sacrifice dimension,
        best level on every compensating attribute."""
        if plan[self.sacrifice] != DOMAINS[self.domain]["levels"][self.sacrifice] - 1:
            return False
        return all(plan[a] == 0 for a in self.compensating)

    def utility(self, plan):
        u = 0.0
        for d in range(len(plan)):
            u += self.weights[d] * _value(self.domain, d, plan[d])
        if self.tradeoff_holds(plan):
            u += self.bonus
        return u

    def in_spec0(self, plan):
        return all(_value(self.domain, d, plan[d]) >= 1
                   for d in self.constrained)


def gen_instances(domain, n, seed):
    """Generate `n` instances. One master stream, in order: weights,
    constrained set, sacrifice (max-weight constrained attr), compensating
    set, then the utility ranking that fixes the acceptable set."""
    rng = RNG(seed)
    plans = universe(domain)
    d = len(DOMAINS[domain]["attrs"])
    instances = []
    for idx in range(n):
        raw = [0.5 + rng.uniform() for _ in range(d)]
        total = sum(raw)
        weights = [w / total for w in raw]
        constrained = sorted(rng.shuffle(list(range(d)))[:N_CONSTRAINED])
        # The trade-off sacrifices the constrained attribute the user
        # cares most about (highest weight): that is the dimension they
        # state as a hard constraint yet will trade in concrete cases.
        sacrifice = max(constrained, key=lambda a: (weights[a], -a))
        others = [a for a in range(d) if a != sacrifice]
        n_comp = 1 + rng.randint(0, 2)
        compensating = sorted(rng.shuffle(others)[:n_comp])
        u_max = sum(weights[a] * _max_value(domain, a) for a in range(d))
        bonus = BONUS_FRAC * u_max
        inst = Instance(idx, domain, plans, weights, constrained,
                        sacrifice, compensating, bonus, frozenset(),
                        frozenset())
        utils = [(inst.utility(p), i) for i, p in enumerate(plans)]
        # Rank by utility desc, plan index asc: exact, tie-safe.
        utils.sort(key=lambda t: (-t[0], t[1]))
        acceptable = frozenset(i for _, i in utils[:N_ACCEPTABLE])
        spec0 = frozenset(i for i, p in enumerate(plans)
                          if inst.in_spec0(p))
        inst.acceptable = acceptable
        inst.spec0 = spec0
        instances.append(inst)
    return instances


# ---------------------------------------------------------------- revision strategies
def _flexibility(weight):
    """Stated flexibility for an attribute, from its marginal weight:
    the more the user cares about a dimension in the abstract, the less
    flexible they say they are about it. Bounded to [0.05, 0.65]."""
    return max(0.05, min(0.65, 0.65 - 1.5 * weight))


def revise_ask(inst, rng, n_questions=None):
    """ASK: ask about constrained attributes in the abstract; relax the
    single most-flexible one, unconditionally. `n_questions` caps how many
    attributes are asked about (sensitivity: asking more questions)."""
    asked = inst.constrained if n_questions is None else \
        inst.constrained[:n_questions]
    flexible = []
    for a in asked:
        if rng.bernoulli(_flexibility(inst.weights[a])):
            flexible.append(a)
    if not flexible:
        return inst.spec0
    # Relax the dimension the user called most flexible (ties: first asked).
    relax = max(flexible, key=lambda a: (_flexibility(inst.weights[a]),
                                         -inst.constrained.index(a)))
    keep = [a for a in inst.constrained if a != relax]
    return frozenset(i for i, p in enumerate(inst.plans)
                     if all(_value(inst.domain, a, p[a]) >= 1 for a in keep))


def _candidate_conditions(inst, d):
    """Candidate compensating conditions for relaxing dimension d:
    none, one other attribute at its best level, or a pair at best."""
    others = [a for a in range(len(inst.plans[0])) if a != d]
    conds = [()]
    for a in others:
        conds.append((a,))
    for i, a in enumerate(others):
        for b in others[i + 1:]:
            conds.append((a, b))
    return conds


def revise_show(inst, rng, k_show=SHOW_K):
    """SHOW: show concrete plans from the single-relaxation neighbourhood
    of S0, learn accept/reject from the hidden utility, and fit the
    revision (relax d, optionally requiring a condition) that best
    explains those labels by F1. Returns the revised space."""
    # Neighbourhood: plans violating exactly one S0 constraint.
    neighbourhood = []
    for i, p in enumerate(inst.plans):
        bad = [a for a in inst.constrained
               if _value(inst.domain, a, p[a]) < 1]
        if len(bad) == 1:
            neighbourhood.append(i)
    shown = rng.sample_without_replacement(neighbourhood, k_show)
    labels = {i: (i in inst.acceptable) for i in shown}
    spec0 = inst.spec0

    def score(space):
        tp = sum(1 for i in shown if labels[i] and i in space)
        fp = sum(1 for i in shown if not labels[i] and i in space)
        fn = sum(1 for i in shown if labels[i] and i not in space)
        if tp == 0:
            return 0.0, 0.0
        prec = tp / (tp + fp)
        rec = tp / (tp + fn)
        return 2 * prec * rec / (prec + rec), prec

    best_space, best_key = spec0, (0.0, 0.0, 0)
    # Enumerate candidates in a fixed order; tie-break on F1, then
    # precision, then smaller space, then earlier enumeration.
    for d in inst.constrained:
        for cond in _candidate_conditions(inst, d):
            region = frozenset(
                i for i, p in enumerate(inst.plans)
                if _value(inst.domain, d, p[d]) < 1
                and all(p[a] == 0 for a in cond))
            space = spec0 | region
            f1, prec = score(space)
            key = (f1, prec, -len(space))
            if key > best_key:
                best_key, best_space = key, space
    return best_space


def revise_oracle(inst):
    """ORACLE: the one correct trade-off — S0 plus exactly the plans
    satisfying the hidden trade-off condition (paper reference arm)."""
    region = frozenset(i for i, p in enumerate(inst.plans)
                       if inst.tradeoff_holds(p))
    return inst.spec0 | region


STRATEGIES = ("none", "ask", "show", "oracle")


def revise(inst, strategy, rng, **kw):
    if strategy == "none":
        return inst.spec0
    if strategy == "ask":
        return revise_ask(inst, rng, **kw)
    if strategy == "show":
        return revise_show(inst, rng, **kw)
    if strategy == "oracle":
        return revise_oracle(inst)
    raise ValueError(f"unknown strategy {strategy!r}")


# ---------------------------------------------------------------- planner + metrics
def plan(inst, space, rng, k=PLANNER_K):
    """Sampling planner: draw k plans without replacement from the revised
    space; succeed iff any drawn plan is acceptable."""
    if not space:
        return False
    drawn = rng.sample_without_replacement(sorted(space), k)
    return any(i in inst.acceptable for i in drawn)


def space_metrics(inst, space):
    """Admission / precision / F1 of a revised space vs the acceptable set."""
    inter = len(space & inst.acceptable)
    admission = inter / len(inst.acceptable) if inst.acceptable else 0.0
    precision = inter / len(space) if space else 0.0
    f1 = (2 * precision * admission / (precision + admission)
          if precision + admission > 0 else 0.0)
    return {"admission": admission, "precision": precision, "f1": f1,
            "n_space": len(space), "n_acceptable": len(inst.acceptable),
            "n_admitted": inter}


def _strategy_seed(seed, idx, strategy):
    salt = {"none": 0x11, "ask": 0x22, "show": 0x33, "oracle": 0x44}[strategy]
    return (seed ^ (idx * 0x9E3779B97F4A7C15) ^ salt) & MASK64


def run_experiment(domain, instances, seed, planner_k=PLANNER_K,
                   show_k=SHOW_K, ask_q=None):
    """Run every strategy on the same instances. Per-(instance, strategy)
    RNG streams make the comparison paired and order-independent."""
    out = {}
    for strategy in STRATEGIES:
        admits, precs, f1s, succs, sizes = [], [], [], [], []
        for inst in instances:
            rng = RNG(_strategy_seed(seed, inst.idx, strategy))
            space = revise(inst, strategy, rng,
                           **({"k_show": show_k} if strategy == "show" else
                              {"n_questions": ask_q} if strategy == "ask"
                              else {}))
            m = space_metrics(inst, space)
            admits.append(m["admission"])
            precs.append(m["precision"])
            f1s.append(m["f1"])
            sizes.append(m["n_space"])
            succs.append(1.0 if plan(inst, space, rng, k=planner_k) else 0.0)
        n = len(instances)
        out[strategy] = {
            "strategy": strategy,
            "n_instances": n,
            "admission": sum(admits) / n,
            "precision": sum(precs) / n,
            "f1": sum(f1s) / n,
            "planner_success": sum(succs) / n,
            "mean_space_size": sum(sizes) / n,
        }
    return out


def pearson(xs, ys):
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    vx = sum((x - mx) ** 2 for x in xs)
    vy = sum((y - my) ** 2 for y in ys)
    if vx == 0 or vy == 0:
        return 0.0
    return cov / math.sqrt(vx * vy)
