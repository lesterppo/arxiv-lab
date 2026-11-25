"""CAPE: Coordinating Calibration and Search for Tool-Use Policies.

arXiv:2611.31938 "CAPE: Coordinating Calibration and Search for
Tool-Use Policies" (Wang, Chen, Yang, Wang).

Core claim implemented here: compiling a tool-use policy for a tool whose
parsing configuration the agent cannot inspect requires both *calibration*
(evaluating candidate calls on candidate configurations) and *search* over
the evaluated calls, sharing one compilation budget. CAPE keeps an
executable policy at all times, directs calibration to the calls that can
improve it, tightens bounds only where doing so ends calibration early,
and certifies optimality when its executable (lower) and optimistic
(upper) values meet at the root.

Formal model (paper Sec. 3.1, 4.1-4.3), reproduced faithfully:

* Configurations H: combinations of parsing aspects (delimiter, date
  order, quoting, missing-value marker, ...). Programs P are
  serializations; a program commits on h iff its encoding matches h on
  every aspect, else it returns an error naming the first mismatching
  aspect. Probes Q are diagnostic fixtures that reveal aspects without
  committing.
* A policy chooses each action from the responses so far, using at most
  C calls per execution. Objective J(pi) = (|S(pi)|, -M(pi)) in
  lexicographic order (completions first, then fewest calls), subject to
  R0 subset S(pi), where R0 is what the initial policy pi0 (programs in
  catalog order) completes.
* A closed action (effects known on every configuration in V) composes
  per Eq. (3): S_a = G_a(V) u union_o S_o, M_a = |V| + sum_o M_o.
* Optimistic bound B(V, b) (Eq. 5) and per-action upper value U_a (Eq. 6)
  with optimistic cells; two-call tightening T2 (Eq. 7). Acquisition
  follows a three-tier frontier: required coverage, then completion
  gain, then call savings, each ranked per missing effect.
* Proposition 1: every accepted root witness preserves R0, witness
  values are nondecreasing, and L == U at the root certifies optimality
  over the full action catalog.

This module implements the compiler, an exact oracle (full-knowledge DP)
for the independently computed optimum, and deterministic baselines
(eager exact, program-only, error-feedback reviser). Pure stdlib,
deterministic given a seed. No model, no network.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

__all__ = [
    "Aspect",
    "World",
    "make_world",
    "EffectTable",
    "compile_policy",
    "exact_optimum",
    "execute_policy",
    "program_only_policy",
    "error_feedback_policy",
    "initial_policy",
    "PAPER_OPTIMAL_RATE",
    "PAPER_TIME_SAVING",
    "PAPER_COMPLETION_GAIN_PP",
]

# Paper headline numbers (abstract / Table 1), for reference in tests.
PAPER_OPTIMAL_RATE = 502 / 504  # CAPE reaches the optimum in 502/504
PAPER_TIME_SAVING = 0.220  # 22.0% less compilation time than eager exact
PAPER_COMPLETION_GAIN_PP = (15.38, 34.66)  # vs LLM error-feedback reviser


# ---------------------------------------------------------------------------
# World model
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Aspect:
    """One parsing aspect of a configuration, e.g. delimiter."""

    name: str
    values: tuple


# Default aspect set: 4 aspects -> 3*2*2*2 = 24 configurations.
DEFAULT_ASPECTS = (
    Aspect("delim", (",", ";", "\t")),
    Aspect("date", ("DMY", "MDY")),
    Aspect("quote", ("minimal", "all")),
    Aspect("missing", ("empty", "NA")),
)

# Extended aspect set: 6 aspects -> 96 configurations.
EXTENDED_ASPECTS = DEFAULT_ASPECTS + (
    Aspect("numeric", ("dot", "comma")),
    Aspect("header", ("yes", "no")),
)


@dataclass
class World:
    """A synthetic ingestion tool: configurations, programs, probes.

    A configuration is a tuple of aspect values. A program is an encoding
    tuple; it commits iff it matches the configuration on every aspect.
    A probe is (primary_aspect, primary_value, reveal_aspect): it submits
    a fixture in `primary_value` encoding that also carries an ambiguous
    date/value for `reveal_aspect`. If the configuration's primary aspect
    differs, the fixture fails to parse; otherwise the response reveals
    the configuration's reveal-aspect value.
    """

    aspects: tuple
    configs: list  # list of config tuples
    programs: list  # list of encoding tuples
    probes: list  # list of (primary_idx, primary_value, reveal_idx)

    def effect(self, action, h):
        """f(a, h) -> (response, commit). Deterministic."""
        kind, payload = action
        if kind == "prog":
            enc = payload
            for i, v in enumerate(enc):
                if h[i] != v:
                    return ("err:%s" % self.aspects[i].name, False)
            return ("ok", True)
        # probe
        pi, pval, ri = payload
        if h[pi] != pval:
            return ("err:%s" % self.aspects[pi].name, False)
        return ("%s:%s" % (self.aspects[ri].name, h[ri]), False)

    def actions(self):
        """Full catalog A = P u Q, programs first (catalog order)."""
        acts = [("prog", p) for p in self.programs]
        acts += [("probe", q) for q in self.probes]
        return acts


def make_world(n_configs=24, n_aspects=4, seed=0):
    """Build a world with `n_configs` sampled from the aspect product.

    Programs are all encodings of the aspect product; probes cover every
    ordered pair (primary, reveal) of distinct aspects and every value
    of the primary aspect.
    """
    aspects = DEFAULT_ASPECTS if n_aspects == 4 else EXTENDED_ASPECTS
    import itertools

    product = list(itertools.product(*[a.values for a in aspects]))
    rng = random.Random(seed)
    if n_configs >= len(product):
        configs = list(product)
    else:
        configs = sorted(rng.sample(product, n_configs))
    probes = []
    for pi in range(len(aspects)):
        for ri in range(len(aspects)):
            if ri == pi:
                continue
            for v in aspects[pi].values:
                probes.append((pi, v, ri))
    return World(aspects, configs, list(product), probes)


# ---------------------------------------------------------------------------
# Effect table (partial knowledge)
# ---------------------------------------------------------------------------

class EffectTable:
    """Acquired effects D subset of A x H."""

    def __init__(self, world):
        self.world = world
        self.known = {}  # (action, h) -> (response, commit)

    def acquire(self, action, h):
        eff = self.world.effect(action, h)
        self.known[(action, h)] = eff
        return eff

    def is_known(self, action, h):
        return (action, h) in self.known

    def missing(self, action, V):
        return sorted(h for h in V if (action, h) not in self.known)

    def closed(self, action, V):
        return all((action, h) in self.known for h in V)

    def complete(self):
        return len(self.known) == len(self.world.actions()) * len(
            self.world.configs
        )


# ---------------------------------------------------------------------------
# Policies as trees; execution
# ---------------------------------------------------------------------------
# A policy tree node is ("stop",) or ("act", action, {response: subtree}).

def execute_policy(world, tree, h, cap):
    """Simulate one execution of `tree` on configuration h.

    Returns (completed: bool, calls: int).
    """
    node = tree
    calls = 0
    while True:
        if node[0] == "stop":
            return (False, calls)
        _, action, children = node
        if calls >= cap:
            return (False, calls)
        resp, commit = world.effect(action, h)
        calls += 1
        if commit:
            return (True, calls)
        nxt = children.get(resp)
        if nxt is None:
            return (False, calls)
        node = nxt


def evaluate_policy(world, tree, cap):
    """Run the policy on every configuration.

    Returns (completed_set, total_calls).
    """
    done = set()
    total = 0
    for h in world.configs:
        ok, calls = execute_policy(world, tree, h, cap)
        total += calls
        if ok:
            done.add(h)
    return done, total


# ---------------------------------------------------------------------------
# Exact oracle: full-knowledge DP over belief states (V, b)
# ---------------------------------------------------------------------------

def _partition(world, table, action, V):
    """Partition V by the (known) response of `action`.

    Returns (committed_set, {response: cell}). Only valid when the action
    is closed on V (or the table is complete). Iterates V sorted: V is a
    frozenset and its iteration order depends on PYTHONHASHSEED, so
    unsorted iteration made acquisition order (and query counts)
    nondeterministic across processes (found 2026-11-25: same world gave
    596 vs 578 queries). Sorting makes the compiler deterministic given
    a seed, as the module docstring promises.
    """
    committed = set()
    cells = {}
    for h in sorted(V):
        resp, commit = table.known[(action, h)]
        if commit:
            committed.add(h)
        else:
            cells.setdefault(resp, set()).add(h)
    return committed, cells


def exact_optimum(world, cap, actions=None, table=None):
    """Optimal (completions, -calls) policy value under full knowledge.

    DP over belief states (V, b) with the R0 coverage constraint, where
    R0 comes from the initial policy. Returns (value, tree) with value =
    (n_completed, -total_calls), or (None, None) if infeasible (cannot
    happen at the root: pi0 itself is feasible).
    """
    if actions is None:
        actions = world.actions()
    if table is None:
        table = EffectTable(world)
        for a in actions:
            for h in world.configs:
                table.acquire(a, h)
    r0, _ = evaluate_policy(world, initial_policy(world, cap), cap)
    memo = {}

    def solve(V, b):
        key = (frozenset(V), b)
        if key in memo:
            return memo[key]
        V = frozenset(V)
        oblig = r0 & set(V)
        best = None  # (value, tree)
        if not oblig:
            best = ((0, 0), ("stop",))
        if b >= 1:
            for a in actions:
                if not table.closed(a, V):
                    continue
                committed, cells = _partition(world, table, a, V)
                # Feasibility: every obligated config must be completed
                # either by this action or in a child cell.
                child_results = {}
                feasible = True
                comp = len(committed)
                calls = len(V)
                for resp, cell in cells.items():
                    sub = solve(cell, b - 1)
                    if sub[0] is None:
                        feasible = False
                        break
                    child_results[resp] = sub
                    comp += sub[0][0]
                    calls += -sub[0][1]
                if not feasible:
                    continue
                # Obligated configs in the committed set are covered;
                # obligated configs in cells are covered iff the child
                # policy covers them (solve() enforces it: it returns
                # None when its own obligation is uncovered).
                if not oblig <= (committed | set().union(
                    *[set(c) for c in cells.values()]
                )):
                    continue
                # Child feasibility already guarantees coverage inside
                # each cell; but a child that returns (0,0) via stop is
                # only feasible when its obligation is empty.
                val = (comp, -calls)
                tree = ("act", a, {r: s[1] for r, s in child_results.items()})
                if best is None or val > best[0]:
                    best = (val, tree)
        memo[key] = best if best is not None else (None, None)
        return memo[key]

    return solve(frozenset(world.configs), cap)


def initial_policy(world, cap):
    """pi0: try the programs in catalog order, up to `cap` calls.

    Each program's failure continuation (whatever the error) moves to
    the next program in catalog order.
    """
    progs = [("prog", p) for p in world.programs]
    node = ("stop",)
    for p in reversed(progs[:cap]):
        children = {}
        for i in range(len(world.aspects)):
            children["err:%s" % world.aspects[i].name] = node
        node = ("act", p, children)
    return node


# ---------------------------------------------------------------------------
# CAPE compiler
# ---------------------------------------------------------------------------

@dataclass
class CapeResult:
    tree: object
    value: tuple  # (n_completed, -total_calls)
    queries: int
    certified: bool
    completed: set
    total_calls: int
    r0: set
    upper: tuple  # root upper value at stop
    acquisitions: list = field(default_factory=list)
    # ordered (action, config) pairs acquired, for audit / parity tracing


def _optimistic_cells(world, table, action, V):
    """Optimistic response cells F_a^+ (paper Sec. 4.2).

    Unknown configurations of a program are assumed to commit; unknown
    configurations of a probe are assumed to return a distinct response
    each (finest partition). Known failures keep actual response cells.
    Returns (optimistic_commit_set, {response: cell}).
    """
    kind, payload = action
    committed = set()
    cells = {}
    unknown = []
    for h in sorted(V):
        if (action, h) in table.known:
            resp, commit = table.known[(action, h)]
            if commit:
                committed.add(h)
            else:
                cells.setdefault(resp, set()).add(h)
        else:
            unknown.append(h)
    if kind == "prog":
        committed |= set(unknown)
    else:
        for h in unknown:
            cells[("?", h)] = {h}
    return committed, cells


class _Cape:
    def __init__(self, world, cap, budget):
        self.world = world
        self.cap = cap
        self.budget = budget
        self.table = EffectTable(world)
        self.queries = 0
        self.r0, _ = evaluate_policy(world, initial_policy(world, cap), cap)
        # Round cache: bounds depend on the effect table, so caches are
        # valid only within one acquisition round (cleared per query).
        self._rcache = {}
        self._validate_initial()

    # -- initial validation: acquire pi0's execution effects, seed witness
    def _validate_initial(self):
        progs = [("prog", p) for p in self.world.programs[: self.cap]]
        remaining = list(self.world.configs)
        for p in progs:
            for h in list(remaining):
                self.table.acquire(p, h)
                self.queries += 1
                resp, commit = self.table.known[(p, h)]
                if commit:
                    remaining.remove(h)

    # -- witness: best feasible composition from closed actions only
    def witness(self, V, b, memo):
        key = (frozenset(V), b)
        if key in memo:
            return memo[key]
        V = frozenset(V)
        oblig = self.r0 & set(V)
        best = None
        if not oblig:
            best = ((0, 0), ("stop",))
        if b >= 1:
            for a in self.world.actions():
                if not self.table.closed(a, V):
                    continue
                committed, cells = _partition(self.world, self.table, a, V)
                child_results = {}
                feasible = True
                comp = len(committed)
                calls = len(V)
                for resp, cell in cells.items():
                    sub = self.witness(cell, b - 1, memo)
                    if sub[0] is None:
                        feasible = False
                        break
                    child_results[resp] = sub
                    comp += sub[0][0]
                    calls += -sub[0][1]
                if not feasible:
                    continue
                if not oblig <= committed | set().union(
                    *[set(c) for c in cells.values()]
                ):
                    continue
                val = (comp, -calls)
                tree = ("act", a, {r: s[1] for r, s in child_results.items()})
                if best is None or val > best[0]:
                    best = (val, tree)
        memo[key] = best if best is not None else (None, None)
        return memo[key]

    # -- optimistic bounds (paper Sec. 4.2-4.3)
    def _bound_B(self, V, b):
        """Cheap upper value B(V, b), Eq. (5); b == 1 special-cased."""
        V = frozenset(V)
        key = ("B", V, b)
        if key in self._rcache:
            return self._rcache[key]
        Vset = set(V)
        oblig = self.r0 & Vset
        gbar = {}
        for a in self.world.actions():
            if a[0] != "prog":
                continue
            s = set()
            for h in Vset:
                if (a, h) in self.table.known:
                    resp, commit = self.table.known[(a, h)]
                    if commit:
                        s.add(h)
                else:
                    s.add(h)
            gbar[a] = s
        union = set().union(*gbar.values()) if gbar else set()
        if not oblig <= union:
            self._rcache[key] = None  # infeasible
            return None
        if not union:
            self._rcache[key] = (0, 0)
            return (0, 0)
        dbar = max(len(s) for s in gbar.values())
        if b <= 1:
            val = (dbar, -len(Vset))
        else:
            val = (len(union), -(len(Vset) + len(union) - dbar))
        self._rcache[key] = val
        return val

    def _action_upper(self, a, V, b, use_t2_children=False):
        """Upper value U_a(V, b), Eq. (6), over optimistic cells."""
        V = frozenset(V)
        key = ("Ua", a, V, b)
        if key in self._rcache:
            return self._rcache[key]
        Vset = set(V)
        committed, cells = _optimistic_cells(self.world, self.table, a, V)
        oblig = self.r0 & Vset
        if not oblig <= committed | set().union(
            *[set(c) for c in cells.values()]
        ):
            self._rcache[key] = None
            return None
        comp = len(committed)
        negcalls = -len(Vset)
        for cell in cells.values():
            if b - 1 <= 0:
                sub = (0, 0)  # no calls left: cell completes nothing
            else:
                sub = self._upper(cell, b - 1, use_t2_children)
            if sub is None:
                self._rcache[key] = None
                return None
            comp += sub[0]
            negcalls += sub[1]
        val = (comp, negcalls)
        self._rcache[key] = val
        return val

    def _upper(self, V, b, use_t2_children=False):
        """Upper value U(V, b): min(B, max_a U_a); b == 2 uses T2."""
        V = frozenset(V)
        key = ("U", V, b)
        if key in self._rcache:
            return self._rcache[key]
        B = self._bound_B(V, b)
        if B is None:
            self._rcache[key] = None
            return None
        if b <= 1:
            self._rcache[key] = B
            return B
        if b == 2:
            val = self._tighten2(V)
            self._rcache[key] = val
            return val
        best = None
        for a in self.world.actions():
            ua = self._action_upper(a, V, b, use_t2_children)
            if ua is None:
                continue
            if best is None or ua > best:
                best = ua
        if best is None:
            self._rcache[key] = B
            return B
        val = min(B, best)
        self._rcache[key] = val
        return val

    def _tighten2(self, V):
        """T2(V): exact optimum of the relaxed two-call problem, Eq. (7)."""
        V = frozenset(V)
        key = ("T2", V)
        if key in self._rcache:
            return self._rcache[key]
        Vset = set(V)
        oblig = self.r0 & Vset
        B = self._bound_B(V, 2)
        if B is None:
            self._rcache[key] = None
            return None
        best = (0, 0) if not oblig else None
        for a in self.world.actions():
            committed, cells = _optimistic_cells(self.world, self.table, a, V)
            if not oblig <= committed | set().union(
                *[set(c) for c in cells.values()]
            ):
                continue
            comp = len(committed)
            negcalls = -len(Vset)
            ok = True
            for cell in cells.values():
                sub = self._bound_B(cell, 1)
                if sub is None:
                    ok = False
                    break
                comp += sub[0]
                negcalls += sub[1]
            if not ok:
                continue
            val = (comp, negcalls)
            if best is None or val > best:
                best = val
        if best is None:
            self._rcache[key] = B
            return B
        val = min(B, best)
        self._rcache[key] = val
        return val

    # -- acquisition frontier (paper Sec. 4.2)
    def _frontier_states(self, root_witness_tree):
        """States reached by the current executable policy, plus states
        along one best optimistic alternative from the root."""
        states = {}

        def walk(tree, V, b):
            if b <= 0:
                return
            states[(frozenset(V), b)] = True
            if tree[0] != "act":
                return
            _, a, children = tree
            if not self.table.closed(a, V):
                return
            _, cells = _partition(self.world, self.table, a, V)
            for resp, cell in cells.items():
                if resp in children:
                    walk(children[resp], cell, b - 1)

        walk(
            root_witness_tree,
            frozenset(self.world.configs),
            self.cap,
        )
        # One best optimistic alternative from the root.
        V = frozenset(self.world.configs)
        b = self.cap
        while b >= 2:
            best_a, best_u = None, None
            for a in self.world.actions():
                ua = self._action_upper(a, V, b)
                if ua is not None and (best_u is None or ua > best_u):
                    best_a, best_u = a, ua
            if best_a is None:
                break
            states[(V, b)] = True
            committed, cells = _optimistic_cells(
                self.world, self.table, best_a, V
            )
            # Follow the largest optimistic cell (lazy expansion).
            if not cells:
                break
            V = frozenset(max(cells.values(), key=len))
            b -= 1
            states[(V, b)] = True
        return list(states.keys())

    def _ranked_candidates(self, frontier):
        """Three-tier ranking of (action, state) acquisitions."""
        cands = []  # (tier, score, action, V, b)
        wit_memo = {}
        for (V, b) in frontier:
            wit = self.witness(V, b, wit_memo)
            wit_val = wit[0] if wit[0] is not None else None
            for a in self.world.actions():
                missing = self.table.missing(a, V)
                if not missing:
                    continue
                ua = self._action_upper(a, V, b)
                if ua is None:
                    continue
                if wit_val is None:
                    # Tier 1: required coverage.
                    score = len(self.r0 & set(V)) / len(missing)
                    cands.append((1, score, a, V, b))
                elif ua[0] > wit_val[0]:
                    # Tier 2: completion gain per missing effect.
                    score = (ua[0] - wit_val[0]) / len(missing)
                    cands.append((2, score, a, V, b))
                elif ua[0] == wit_val[0] and ua[1] > wit_val[1]:
                    # Tier 3: call savings per missing effect.
                    score = (ua[1] - wit_val[1]) / len(missing)
                    cands.append((3, score, a, V, b))
        cands.sort(key=lambda c: (c[0], -c[1]))
        return cands

    def run(self):
        """Compile; return a CapeResult."""
        acquisitions = []
        while self.queries < self.budget:
            self._rcache = {}  # effect table changed: bounds are stale
            memo = {}
            root_wit = self.witness(
                frozenset(self.world.configs), self.cap, memo
            )
            root_tree = root_wit[1] if root_wit[0] is not None else None
            if root_tree is None:
                # No feasible witness yet (cannot happen after pi0
                # validation, but guard anyway): acquire pi0 effects.
                break
            upper = self._upper(frozenset(self.world.configs), self.cap)
            if upper is not None and root_wit[0] == upper:
                break  # certified optimal
            frontier = self._frontier_states(root_tree)
            cands = self._ranked_candidates(frontier)
            if not cands:
                break
            _, _, action, V, b = cands[0]
            # Within the selected action, query required configs first.
            missing = self.table.missing(action, V)
            required = [h for h in missing if h in self.r0]
            target = required[0] if required else missing[0]
            self.table.acquire(action, target)
            self.queries += 1
            acquisitions.append((action, target))
            # Stop acquiring this action when it is closed, infeasible,
            # or bounded by the available witness (checked next round
            # via re-ranking: a bounded action drops out of tiers 2-3
            # because U_a <= witness value).
        memo = {}
        self._rcache = {}  # table changed since the last round
        root_wit = self.witness(frozenset(self.world.configs), self.cap, memo)
        upper = self._upper(frozenset(self.world.configs), self.cap)
        certified = upper is not None and root_wit[0] == upper
        if self.table.complete():
            # Effect table complete: exact-solve on the remainder.
            val, tree = exact_optimum(
                self.world, self.cap, table=self.table
            )
            if val is not None:
                root_wit = (val, tree)
                certified = True
        completed, total_calls = evaluate_policy(
            self.world, root_wit[1], self.cap
        )
        return CapeResult(
            tree=root_wit[1],
            value=root_wit[0],
            queries=self.queries,
            certified=certified,
            completed=completed,
            total_calls=total_calls,
            r0=self.r0,
            upper=upper,
            acquisitions=acquisitions,
        )


def compile_policy(world, cap=3, budget=400):
    """Compile a tool-use policy with CAPE (paper Sec. 4)."""
    return _Cape(world, cap, budget).run()


# ---------------------------------------------------------------------------
# Baselines
# ---------------------------------------------------------------------------

def program_only_policy(world, cap):
    """Best program-only policy (no probes) under full knowledge."""
    progs = [("prog", p) for p in world.programs]
    val, tree = exact_optimum(world, cap, actions=progs)
    return val, tree


def error_feedback_policy(world, cap, seed=0):
    """Deterministic proxy for the paper's LLM error-feedback reviser.

    Starts from a default encoding; on an error naming aspect X, tries
    the next value of X (round-robin); on a commit, stops. One policy
    tree per execution path is not possible for an adaptive reviser, so
    this is simulated per configuration and reported as completions /
    calls directly. NOT an LLM: labelled a proxy in the session README.
    """
    rng = random.Random(seed)
    default = tuple(a.values[0] for a in world.aspects)
    completed = set()
    total_calls = 0
    for h in world.configs:
        enc = list(default)
        ok = False
        for _ in range(cap):
            total_calls += 1
            resp, commit = world.effect(("prog", tuple(enc)), h)
            if commit:
                ok = True
                break
            if resp.startswith("err:"):
                aspect = resp[4:]
                idx = next(
                    i for i, a in enumerate(world.aspects)
                    if a.name == aspect
                )
                vals = world.aspects[idx].values
                cur = vals.index(enc[idx])
                enc[idx] = vals[(cur + 1) % len(vals)]
            else:
                break
        if ok:
            completed.add(h)
    return completed, total_calls
