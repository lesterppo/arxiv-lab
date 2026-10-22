"""Parallel AI agent collaboration as graph-structured distributed block optimization.

Reproduction of the core mechanism of arXiv 2610.30074 ("Parallel AI Agent
Collaboration as Graph-Structured Distributed Block Optimization").

Model
-----
A shared discrete solution ``x in {0,1}^n`` and an exactly evaluable objective::

    f(x) = sum_i c_i * x_i + sum_{(i,j) in E} w_ij * [x_i == x_j]

with a *known dependency graph* ``G = (V, E)``.  Parallel agents propose block
edits (reassignments of subsets of variables).  Each proposal's *verified local
gain* is measured against the same current solution ``x``.

Key mechanism reproduced
------------------------
1. **Independent-set certification**: proposals whose supports are pairwise
   non-adjacent in ``G`` (no shared vertex, no edge between supports) are
   compatible; their verified gains *add exactly* when merged.
2. **Conflict graph** ``H`` over proposals (edge = interacting edits).
3. **Fractional-coloring merger**: an optimal fractional coloring
   ``{(I_k, lambda_k)}`` of ``H`` (``sum_{k: v in I_k} lambda_k >= 1`` for all
   ``v``, minimizing ``sum lambda_k = chi_f(H)``) merged as the max-weight
   independent set preserves at least ``1 / chi_f(H)`` of the total verified
   local improvement; this is tight among mergers that select a subset of the
   same proposals.
4. **Geometric convergence**: under conditional local-improvement guarantees
   (each round's total verified gain ``>= rho * gap``), repeated collaboration
   contracts the objective gap geometrically:
   ``gap_{t+1} <= (1 - rho / chi_f_max) * gap_t``.

The fractional chromatic number is computed exactly on small instances via a
compact two-phase simplex (dense tableau); no external LP solver needed.

Only stdlib + numpy.
"""

import itertools

try:
    import numpy as np
    _HAVE_NUMPY = True
except ImportError:  # pragma: no cover
    np = None
    _HAVE_NUMPY = False


def _need_numpy():
    if not _HAVE_NUMPY:
        raise ImportError("numpy is required for block_opt_merge")

__all__ = [
    "BlockObjective",
    "dependency_adjacency",
    "proposals_conflict",
    "conflict_graph",
    "maximal_independent_sets",
    "fractional_chromatic_number",
    "fractional_coloring",
    "fractional_merger",
    "naive_merge",
    "brute_force_optimum",
    "collaborate",
]


# ---------------------------------------------------------------------------
# Objective and edits
# ---------------------------------------------------------------------------

class BlockObjective:
    """f(x) = sum_i c_i x_i + sum_{(i,j)} w_ij [x_i == x_j], x in {0,1}^n."""

    def __init__(self, unary, edges):
        _need_numpy()
        self.unary = np.asarray(unary, dtype=float)
        self.n = len(self.unary)
        # edges: iterable of (i, j, w); canonicalize i < j, sum duplicate weights
        self.edge = {}
        for i, j, w in edges:
            if i == j:
                continue
            a, b = (i, j) if i < j else (j, i)
            self.edge[(a, b)] = self.edge.get((a, b), 0.0) + float(w)
        self.adj = dependency_adjacency(self.n, list(self.edge))

    def f(self, x):
        _need_numpy()
        x = np.asarray(x, dtype=int)
        val = float(np.dot(self.unary, x))
        for (a, b), w in self.edge.items():
            if x[a] == x[b]:
                val += w
        return val

    def apply(self, x, edit):
        """edit: dict {block_index: new_value (0/1)}."""
        _need_numpy()
        y = np.asarray(x, dtype=int).copy()
        for i, v in edit.items():
            y[i] = v
        return y

    def gain(self, x, edit):
        """Verified local gain of `edit` measured against the current `x`."""
        _need_numpy()
        return self.f(self.apply(x, edit)) - self.f(x)


def dependency_adjacency(n, edges):
    """Adjacency sets of the dependency graph G over n blocks."""
    _need_numpy()
    adj = [set() for _ in range(n)]
    for a, b in edges:
        adj[a].add(b)
        adj[b].add(a)
    return adj


# ---------------------------------------------------------------------------
# Conflict structure over proposals
# ---------------------------------------------------------------------------

def proposals_conflict(e1, e2, adj):
    """Two edits interact iff their supports share a vertex or are joined by
    an edge of the dependency graph (then gains need not add)."""
    _need_numpy()
    s1, s2 = set(e1), set(e2)
    if s1 & s2:
        return True
    for u in s1:
        if adj[u] & s2:
            return True
    return False


def conflict_graph(proposals, adj):
    """Adjacency sets of the conflict graph H over proposal indices."""
    _need_numpy()
    m = len(proposals)
    hadj = [set() for _ in range(m)]
    for i in range(m):
        for j in range(i + 1, m):
            if proposals_conflict(proposals[i], proposals[j], adj):
                hadj[i].add(j)
                hadj[j].add(i)
    return hadj


def maximal_independent_sets(adj):
    """All maximal independent sets via Bron-Kerbosch (no pivot, small graphs)."""
    _need_numpy()
    n = len(adj)
    out = []

    def bk(r, p, x):
        _need_numpy()
        if not p and not x:
            out.append(frozenset(r))
            return
        for v in list(p):
            nbrs = adj[v]
            bk(r | {v}, p - nbrs - {v}, x - nbrs - {v})
            p = p - {v}
            x = x | {v}

    bk(set(), set(range(n)), set())
    return out


# ---------------------------------------------------------------------------
# Exact fractional chromatic number via two-phase simplex
# ---------------------------------------------------------------------------

def _simplex_min(c, A, b, tol=1e-9, max_iter=10000):
    """min c^T x s.t. A x >= b, x >= 0 (dense two-phase simplex).

    Returns (x_opt, objective).  Assumes b > 0 elementwise (true for the
    fractional-coloring LP, whose constraint RHS is all ones).
    """
    _need_numpy()
    m, n = A.shape
    if m == 0:
        return np.zeros(n), 0.0
    assert np.all(b > tol), "RHS must be positive"
    nxs = n + m                        # x vars + surplus vars
    N = nxs + m                        # + artificial vars
    T = np.zeros((m + 1, N + 1))
    T[:m, :n] = A
    T[:m, n:nxs] = -np.eye(m)          # surplus
    T[:m, nxs:N] = np.eye(m)           # artificials
    T[:m, N] = b
    basis = list(range(nxs, N))

    def pivot(r, col):
        piv = T[r, col]
        T[r] /= piv
        for i in range(m + 1):
            if i != r and abs(T[i, col]) > 1e-300:
                T[i] -= T[i, col] * T[r]
        basis[r] = col

    def run_phase(cost, nvar):
        T[m, :nvar] = cost[:nvar]  # reduced costs: c - c_B B^{-1} A
        for i, j in enumerate(basis):
            if abs(cost[j]) > 1e-300:
                T[m] -= cost[j] * T[i]
        for i, j in enumerate(basis):  # basic reduced costs are exactly 0
            T[m, j] = 0.0
        it = 0
        while it < max_iter:
            row = T[m, :nvar]
            entering = int(np.argmin(row))   # Bland's rule (smallest index)
            if row[entering] >= -tol:
                break
            col = T[:m, entering]
            mask = col > tol
            if not np.any(mask):
                raise ArithmeticError("LP unbounded")
            ratios = np.where(mask, T[:m, N] / np.maximum(col, tol), np.inf)
            leaving = int(np.argmin(ratios))
            pivot(leaving, entering)
            it += 1
        else:
            raise ArithmeticError("simplex iteration limit")
        x = np.zeros(nvar)
        for i, j in enumerate(basis):
            x[j] = T[i, N]
        return x, -T[m, N]

    # Phase I: minimize sum of artificials
    cost1 = np.zeros(N)
    cost1[nxs:N] = 1.0
    _, v1 = run_phase(cost1, N)
    if v1 > 1e-6:
        raise ArithmeticError("LP infeasible")

    # Purge artificials before Phase II (a zero-cost artificial re-entering
    # the basis would mask infeasibility). A basic artificial is 0-valued;
    # pivot it out on a non-artificial column, else the row is redundant.
    for i in range(m):
        if basis[i] >= nxs:
            for jj in range(nxs):
                if abs(T[i, jj]) > tol:
                    pivot(i, jj)
                    break
    keep = [i for i in range(m) if basis[i] < nxs]
    m2 = len(keep)
    T2 = np.zeros((m2 + 1, nxs + 1))
    T2[:m2, :nxs] = T[keep][:, :nxs]
    T2[:m2, nxs] = T[keep][:, N]
    T = T2
    basis = [basis[i] for i in keep]

    # Phase II on the purged tableau
    m = m2
    N = nxs

    def pivot2(r, col):
        piv = T[r, col]
        T[r] /= piv
        for i in range(m + 1):
            if i != r and abs(T[i, col]) > 1e-300:
                T[i] -= T[i, col] * T[r]
        basis[r] = col

    cost2 = np.zeros(N)
    cost2[:n] = c
    T[m, :N] = cost2  # reduced costs: c - c_B B^{-1} A
    for i, j in enumerate(basis):
        if abs(cost2[j]) > 1e-300:
            T[m] -= cost2[j] * T[i]
    for i, j in enumerate(basis):
        T[m, j] = 0.0
    it = 0
    while it < max_iter:
        row = T[m, :N]
        entering = int(np.argmin(row))
        if row[entering] >= -tol:
            break
        col = T[:m, entering]
        mask = col > tol
        if not np.any(mask):
            raise ArithmeticError("LP unbounded")
        ratios = np.where(mask, T[:m, N] / np.maximum(col, tol), np.inf)
        leaving = int(np.argmin(ratios))
        pivot2(leaving, entering)
        it += 1
    else:
        raise ArithmeticError("simplex iteration limit")
    x = np.zeros(N)
    for i, j in enumerate(basis):
        x[j] = T[i, N]
    return x[:n], -T[m, N]

def fractional_chromatic_number(hadj):
    """Exact chi_f of the conflict graph: min sum lambda_I s.t. every vertex
    is covered to total weight >= 1 over maximal independent sets I."""
    _need_numpy()
    mis = maximal_independent_sets(hadj)
    m = len(hadj)
    if m == 0:
        return 0.0
    A = np.zeros((m, len(mis)))
    for k, s in enumerate(mis):
        for v in s:
            A[v, k] = 1.0
    _, chi_f = _simplex_min(np.ones(len(mis)), A, np.ones(m))
    return chi_f


def fractional_coloring(hadj):
    """Optimal fractional coloring: list of (independent_set, weight) with
    sum of weights == chi_f(H).

    Restricting to *maximal* independent sets is without loss of optimality:
    any non-maximal set can be grown to a maximal superset at the same weight
    without breaking coverage or changing the objective."""
    _need_numpy()
    mis = maximal_independent_sets(hadj)
    m = len(hadj)
    if m == 0:
        return 0.0, []
    A = np.zeros((m, len(mis)))
    for k, s in enumerate(mis):
        for v in s:
            A[v, k] = 1.0
    lam, chi_f = _simplex_min(np.ones(len(mis)), A, np.ones(m))
    coloring = [(set(mis[k]), float(lam[k])) for k in range(len(mis)) if lam[k] > 1e-9]
    return chi_f, coloring


# ---------------------------------------------------------------------------
# Mergers
# ---------------------------------------------------------------------------

def fractional_merger(proposals, gains, hadj):
    """Merge via the max-weight class of an optimal fractional coloring.

    Returns (chosen_indices, preserved_gain, chi_f).  The paper's guarantee:
    preserved_gain >= total_verified_gain / chi_f.
    """
    _need_numpy()
    gains = np.asarray(gains, dtype=float)
    chi_f, coloring = fractional_coloring(hadj)
    if not coloring:
        return [], 0.0, chi_f
    best, best_w = None, -np.inf
    for s, _ in coloring:
        w = float(gains[list(s)].sum()) if s else 0.0
        if w > best_w:
            best, best_w = set(s), w
    return sorted(best), best_w, chi_f


def naive_merge(proposals):
    """Apply every proposal (later proposals overwrite earlier ones)."""
    _need_numpy()
    merged = {}
    for e in proposals:
        merged.update(e)
    return merged


def brute_force_optimum(obj):
    """Exact optimum of f by enumeration (n <= ~20)."""
    _need_numpy()
    best, bestx = -np.inf, None
    for bits in itertools.product((0, 1), repeat=obj.n):
        v = obj.f(bits)
        if v > best:
            best, bestx = v, np.array(bits)
    return best, bestx


def collaborate(obj, x0, oracle, rounds, fstar=None):
    """Repeated parallel collaboration.

    oracle(x) -> list of proposal dicts (edits).  Each round: verify gains
    against the current x, build the conflict graph, merge via the
    fractional-coloring merger, apply (gains add exactly on the independent
    set).  Returns a history list of dicts.
    """
    _need_numpy()
    x = np.array(x0, dtype=int)
    hist = []
    for t in range(rounds):
        proposals = oracle(x)
        gains = [obj.gain(x, e) for e in proposals]
        total = float(np.sum([g for g in gains if g > 0]))
        hadj = conflict_graph(proposals, obj.adj)
        chosen, preserved, chi_f = fractional_merger(proposals, gains, hadj)
        merged = {}
        for i in chosen:
            merged.update(proposals[i])
        x = obj.apply(x, merged)
        rec = {"round": t, "n_proposals": len(proposals), "total_verified": total,
               "chi_f": chi_f, "chosen": chosen, "preserved": preserved,
               "f": obj.f(x)}
        if fstar is not None:
            rec["gap"] = fstar - obj.f(x)
        hist.append(rec)
        if fstar is not None and rec["gap"] <= 1e-9:
            break
    return hist
