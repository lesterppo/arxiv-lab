"""Causal Memory Policy retrieval-intervention scoring (arXiv:2610.02070).

Paper claim implemented: memory-utility estimates that intervene only at
the *store* level (remove a memory, measure the performance change) suffer a
retrieval-level positivity violation — a memory that is never retrieved has
estimated utility 0 no matter how useful it would be. CMP restores
identification by intervening on *retrieval itself*: a fixed number of
context slots are reserved for memories sampled with known propensities, and
utility is estimated by (self-normalized) inverse-propensity weighting under
a balanced assignment design.

Provides a synthetic memory store (``MemoryStore``), the naive
store-intervention estimator (``naive_store_intervention``), the CMP balanced
design estimator (``cmp_balanced_design``), and ``auc``.

Requires numpy (optional dependency): ``pip install numpy``.
"""

try:
    import numpy as np
except ImportError:  # pragma: no cover - optional dependency
    np = None


def _need_numpy():
    if np is None:
        raise ImportError(
            "arxiv_lab.memory requires numpy (optional dependency); "
            "install it with `pip install numpy`"
        )


def _trapz(y, x):
    # np.trapz was removed in numpy 2.0 (renamed np.trapezoid).
    f = getattr(np, "trapezoid", None) or getattr(np, "trapz")
    return f(y, x)


class MemoryStore:
    """Synthetic memory store with a retrieval-level positivity violation.

    ``n_required`` memories carry real per-query utility; a
    ``frac_never_retrieved`` share of them is suppressed from retrieval
    scores (the paper reports 54% on LongMemEval), so they are never
    retrieved no matter the query.
    """

    def __init__(self, rng, n_mem=60, n_queries=40, n_required=20,
                 frac_never_retrieved=0.55, top_k=8):
        _need_numpy()
        self.n_mem, self.n_queries, self.top_k = n_mem, n_queries, top_k
        self.true_u = np.zeros((n_mem, n_queries))
        self.required = np.arange(n_required)
        for m in self.required:
            qs = rng.choice(n_queries, size=6, replace=False)
            self.true_u[m, qs] = rng.uniform(0.6, 1.0, size=6)
        self.true_u += rng.uniform(0, 0.05, size=(n_mem, n_queries))
        # noisy retrieval scores; suppress a share of required memories ->
        # retrieval-level positivity violation
        self.scores = self.true_u + rng.normal(0, 0.25, size=(n_mem, n_queries))
        n_supp = int(round(frac_never_retrieved * n_required))
        self.suppressed = self.required[:n_supp]
        self.scores[self.suppressed, :] = -10.0

    def retrieve(self, q, k):
        """Top-k memory ids for query q by retrieval score."""
        return np.argsort(-self.scores[:, q])[:k]

    def perf(self, mem_set, q):
        """Task performance: sum of true utilities of retrieved memories that
        are present in ``mem_set``."""
        ctx = [m for m in self.retrieve(q, self.top_k) if m in mem_set]
        return float(self.true_u[ctx, q].sum()) if ctx else 0.0


def naive_store_intervention(store):
    """Remove each memory from the store; mean perf change over queries where
    it was retrieved. Never-retrieved memories score exactly 0 (the
    positivity violation: identification fails)."""
    _need_numpy()
    n = store.n_mem
    est = np.zeros(n)
    for m in range(n):
        d, cnt = 0.0, 0
        for q in range(store.n_queries):
            if m in store.retrieve(q, store.top_k):
                d += (store.perf(set(range(n)), q)
                      - store.perf(set(range(n)) - {m}, q))
                cnt += 1
        est[m] = d / cnt if cnt else 0.0
    return est


def cmp_balanced_design(store, rng, reserved=2, n_rep=80):
    """CMP: per query keep top-(K-R) score slots; each remaining candidate is
    independently included with known propensity 0.5 (balanced assignment).
    Utility = SNIPW estimate of E[perf | included] - E[perf | excluded].

    Returns ``(ate, valid)``: per-(memory, query) average treatment effect
    and its validity mask.
    """
    _need_numpy()
    M, Q = store.n_mem, store.n_queries
    num_in = np.zeros((M, Q)); den_in = np.zeros((M, Q))
    num_out = np.zeros((M, Q)); den_out = np.zeros((M, Q))
    for q in range(Q):
        base = list(store.retrieve(q, store.top_k - reserved))
        base_set = set(base)
        cand = [m for m in range(M) if m not in base_set]
        for _ in range(n_rep):
            mask = rng.random(len(cand)) < 0.5
            ctx = base + [cand[j] for j in range(len(cand)) if mask[j]]
            perf = float(store.true_u[ctx, q].sum())
            for j, m in enumerate(cand):
                if mask[j]:
                    num_in[m, q] += perf / 0.5
                    den_in[m, q] += 1 / 0.5
                else:
                    num_out[m, q] += perf / 0.5
                    den_out[m, q] += 1 / 0.5
    valid = (den_in > 0) & (den_out > 0)
    ate = np.full((M, Q), np.nan)
    ate[valid] = num_in[valid] / den_in[valid] - num_out[valid] / den_out[valid]
    return ate, valid


def auc(scores, labels):
    """Area under the ROC curve for scores against binary labels."""
    _need_numpy()
    order = np.argsort(-np.asarray(scores, dtype=float))
    tp = np.cumsum(labels[order]); fp = np.cumsum(1 - labels[order])
    return float(_trapz(tp / tp[-1], fp / fp[-1]))
