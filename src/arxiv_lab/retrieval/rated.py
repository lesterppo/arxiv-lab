"""RATED: Reliability-Aware AI Agent Tool Retrieval via Execution Dynamics.

arXiv:2610.13889 — "RATED: Reliability-Aware AI Agent Tool Retrieval via
Execution Dynamics" (Chen, Liao). Reference implementation:
https://github.com/Elliotes/EcoAgent (Apache-2.0).

Paper mechanism implemented here
(``scripts/training/train_bandit_warmup.py``,
``scripts/evaluation/evaluate_rated_full.py``):

* frozen semantic stage: top-k candidates by cosine similarity
  (dual-encoder proxy);
* a lightweight **discounted LinUCB contextual bandit**, one arm per tool,
  context = query embedding, warmed up from static reliability scores and
  updated online from shaped execution rewards
  (``0.1 * sem * static + 0.9 * exec_signal`` with ``exec_signal`` =
  +1.0 success / -2.0 picked-a-failing-tool / 0.0 otherwise);
* final ranking by the **hybrid score**
  ``0.75 * semantic + 0.10 * static_reliability + 0.15 * bandit_score`` —
  additive, so the bandit nudges rather than overrides the semantic order.

Paper claims: the bandit adapts to tool outages without retraining (fault
selection 86.2% -> 2.9% under a severe outage in their eval), MRR improves
over the ToolLLM API retriever, re-rank overhead < 1 ms.

Requires numpy (optional dependency): ``pip install numpy``.
"""

try:
    import numpy as np
except ImportError:  # pragma: no cover - optional dependency
    np = None

__all__ = [
    "DiscountedLinUCB",
    "build_hybrid_score",
    "cosine_topk",
    "fault_selection_rate",
    "mrr_at_k",
    "rerank_hybrid",
]

# Default hybrid weights from evaluate_rated_full.py
W_SEMANTIC = 0.75
W_STATIC = 0.10
W_DYNAMIC = 0.15

# Default reward shaping from evaluate_rated_full.py
R_SUCCESS = 1.0
R_FAILED_TOOL = -2.0
R_WRONG_TOOL = 0.0
R_SEM_WEIGHT = 0.1
R_EXEC_WEIGHT = 0.9


def _need_numpy():
    if np is None:  # pragma: no cover
        raise RuntimeError("arxiv_lab.retrieval.rated requires numpy")


def cosine_topk(query_vec, tool_matrix, k):
    """Top-k tool indices by cosine similarity (frozen semantic stage).

    ``query_vec``: (d,) array; ``tool_matrix``: (n_tools, d) array.
    Returns (indices, scores) sorted by descending score.
    """
    _need_numpy()
    q = np.asarray(query_vec, dtype=float)
    m = np.asarray(tool_matrix, dtype=float)
    qn = q / (np.linalg.norm(q) + 1e-12)
    mn = m / (np.linalg.norm(m, axis=1, keepdims=True) + 1e-12)
    scores = mn @ qn
    k = min(k, len(scores))
    idx = np.argsort(-scores, kind="stable")[:k]
    return idx, scores[idx]


class DiscountedLinUCB:
    """Per-tool discounted LinUCB arms (the paper's bandit module).

    Each tool owns an arm ``(A, b)``; context ``x`` is the query embedding.
    ``score`` returns the UCB value ``mu + alpha * sigma``; ``mean``
    returns the posterior mean ``mu`` only. ``update`` applies the
    discount ``gamma`` then the rank-1 reward update. Arms are created
    lazily. ``initialize`` seeds an arm from a static reliability score,
    mirroring the paper's warm-up (``b = r_static * default_context``).
    """

    def __init__(self, dim, alpha=0.1, gamma=0.95, ridge=1.0):
        self.dim = int(dim)
        self.alpha = float(alpha)
        self.gamma = float(gamma)
        self.ridge = float(ridge)
        self._A = {}
        self._b = {}

    def _arm(self, tool_id):
        tid = int(tool_id)
        if tid not in self._A:
            _need_numpy()
            self._A[tid] = np.eye(self.dim) * self.ridge
            self._b[tid] = np.zeros(self.dim)
        return self._A[tid], self._b[tid]

    def initialize(self, tool_id, static_reliability, default_context=None):
        """Warm-start an arm from a static reliability score in [0, 1]."""
        _need_numpy()
        A, _ = self._arm(tool_id)
        if default_context is None:
            default_context = np.ones(self.dim) / np.sqrt(self.dim)
        self._b[int(tool_id)] = (float(static_reliability)
                                 * np.asarray(default_context, dtype=float))

    def _theta(self, tool_id, x):
        _need_numpy()
        A, b = self._arm(tool_id)
        x = np.asarray(x, dtype=float)
        # solve instead of inverting, same value, better conditioning
        theta = np.linalg.solve(A, b)
        a_inv_x = np.linalg.solve(A, x)
        mu = float(theta @ x)
        sigma = float(np.sqrt(max(x @ a_inv_x, 0.0)))
        return mu, sigma

    def mean(self, tool_id, x):
        """Posterior-mean predicted reward for (tool, context)."""
        mu, _ = self._theta(tool_id, x)
        return mu

    def score(self, tool_id, x):
        """UCB score mu + alpha * sigma for (tool, context)."""
        mu, sigma = self._theta(tool_id, x)
        return mu + self.alpha * sigma

    def update(self, tool_id, x, reward):
        """Discounted rank-1 update of one arm."""
        _need_numpy()
        A, b = self._arm(tool_id)
        x = np.asarray(x, dtype=float)
        r = float(reward)
        self._A[int(tool_id)] = self.gamma * A + np.outer(x, x)
        self._b[int(tool_id)] = self.gamma * b + r * x

    @property
    def n_arms(self):
        return len(self._A)


def build_hybrid_score(semantic_score, static_reliability, dynamic_reliability,
                       w_semantic=W_SEMANTIC, w_static=W_STATIC,
                       w_dynamic=W_DYNAMIC):
    """Additive hybrid re-rank score (paper's evaluate_rated_full.py)."""
    return (w_semantic * semantic_score
            + w_static * static_reliability
            + w_dynamic * dynamic_reliability)


def execution_reward(semantic_score, static_reliability, success,
                     picked_failed_tool):
    """Shaped online reward (paper's evaluate_rated_full.py).

    ``success``: picked tool was a working positive. ``picked_failed_tool``:
    picked tool is in the currently-failing set.
    """
    if success:
        exec_signal = R_SUCCESS
    elif picked_failed_tool:
        exec_signal = R_FAILED_TOOL
    else:
        exec_signal = R_WRONG_TOOL
    return (R_SEM_WEIGHT * semantic_score * static_reliability
            + R_EXEC_WEIGHT * exec_signal)


def rerank_hybrid(candidate_ids, sem_scores, bandit, context,
                  static_rel=0.75, use_bandit=True):
    """Re-rank candidates by the hybrid score (descending).

    ``static_rel``: scalar or per-candidate static reliability (the paper
    defaults to 0.75 when no reliability column exists). With
    ``use_bandit=False`` this reduces to the semantic-only baseline (the
    paper's ``--disable-bandit`` mode). Returns (reranked_ids,
    hybrid_scores).
    """
    _need_numpy()
    ids = np.asarray(candidate_ids)
    sem = np.asarray(sem_scores, dtype=float)
    x = np.asarray(context, dtype=float)
    if np.isscalar(static_rel):
        st = np.full(len(ids), float(static_rel))
    else:
        st = np.asarray(static_rel, dtype=float)
    if use_bandit:
        dyn = np.array([bandit.score(i, x) for i in ids])
    else:
        dyn = np.zeros(len(ids))
        # baseline: pure semantic order (paper's --disable-bandit)
        order = np.argsort(-sem, kind="stable")
        return ids[order], sem[order]
    hybrid = (W_SEMANTIC * sem + W_STATIC * st + W_DYNAMIC * dyn)
    order = np.argsort(-hybrid, kind="stable")
    return ids[order], hybrid[order]


def mrr_at_k(ranked_ids, relevant_working, k=10):
    """MRR where a hit must be a *working* relevant tool.

    Mirrors the paper's dynamic-reliability evaluation: ranking a broken
    tool first earns nothing — the agent would have to fall back anyway.
    """
    rel = set(int(i) for i in relevant_working)
    for rank, tid in enumerate(ranked_ids[:k], start=1):
        if int(tid) in rel:
            return 1.0 / rank
    return 0.0


def fault_selection_rate(executed_ids, failed_ids):
    """Fraction of executed tool calls that hit a failing tool."""
    executed = list(executed_ids)
    if not executed:
        return 0.0
    failed = set(int(i) for i in failed_ids)
    return sum(1 for i in executed if int(i) in failed) / len(executed)
