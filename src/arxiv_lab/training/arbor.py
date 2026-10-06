"""ARBOR: conditional rank allocation for taxonomy-aware adaptation.

Paper: arXiv 2610.06765 "Conditional Rank Allocation for Taxonomy-Aware
Medical Language Model Adaptation".

One-line claim: selecting rank-one components from a shared low-rank basis
*per question* — via an additive gate over question representations,
specialty/operation tags, and their interaction — avoids the approximation
floor a fixed update of the same active rank faces under orthogonal,
equiprobable subtasks, and beats LoRA r16 / MoELoRA on medical QA.

This module implements the paper's *illustrative separation* construction
(the theoretical motivation): K orthogonal subtasks, a shared basis of
rank-one atoms, a fixed rank-r update vs conditional per-question atom
selection with the same active rank. numpy-guarded per repo conventions.
"""

try:
    import numpy as np
except ImportError:  # pragma: no cover
    np = None


def _need_numpy():
    if np is None:
        raise RuntimeError("numpy is required for arxiv_lab.training.arbor")


def make_orthogonal_tasks(K=4, dim=16, rank_per_task=2, seed=0):
    """K subtasks with disjoint coordinate support (hence orthogonal).

    Returns (deltas, atoms): deltas[k] is the (dim x dim) ideal rank-
    `rank_per_task` update for subtask k; atoms is the shared basis of
    K*rank_per_task rank-one components (2 per subtask here).
    """
    _need_numpy()
    rng = np.random.RandomState(seed)
    block = dim // K
    deltas = []
    atoms = []  # list of (u, v) with delta_k = sum_i u_i v_i^T
    for k in range(K):
        delta = np.zeros((dim, dim))
        comps = []
        for _ in range(rank_per_task):
            u = np.zeros(dim)
            v = np.zeros(dim)
            u[k * block : (k + 1) * block] = rng.randn(block)
            v[k * block : (k + 1) * block] = rng.randn(block)
            u /= np.linalg.norm(u)
            v /= np.linalg.norm(v)
            delta += np.outer(u, v)
            comps.append((u, v))
        deltas.append(delta)
        atoms.append(comps)
    return deltas, atoms


def fixed_update_floor(deltas, active_rank):
    """Best single rank-`active_rank` update shared by all subtasks.

    The optimal fixed update is the top-`active_rank` SVD truncation of the
    stacked task matrix; its mean residual is the approximation floor a
    fixed update cannot go below. Returns (delta_fixed, floor_mse).
    """
    _need_numpy()
    stacked = np.concatenate([d.reshape(1, -1) for d in deltas], axis=0)
    U, s, Vt = np.linalg.svd(stacked, full_matrices=False)
    approx = (U[:, :active_rank] * s[:active_rank]) @ Vt[:active_rank, :]
    floor = float(np.mean([(d.reshape(-1) - approx[k]) ** 2
                           for k, d in enumerate(deltas)]))
    delta_fixed = approx[0].reshape(deltas[0].shape)  # representative
    return delta_fixed, floor


def additive_gate(question, tag, Wq, Wt, Wx):
    """The paper's additive gate: q-repr + tag + interaction -> atom scores.

    question: (dq,) vector; tag: one-hot (K,); returns softmax scores over
    the K*rank_per_task shared atoms.
    """
    _need_numpy()
    interact = np.outer(question, tag).reshape(-1)
    logits = Wq @ question + Wt @ tag + Wx @ interact
    e = np.exp(logits - logits.max())
    return e / e.sum()


def conditional_update(question, tag, atoms, Wq, Wt, Wx, active_rank,
                       coeff=1.0):
    """ARBOR update: gate selects `active_rank` atoms for this question."""
    _need_numpy()
    scores = additive_gate(question, tag, Wq, Wt, Wx)
    flat = [uv for comps in atoms for uv in comps]
    top = np.argsort(scores)[-active_rank:]
    delta = np.zeros_like(flat[0][0][:, None] * flat[0][1][None, :])
    for i in top:
        u, v = flat[i]
        delta = delta + coeff * scores[i] * np.outer(u, v)
    return delta, scores


def oracle_gate_weights(K, rank_per_task, dim_q=8, seed=1):
    """Gate weights that route each subtask's questions to its own atoms.

    Models a *learned* gate after training (the paper reports atom clusters
    align with specialty labels, ARI 0.62): questions carry a subtask
    signature the gate exploits. Returns (Wq, Wt, Wx).
    """
    _need_numpy()
    rng = np.random.RandomState(seed)
    n_atoms = K * rank_per_task
    dq = dim_q
    Wq = np.zeros((n_atoms, dq))
    for k in range(K):
        # question signature: one-hot-ish in first K dims + noise dims
        Wq[k * rank_per_task : (k + 1) * rank_per_task, k] = 6.0
    Wt = np.zeros((n_atoms, K))
    for k in range(K):
        Wt[k * rank_per_task : (k + 1) * rank_per_task, k] = 6.0
    Wx = rng.randn(n_atoms, dq * K) * 0.01  # near-zero interaction
    return Wq, Wt, Wx


def question_for_task(k, K, dim_q=8, seed=2, noise=0.1):
    """A question representation whose signature marks subtask k."""
    _need_numpy()
    rng = np.random.RandomState(seed + k)
    q = np.zeros(dim_q)
    q[k] = 1.0
    q += rng.randn(dim_q) * noise
    tag = np.zeros(K)
    tag[k] = 1.0
    return q, tag
