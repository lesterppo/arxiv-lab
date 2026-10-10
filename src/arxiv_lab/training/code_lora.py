"""CoDe-LoRA (arXiv 2610.08312) — consolidation/decoupling continual-learning mechanism.

Paper claim: strict orthogonal isolation (O-LoRA-style) discards a rho_t
fraction of each new task's update energy, which cripples transfer between
semantically related tasks (the "orthogonality dilemma"); CoDe-LoRA's dual
branch — a null-projected, dynamically-scaled consolidation branch plus a
pool of prototype-routed task experts with confidence fallback — preserves
retention while recovering transfer. numpy-guarded per repo conventions.
"""

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:  # pragma: no cover
    np = None
    HAS_NUMPY = False


def _need_numpy():
    if not HAS_NUMPY:
        raise RuntimeError("numpy is required for arxiv_lab.training.code_lora")


# Module-level constants are plain Python (numpy-free) per repo conventions.
DEFAULT_TAU = 0.75   # routing confidence threshold (paper Sec. 4.4)
DEFAULT_DV = 6       # null-space basis rank dv (= LoRA rank r in the paper)


def dynamic_scaling(t):
    """Dynamic scaling coefficients (paper Sec. 4.2 / App. A.2).

    c_t = sqrt((t-1)/t), s_t = sqrt(1/t) for 1-based task index t,
    satisfying the normalization identity c_t^2 + s_t^2 == 1.
    """
    _need_numpy()
    t = int(t)
    if t < 1:
        raise ValueError("task index t must be >= 1")
    c = float(np.sqrt((t - 1.0) / t))
    s = float(np.sqrt(1.0 / t))
    return c, s


def top_column_basis(W_acc, dv=DEFAULT_DV):
    """Top-dv left singular vectors U of W_acc (paper Sec. 4.2).

    U is an orthonormal basis of the accumulated column space. An all-zero
    W_acc (task 1, W_acc^(0) = 0) yields an empty basis so the projection is
    a no-op instead of projecting onto arbitrary SVD directions.
    """
    _need_numpy()
    W_acc = np.asarray(W_acc, dtype=float)
    d_out = W_acc.shape[0]
    if not np.any(W_acc):
        return np.zeros((d_out, 0))
    U, _, _ = np.linalg.svd(W_acc, full_matrices=False)
    k = max(0, min(int(dv), U.shape[1]))
    return U[:, :k]


def consolidate_null_project(delta_W_shared, W_acc_prev, dv=DEFAULT_DV):
    """Adaptive null-space projection, paper Eq. (3).

    ΔW_null^(t) = ΔW_shared^(t) − U Uᵀ ΔW_shared^(t),
    where U holds the top-dv left singular vectors of W_acc^(t-1).
    Keeps only what is new in each task so earlier consolidated
    representations are not overwritten.
    """
    _need_numpy()
    delta = np.asarray(delta_W_shared, dtype=float)
    U = top_column_basis(W_acc_prev, dv)
    if U.shape[1] == 0:
        return delta.copy()
    return delta - U @ (U.T @ delta)


def update_overlap_rho(delta_W_shared, W_acc_prev, dv=DEFAULT_DV):
    """Update-overlap ratio, paper Eq. (2) style.

    rho = ||U Uᵀ ΔW||_F^2 / ||ΔW||_F^2 in [0, 1]: the fraction of the
    task update's Frobenius energy lying in the prior column subspace —
    i.e. the fraction a strict null projection would discard.
    """
    _need_numpy()
    delta = np.asarray(delta_W_shared, dtype=float)
    denom = float(np.sum(delta * delta))
    if denom == 0.0:
        return 0.0
    U = top_column_basis(W_acc_prev, dv)
    if U.shape[1] == 0:
        return 0.0
    proj = U @ (U.T @ delta)
    return float(np.sum(proj * proj) / denom)


def code_lora_step(W_acc_prev, delta_W_shared, t, dv=DEFAULT_DV):
    """One consolidation step, paper Eq. (4).

    W_acc^(t) = c_t W_acc^(t-1) + s_t ΔW_null^(t).
    Returns (W_acc_new, delta_W_null, (c_t, s_t)).
    """
    _need_numpy()
    c, s = dynamic_scaling(t)
    W_prev = np.asarray(W_acc_prev, dtype=float)
    delta_null = consolidate_null_project(delta_W_shared, W_prev, dv)
    return c * W_prev + s * delta_null, delta_null, (c, s)


def build_prototype(X_task, n_proto=10, rng=None):
    """Task prototype, paper Sec. 4.3: L2-normalized mean of n_proto embeddings.

    (Here the "frozen-backbone embedding" is the raw input; the mechanism —
    normalized mean of per-task samples — is unchanged.)
    """
    _need_numpy()
    X = np.asarray(X_task, dtype=float)
    n = min(int(n_proto), X.shape[0])
    if n <= 0:
        raise ValueError("need at least one sample for a prototype")
    idx = range(n) if rng is None else rng.choice(X.shape[0], n, replace=False)
    proto = X[idx].mean(axis=0)
    norm = float(np.linalg.norm(proto))
    if norm == 0.0:
        raise ValueError("degenerate zero prototype")
    return proto / norm


def route_prototype(x, prototypes, tau=DEFAULT_TAU):
    """Prototype routing with confidence fallback, paper Sec. 4.4.

    k* = argmax_k cosine(x, P_k); confidence = max cosine.
    Returns (k_star, confidence, use_expert) with
    use_expert = (confidence > tau); otherwise fall back to the shared
    (consolidation) branch.
    """
    _need_numpy()
    x = np.asarray(x, dtype=float)
    xn = float(np.linalg.norm(x))
    if xn == 0.0:
        raise ValueError("degenerate zero query")
    sims = []
    for p in prototypes:
        p = np.asarray(p, dtype=float)
        pn = float(np.linalg.norm(p))
        sims.append(float(np.dot(x, p) / (xn * pn)) if pn > 0 else 0.0)
    k_star = int(np.argmax(sims))
    confidence = sims[k_star]
    return k_star, confidence, bool(confidence > tau)
