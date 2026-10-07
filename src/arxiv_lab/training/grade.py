"""GRADE: gradient admission for data-centric SLM finetuning.

Paper: arXiv 2610.07553 "Which and When to Admit: Gradient Admission for
Data-Centric Small Language Model Finetuning".

One-line claim: controlling *which* data-induced gradients enter the LoRA
subspace (state-aware gradient-aligned admission) and *when* updates are
committed (self-calibrating step-level gate) is the only recipe that improves
consistently over standard LoRA across architectures on a heterogeneous
instruction pool, with more coherent gradient trajectories and less
destructive overwrite.

This module implements the paper's closed-loop admission math in miniature
(the mechanism, not the LLM harness):

  Mechanism 1 (Sec. 4.1): keep fraction from pool geometry
      r* = cos_intra / (cos_intra + (T-1) * cos_inter)
      with the paper's three safeguards (inter-cosine floor at 1e-8,
      hard fallback to 0.5, global clamp to [0.1, 1.0]).
      Forward-probe admission score a_i = d_i * d_ref (magnitude-weighted
      projection of per-sample finite differences along a shared random
      probe direction), top-kb admission.

  Mechanism 2 (Sec. 4.2): self-calibrating step-level gate.
      Probe loss L_probe(t) (admitted-batch loss) vs its EMA
      Lbar(t) = (1-a) Lbar(t-1) + a L_probe(t). The gate latches once the
      relative EMA slope plateaus: (Lbar(t-1) - Lbar(t)) / L_probe(0) < eps_rel.
      After latching: commit the step iff L_probe(t) <= Lbar(t), else SKIP
      (truncate the update).

numpy-guarded per repo conventions.
"""

try:
    import numpy as np
except ImportError:  # pragma: no cover
    np = None


def _need_numpy():
    if np is None:
        raise RuntimeError("numpy is required for arxiv_lab.training.grade")


# --------------------------------------------------------------------------
# Mechanism 1: keep fraction + probe admission
# --------------------------------------------------------------------------

def r_star(cos_intra, cos_inter, T, floor=1e-8, fallback=0.5,
           lo=0.1, hi=1.0):
    """Closed-form keep fraction (paper Eq. 2) with the three safeguards.

    r* = cos_intra / (max(cos_inter, floor) + ...); degenerate denominator
    -> fallback; result clamped to [lo, hi].
    """
    _need_numpy()
    denom_inner = max(cos_inter, floor)
    denom = denom_inner * 0 + cos_intra + (T - 1) * denom_inner
    # NOTE: denom = cos_intra + (T-1)*max(cos_inter, floor)
    if denom <= 0:
        r = fallback
    else:
        r = cos_intra / denom
    return float(min(hi, max(lo, r)))


def cosine_stats(grads, labels):
    """Mean within-dataset and across-dataset gradient cosines.

    grads: (n, q) array of per-sample LoRA-subspace gradients.
    labels: (n,) integer dataset ids.
    Returns (cos_intra, cos_inter).
    """
    _need_numpy()
    n = grads.shape[0]
    norms = np.linalg.norm(grads, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    G = grads / norms
    C = G @ G.T
    intra, inter = [], []
    for i in range(n):
        for j in range(i + 1, n):
            if labels[i] == labels[j]:
                intra.append(C[i, j])
            else:
                inter.append(C[i, j])
    cos_intra = float(np.mean(intra)) if intra else 0.0
    cos_inter = float(np.mean(inter)) if inter else 0.0
    return cos_intra, cos_inter


def probe_scores(d_i, d_ref):
    """Forward-probe admission scores a_i = d_i * d_ref (scalar product).

    d_i: (m,) per-candidate finite differences along the shared probe dir.
    d_ref: scalar reference probe (mean finite difference over D_ref).
    E_u[d_i * d_ref] = (1/q) g_i . g_ref  (unbiased inner-product estimate).
    """
    _need_numpy()
    return np.asarray(d_i, dtype=float) * float(d_ref)


def admit_topk(scores, kb):
    """Indices of the top-kb candidates by admission score."""
    _need_numpy()
    scores = np.asarray(scores, dtype=float)
    kb = int(min(max(kb, 1), len(scores)))
    return np.argsort(scores)[-kb:][::-1]


# --------------------------------------------------------------------------
# Mechanism 2: self-calibrating step-level gate
# --------------------------------------------------------------------------

class AdmissionGate:
    """Step-level admission gate (paper Sec. 4.2).

    Tracks the probe-loss EMA; latches once the relative EMA slope
    plateaus; afterwards commits a step iff probe <= EMA, else skips.
    """

    def __init__(self, alpha=0.1, eps_rel=0.01):
        _need_numpy()
        self.alpha = float(alpha)
        self.eps_rel = float(eps_rel)
        self.ema = None
        self.ema_prev = None
        self.probe0 = None
        self.latched = False
        self.latch_step = None
        self.n_skipped = 0
        self.n_committed = 0

    def observe(self, probe_loss, step):
        """Feed one step's admitted-batch probe loss.

        Returns 'commit' or 'skip'. Before latching, always commits.
        """
        p = float(probe_loss)
        if self.ema is None:
            self.ema = p
            self.probe0 = p
            self.n_committed += 1
            return "commit"
        self.ema_prev = self.ema
        self.ema = (1 - self.alpha) * self.ema + self.alpha * p
        if not self.latched:
            rel_slope = (self.ema_prev - self.ema) / max(self.probe0, 1e-12)
            if rel_slope < self.eps_rel:
                self.latched = True
                self.latch_step = step
            self.n_committed += 1
            return "commit"
        if p <= self.ema:
            self.n_committed += 1
            return "commit"
        self.n_skipped += 1
        return "skip"


# --------------------------------------------------------------------------
# Miniature end-to-end: synthetic multi-task gradient field
# --------------------------------------------------------------------------

def make_task_field(K=4, q=32, seed=0):
    """K tasks with near-orthogonal mean gradient directions in R^q."""
    _need_numpy()
    rng = np.random.default_rng(seed)
    dirs, _ = np.linalg.qr(rng.standard_normal((q, K)))
    return dirs.T.copy()  # (K, q), orthonormal rows


def simulate_admission(K=4, q=32, n_cand=40, kb=12, seed=0, noise=0.3,
                      conflict_task=None, n_probes=1):
    """One miniature admission round.

    Each candidate belongs to a task; its gradient = task dir + noise.
    Reference direction = mean task direction (heterogeneous pool).
    With ``conflict_task=k``, task k's direction is set to oppose the
    pool mean (the heterogeneous-conflict regime of the paper).
    Probe: shared random unit u; d_i, d_ref finite differences of a
    quadratic loss (exact here: d = g . u). Scores averaged over
    ``n_probes`` independent probe directions (variance reduction; the
    per-probe estimator is unbiased by App. M). Returns the fraction of
    admitted candidates whose task aligns positively with the reference.
    """
    _need_numpy()
    rng = np.random.default_rng(seed)
    dirs = make_task_field(K, q, seed=seed)
    if conflict_task is not None:
        keep = [d for j, d in enumerate(dirs) if j != conflict_task]
        opp = -np.sum(keep, axis=0)
        opp /= np.linalg.norm(opp)
        dirs = np.stack(keep + [opp])  # conflict task last
        conflict_idx = K - 1
    else:
        conflict_idx = None
    g_ref = dirs.mean(axis=0)
    task_of = rng.integers(0, K, size=n_cand)
    grads = np.stack([dirs[t] + noise * rng.standard_normal(q)
                      for t in task_of])
    scores = np.zeros(n_cand)
    for _ in range(n_probes):
        u = rng.standard_normal(q)
        u /= np.linalg.norm(u)
        d_i = grads @ u
        d_ref = float(g_ref @ u)
        scores += probe_scores(d_i, d_ref)
    scores /= n_probes
    adm = admit_topk(scores, kb)
    true_align = (grads @ g_ref) > 0
    out = {
        "admitted_align_frac": float(true_align[adm].mean()),
        "pool_align_frac": float(true_align.mean()),
        "scores": scores,
    }
    if conflict_idx is not None:
        out["admitted_conflict_frac"] = float((task_of[adm] == conflict_idx).mean())
        out["pool_conflict_frac"] = float((task_of == conflict_idx).mean())
    return out
