"""SST: Subspace Tuning for mitigating catastrophic forgetting.

arXiv:2610.22293 — "Towards Continual Learning: Mitigating Forgetting With
Subspace Tuning".

Paper mechanism implemented here:

* Fine-tuning moves weights far along the *principal* singular directions of
  each weight matrix; SFT concentrates updates there and forgets, while
  on-policy methods avoid those directions and forget less.
* **Subspace Tuning** restricts each weight update to the *bottom* singular
  subspace: before applying a gradient step to a weight matrix ``W``, every
  component of the gradient along the top-``r`` singular directions of
  ``W`` is removed — on both sides::

      W = U S Vᵀ ;  G_sst = (I - U_r U_rᵀ) G (I - V_r V_rᵀ)

  with ``r = protect_frac * min(m, n)``. The update then lives entirely in
  the bottom-bottom singular block: no change along any principal
  direction. Protecting up to 99% of the subspace substantially improves
  retention while preserving strong target-task performance; it is
  optimizer-agnostic and composes with SFT and self-distillation recipes.
  (An earlier draft of this module removed only the top-top block
  ``U_r (U_rᵀ G V_r) V_rᵀ``; the toy reproduction showed that reading has
  no retention effect — the top-bottom/bottom-top blocks carry enough
  interference to destroy retention. The two-sided projection above is
  the faithful reading of "eliminating changes along the principal
  directions".)
* **Refresh**: the SVD is recomputed periodically as ``W`` drifts (every
  step at toy scale; every K steps in the staged Colab script).

Paper claims: SST advances the learning-forgetting Pareto frontier, keeps
the model 2-4x closer (KL) to its base distribution, and nearly eliminates
forgetting in the large-model regime.

Requires numpy (optional dependency): ``pip install numpy``.
"""

try:
    import numpy as np
except ImportError:  # pragma: no cover - optional dependency
    np = None

__all__ = [
    "project_grad_to_bottom_subspace",
    "sst_update",
    "principal_directions",
]


def _need_numpy():
    if np is None:
        raise ImportError("numpy is required for arxiv_lab.training.sst")


def principal_directions(weight, protect_frac):
    """Top-``r`` left/right singular vectors of ``weight``.

    ``r = max(1, round(protect_frac * min(m, n)))``. Returns ``(U_r, V_r)``
    with shapes ``(m, r)`` and ``(n, r)``.
    """
    _need_numpy()
    w = np.asarray(weight, dtype=float)
    m, n = w.shape
    r = max(1, int(round(protect_frac * min(m, n))))
    r = min(r, min(m, n) - 1)  # always keep at least one free direction
    u, _s, vt = np.linalg.svd(w, full_matrices=False)
    return u[:, :r], vt[:r, :].T


def project_grad_to_bottom_subspace(grad, weight, protect_frac):
    """Remove all gradient components along the top singular directions of W.

    Returns ``(I - U_r U_rᵀ) G (I - V_r V_rᵀ)``: the update is restricted to
    the bottom-bottom singular block, i.e. it makes no change along any
    principal direction of ``W`` (neither on the left nor on the right).
    """
    _need_numpy()
    g = np.asarray(grad, dtype=float)
    u_r, v_r = principal_directions(weight, protect_frac)
    m, n = g.shape
    p_left = np.eye(m) - u_r @ u_r.T
    p_right = np.eye(n) - v_r @ v_r.T
    return p_left @ g @ p_right


def sst_update(params, grads, lr, protect_frac):
    """One SGD step with Subspace Tuning applied to 2-D weight matrices.

    ``params``/``grads`` are parallel lists; 1-D entries (biases) are
    updated plainly. Returns the new parameter list.
    """
    _need_numpy()
    new_params = []
    for p, g in zip(params, grads):
        p = np.asarray(p, dtype=float)
        g = np.asarray(g, dtype=float)
        if g.ndim == 2:
            g = project_grad_to_bottom_subspace(g, p, protect_frac)
        new_params.append(p - lr * g)
    return new_params
