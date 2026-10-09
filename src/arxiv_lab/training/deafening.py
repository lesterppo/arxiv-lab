"""A Deafening Silence (arXiv 2610.09835) — Adam-epsilon intervention, CPU miniature.

Paper mechanism: tokens absent from the new corpus receive persistent
one-sided softmax gradients. Adam's second-moment (sqrt(v_hat))
normalization amplifies these tiny gradients into full-sized updates,
so forgetting concentrates in the output embeddings of rare tokens.
Intervention: raise Adam's epsilon *exclusively for the output
projection* — dampening the tiny-gradient regime while leaving
normal-sized gradients (real learning) essentially untouched.

This module simulates that mechanism on a single scalar (one output
embedding row element) receiving a persistent one-sided gradient, with
exact Adam bias correction. numpy-guarded per repo conventions.
"""

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:  # pragma: no cover
    np = None
    HAS_NUMPY = False


def _need_numpy():
    if not HAS_NUMPY:
        raise RuntimeError("numpy is required for arxiv_lab.training.deafening")


def adam_row_drift(g, steps, lr=2e-4, beta1=0.9, beta2=0.999, eps=1e-8):
    """Cumulative |drift| of one scalar under persistent one-sided gradient g.

    Exact Adam (with bias correction), the optimizer the paper analyzes.
    Returns total absolute displacement after `steps` updates.
    """
    _need_numpy()
    m = 0.0
    v = 0.0
    x = 0.0
    for t in range(1, steps + 1):
        m = beta1 * m + (1 - beta1) * g
        v = beta2 * v + (1 - beta2) * g * g
        m_hat = m / (1 - beta1 ** t)
        v_hat = v / (1 - beta2 ** t)
        x -= lr * m_hat / (np.sqrt(v_hat) + eps)
    return abs(x)


def steady_update_magnitude(g, lr=2e-4, beta1=0.9, beta2=0.999, eps=1e-8,
                            burn_in=2000):
    """Per-step |update| once Adam's moments have converged (m->g, v->g^2).

    Closed form of the steady state: lr * |g| / (|g| + eps).
    """
    _need_numpy()
    return lr * abs(g) / (abs(g) + eps)


def dampening_ratio(g_small=1e-4, eps_low=1e-8, eps_high=1e-2, steps=500,
                    lr=2e-4):
    """drift(eps_high) / drift(eps_low) for a persistent tiny gradient.

    The paper's claim in miniature: raising eps for the output projection
    removes most of the rare-token drift.
    """
    _need_numpy()
    d_low = adam_row_drift(g_small, steps, lr=lr, eps=eps_low)
    d_high = adam_row_drift(g_small, steps, lr=lr, eps=eps_high)
    return d_high / d_low if d_low > 0 else float("nan")
