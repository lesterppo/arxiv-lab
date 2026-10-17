"""Training-Free On-Policy Distillation, TF-OPD (arXiv:2610.22236).

Paper: "Training-Free On-Policy Distillation" (Yang et al.).

One-line claim: specialists that share a base model can be consolidated
into one model with forward passes only — each linear layer of the fused
model is the least-squares solution that reproduces every specialist's
output on the inputs that specialist receives while reading its own
answers; the solution needs only the second moments (Gram matrices) of
those inputs. What the rollouts contain matters more than the solver:
answers truncated in repetition loops must be filtered out.

Implements the paper's core:

- ``accumulate_gram(X)``: G = X^T X / n over rollout token rows (§3.2:
  the sum divided by the number of tokens so long answers do not
  dominate).
- ``shrink_gram(G, alpha=0.9)``: paper Eq. 3, off-diagonal shrinkage
  toward diag(G) for conditioning.
- ``fuse_linear_layer(W_base, deltas, grams, weights=None, alpha=0.9)``:
  paper Eq. 2, solved in float64,
      W_* = W_b + (sum_d w_d Delta_d G~_d)(sum_d w_d G~_d)^{-1},
  with a pseudo-inverse fallback (the paper takes the inverse as the
  pseudo-inverse when ranges are orthogonal, Prop. 3.1(iii)).
- ``unreproduced_share(Delta_star, Delta_d, gram_tilde_d)``: paper Eq. 4,
  the share of teacher d's update the fused layer leaves unreproduced.
- ``normal_equation_residual(...)``: exactness diagnostic of the solve.
- ``detect_repetition_loop(text)``: paper §3.4 — an answer is flagged
  when its last 20,000 characters compress under DEFLATE to < 5% of
  their length.
- ``fuse_layer_dict(...)``: convenience fusion over a dict of named
  layers; biases/non-linear-map params combine by task arithmetic
  (paper §3.3: they "change little").

Weight convention: W has shape (d_in, d_out), Y = X @ W.

numpy-guarded per repo conventions.
"""

from __future__ import annotations

import math
import zlib

try:
    import numpy as np
except Exception:  # pragma: no cover
    np = None


def _need_numpy():
    if np is None:
        raise RuntimeError("numpy is required for arxiv_lab.training.tf_opd")


# ---------------------------------------------------------------------------
# Statistics: second moments of layer inputs on a specialist's own rollouts
# ---------------------------------------------------------------------------

def accumulate_gram(X):
    """Gram matrix G = X^T X / n over rollout rows (§3.2).

    ``X``: 2-D array, one row per token position of the layer's input.
    Returns the second-moment matrix, divided by the number of rows so
    that long answers do not dominate the statistics.
    """
    _need_numpy()
    X = np.asarray(X, dtype=np.float64)
    n = X.shape[0]
    if n == 0:
        d = X.shape[1]
        return np.zeros((d, d), dtype=np.float64)
    return (X.T @ X) / float(n)


def shrink_gram(G, alpha=0.9):
    """Off-diagonal shrinkage toward the diagonal, paper Eq. 3.

    G~ = alpha * G + (1 - alpha) * diag(G).  Makes the system well
    conditioned; equivalent to a per-coordinate penalty pulling W
    toward each teacher.
    """
    _need_numpy()
    G = np.asarray(G, dtype=np.float64)
    return alpha * G + (1.0 - alpha) * np.diag(np.diag(G))


# ---------------------------------------------------------------------------
# The exact layer-wise solution (paper Eq. 1 -> Eq. 2)
# ---------------------------------------------------------------------------

def fuse_linear_layer(W_base, deltas, grams, weights=None, alpha=0.9):
    """Fuse one linear layer from K specialists sharing a base, Eq. 2.

    Paper (column-vector form)::
        W_* = W_b + (sum_d w_d Delta_d G~_d)(sum_d w_d G~_d)^{-1}
    Implemented in the row-vector convention Y = X @ W, W (d_in, d_out)::
        W_* = W_b + (sum_d w_d G~_d)^{-1}(sum_d w_d G~_d Delta_d)

    ``W_base``: (d_in, d_out) base weight.
    ``deltas``: list of (d_in, d_out) Delta_d = W_d - W_b.
    ``grams``:  list of (d_in, d_in) second-moment matrices G_d.
    ``weights``: optional domain weights w_d (default: all 1).
    ``alpha``: off-diagonal shrinkage (Eq. 3), default 0.9.

    Solved in float64 (§3.5); returns (W_star, Delta_star, info) where
    info carries the shrunk grams, S, T and the normal-equation
    residual.
    """
    _need_numpy()
    Wb = np.asarray(W_base, dtype=np.float64)
    Ds = [np.asarray(d, dtype=np.float64) for d in deltas]
    Gs = [np.asarray(g, dtype=np.float64) for g in grams]
    k = len(Ds)
    ws = [1.0] * k if weights is None else list(weights)
    Gt = [w * shrink_gram(G, alpha) for w, G in zip(ws, Gs)]
    S = sum(Gt)
    T = sum(w * (G @ D) for w, D, G in zip(ws, Ds, Gt))
    # Pseudo-inverse: exact inverse when S is well conditioned, the
    # orthogonal-range limit of Prop. 3.1(iii) otherwise.
    Sinv = np.linalg.pinv(S)
    Delta_star = Sinv @ T
    W_star = Wb + Delta_star
    resid = np.linalg.norm(S @ Delta_star - T)
    scale = np.linalg.norm(S @ Delta_star) + np.linalg.norm(T)
    info = {
        "grams_tilde": Gt,
        "S": S,
        "T": T,
        "normal_eq_rel_residual": float(resid / scale) if scale > 0 else 0.0,
        "alpha": alpha,
        "weights": ws,
    }
    return W_star, Delta_star, info


def unreproduced_share(Delta_star, Delta_d, gram_tilde_d):
    """Paper Eq. 4: share of teacher d's update the fused layer leaves
    unreproduced,

        eps_d = tr[(D* - D_d)^T G~_d (D* - D_d)] / tr[D_d^T G~_d D_d].

    (Row-vector convention Y = X @ W; the paper's column form is the
    transpose of this.) Zero when the fused layer matches the teacher on
    its own inputs.
    """
    _need_numpy()
    Ddiff = np.asarray(Delta_star, dtype=np.float64) - np.asarray(
        Delta_d, dtype=np.float64
    )
    G = np.asarray(gram_tilde_d, dtype=np.float64)
    num = float(np.trace(Ddiff.T @ G @ Ddiff))
    Dd = np.asarray(Delta_d, dtype=np.float64)
    den = float(np.trace(Dd.T @ G @ Dd))
    return num / den if den > 0 else 0.0


def weighted_objective(W, W_base, deltas, grams_tilde, weights=None):
    """The distillation objective of Eq. 1 with the (shrunk) grams:
    J(W) = sum_d w_d tr[(W - W_d)^T G~_d (W - W_d)]."""
    _need_numpy()
    W = np.asarray(W, dtype=np.float64)
    Wb = np.asarray(W_base, dtype=np.float64)
    k = len(deltas)
    ws = [1.0] * k if weights is None else list(weights)
    total = 0.0
    for w, D, G in zip(ws, deltas, grams_tilde):
        Dd = np.asarray(D, dtype=np.float64)
        Wdiff = (W - Wb) - Dd
        G = np.asarray(G, dtype=np.float64)
        total += w * float(np.trace(Wdiff.T @ G @ Wdiff))
    return total


# ---------------------------------------------------------------------------
# Rollout filtering: repetition-loop detection (paper §3.4)
# ---------------------------------------------------------------------------

def detect_repetition_loop(text, window=20000, ratio=0.05):
    """Flag an answer truncated in a repetition loop, without supervision.

    Paper §3.4: an answer is flagged when the last ``window`` characters
    compress under DEFLATE to less than ``ratio`` of their length — far
    below ordinary reasoning text. ``zlib`` implements DEFLATE.
    """
    if not text:
        return False
    tail = text[-window:]
    raw = tail.encode("utf-8", errors="ignore")
    if len(raw) == 0:
        return False
    compressed = zlib.compress(raw, 9)
    return (len(compressed) / len(raw)) < ratio


def compression_ratio(text, window=20000):
    """Helper: DEFLATE size of the last ``window`` chars as a fraction of
    raw size (diagnostic for the filter threshold)."""
    if not text:
        return 1.0
    tail = text[-window:]
    raw = tail.encode("utf-8", errors="ignore")
    if len(raw) == 0:
        return 1.0
    return len(zlib.compress(raw, 9)) / len(raw)


# ---------------------------------------------------------------------------
# Convenience: fuse a whole named layer dict
# ---------------------------------------------------------------------------

def fuse_layer_dict(base_layers, specialist_deltas, specialist_grams,
                    bias_policy="task_arithmetic", alpha=0.9, weights=None):
    """Fuse a dict of named layers.

    ``base_layers``: {name: W_base}. ``specialist_deltas`` /
    ``specialist_grams``: lists (one per specialist) of {name: array}.
    Names present in ``specialist_grams`` are fused with Eq. 2; all other
    names (biases, norms — "not linear maps of a hidden state", §3.3)
    combine by task arithmetic (``bias_policy``).
    Returns {name: fused array}.
    """
    _need_numpy()
    fused = {}
    names = set(base_layers)
    for name in names:
        Wb = np.asarray(base_layers[name], dtype=np.float64)
        if all(name in g for g in specialist_grams):
            Ds = [np.asarray(d[name], dtype=np.float64)
                  for d in specialist_deltas]
            Gs = [np.asarray(g[name], dtype=np.float64)
                  for g in specialist_grams]
            W_star, _, _ = fuse_linear_layer(Wb, Ds, Gs,
                                             weights=weights, alpha=alpha)
            fused[name] = W_star
        else:
            if bias_policy == "task_arithmetic":
                acc = Wb.copy()
                for d in specialist_deltas:
                    if name in d:
                        acc = acc + np.asarray(d[name], dtype=np.float64)
                fused[name] = acc
            elif bias_policy == "average":
                tot = Wb.copy()
                n = 1
                for d in specialist_deltas:
                    if name in d:
                        tot = tot + (Wb + np.asarray(d[name],
                                                   dtype=np.float64))
                        n += 1
                fused[name] = tot / n
            else:
                raise ValueError("unknown bias_policy: %r" % bias_policy)
    return fused


# ---------------------------------------------------------------------------
# Toy harness helpers (used by the CPU test; stdlib + numpy only)
# ---------------------------------------------------------------------------

def mlp_forward(X, params):
    """2-layer MLP: tanh(W1) -> linear(W2). params: W1,b1,W2,b2."""
    _need_numpy()
    z1 = X @ params["W1"] + params["b1"]
    h = np.tanh(z1)
    y = h @ params["W2"] + params["b2"]
    return y, {"z1": z1, "h": h}


def train_specialist_sgd(params, X, y, steps=400, lr=0.05, seed=0):
    """Deterministic full-batch gradient descent on MSE from ``params``.

    Returns the trained params dict. Pure numpy, seeded (only used to
    shuffle is off — fully deterministic already; seed kept for API).
    """
    _need_numpy()
    rng = np.random.default_rng(seed)  # reserved; kept for determinism contract
    P = {k: np.asarray(v, dtype=np.float64).copy() for k, v in params.items()}
    n = X.shape[0]
    for _ in range(steps):
        yhat, cache = mlp_forward(X, P)
        err = (yhat - y) / n                      # d(MSE)/dyhat
        dW2 = cache["h"].T @ err
        db2 = err.sum(axis=0)
        dh = err @ P["W2"].T
        dz1 = dh * (1.0 - np.tanh(cache["z1"]) ** 2)
        dW1 = X.T @ dz1
        db1 = dz1.sum(axis=0)
        P["W2"] -= lr * dW2
        P["b2"] -= lr * db2
        P["W1"] -= lr * dW1
        P["b1"] -= lr * db1
    return P


def collect_layer_inputs(params, answers, layer_names=("W1", "W2")):
    """Record each linear layer's input rows while the model "reads its
    own answers" (one forward pass per answer).

    ``answers``: list of 2-D arrays (rows = token positions).
    Returns {name: stacked input matrix}.
    """
    _need_numpy()
    bufs = {name: [] for name in layer_names}
    for A in answers:
        A = np.asarray(A, dtype=np.float64)
        if "W1" in bufs:
            bufs["W1"].append(A)
        if "W2" in bufs:
            _, cache = mlp_forward(A, params)
            bufs["W2"].append(cache["h"])
    return {name: (np.vstack(b) if b else np.zeros((0, 0)))
            for name, b in bufs.items()}
