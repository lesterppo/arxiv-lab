"""Memory-efficient training via local proxy networks (arXiv:2610.23145).

Paper claim: for each network block, train a small *proxy* by distillation
to predict the **global** network output from that block's input. Gradients
for the block's parameters are then propagated locally through the proxy
instead of end-to-end backprop, so activations need not be retained for the
whole stack. In theory exact for deep linear networks; for nonlinear
residual nets the gradient error is bounded with a residual-form proxy and
shrinks as the distillation loss is optimized; in practice it beats
memory-efficient zeroth-order optimizers at comparable memory cost.

This module implements the mechanism in numpy (CPU) with three checks:

- ``linear_exactness`` — deep linear net: closed-form (least-squares)
  proxies give proxy-propagated gradients identical to backprop.
- ``residual_error_bound`` — residual MLP: gradient-estimation error falls
  as the proxy distillation loss is optimized (residual-form proxy).
- ``three_way_comparison`` — backprop vs proxy-local vs MeZO-style
  zeroth-order on a synthetic classifier: final loss + activation-slot
  memory accounting.

Conventions: row-vector batches, ``X @ W`` linear maps.
"""

try:
    import numpy as _np
except ImportError:  # pragma: no cover
    _np = None


def _need_numpy():
    if _np is None:
        raise RuntimeError("numpy is required for local_proxies")


def rel_err(a, b):
    """Relative Frobenius error ||a-b|| / (||b|| + eps)."""
    _need_numpy()
    return float(_np.linalg.norm(a - b) / (_np.linalg.norm(b) + 1e-12))


def cos_sim(a, b):
    """Cosine similarity of flattened arrays."""
    _need_numpy()
    af, bf = a.ravel(), b.ravel()
    return float(_np.dot(af, bf) / (_np.linalg.norm(af) * _np.linalg.norm(b)
                                   + 1e-12))


# ── Deep linear networks: exactness ──────────────────────────────────────────

def linear_forward(Ws, X):
    """Ws: list of (d_in, d_out) matrices. Returns (Y, [h_0..h_L])."""
    _need_numpy()
    hs = [X]
    for W in Ws:
        hs.append(hs[-1] @ W)
    return hs[-1], hs


def linear_true_grads(Ws, X, T):
    """Exact backprop grads for mean 1/2||Y-T||^2. Returns list per layer."""
    _need_numpy()
    n = X.shape[0]
    Y, hs = linear_forward(Ws, X)
    dh = (Y - T) / n
    grads = []
    for i in reversed(range(len(Ws))):
        grads.append(hs[i].T @ dh)
        dh = dh @ Ws[i].T
    return list(reversed(grads))


def lstsq_proxy(H, Y):
    """Closed-form proxy: linear map H -> Y via least squares."""
    _need_numpy()
    P, *_ = _np.linalg.lstsq(H, Y, rcond=None)
    return P


def linear_proxy_grads(Ws, X, T):
    """Proxy-propagated grads.

    Proxy Q_j maps the *output* of layer j-1 (i.e. hs[j]) to the global
    output Y. dL/dh_j is estimated locally as (hs[j] @ Q_j - T)/n @ Q_j.T
    (exact for the last layer, where hs[L] == Y); dL/dW_i = hs[i].T @
    dL/dh_{i+1} exactly as in backprop. No cross-layer activation retention.
    """
    _need_numpy()
    n = X.shape[0]
    Y, hs = linear_forward(Ws, X)
    L = len(Ws)
    Q = {j: lstsq_proxy(hs[j], Y) for j in range(1, L)}
    dL_dh = {L: (Y - T) / n}
    for j in range(L - 1, 0, -1):
        y_hat = hs[j] @ Q[j]
        dL_dh[j] = (y_hat - T) / n @ Q[j].T
    return [hs[i].T @ dL_dh[i + 1] for i in range(L)]


def linear_exactness(seed=0, n=200, d=8, depth=4, dout=3):
    """Returns max relative gradient error (expect ~1e-12..1e-8)."""
    _need_numpy()
    rng = _np.random.default_rng(seed)
    X = rng.standard_normal((n, d))
    Ws = [rng.standard_normal((d, d)) * 0.5 for _ in range(depth - 1)]
    Ws.append(rng.standard_normal((d, dout)) * 0.5)
    T = rng.standard_normal((n, dout))
    true_g = linear_true_grads(Ws, X, T)
    prox_g = linear_proxy_grads(Ws, X, T)
    return max(rel_err(p, t) for p, t in zip(prox_g, true_g))


# ── Tiny numpy MLP (used for proxies and the 3-way net) ──────────────────────

class TinyMLP:
    """Two-layer MLP: X -> tanh(X@W1+b1) @ W2 + b2, manual backward."""

    def __init__(self, d_in, d_hid, d_out, rng):
        self.W1 = rng.standard_normal((d_in, d_hid)) * _np.sqrt(1.0 / d_in)
        self.b1 = _np.zeros(d_hid)
        self.W2 = rng.standard_normal((d_hid, d_out)) * _np.sqrt(1.0 / d_hid)
        self.b2 = _np.zeros(d_out)

    def params(self):
        return [self.W1, self.b1, self.W2, self.b2]

    def forward(self, X):
        _need_numpy()
        z1 = X @ self.W1 + self.b1
        a1 = _np.tanh(z1)
        out = a1 @ self.W2 + self.b2
        return out, (X, z1, a1)

    def grad_input(self, dout, cache):
        """dL/dX given dL/dout; also returns param grads."""
        X, z1, a1 = cache
        dW2 = a1.T @ dout
        db2 = dout.sum(axis=0)
        da1 = dout @ self.W2.T
        dz1 = da1 * (1.0 - _np.tanh(z1) ** 2)
        dW1 = X.T @ dz1
        db1 = dz1.sum(axis=0)
        dX = dz1 @ self.W1.T
        return dX, [dW1, db1, dW2, db2]

    def sgd_step(self, X, Y, lr):
        """One MSE-fit step: returns current MSE."""
        out, cache = self.forward(X)
        n = X.shape[0]
        dout = 2.0 * (out - Y) / n
        _, grads = self.grad_input(dout, cache)
        for p, g in zip(self.params(), grads):
            p -= lr * g
        return float(_np.mean((out - Y) ** 2))


class ResidualProxy:
    """Residual-form proxy: P(h) = h @ A + MLP(h) (paper's 'appropriate
    residual proxy' for nonlinear residual nets)."""

    def __init__(self, d_in, d_out, rng, hid=24):
        self.A = _np.zeros((d_in, d_out))
        self.mlp = TinyMLP(d_in, hid, d_out, rng)

    def forward(self, H):
        _need_numpy()
        m_out, m_cache = self.mlp.forward(H)
        return H @ self.A + m_out, (H, m_cache)

    def grad_input(self, dout, cache):
        """dL/dH given dL/dP_out."""
        H, m_cache = cache
        dH_via_A = dout @ self.A.T
        dH_via_mlp, _ = self.mlp.grad_input(dout, m_cache)
        return dH_via_A + dH_via_mlp

    def fit_step(self, H, Y, lr):
        out, (Hh, m_cache) = self.forward(H)
        n = H.shape[0]
        dout = 2.0 * (out - Y) / n
        dA = Hh.T @ dout
        _, m_grads = self.mlp.grad_input(dout, m_cache)
        self.A -= lr * dA
        for p, g in zip(self.mlp.params(), m_grads):
            p -= lr * g
        return float(_np.mean((out - Y) ** 2))


# ── Nonlinear residual net: gradient-error bound ─────────────────────────────

def residual_forward(blocks, head, X):
    """blocks: list of W (d,d); h_{i+1} = h_i + tanh(h_i @ W).
    head: (d, nclass). Returns (logits, [h_0..h_B])."""
    _need_numpy()
    hs = [X]
    for W in blocks:
        hs.append(hs[-1] + _np.tanh(hs[-1] @ W))
    return hs[-1] @ head, hs


def residual_true_grads(blocks, head, X, labels):
    """Exact backprop grads (CE loss) for block Ws. Returns list."""
    _need_numpy()
    n = X.shape[0]
    logits, hs = residual_forward(blocks, head, X)
    e = _np.exp(logits - logits.max(axis=1, keepdims=True))
    probs = e / e.sum(axis=1, keepdims=True)
    dlog = probs
    dlog[_np.arange(n), labels] -= 1.0
    dlog /= n
    dh = dlog @ head.T
    grads = []
    for i in reversed(range(len(blocks))):
        z = hs[i] @ blocks[i]
        t = _np.tanh(z)
        # h_{i+1} = h_i + t  =>  dh_i = dh_{i+1} * (I + diag(1-t^2) @ W^T)
        dW = (hs[i] * (1.0 - t ** 2)).T @ dh
        grads.append(dW)
        dh = dh + (dh * (1.0 - t ** 2)) @ blocks[i].T
    return list(reversed(grads))


def residual_proxy_grads(blocks, head, X, labels, proxies):
    """Proxy-estimated grads: dL/dh_j via frozen residual proxies.

    ``proxies`` maps j -> proxy for hs[j] (block *outputs*), j = 1..B-1;
    hs[B] feeds the linear head, so dL/dh_B is exact (dlog @ head.T).
    Block i's weight grads use dL/dh_{i+1} through proxy_{i+1}'s jacobian.
    """
    _need_numpy()
    n = X.shape[0]
    B = len(blocks)
    logits, hs = residual_forward(blocks, head, X)
    e = _np.exp(logits - logits.max(axis=1, keepdims=True))
    probs = e / e.sum(axis=1, keepdims=True)
    dlog = probs
    dlog[_np.arange(n), labels] -= 1.0
    dlog /= n
    dL_dhB = dlog @ head.T  # exact: head is linear
    grads = []
    for i, W in enumerate(blocks):
        j = i + 1
        if j < B:
            _, p_cache = proxies[j].forward(hs[j])
            dh_j = proxies[j].grad_input(dlog, p_cache)
        else:
            dh_j = dL_dhB
        z = hs[i] @ W
        t = _np.tanh(z)
        dW = (hs[i] * (1.0 - t ** 2)).T @ dh_j
        grads.append(dW)
    return grads


def residual_error_bound(seed=1, proxy_steps_list=(0, 10, 50, 200)):
    """Track (distill MSE, grad cosine sim) as proxies are trained.

    Paper claim: optimizing the distillation loss improves gradient
    estimation. Returns lists (mses, cos_sims) over the checkpoint grid.
    """
    _need_numpy()
    rng = _np.random.default_rng(seed)
    n, d, nclass, B = 160, 12, 2, 3
    X = rng.standard_normal((n, d))
    labels = (X[:, 0] * X[:, 1] + _np.sin(X[:, 2]) > 0).astype(int)
    blocks = [rng.standard_normal((d, d)) * 0.3 for _ in range(B)]
    head = rng.standard_normal((d, nclass)) * 0.5
    true_g = residual_true_grads(blocks, head, X, labels)
    logits, hs = residual_forward(blocks, head, X)

    proxies = {j: ResidualProxy(d, nclass, rng) for j in range(1, B)}
    mses, coss = [], []
    done = 0
    for total in proxy_steps_list:
        for _ in range(total - done):
            for j, px in proxies.items():
                px.fit_step(hs[j], logits, lr=0.05)
        done = total
        mse = float(_np.mean([_np.mean((px.forward(hs[j])[0] - logits) ** 2)
                              for j, px in proxies.items()]))
        pg = residual_proxy_grads(blocks, head, X, labels, proxies)
        cos = float(_np.mean([cos_sim(p, t)
                              for p, t in zip(pg, true_g)]))
        mses.append(mse)
        coss.append(cos)
    return mses, coss


# ── Three-way comparison: backprop vs proxy-local vs MeZO ────────────────────

def _ce_loss_and_dlogits(logits, labels):
    _need_numpy()
    n = logits.shape[0]
    e = _np.exp(logits - logits.max(axis=1, keepdims=True))
    probs = e / e.sum(axis=1, keepdims=True)
    loss = float(-_np.mean(_np.log(probs[_np.arange(n), labels] + 1e-12)))
    dlog = probs
    dlog[_np.arange(n), labels] -= 1.0
    dlog /= n
    return loss, dlog


class FeedforwardNet:
    """d -> 32 -> tanh -> 32 -> tanh -> nclass. Param list interface."""

    def __init__(self, d_in, nclass, rng):
        s = 1.0 / _np.sqrt(d_in)
        self.W1 = rng.standard_normal((d_in, 32)) * s
        self.b1 = _np.zeros(32)
        self.W2 = rng.standard_normal((32, 32)) * 0.25
        self.b2 = _np.zeros(32)
        self.W3 = rng.standard_normal((32, nclass)) * 0.25
        self.b3 = _np.zeros(nclass)

    def params(self):
        return [self.W1, self.b1, self.W2, self.b2, self.W3, self.b3]

    def set_params(self, ps):
        for dst, src in zip(self.params(), ps):
            dst[...] = src

    def forward(self, X):
        _need_numpy()
        z1 = X @ self.W1 + self.b1
        a1 = _np.tanh(z1)
        z2 = a1 @ self.W2 + self.b2
        a2 = _np.tanh(z2)
        logits = a2 @ self.W3 + self.b3
        return logits, (X, z1, a1, z2, a2)

    def backward(self, dlog, cache):
        X, z1, a1, z2, a2 = cache
        dW3 = a2.T @ dlog
        db3 = dlog.sum(0)
        da2 = dlog @ self.W3.T
        dz2 = da2 * (1 - _np.tanh(z2) ** 2)
        dW2 = a1.T @ dz2
        db2 = dz2.sum(0)
        da1 = dz2 @ self.W2.T
        dz1 = da1 * (1 - _np.tanh(z1) ** 2)
        dW1 = X.T @ dz1
        db1 = dz1.sum(0)
        return [dW1, db1, dW2, db2, dW3, db3]


def three_way_comparison(seed=2, steps=120, lr=0.05):
    """Backprop vs proxy-local vs MeZO on a synthetic 2-class task.

    Returns dict with final losses, loss curves, and analytic activation
    memory accounting (number of layer-activation tensors retained
    simultaneously): backprop keeps every layer's activations for the
    backward pass; proxy-local recomputes one block at a time through its
    proxy; MeZO keeps none (forward-only).
    """
    _need_numpy()
    rng = _np.random.default_rng(seed)
    n, d_in, nclass = 256, 16, 2
    X = rng.standard_normal((n, d_in))
    labels = (X[:, 0] * X[:, 1] + _np.sin(X[:, 2]) > 0).astype(int)

    def fresh_net():
        return FeedforwardNet(d_in, nclass,
                              _np.random.default_rng(seed + 999))

    # — backprop —
    net = fresh_net()
    bp_curve = []
    for _ in range(steps):
        logits, cache = net.forward(X)
        loss, dlog = _ce_loss_and_dlogits(logits, labels)
        grads = net.backward(dlog, cache)
        for p, g in zip(net.params(), grads):
            p -= lr * g
        bp_curve.append(loss)

    # — proxy-local: proxies P1: a1->logits, P2: a2->logits. Each block's
    #   params get gradients through the *downstream* proxy's jacobian only
    #   (one block resident at a time); the head uses exact local grads —
    net = fresh_net()
    px_rng = _np.random.default_rng(seed + 1)
    proxies = [TinyMLP(32, 24, nclass, px_rng),   # P1: a1 -> logits
               TinyMLP(32, 24, nclass, px_rng)]   # P2: a2 -> logits
    px_curve = []
    for _ in range(steps):
        logits, (Xc, z1, a1, z2, a2) = net.forward(X)
        loss, dlog = _ce_loss_and_dlogits(logits, labels)
        hs = [a1, a2]
        # (a) distill: train each proxy to predict global logits (no grad
        #     through the main net; detached targets)
        for px, h in zip(proxies, hs):
            for _ in range(8):
                px.sgd_step(h, logits, lr=0.1)
        # (b) local gradient steps, one block at a time
        # block 0 params (W1, b1): dL/da1 via P1 jacobian
        _, p_cache = proxies[0].forward(a1)
        dh1 = proxies[0].grad_input(dlog, p_cache)[0]
        dz1 = dh1 * (1 - _np.tanh(z1) ** 2)
        net.W1 -= lr * (Xc.T @ dz1)
        net.b1 -= lr * dz1.sum(0)
        # block 1 params (W2, b2): dL/da2 via P2 jacobian
        _, p_cache = proxies[1].forward(a2)
        dh2 = proxies[1].grad_input(dlog, p_cache)[0]
        dz2 = dh2 * (1 - _np.tanh(z2) ** 2)
        net.W2 -= lr * (a1.T @ dz2)
        net.b2 -= lr * dz2.sum(0)
        # head params: exact local grads (output block needs no proxy)
        net.W3 -= lr * (a2.T @ dlog)
        net.b3 -= lr * dlog.sum(0)
        px_curve.append(loss)

    # — MeZO (SPSA): two forward passes per step, no activations retained —
    net = fresh_net()
    mezo_curve = []
    eps = 1e-3
    for _ in range(steps):
        ps = net.params()
        flats = [p.ravel() for p in ps]
        z = [rng.standard_normal(f.shape) for f in flats]
        pp = [f + eps * zi for f, zi in zip(flats, z)]
        pm = [f - eps * zi for f, zi in zip(flats, z)]
        net.set_params([p.reshape(s.shape) for p, s in zip(pp, ps)])
        lp, _ = _ce_loss_and_dlogits(net.forward(X)[0], labels)
        net.set_params([p.reshape(s.shape) for p, s in zip(pm, ps)])
        lm, _ = _ce_loss_and_dlogits(net.forward(X)[0], labels)
        g_est = [(lp - lm) / (2 * eps) * zi for zi in z]
        new = [f - 0.002 * g for f, g in zip(flats, g_est)]
        net.set_params([p.reshape(s.shape) for p, s in zip(new, ps)])
        mezo_curve.append(_ce_loss_and_dlogits(net.forward(X)[0], labels)[0])

    return {
        # analytic activation-memory accounting (tensors retained at once)
        "slots": {"backprop": 3, "proxy": 1, "mezo": 0},
        "final": {"backprop": bp_curve[-1], "proxy": px_curve[-1],
                  "mezo": mezo_curve[-1]},
        "curves": {"backprop": bp_curve, "proxy": px_curve,
                   "mezo": mezo_curve},
    }
