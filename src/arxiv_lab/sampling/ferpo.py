"""FERPO forward-KL mode-covering sampler (arXiv:2610.02198).

Paper claims implemented: FERPO derives an optimal target action distribution
from an entropy/KL-regularized policy-improvement objective, then fits the
actor by minimizing a *forward-KL* objective estimated with self-normalized
importance sampling (SNIS) over actions drawn from the rollout policy.

  A. By limiting the target distribution's deviation from the rollout
     policy, the KL regularization keeps the SNIS importance weights well
     behaved (effective sample size rises with the regularization
     temperature).
  B. In contrast to reverse-KL objectives, which can favor a subset of the
     target distribution's modes, the forward-KL objective encourages
     coverage of multiple high-value modes.

Setup: 1-D continuous actions; target = 3-mode Gaussian mixture (the
entropy-regularized improvement target); rollout policy = single broad
Gaussian. Forward-KL fit via weighted EM on SNIS samples (exact forward-KL
minimizer for the mixture family); reverse-KL fit via Monte-Carlo
finite-difference Adam with common random numbers.

Requires numpy (optional dependency): ``pip install numpy``.
"""

try:
    import numpy as np
except ImportError:  # pragma: no cover - optional dependency
    np = None


def _need_numpy():
    if np is None:
        raise ImportError(
            "arxiv_lab.sampling requires numpy (optional dependency); "
            "install it with `pip install numpy`"
        )


# Target mixture constants as plain tuples so the module imports without
# numpy; converted to arrays inside the functions that need them.
TGT_MEANS = (-3.0, 0.5, 4.0)
TGT_SIGS = (0.5, 0.5, 0.5)
TGT_WTS = (0.55, 0.30, 0.15)
ROLL_MU, ROLL_SIG = 0.0, 2.0


def log_target(a, alpha=1.0):
    """Log of the regularized improvement target (temperature alpha: smaller
    alpha = spikier target = less KL regularization toward the rollout)."""
    _need_numpy()
    means = np.asarray(TGT_MEANS)[:, None]
    sigs = np.asarray(TGT_SIGS)[:, None]
    wts = np.asarray(TGT_WTS)[:, None]
    comps = (-0.5 * ((a[None, :] - means) / sigs) ** 2
             - np.log(sigs) + np.log(wts)) / alpha
    m = comps.max(0)
    return m + np.log(np.exp(comps - m).sum(0))


def log_rollout(a):
    """Log density of the rollout policy (single broad Gaussian)."""
    _need_numpy()
    return -0.5 * ((a - ROLL_MU) / ROLL_SIG) ** 2 - np.log(ROLL_SIG)


def snis_weights(rng, n, alpha=1.0):
    """SNIS weights for actions drawn from the rollout policy.

    Returns ``(actions, normalized_weights, ess_fraction)``.
    """
    _need_numpy()
    a = rng.normal(ROLL_MU, ROLL_SIG, n)
    lw = log_target(a, alpha) - log_rollout(a)
    lw -= lw.max()
    w = np.exp(lw)
    w /= w.sum()
    ess_frac = float(1.0 / np.sum(w ** 2) / n)
    return a, w, ess_frac


def forward_kl_fit(a, w, n_comp, iters=300):
    """Weighted EM = minimizes KL(target || q) over Gaussian mixtures.

    Returns ``(mu, sigma, pi)`` sorted by component mean.
    """
    _need_numpy()
    rng = np.random.default_rng(0)
    mu = np.linspace(a.min(), a.max(), n_comp)
    sg = np.full(n_comp, a.std())
    pi = np.full(n_comp, 1.0 / n_comp)
    for _ in range(iters):
        logc = (-0.5 * ((a[None, :] - mu[:, None]) / sg[:, None]) ** 2
                - np.log(sg[:, None]) + np.log(pi[:, None]))
        logc -= logc.max(0)
        resp = np.exp(logc)
        resp /= resp.sum(0)
        rw = resp * w[None, :]
        nk = rw.sum(1)
        pi = nk / nk.sum()
        mu = (rw * a[None, :]).sum(1) / nk
        sg = np.sqrt((rw * (a[None, :] - mu[:, None]) ** 2).sum(1) / nk)
    order = np.argsort(mu)
    return mu[order], sg[order], pi[order]


def reverse_kl_fit(rng, n_comp, n_mc=40000, iters=200, lr=0.05, init_mu=None):
    """Minimize KL(q || target) for a Gaussian mixture via MC
    finite-difference Adam with common random numbers (deterministic given
    the rng seed). Returns ``(mu, sigma, pi)`` sorted by component mean.
    """
    _need_numpy()
    z = rng.normal(size=(n_mc, n_comp))
    u = rng.random(n_mc)

    def unpack(p):
        mu_ = p[0:n_comp]
        sg_ = np.exp(p[n_comp:2 * n_comp])
        lg = p[2 * n_comp:]
        ex = np.exp(lg - lg.max())
        return mu_, sg_, ex / ex.sum()

    def revkl(p):
        mu_, sg_, pi_ = unpack(p)
        kidx = np.searchsorted(np.cumsum(pi_), u)
        samp = mu_[kidx] + sg_[kidx] * z[np.arange(n_mc), kidx]
        logq = np.logaddexp.reduce(
            -0.5 * ((samp[None, :] - mu_[:, None]) / sg_[:, None]) ** 2
            - np.log(sg_[:, None]) + np.log(pi_[:, None]), axis=0)
        return float(np.mean(logq - log_target(samp)))

    if init_mu is None:
        init_mu = np.full(n_comp, TGT_MEANS[0])
    p = np.concatenate([np.asarray(init_mu, dtype=float),
                        np.zeros(n_comp), np.zeros(n_comp)])
    m_ = np.zeros_like(p)
    v_ = np.zeros_like(p)
    for t in range(1, iters + 1):
        f0 = revkl(p)
        g = np.zeros_like(p)
        h = 1e-4
        for j in range(len(p)):
            dp = np.zeros_like(p)
            dp[j] = h
            g[j] = (revkl(p + dp) - f0) / h
        m_ = 0.9 * m_ + 0.1 * g
        v_ = 0.999 * v_ + 0.001 * g * g
        p = p - lr * np.sqrt(1 - 0.999 ** t) / (1 - 0.9 ** t) * m_ / (np.sqrt(v_) + 1e-8)
    mu_, sg_, pi_ = unpack(p)
    order = np.argsort(mu_)
    return mu_[order], sg_[order], pi_[order]


def mode_coverage(mu, pi, tol=1.0, min_w=0.08):
    """How many of the 3 target modes have a fitted component with weight >
    ``min_w`` within ``tol`` of the mode center."""
    _need_numpy()
    return sum(bool(np.any((np.abs(mu - tm) < tol) & (pi > min_w)))
               for tm in TGT_MEANS)


def expected_log_target(rng, mu, sg, pi, n=60000):
    """Monte-Carlo estimate of E[log target] under the fitted mixture."""
    _need_numpy()
    k = rng.choice(len(pi), size=n, p=pi)
    s = rng.normal(mu[k], sg[k])
    return float(log_target(s).mean())
