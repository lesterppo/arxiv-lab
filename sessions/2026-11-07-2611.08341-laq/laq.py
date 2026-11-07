"""Locality-aware quantization for compressed agent memory.

arXiv:2611.08341 — "Enhancing Local and Global Structure in Compressed
AI Agent Memory". Core claims implemented here:
  C1  A compression can keep local fidelity L (10-NN survival) high while
      destroying global structure: rescored binary codes reach L=0.97 yet
      move 10.1% of passages to another theme; PCA+8-bit codes reach only
      L=0.77 but move 1.7%.
  C2  Bound: across the global-fidelity band, 1-nu_k <= 2..2.18 * sqrt(r),
      where r is the near-far rank-violation rate; r follows the distance
      noise of the compression (paper: correlation 0.979).
  C3  Locality-aware quantization (LAQ): a hinge objective on the
      codebooks raises global fidelity G where reconstruction alone does
      not — reconstruction raises G in 0/52 held-out runs, the far hinge
      in all 52 (p=8.5e-4 over 13 memory-encoder groups).

Bounded CPU adaptation (documented): synthetic themed memories
(content-group near-duplicates + soft Dirichlet theme mixtures, unit
directions) stand in for passage embeddings; PCA rotation + per-dimension
uniform 8-bit quantizer stands in for the OPQ index (same bytes,
rotation, and search); the LAQ hinge is optimized over per-dimension
quantizer clip ranges (the codebook knob) and, as a second arm, over
per-dimension distance weights — both restricted diagonal forms of the
paper's full OPQ codebook optimization, which is the stated limit of
this test. The reconstruction baseline optimizes the SAME clip knobs
for MSE, so the comparison isolates the OBJECTIVE, not the search; and
the hinge starts from the MSE-OPTIMAL clips, because OPQ already
minimizes reconstruction error — the paper's claim is that the hinge
adds global fidelity BEYOND reconstruction. Two guards found necessary
while building this (both recorded in the session README): the hinge
margin is FIXED once from the initial state (a recomputed margin admits
a scale degeneracy), and the clip hinge is accepted only within 1% of
the starting MSE (the reconstruction anchor). Global fidelity G is
operationalized as far-set overlap: for each item, the set of items
ranked 30-50% by distance in the original space, and the fraction of
that set surviving in compressed space. The bound's rank-violation
rate r is the FAR violation rate: for a far-band item k and an outer
(50-70%) item m, P(d_comp(i,m) < d_comp(i,k)) — near-vs-far flips are
~0 once content groups exist, so they cannot govern G; the far/outer
boundary is what G measures. The bound constant is therefore checked
as a measured ratio, not assumed.
"""
import numpy as np

__all__ = [
    "make_memory", "pairwise_sq", "local_fidelity", "global_fidelity",
    "theme_move_rate", "rank_violation_rate", "distance_noise",
    "binary_codes", "binary_distance", "fit_pca", "project",
    "quantize_dequantize", "mse_of", "sample_triples", "hinge_loss",
    "fit_clip_mse", "fit_clip_hinge", "fit_weights_hinge", "apply_weights",
    "fit_weights_viol",
    "sample_far_triples", "far_violation_rate",
]

_RNG = np.random.default_rng


def make_memory(seed, n=2000, d=96, n_themes=8, spread=1.0, noise=1.0,
                alpha=0.7, content_weight=1.0, theme_weight=0.8,
                group_size=8):
    """Themed synthetic memory built from UNIT directions, so the
    signal budget is explicit and dimension-free:
      X = cw * content_dir + tw * (Dirichlet theme mixture) + nw * noise_dir
    - content_dir: one random unit direction per near-duplicate group
      (group_size passages share it): the local structure that makes
      10-NN meaningful, spread across ALL dimensions, so a 96-dim PCA
      projection loses content detail while 768 random 1-bit projections
      keep it (the paper's binary-vs-PCA asymmetry on L).
    - theme mixture: SOFT themes; centroids sit in a few super-clusters
      (overlapping themes, no clean gaps); the far structure (30-50%
      band) depends on theme-mixture similarity and is fragile under
      quantization.
    - noise_dir: a random unit direction scaled by `noise`.
    Rows are unit-normalized. Theme label = argmax mix.
    `spread` scales the super-cluster separation of theme centroids."""
    rng = _RNG(seed)
    # content groups: near-duplicate passages share a direction
    n_groups = max(1, n // group_size)
    U = rng.normal(size=(n_groups, d))
    U /= np.linalg.norm(U, axis=1, keepdims=True)
    group_id = np.repeat(np.arange(n_groups), group_size)[:n]
    rng.shuffle(group_id)
    content = U[group_id]
    # soft themes in super-clusters
    n_super = max(2, n_themes // 3)
    supers = rng.normal(size=(n_super, d))
    supers /= np.linalg.norm(supers, axis=1, keepdims=True)
    super_id = rng.integers(0, n_super, size=n_themes)
    centroids = supers[super_id] + rng.normal(0.0, 0.6 * spread,
                                               size=(n_themes, d))
    centroids /= np.linalg.norm(centroids, axis=1, keepdims=True)
    W = rng.dirichlet(alpha * np.ones(n_themes), size=n)
    theme_part = W @ centroids
    noise_dir = rng.normal(size=(n, d))
    noise_dir /= np.linalg.norm(noise_dir, axis=1, keepdims=True)
    X = (content_weight * content + theme_weight * theme_part
         + noise * noise_dir)
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    return X, W.argmax(axis=1)


def pairwise_sq(X):
    """Squared Euclidean distances (n, n), diagonal set to +inf."""
    sq = (X * X).sum(axis=1, keepdims=True)
    D = sq + sq.T - 2.0 * (X @ X.T)
    np.maximum(D, 0.0, out=D)
    np.fill_diagonal(D, np.inf)
    return D


def _knn_sets(D, k):
    idx = np.argpartition(D, k, axis=1)[:, :k]
    return [set(row.tolist()) for row in idx]


def local_fidelity(D_orig, D_comp, k=10):
    """Mean fraction of each item's true k-NN surviving compression."""
    orig = _knn_sets(D_orig, k)
    comp = _knn_sets(D_comp, k)
    return float(np.mean([len(a & b) / k for a, b in zip(orig, comp)]))


def global_fidelity(D_orig, D_comp, lo=0.30, hi=0.50):
    """Far-set overlap: for each item, items ranked lo..hi of n by
    distance in the original space; fraction surviving in compressed
    space. This is the G of the paper's 30-50% band."""
    n = D_orig.shape[0]
    a, b = int(lo * n), int(hi * n)
    order_o = np.argsort(D_orig, axis=1)
    order_c = np.argsort(D_comp, axis=1)
    far_o = [set(row[a:b].tolist()) for row in order_o]
    far_c = [set(row[a:b].tolist()) for row in order_c]
    return float(np.mean([len(x & y) / (b - a) for x, y in zip(far_o, far_c)]))


def theme_move_rate(X_orig, X_comp, themes):
    """Fraction of items whose nearest theme centroid changes under
    compression. Centroids are transformed with the data: for a
    quantizer, centroids are quantized the same way."""
    cents = np.stack([X_orig[themes == t].mean(axis=0)
                      for t in np.unique(themes)])
    cents /= np.linalg.norm(cents, axis=1, keepdims=True)
    cents_c = np.stack([X_comp[themes == t].mean(axis=0)
                        for t in np.unique(themes)])
    cents_c /= np.linalg.norm(cents_c, axis=1, keepdims=True)
    near_o = np.argmin(((X_orig[:, None, :] - cents[None]) ** 2).sum(-1), axis=1)
    near_c = np.argmin(((X_comp[:, None, :] - cents_c[None]) ** 2).sum(-1), axis=1)
    return float(np.mean(near_o != near_c))


def sample_triples(D_orig, rng, n_triples=2000, k_near=10, lo=0.30, hi=0.50):
    """Triples (i, j, k): j in i's k-NN, k in i's lo..hi far band.
    Diagnostic only (near-vs-far separation is robust in practice)."""
    n = D_orig.shape[0]
    order = np.argsort(D_orig, axis=1)
    a, b = int(lo * n), int(hi * n)
    ii = rng.integers(0, n, size=n_triples)
    jj = np.array([order[i, rng.integers(0, k_near)] for i in ii])
    kk = np.array([order[i, rng.integers(a, b)] for i in ii])
    return ii, jj, kk


def sample_far_triples(D_orig, rng, n_triples=2000, lo=0.30, hi=0.50,
                       outer_hi=0.70):
    """Triples (i, k, m): k in i's far band (lo..hi of n by distance),
    m in the adjacent outer band (hi..outer_hi). The hinge and the
    bound's violation rate are defined on THESE: G (far-set overlap)
    is governed by whether outer items invade the far set, not by
    near-vs-far flips (which are ~0 once content groups exist)."""
    n = D_orig.shape[0]
    order = np.argsort(D_orig, axis=1)
    a, b, c = int(lo * n), int(hi * n), int(outer_hi * n)
    ii = rng.integers(0, n, size=n_triples)
    kk = np.array([order[i, rng.integers(a, b)] for i in ii])
    mm = np.array([order[i, rng.integers(b, c)] for i in ii])
    return ii, kk, mm


def far_violation_rate(D_orig, D_comp, triples):
    """Fraction of far triples where compression lets an outer item
    invade the far set: d_comp(i,m) < d_comp(i,k)."""
    ii, kk, mm = triples
    return float(np.mean(D_comp[ii, mm] < D_comp[ii, kk]))


def rank_violation_rate(D_orig, D_comp, triples):
    """Fraction of triples where compression flips near/far order:
    d_orig(i,j) < d_orig(i,k) but d_comp(i,j) > d_comp(i,k)."""
    ii, jj, kk = triples
    return float(np.mean(D_comp[ii, jj] > D_comp[ii, kk]))


def distance_noise(D_orig, D_comp, rng, n_pairs=5000):
    """Std of relative distance error over sampled pairs."""
    n = D_orig.shape[0]
    ii = rng.integers(0, n, size=n_pairs)
    jj = rng.integers(0, n, size=n_pairs)
    keep = ii != jj
    ii, jj = ii[keep], jj[keep]
    rel = (np.sqrt(D_comp[ii, jj]) - np.sqrt(D_orig[ii, jj])) / np.sqrt(D_orig[ii, jj])
    return float(np.std(rel))


def binary_codes(X, seed, n_bits=768):
    """Random-projection sign codes (768 bits = 96 bytes/passage).
    Returns (codes {-1,+1}, Q_proj) where Q_proj is the real-valued
    projection used for asymmetric distances."""
    rng = _RNG(seed)
    W = rng.normal(size=(X.shape[1], n_bits)) / np.sqrt(X.shape[1])
    Q_proj = X @ W
    return np.sign(Q_proj), Q_proj


def binary_distance(Q_proj, codes):
    """Squared asymmetric distance between projected queries (n, B) and
    sign codes (n, B): sum_j (q_j - c_j)^2. Diagonal set to +inf (a
    memory compared with itself)."""
    q2 = (Q_proj * Q_proj).sum(axis=1, keepdims=True)
    c2 = (codes * codes).sum(axis=1, keepdims=True).T  # = B per code
    D = q2 + c2 - 2.0 * (Q_proj @ codes.T)
    np.maximum(D, 0.0, out=D)
    np.fill_diagonal(D, np.inf)
    return D


def fit_pca(X, n_keep=None):
    """PCA rotation (top n_keep components; all d if None) + per-dim
    mean/std of projections."""
    mu = X.mean(axis=0)
    Xc = X - mu
    cov = (Xc.T @ Xc) / max(1, X.shape[0] - 1)
    vals, vecs = np.linalg.eigh(cov)
    order = np.argsort(vals)[::-1]
    if n_keep is not None:
        order = order[:n_keep]
    R = vecs[:, order]  # (d, k) columns = components
    Z = Xc @ R
    return mu, R, Z.mean(axis=0), Z.std(axis=0)


def project(X, mu, R):
    return (X - mu) @ R


def quantize_dequantize(Z, z_mean, z_std, clip):
    """Uniform 8-bit quantizer per dim. clip: per-dim half-range in
    units of z_std (array, shape (d,)). Returns dequantized coords."""
    lo = z_mean - clip * z_std
    step = (2.0 * clip * z_std) / 255.0
    step = np.where(step > 0, step, 1.0)
    q = np.round((Z - lo) / step)
    np.clip(q, 0, 255, out=q)
    return lo + (q + 0.5) * step


def mse_of(Z, z_mean, z_std, clip):
    Zhat = quantize_dequantize(Z, z_mean, z_std, clip)
    return float(np.mean((Z - Zhat) ** 2))


def hinge_loss(Z, z_mean, z_std, clip, triples, margin=0.05, margin_abs=None):
    """Mean hinge over FAR triples (i, k, m): max(0, d_ik - d_im + m),
    distances in dequantized space. margin is a fraction of the median
    far distance unless margin_abs is given (the fits pass a FIXED
    margin: a margin recomputed from the current state admits a scale
    degeneracy — shrinking all distances shrinks the margin with it —
    so the fits anchor the margin once, at the initial state)."""
    ii, kk, mm = triples
    Zhat = quantize_dequantize(Z, z_mean, z_std, clip)
    dik = np.sqrt(((Zhat[ii] - Zhat[kk]) ** 2).sum(axis=1))
    dim = np.sqrt(((Zhat[ii] - Zhat[mm]) ** 2).sum(axis=1))
    m = margin_abs if margin_abs is not None else margin * float(np.median(dik))
    return float(np.mean(np.maximum(0.0, dik - dim + m)))


_CLIP_GRID = (1.5, 2.0, 2.5, 3.0, 4.0, 6.0)


def _quant_col(z_col, mean, std, clip):
    """Dequantized values of one projected column under a clip range."""
    lo = mean - clip * std
    step = (2.0 * clip * std) / 255.0
    if step <= 0:
        step = 1.0
    q = np.round((z_col - lo) / step)
    np.clip(q, 0, 255, out=q)
    return lo + (q + 0.5) * step


def fit_clip_mse(Z, z_mean, z_std, sweeps=2):
    """Reconstruction-only baseline: same knobs, MSE objective.
    Incremental: changing one dimension's clip only changes that
    column's MSE, so each candidate costs O(n), not O(n*d)."""
    d = Z.shape[1]
    clip = np.full(d, 3.0)
    col_mse = np.array([float(np.mean((Z[:, j] - _quant_col(
        Z[:, j], z_mean[j], z_std[j], clip[j])) ** 2)) for j in range(d)])
    best = float(col_mse.sum())
    for _ in range(sweeps):
        for j in range(d):
            cur = clip[j]
            for cand in _CLIP_GRID:
                if cand == cur:
                    continue
                m = float(np.mean((Z[:, j] - _quant_col(
                    Z[:, j], z_mean[j], z_std[j], cand)) ** 2))
                if col_mse.sum() - col_mse[j] + m < best - 1e-12:
                    best = col_mse.sum() - col_mse[j] + m
                    col_mse[j] = m
                    cur = cand
            clip[j] = cur
    return clip, best / d


_WEIGHT_GRID = (0.5, 0.75, 1.0, 1.5, 2.0)


def fit_weights_hinge(Z, z_mean, z_std, clip, triples, sweeps=2):
    """LAQ analogue, stronger knob: per-dimension DISTANCE weights on
    the dequantized coords, tuned by the far hinge. Folding a weight
    into its codebook dimension is what a codebook-learning hinge does
    in diagonal form: it upweights the dimensions that separate far
    items and downweights the ones that confuse near/far order.
    clip is held fixed (3.0 in the test); weights start at 1.
    Two guards against the hinge's scale degeneracy (shrinking all
    distances shrinks a recomputed margin with it): the margin is
    FIXED once from the initial state, and the weights are renormalized
    to mean 1 after every sweep, so the search can only REDISTRIBUTE
    weight, never shrink the space.
    Returns (weights, best hinge loss)."""
    ii, kk, mm = triples
    Zhat = quantize_dequantize(Z, z_mean, z_std, clip)
    Dij = (Zhat[ii] - Zhat[kk]) ** 2  # (T, d)
    Dik = (Zhat[ii] - Zhat[mm]) ** 2
    d = Z.shape[1]
    w = np.ones(d)
    sij = Dij.sum(axis=1)
    sik = Dik.sum(axis=1)
    m_fix = 0.05 * float(np.median(np.sqrt(np.maximum(sij, 0.0))))

    def loss_from(sij_, sik_):
        dij = np.sqrt(np.maximum(sij_, 0.0))
        dik = np.sqrt(np.maximum(sik_, 0.0))
        return float(np.mean(np.maximum(0.0, dij - dik + m_fix)))

    best = loss_from(sij, sik)
    for _ in range(sweeps):
        for j in range(d):
            cur = w[j]
            for cand in _WEIGHT_GRID:
                if cand == cur:
                    continue
                val = loss_from(sij + (cand - cur) * Dij[:, j],
                                sik + (cand - cur) * Dik[:, j])
                if val < best - 1e-12:
                    best = val
                    sij = sij + (cand - cur) * Dij[:, j]
                    sik = sik + (cand - cur) * Dik[:, j]
                    cur = cand
            w[j] = cur
        # scale anchor: redistribute, never shrink
        w = w / w.mean()
        sij = Dij @ w
        sik = Dik @ w
        best = loss_from(sij, sik)
    return w, best


def fit_weights_viol(Z, z_mean, z_std, clip, triples, sweeps=2):
    """Distance weights tuned by the FAR VIOLATION RATE itself (the
    quantity the paper's bound controls), not by the hinge: the mean
    hinge optimizes the TAIL (magnitude of the worst violations) and
    can trade many small new violations for a few large fixes, which
    lowers the hinge while RAISING the violation rate and collapsing
    G. The violation rate is a step function, but coordinate descent
    needs no gradients — a candidate is accepted only if it strictly
    reduces the violated-triple count on the training triples. The
    rate is ordering-based, hence scale-invariant by construction.
    Returns (weights, best violation count)."""
    ii, kk, mm = triples
    Zhat = quantize_dequantize(Z, z_mean, z_std, clip)
    Dij = (Zhat[ii] - Zhat[kk]) ** 2  # (T, d)
    Dik = (Zhat[ii] - Zhat[mm]) ** 2
    d = Z.shape[1]
    w = np.ones(d)
    sij = Dij.sum(axis=1)
    sik = Dik.sum(axis=1)

    def viol_count(sij_, sik_):
        return int(np.sum(sik_ < sij_))

    best = viol_count(sij, sik)
    for _ in range(sweeps):
        for j in range(d):
            cur = w[j]
            for cand in _WEIGHT_GRID:
                if cand == cur:
                    continue
                val = viol_count(sij + (cand - cur) * Dij[:, j],
                                 sik + (cand - cur) * Dik[:, j])
                if val < best:
                    best = val
                    sij = sij + (cand - cur) * Dij[:, j]
                    sik = sik + (cand - cur) * Dik[:, j]
                    cur = cand
            w[j] = cur
    return w, best


def apply_weights(Zhat, w):
    """Scale dequantized coords by sqrt(weights) for distance use."""
    return Zhat * np.sqrt(w)[None, :]


def fit_clip_hinge(Z, z_mean, z_std, triples, sweeps=2, clip_start=None):
    """LAQ analogue: same knobs (per-dim clip ranges), far-hinge
    objective. Incremental: changing one dimension's clip only changes
    that column's squared differences, so each candidate costs O(T)
    instead of O(T*d). Two guards (see fit_weights_hinge): the margin
    is FIXED once from the initial state, and a candidate is accepted
    only if it keeps total reconstruction MSE within 1% of the
    STARTING quantizer — the reconstruction anchor the paper's joint
    (hinge + reconstruction) codebook objective provides. Start from
    the MSE-optimal clips (the OPQ-like baseline): the paper's claim is
    that the hinge adds global fidelity BEYOND reconstruction, so the
    hinge must start where reconstruction stopped."""
    ii, kk, mm = triples
    d = Z.shape[1]
    clip = np.array(clip_start, dtype=float) if clip_start is not None \
        else np.full(d, 3.0)
    Zhat = quantize_dequantize(Z, z_mean, z_std, clip)
    Dij = (Zhat[ii] - Zhat[kk]) ** 2  # (T, d) per-dim squared diffs
    Dik = (Zhat[ii] - Zhat[mm]) ** 2
    sij = Dij.sum(axis=1)
    sik = Dik.sum(axis=1)
    m_fix = 0.05 * float(np.median(np.sqrt(np.maximum(sij, 0.0))))
    col_mse = np.array([float(np.mean((Z[:, j] - Zhat[:, j]) ** 2))
                        for j in range(d)])
    mse0 = float(col_mse.sum())

    def loss_from(sij_, sik_):
        dij = np.sqrt(np.maximum(sij_, 0.0))
        dik = np.sqrt(np.maximum(sik_, 0.0))
        return float(np.mean(np.maximum(0.0, dij - dik + m_fix)))

    best = loss_from(sij, sik)
    for _ in range(sweeps):
        for j in range(d):
            cur = clip[j]
            old_ij = Dij[:, j].copy()
            old_ik = Dik[:, j].copy()
            for cand in _CLIP_GRID:
                if cand == cur:
                    continue
                zhat_j = _quant_col(Z[:, j], z_mean[j], z_std[j], cand)
                m_col = float(np.mean((Z[:, j] - zhat_j) ** 2))
                if col_mse.sum() - col_mse[j] + m_col > 1.01 * mse0:
                    continue  # reconstruction anchor: reject
                new_ij = (zhat_j[ii] - zhat_j[kk]) ** 2
                new_ik = (zhat_j[ii] - zhat_j[mm]) ** 2
                val = loss_from(sij - old_ij + new_ij,
                                sik - old_ik + new_ik)
                if val < best - 1e-12:
                    best = val
                    cur = cand
                    Dij[:, j] = new_ij
                    Dik[:, j] = new_ik
                    sij = sij - old_ij + new_ij
                    sik = sik - old_ik + new_ik
                    old_ij, old_ik = new_ij, new_ik
                    col_mse[j] = m_col
            clip[j] = cur
    return clip, best
