"""FlyHash for LLM agent memory (arXiv:2611.05791).

Core claim (three questions the paper answers precisely):
  Q1 quality — under a correctly *matched-bit* protocol, FlyHash
       (sparse random projection + winner-take-all) preserves neighbour
       structure better per bit than SimHash (random-hyperplane signs)
       for agent-memory retrieval.
  Q2 cost — at equal bits FlyHash codes are larger and slower to compare
       than SimHash codes, and dense-vector (HNSW-style) indexing dominates
       both hash methods at ordinary scale; a sparse-set comparison
       recovers most but not all of a naive latency gap, so hashing's
       practical case is narrower than the quality result alone implies.
  Q3 labels — the quality advantage survives genuine, non-circular
       agent-memory relevance labels (labels from fact identity, not
       from nearest-neighbour-by-embedding proximity).

Mechanism (Dasgupta et al. 2017, as applied by the paper):
  FlyHash: each of ``m`` Kenyon cells sums ``s`` randomly sampled input
  coordinates (Gaussian weights); exactly the top-``k`` activations stay
  on (winner-take-all). Code: a sparse ``m``-bit vector with ``k`` ones.
  SimHash: ``m`` dense Gaussian hyperplanes; code bit = sign of the
  projection. Both codes are ``m`` bits — that matched-bit protocol is
  the paper's corrected comparison; comparing a longer FlyHash code
  against a shorter SimHash code (or Hamming on raw sparse sets against
  dense bits without matching) is the pitfall this module avoids by
  construction: every comparison below takes ``m`` as a shared argument.

Requires numpy (optional dependency): ``pip install numpy``.
"""

import math

try:
    import numpy as np
except ImportError:  # pragma: no cover
    np = None


def _need_numpy():
    if np is None:
        raise ImportError("numpy is required for flyhash")


def flyhash_codes(X, m=512, k=26, s=10, seed=0):
    """FlyHash codes for rows of X (shape (n, d)) -> bool array (n, m).

    Each Kenyon cell samples ``s`` distinct input coordinates and sums
    them with Gaussian weights; the top-``k`` activations per row win.
    """
    _need_numpy()
    X = np.asarray(X, dtype=float)
    n, d = X.shape
    rng = np.random.default_rng(seed)
    W = np.zeros((m, d))
    for row in range(m):
        idx = rng.choice(d, size=min(s, d), replace=False)
        W[row, idx] = rng.standard_normal(len(idx))
    act = X @ W.T
    codes = np.zeros((n, m), dtype=bool)
    # Top-k per row: partition, then threshold on the k-th largest value.
    kth = np.partition(act, m - k, axis=1)[:, m - k:m - k + 1]
    codes = act >= kth
    # Ties at the threshold could give >k ones; break them deterministically
    # by keeping, per row, the first k activations in descending order.
    for i in range(n):
        on = np.flatnonzero(codes[i])
        if len(on) > k:
            order = on[np.argsort(-act[i, on], kind="stable")]
            codes[i, :] = False
            codes[i, order[:k]] = True
    return codes


def simhash_codes(X, m=512, seed=0):
    """SimHash codes for rows of X -> bool array (n, m), same bit budget."""
    _need_numpy()
    X = np.asarray(X, dtype=float)
    rng = np.random.default_rng(seed)
    R = rng.standard_normal((m, X.shape[1]))
    return (X @ R.T) >= 0


_POPCOUNT = None


def _popcount_table():
    global _POPCOUNT
    if _POPCOUNT is None:
        _POPCOUNT = np.array([bin(i).count("1") for i in range(256)],
                             dtype=np.int32)
    return _POPCOUNT


def hamming_topk(db_codes, q_codes, topk=10):
    """Top-``topk`` database rows per query by Hamming distance (packed bytes)."""
    _need_numpy()
    table = _popcount_table()
    db = np.packbits(db_codes, axis=1)
    q = np.packbits(q_codes, axis=1)
    out = []
    for qi in range(len(q)):
        dist = table[db ^ q[qi]].sum(axis=1)
        out.append(np.argsort(dist, kind="stable")[:topk])
    return out


def overlap_topk(db_codes, q_codes, topk=10):
    """Top-``topk`` database rows per query by sparse-set overlap size."""
    _need_numpy()
    sim = q_codes.astype(np.int32) @ db_codes.astype(np.int32).T
    return [np.argsort(-sim[i], kind="stable")[:topk] for i in range(len(q_codes))]


def cosine_topk(db, q, topk=10):
    """Exact cosine top-``topk`` (ground truth for the synthetic sets)."""
    _need_numpy()
    db = np.asarray(db, dtype=float)
    q = np.asarray(q, dtype=float)
    dn = db / np.maximum(np.linalg.norm(db, axis=1, keepdims=True), 1e-12)
    qn = q / np.maximum(np.linalg.norm(q, axis=1, keepdims=True), 1e-12)
    sim = qn @ dn.T
    return [np.argsort(-sim[i], kind="stable")[:topk] for i in range(len(q))]


def recall_at_k(retrieved, truth, topk=10):
    """Mean |retrieved ∩ truth| / topk over queries."""
    scores = []
    for r, t in zip(retrieved, truth):
        scores.append(len(set(int(x) for x in r) & set(int(x) for x in t)) / topk)
    return float(sum(scores) / max(len(scores), 1))


def sparse_set_bytes(k, m):
    """Bytes to store one FlyHash code as sorted cell indices (uint16 if possible)."""
    return k * (2 if m <= 65535 else 4)


def dense_bytes(d):
    """Bytes to store one float32 dense vector of dimension d."""
    return 4 * d


def inverted_index_candidates(db_codes, q_code):
    """Candidate database rows sharing >=1 Kenyon cell with the query.

    Returns (candidate indices, candidate fraction). This is the sparse-set
    comparison the paper introduces: retrieval touches only rows that
    share a cell, not the whole store.
    """
    _need_numpy()
    postings = {}
    for i, row in enumerate(db_codes):
        for cell in np.flatnonzero(row):
            postings.setdefault(int(cell), []).append(i)
    cand = set()
    for cell in np.flatnonzero(q_code):
        cand.update(postings.get(int(cell), []))
    return sorted(cand), len(cand) / max(len(db_codes), 1)


def make_clustered(n=1500, d=48, n_clusters=30, n_queries=150, seed=0,
                   noise=0.35):
    """Clustered Gaussian vectors; queries are noisy copies of centres.

    Ground truth = exact cosine neighbours, so labels are geometric, not
    circular: the generator knows the clusters but recall is scored
    against brute-force cosine, not cluster identity.
    """
    _need_numpy()
    rng = np.random.default_rng(seed)
    centres = rng.standard_normal((n_clusters, d))
    centres /= np.maximum(np.linalg.norm(centres, axis=1, keepdims=True), 1e-12)
    assign = rng.integers(0, n_clusters, size=n)
    db = centres[assign] + noise * rng.standard_normal((n, d))
    q_assign = rng.integers(0, n_clusters, size=n_queries)
    q = centres[q_assign] + noise * rng.standard_normal((n_queries, d))
    return db, q


def make_sparse_binary(n=1500, d=200, density=0.08, n_queries=150, seed=0):
    """Sparse binary (bag-like) vectors, second modality.

    Queries are database rows with 20% of their ones flipped off plus
    5% spurious ones — a noisy-copy retrieval task with exact-cosine
    ground truth, mimicking term-vector agent memories.
    """
    _need_numpy()
    rng = np.random.default_rng(seed)
    db = (rng.random((n, d)) < density).astype(float)
    # An (astronomically rare) all-zero row carries no signal; give it a
    # single active coordinate rather than leaving a zero vector.
    db = np.where(db.sum(axis=1, keepdims=True) == 0, 1.0, db)
    idx = rng.integers(0, n, size=n_queries)
    q = db[idx].copy()
    drop = (rng.random(q.shape) < 0.20) & (q > 0)
    add = (rng.random(q.shape) < 0.05) & (q == 0)
    q = np.where(drop, 0.0, q)
    q = np.where(add, 1.0, q)
    return db, q, idx
