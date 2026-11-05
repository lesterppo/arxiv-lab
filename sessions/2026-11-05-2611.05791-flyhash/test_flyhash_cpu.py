"""CPU test: FlyHash for LLM agent memory (arXiv:2611.05791).

Outcome (measured, see the session README for the full sweep): NEGATIVE
for the paper's Q1 quality claim under a matched-bit protocol on synthetic
data — SimHash beats FlyHash on every seed and every dataset below, and
the gap only closes when k reaches 50% of m, where the code is no longer
sparse. Q3 (non-circular fact-id labels) shows the same ordering. Q2's
nuance is confirmed in one respect: at m=512, k=26, inverted-index
candidate pruning barely prunes (candidate fraction ~0.86), so the
sparse-set comparison does not rescue latency at ordinary scale.

Assertions encode the measured result, not the paper's claim: they fail
if a future implementation change flips the ordering, which is the point
of banking a negative.
"""

import math
import os
import random
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

from flyhash import (
    cosine_topk,
    dense_bytes,
    flyhash_codes,
    hamming_topk,
    inverted_index_candidates,
    make_clustered,
    make_sparse_binary,
    overlap_topk,
    recall_at_k,
    simhash_codes,
    sparse_set_bytes,
)


def _sweep(name, maker):
    fly, sim = [], []
    for seed in range(10):
        if name == "clustered":
            db, q = maker(seed=seed)
        else:
            db, q, _ = maker(seed=seed)
        truth = cosine_topk(db, q)
        fc = flyhash_codes(db, seed=1000 + seed)
        fq = flyhash_codes(q, seed=1000 + seed)
        sc = simhash_codes(db, seed=2000 + seed)
        sq = simhash_codes(q, seed=2000 + seed)
        fly.append(recall_at_k(hamming_topk(fc, fq), truth))
        sim.append(recall_at_k(hamming_topk(sc, sq), truth))
    return float(np.mean(fly)), float(np.mean(sim)), fly, sim


def _fact_data(seed):
    rng = random.Random(seed)
    vocab = ["w%d" % i for i in range(300)]
    facts = [rng.sample(vocab, 6) for _ in range(60)]

    def doc(kw):
        words = kw + rng.sample(vocab, 12)
        rng.shuffle(words)
        return words

    db_docs, db_labels = [], []
    for fi, kw in enumerate(facts):
        for _ in range(5):
            db_docs.append(doc(kw))
            db_labels.append(fi)
    q_docs, q_labels = [], []
    for fi, kw in enumerate(facts):
        for _ in range(2):
            q_docs.append(doc(kw))
            q_labels.append(fi)
    df = Counter()
    for d in db_docs + q_docs:
        for t in set(d):
            df[t] += 1
    n_all = len(db_docs) + len(q_docs)
    idx = {t: i for i, t in enumerate(vocab)}

    def vec(d):
        c = Counter(d)
        v = np.zeros(len(vocab))
        for t, cnt in c.items():
            v[idx[t]] = cnt * math.log(n_all / df[t])
        return v

    return (np.array([vec(d) for d in db_docs]),
            np.array([vec(d) for d in q_docs]), db_labels, q_labels)


def _fact_recall(retrieved, db_labels, q_labels):
    scores = []
    for r, ql in zip(retrieved, q_labels):
        rel = {i for i, lab in enumerate(db_labels) if lab == ql}
        scores.append(len(set(int(x) for x in r) & rel) / len(rel))
    return float(np.mean(scores))


def main():
    # Q1: matched 512-bit codes, recall@10 vs exact cosine truth.
    cl_fly, cl_sim, cl_fly_seeds, cl_sim_seeds = _sweep("clustered",
                                                         make_clustered)
    sp_fly, sp_sim, sp_fly_seeds, sp_sim_seeds = _sweep("sparse",
                                                         make_sparse_binary)
    print("clustered: fly=%.4f sim=%.4f (10 seeds)" % (cl_fly, cl_sim))
    print("sparse:    fly=%.4f sim=%.4f (10 seeds)" % (sp_fly, sp_sim))
    assert all(s < t for s, t in zip(cl_fly_seeds, cl_sim_seeds)), \
        "clustered: FlyHash beat SimHash on some seed"
    assert all(s < t for s, t in zip(sp_fly_seeds, sp_sim_seeds)), \
        "sparse: FlyHash beat SimHash on some seed"
    assert cl_sim - cl_fly > 0.10, "clustered gap too small to bank"
    assert sp_sim - sp_fly > 0.10, "sparse gap too small to bank"

    # Fixed-k identity: Hamming distance = 2k - 2*overlap, so the two
    # rankings coincide exactly. The paper's Hamming-vs-overlap contrast
    # is therefore not a second experiment for fixed-k codes.
    db, q = make_clustered(seed=0)
    fc = flyhash_codes(db, seed=1000)
    fq = flyhash_codes(q, seed=1000)
    ham = hamming_topk(fc, fq)
    ov = overlap_topk(fc, fq)
    assert all(list(map(int, h)) == list(map(int, o)) for h, o in zip(ham, ov)), \
        "Hamming and overlap rankings differ for fixed-k codes"

    # Q2 storage: sparse FlyHash < SimHash < dense float32.
    assert sparse_set_bytes(26, 512) < 512 // 8 < dense_bytes(48)
    assert sparse_set_bytes(26, 512) == 52

    # Q2 pruning at ordinary scale (m=512, k=26): barely prunes.
    fracs = []
    for qi in range(len(q)):
        _, frac = inverted_index_candidates(fc, fq[qi])
        fracs.append(frac)
    mean_frac = float(np.mean(fracs))
    print("inverted-index mean candidate fraction=%.4f" % mean_frac)
    assert 0.70 < mean_frac < 0.95, "candidate fraction outside measured band"

    # Q3: non-circular fact-id labels, 5 seeds.
    q3_fly, q3_sim = [], []
    for seed in range(5):
        db2, q2, dl, ql = _fact_data(seed)
        fc2 = flyhash_codes(db2, seed=1000 + seed)
        fq2 = flyhash_codes(q2, seed=1000 + seed)
        sc2 = simhash_codes(db2, seed=2000 + seed)
        sq2 = simhash_codes(q2, seed=2000 + seed)
        q3_fly.append(_fact_recall(hamming_topk(fc2, fq2), dl, ql))
        q3_sim.append(_fact_recall(hamming_topk(sc2, sq2), dl, ql))
    print("fact-id recall: fly=%.4f sim=%.4f (5 seeds)"
          % (float(np.mean(q3_fly)), float(np.mean(q3_sim))))
    assert float(np.mean(q3_sim)) - float(np.mean(q3_fly)) > 0.10

    print("FLYHASH CPU TEST: NEGATIVE reproduced "
          "(fly < sim on clustered, sparse, and fact-id tasks)")
    print("RESULTS clustered_fly=%.4f clustered_sim=%.4f "
          "sparse_fly=%.4f sparse_sim=%.4f fact_fly=%.4f fact_sim=%.4f "
          "candidate_frac=%.4f"
          % (cl_fly, cl_sim, sp_fly, sp_sim,
             float(np.mean(q3_fly)), float(np.mean(q3_sim)), mean_frac))


if __name__ == "__main__":
    main()
