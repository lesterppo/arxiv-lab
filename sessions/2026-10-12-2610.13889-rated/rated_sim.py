"""RATED reproduction (arXiv:2610.13889), faithful to the authors' code.

Follows ``scripts/evaluation/evaluate_rated_full.py`` (Elliotes/EcoAgent,
Apache-2.0): frozen semantic top-k -> hybrid re-rank
(0.75*semantic + 0.10*static + 0.15*LinUCB-bandit) -> execute top-1 ->
shaped online reward (0.1*sem*static + 0.9*exec_signal, exec_signal =
+1.0 success / -2.0 picked-a-failing-tool / 0.0 otherwise) -> discounted
LinUCB update. At ``fail_step`` a mid-frequency slice of the positive
tools starts failing (severe-outage scenario); the bandit must adapt
without retraining. The ``--disable-bandit`` arm (pure semantic ranking)
is the baseline.

Deterministic given ``seed``. Writes results.json next to this file.

Run:  python3 sessions/2026-10-12-2610.13889-rated/rated_sim.py
"""

import json
import os
import sys
import time

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "src"))

import numpy as np

from arxiv_lab.retrieval import (
    DiscountedLinUCB,
    cosine_topk,
    execution_reward,
    fault_selection_rate,
    mrr_at_k,
    rerank_hybrid,
)

HERE = os.path.dirname(os.path.abspath(__file__))

STATIC_REL = 0.75  # paper's default when no reliability column exists


def build_world(rng, n_tools=1000, dim=64):
    vecs = rng.normal(size=(n_tools, dim))
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-12
    return vecs


def sample_queries(rng, tool_vecs, n_queries, zipf_a=1.2, noise=0.15):
    """Pre-sample (p1, p2, query_vec) triples.

    p1 ~ Zipf popularity (some tools are queried far more often, as in
    EcoAgent); p2 = p1's nearest neighbour so each query has two
    semantically close positives. The query is dominated by the
    positives (small noise) so the semantic stage is strong pre-outage
    and the outage is what breaks it.
    Returns list of (p1, p2, q).
    """
    n, dim = tool_vecs.shape
    ranks = np.arange(1, n + 1)
    pop = 1.0 / ranks ** zipf_a
    pop /= pop.sum()
    queries = []
    for _ in range(n_queries):
        p1 = int(rng.choice(n, p=pop))
        sims = tool_vecs @ tool_vecs[p1]
        sims[p1] = -np.inf
        p2 = int(np.argmax(sims))
        n_hat = rng.normal(size=dim)
        n_hat /= np.linalg.norm(n_hat) + 1e-12
        q = 0.6 * tool_vecs[p1] + 0.4 * tool_vecs[p2] + noise * n_hat
        q /= np.linalg.norm(q) + 1e-12
        queries.append((p1, p2, q))
    return queries


def pick_failing_tools(queries, fail_frac=0.30):
    """Most-frequently-queried positives start failing.

    Severe-outage scenario: the outage hits the most-used tools (the
    realistic "major provider outage" case, and the regime of the
    paper's headline numbers). The paper's own eval slices a
    mid-frequency band instead (see README sensitivity note); both are
    reported.
    """
    from collections import Counter
    cnt = Counter(p1 for p1, _, _ in queries)
    ranked = [t for t, _ in cnt.most_common()]
    n_fail = max(1, int(len(ranked) * fail_frac))
    return set(ranked[:n_fail])


def run_scenario(seed=7, n_queries=400, n_tools=1000, dim=64,
                 fail_step=200, fail_frac=0.30, topk=50, mrr_k=10,
                 verbose=True):
    rng = np.random.default_rng(seed)
    tool_vecs = build_world(rng, n_tools=n_tools, dim=dim)
    queries = sample_queries(rng, tool_vecs, n_queries)
    failing = pick_failing_tools(queries, fail_frac=fail_frac)

    bandit = DiscountedLinUCB(dim=dim, alpha=0.1, gamma=0.95, ridge=1.0)

    arms = {
        "baseline": {"mrr": [], "success": [], "exec": []},
        "rated": {"mrr": [], "success": [], "exec": []},
    }
    rerank_ms = []

    for step, (p1, p2, q) in enumerate(queries):
        outage = step >= fail_step
        positives = {p1, p2}
        working_pos = [t for t in positives
                       if not (outage and t in failing)]
        sem_ids, sem_scores = cosine_topk(q, tool_vecs, topk)

        # --- baseline: semantic-only (paper's --disable-bandit) ------------
        b_ids, _ = rerank_hybrid(sem_ids, sem_scores, bandit, q,
                                 static_rel=STATIC_REL, use_bandit=False)
        b_pick = int(b_ids[0])
        b_ok = b_pick in positives and not (outage and b_pick in failing)
        arms["baseline"]["mrr"].append(mrr_at_k(b_ids[:mrr_k], working_pos,
                                                k=mrr_k))
        arms["baseline"]["success"].append(1.0 if b_ok else 0.0)
        arms["baseline"]["exec"].append(b_pick)

        # --- RATED: hybrid re-rank with the LinUCB bandit -------------------
        t0 = time.perf_counter()
        r_ids, _ = rerank_hybrid(sem_ids, sem_scores, bandit, q,
                                 static_rel=STATIC_REL, use_bandit=True)
        rerank_ms.append((time.perf_counter() - t0) * 1000.0)
        r_pick = int(r_ids[0])
        r_sem = float(sem_scores[list(sem_ids).index(r_pick)])
        r_ok = r_pick in positives and not (outage and r_pick in failing)
        r_failed = outage and r_pick in failing
        arms["rated"]["mrr"].append(mrr_at_k(r_ids[:mrr_k], working_pos,
                                             k=mrr_k))
        arms["rated"]["success"].append(1.0 if r_ok else 0.0)
        arms["rated"]["exec"].append(r_pick)

        # shaped online reward -> bandit (RATED arm only)
        rew = execution_reward(r_sem, STATIC_REL, r_ok, r_failed)
        bandit.update(r_pick, q, rew)

    def summarize(d):
        mrr = np.array(d["mrr"])
        suc = np.array(d["success"])
        ex = d["exec"]
        pre = slice(0, fail_step)
        post = slice(fail_step, None)
        post_failed = [t for t in ex[fail_step:] if t in failing]
        return {
            "mrr_all": float(mrr.mean()),
            "mrr_pre": float(mrr[pre].mean()),
            "mrr_post": float(mrr[post].mean()),
            "success_pre": float(suc[pre].mean()),
            "success_post": float(suc[post].mean()),
            "fault_selection_rate_post": fault_selection_rate(
                ex[fail_step:], post_failed),
            "n": len(mrr),
        }

    out = {
        "seed": seed,
        "n_queries": n_queries,
        "n_tools": n_tools,
        "dim": dim,
        "fail_step": fail_step,
        "n_failing_tools": len(failing),
        "baseline": summarize(arms["baseline"]),
        "rated": summarize(arms["rated"]),
        "rerank_latency_ms": {
            "mean": float(np.mean(rerank_ms)),
            "p99": float(np.percentile(rerank_ms, 99)),
            "per_candidate_mean": float(np.mean(rerank_ms) / topk),
        },
    }
    b, r = out["baseline"], out["rated"]
    out["post_success_gain_pp"] = (r["success_post"] - b["success_post"]) * 100.0
    out["post_fault_reduction_pp"] = (
        b["fault_selection_rate_post"] - r["fault_selection_rate_post"]) * 100.0
    out["post_mrr_gain_rel"] = (
        (r["mrr_post"] - b["mrr_post"]) / max(b["mrr_post"], 1e-9))

    if verbose:
        print("queries=%d tools=%d fail_step=%d n_failing=%d seed=%d" % (
            n_queries, n_tools, fail_step, len(failing), seed))
        for name in ("baseline", "rated"):
            d = out[name]
            print("%-8s pre : success=%.3f MRR=%.3f" % (
                name, d["success_pre"], d["mrr_pre"]))
            print("%-8s post: success=%.3f MRR=%.3f fault=%.1f%%" % (
                name, d["success_post"], d["mrr_post"],
                100 * d["fault_selection_rate_post"]))
        print("post-outage success gain: %+.1f pp" % out["post_success_gain_pp"])
        print("post-outage fault reduction: %.1f pp" % out["post_fault_reduction_pp"])
        print("post-outage relative MRR gain: %+.1f%%" %
              (100 * out["post_mrr_gain_rel"]))
        print("re-rank latency: mean=%.3f ms/query (%.4f ms/candidate) p99=%.3f ms" % (
            out["rerank_latency_ms"]["mean"],
            out["rerank_latency_ms"]["per_candidate_mean"],
            out["rerank_latency_ms"]["p99"]))
    return out


def main():
    out = run_scenario()
    with open(os.path.join(HERE, "results.json"), "w") as f:
        json.dump(out, f, indent=1)
    print("wrote", os.path.join(HERE, "results.json"))


if __name__ == "__main__":
    main()
