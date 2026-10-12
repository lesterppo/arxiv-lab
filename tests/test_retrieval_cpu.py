"""CPU tests for arxiv_lab.retrieval (RATED, arXiv:2610.13889).

Ports sessions/2026-10-12-2610.13889-rated/rated_sim.py.

Claim A — fault adaptation: under a severe outage of popular tools, the
hybrid re-rank (0.75*semantic + 0.10*static + 0.15*discounted-LinUCB)
cuts the post-outage fault selection rate far below the semantic-only
baseline, without retraining.
Claim B — ranking quality: post-outage success rate and MRR (hits must
be *working* positives) both beat the semantic-only baseline.
Claim C — latency: re-rank cost per candidate stays under 1 ms
(the paper's <1 ms overhead claim; per-query cost scales with the
candidate count).

Seeded and deterministic. Requires numpy (optional dependency); skipped
gracefully without it.

Run:  python3 tests/test_retrieval_cpu.py
"""

import os
import sys
import time

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "sessions", "2026-10-12-2610.13889-rated"))

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

from arxiv_lab.retrieval import (
    DiscountedLinUCB,
    build_hybrid_score,
    cosine_topk,
    execution_reward,
    fault_selection_rate,
    mrr_at_k,
    rerank_hybrid,
)


def test_cosine_topk_ranks_by_similarity():
    rng = np.random.default_rng(0)
    m = rng.normal(size=(50, 8))
    q = m[7] + 0.01 * rng.normal(size=8)
    idx, scores = cosine_topk(q, m, 5)
    assert int(idx[0]) == 7, "nearest vector must rank first"
    assert all(a >= b for a, b in zip(scores, scores[1:])), \
        "scores must be descending"
    print("  topk ok: first=%d score=%.4f" % (idx[0], scores[0]))


def test_linucb_unseen_arm_scores_alpha():
    b = DiscountedLinUCB(dim=8, alpha=0.1, gamma=0.95, ridge=1.0)
    x = np.ones(8) / np.sqrt(8)
    s = b.score(3, x)
    assert abs(s - 0.1) < 1e-9, \
        "unseen arm: mu=0, sigma=|x|=1 -> score must equal alpha"
    assert b.n_arms == 1
    print("  unseen-arm score=%.4f ok" % s)


def test_linucb_update_moves_mean_toward_reward():
    b = DiscountedLinUCB(dim=8, alpha=0.0, gamma=1.0, ridge=1.0)
    rng = np.random.default_rng(1)
    x = rng.normal(size=8)
    x /= np.linalg.norm(x)
    b.update(5, x, 1.0)
    m1 = b.mean(5, x)
    assert m1 > 0.3, "one +1 reward must pull the mean clearly positive"
    b.update(5, x, -2.0)
    m2 = b.mean(5, x)
    assert m2 < m1, "a -2 reward must pull the mean back down"
    assert m2 < 0.0, "net reward is negative -> mean must go negative"
    print("  means: +1 -> %.3f, then -2 -> %.3f ok" % (m1, m2))


def test_linucb_is_contextual():
    # same arm, two orthogonal contexts: learning on one must barely move
    # the other (this is the "contextual" in contextual bandit)
    b = DiscountedLinUCB(dim=8, alpha=0.0, gamma=1.0, ridge=1.0)
    x1 = np.zeros(8)
    x1[0] = 1.0
    x2 = np.zeros(8)
    x2[1] = 1.0
    for _ in range(5):
        b.update(9, x1, -2.0)
    assert b.mean(9, x1) < -0.5, "trained context must be demoted"
    assert abs(b.mean(9, x2)) < 0.05, \
        "orthogonal context must stay near zero (no cross-query leakage)"
    print("  contextual ok: trained=%.3f orthogonal=%.3f"
          % (b.mean(9, x1), b.mean(9, x2)))


def test_hybrid_score_weights():
    h = build_hybrid_score(0.8, 0.75, 0.5)
    assert abs(h - (0.75 * 0.8 + 0.10 * 0.75 + 0.15 * 0.5)) < 1e-12
    print("  hybrid=%.4f ok" % h)


def test_execution_reward_shaping():
    r_ok = execution_reward(0.9, 0.75, True, False)
    r_bad = execution_reward(0.9, 0.75, False, True)
    r_wrong = execution_reward(0.9, 0.75, False, False)
    assert r_ok > 0.5 > r_wrong > r_bad, \
        "success > irrelevant > picked-failing-tool"
    assert r_bad < -1.0, "picking a failing tool must be strongly negative"
    print("  rewards: ok=%.3f wrong=%.3f failed=%.3f ok"
          % (r_ok, r_wrong, r_bad))


def test_metrics():
    assert mrr_at_k([5, 3, 8], [3], k=10) == 0.5
    assert mrr_at_k([5, 7], [3], k=10) == 0.0
    # broken tool ranked first earns nothing even if "relevant"
    assert mrr_at_k([3, 9], [9], k=10) == 0.5
    assert abs(fault_selection_rate([1, 2, 3, 4], [2, 4]) - 0.5) < 1e-12
    assert fault_selection_rate([], []) == 0.0
    print("  metrics ok")


def test_rerank_hybrid_baseline_is_semantic():
    rng = np.random.default_rng(2)
    bandit = DiscountedLinUCB(dim=8)
    ids = np.arange(10)
    sem = rng.random(10)
    x = rng.normal(size=8)
    r_ids, _ = rerank_hybrid(ids, sem, bandit, x, use_bandit=False)
    assert list(r_ids) == list(np.argsort(-sem, kind="stable")), \
        "disabled bandit must reproduce the semantic order exactly"
    print("  disable-bandit == semantic order ok")


def test_full_scenario_reproduces_paper_claims():
    from rated_sim import run_scenario
    out = run_scenario(seed=7, n_queries=700, fail_step=200, verbose=False)
    b, r = out["baseline"], out["rated"]
    print("  baseline post: success=%.3f MRR=%.3f fault=%.1f%%" % (
        b["success_post"], b["mrr_post"],
        100 * b["fault_selection_rate_post"]))
    print("  rated    post: success=%.3f MRR=%.3f fault=%.1f%%" % (
        r["success_post"], r["mrr_post"],
        100 * r["fault_selection_rate_post"]))
    # Claim A: fault adaptation without retraining
    assert out["post_fault_reduction_pp"] >= 15.0, \
        "bandit must cut the post-outage fault rate substantially"
    # Claim B: ranking quality preserved / improved
    assert out["post_success_gain_pp"] > 0, \
        "post-outage success must beat the semantic-only baseline"
    assert out["post_mrr_gain_rel"] > 0, \
        "post-outage MRR must beat the semantic-only baseline"
    # Claim C: <1 ms re-rank overhead per candidate
    assert out["rerank_latency_ms"]["per_candidate_mean"] < 1.0, \
        "per-candidate re-rank must stay under 1 ms"
    print("  scenario claims A/B/C ok")


def test_scenario_determinism():
    from rated_sim import run_scenario
    a = run_scenario(seed=123, n_queries=120, fail_step=40, verbose=False)
    b = run_scenario(seed=123, n_queries=120, fail_step=40, verbose=False)
    assert a["rated"]["success_post"] == b["rated"]["success_post"]
    assert (a["rated"]["fault_selection_rate_post"]
            == b["rated"]["fault_selection_rate_post"])
    print("  determinism ok")


def main():
    if not HAS_NUMPY:
        print("SKIPPED: numpy not installed (optional dependency)")
        return
    t0 = time.time()
    test_cosine_topk_ranks_by_similarity()
    test_linucb_unseen_arm_scores_alpha()
    test_linucb_update_moves_mean_toward_reward()
    test_linucb_is_contextual()
    test_hybrid_score_weights()
    test_execution_reward_shaping()
    test_metrics()
    test_rerank_hybrid_baseline_is_semantic()
    test_full_scenario_reproduces_paper_claims()
    test_scenario_determinism()
    print("ALL RATED CPU TESTS PASSED in %.1fs" % (time.time() - t0))


if __name__ == "__main__":
    main()
