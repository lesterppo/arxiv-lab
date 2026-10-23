"""CPU tests for arXiv:2610.30755 (Raw-RAG vs transformative agentic memory).

Paper claims under test:
  1. Naive RAG over the RAW history ranks first among the memory systems.
  2. The gap is driven by *coverage* of the necessary evidence (answer
     values plus relation/dependency paths), not by utilization.
  3. Even complete access to a transformed store leaves an accuracy gap
     versus raw evidence (loss is in the transformation, not retrieval).
  4. Mutation (overwrite) write semantics destroy historical/temporal
     evidence while preserving current state.

All tests are deterministic (seeded worlds, TF-IDF retriever, oracle
reader); numpy is required (guarded like the other numpy tests).
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

try:
    import numpy  # noqa: F401
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

if HAS_NUMPY:
    from arxiv_lab.memory.rawrag import (
        ConversationWorld,
        compare,
        evaluate_profile,
        run_setting,
        _chunks_profile,
        _chunks_raw,
        window_chunks,
    )

SEEDS = list(range(8))


def _skip():
    if not HAS_NUMPY:
        print("SKIP (numpy not installed)")
        return True
    return False


def test_determinism():
    if _skip():
        return
    assert run_setting(3) == run_setting(3)


def test_world_ground_truth_complete():
    """Every question's required evidence exists in the world."""
    if _skip():
        return
    w = ConversationWorld(0)
    tids = {t.tid for t in w.turns}
    for q in w.questions:
        assert all(e in tids for e in q.evidence), q.text
        assert len(q.keys) >= 1


def test_raw_rag_ranks_first_every_seed():
    """Claim 1: raw-RAG EM strictly above every transformed arm, all seeds."""
    if _skip():
        return
    for seed in SEEDS:
        r = run_setting(seed)
        raw = r["raw_rag"]["em"]
        assert raw == 1.0, (seed, raw)
        for arm in ("window_rag", "window_full", "profile_rag"):
            assert raw > r[arm]["em"], (seed, arm, r[arm]["em"])


def test_gap_is_coverage_not_utilization():
    """Claim 2: the EM gap between raw and transformed arms equals the
    coverage gap (utilization is 1.0 for all arms under the oracle reader,
    so coverage is the entire driver)."""
    if _skip():
        return
    for seed in SEEDS:
        r = run_setting(seed)
        for arm in ("window_rag", "profile_rag"):
            em_gap = r["raw_rag"]["em"] - r[arm]["em"]
            cov_gap = r["raw_rag"]["coverage"] - r[arm]["coverage"]
            assert abs(em_gap - cov_gap) < 1e-9, (seed, arm, em_gap, cov_gap)
            assert r[arm]["utilization"] == 1.0


def test_transformation_not_retrieval_is_the_loss():
    """Claim 3: complete access to the transformed (windowed) store scores
    identically to retrieving from it — the loss is in the transformation."""
    if _skip():
        return
    for seed in SEEDS:
        r = run_setting(seed)
        assert r["window_full"]["em"] == r["window_rag"]["em"] == \
            r["window_full"]["coverage"]
        assert r["window_full"]["em"] < r["raw_full"]["em"]


def test_mutation_destroys_temporal_preserves_current():
    """Claim 4: overwrite (mutation) semantics keep current-state facts
    (single-hop 1.0, multi-hop join 1.0) but destroy history
    (temporal 0.0)."""
    if _skip():
        return
    for seed in SEEDS:
        r = run_setting(seed)
        kinds = r["profile_rag"]["kind_em"]
        assert kinds["single"] == 1.0, (seed, kinds)
        assert kinds["multi"] == 1.0, (seed, kinds)
        assert kinds["temporal"] == 0.0, (seed, kinds)


def test_relation_paths_are_the_fragile_case():
    """Multi-hop (relation-path) questions suffer most under the recency
    window: both hops must survive the budget."""
    if _skip():
        return
    agg = compare(SEEDS)
    w_single = agg["window_rag"]["kind_em"]["single"][0]
    w_multi = agg["window_rag"]["kind_em"]["multi"][0]
    assert w_multi < w_single, (w_multi, w_single)


def test_window_budget_monotonic():
    """Sanity: a larger per-person window keeps more evidence."""
    if _skip():
        return
    w = ConversationWorld(1)
    c4 = window_chunks(w, per_person=4)
    c6 = window_chunks(w, per_person=6)
    assert len(c6) > len(c4)
    assert len(c4) == 6 * 4  # 6 persons x 4 turns


if __name__ == "__main__":
    for name, fn in sorted(
            [(k, v) for k, v in globals().items() if k.startswith("test_")]):
        fn()
        print(f"PASS {name}")
    print("all rawrag CPU tests passed")
