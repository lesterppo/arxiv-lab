"""CPU tests for arxiv_lab.retrieval.certify (arXiv:2610.34684).

Claims under test (Certify: requirement-directed search with Horn-closure
certificates):
A. Soundness + completeness (given correct requirements/links): Certify
   never answers an unanswerable twin and never abstains on an answerable
   question — decision accuracy 1.000 on every seed.
B. The entity-naming trap: a holistic judge that answers whenever every
   entity is named answers every twin (decision 0.500) — naming every
   entity is not sufficiency.
C. Entity checks matter: Certify without the entity check scores strictly
   below entity-checked Certify (wrong-entity passages falsely complete
   closures); the gain's magnitude is regime-dependent (wrong-entity
   passages are dense in this simulation), so only the direction is
   asserted, not the paper's 4-13 point interval.
D. Certificate answers beat all-passages answers: filtering to verified
   links removes misleading-distractor errors (intrinsic hard-case errors
   remain in both, so neither rate is 1.0).
E. Abstentions name the open requirement: on every twin, the abstention's
   missing leaf facts are exactly the removed gold leaf.

Seeded and deterministic; stdlib only. The simulation lives in the session
directory (imported below, as test_retrieval_cpu.py does for RATED).

Run:  python3 tests/test_retrieval_certify_cpu.py
"""

import os
import sys

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "sessions", "2026-10-27-2610.34684-certify"))

from arxiv_lab.retrieval.certify import (  # noqa: E402
    Passage, Requirement, Rule, certify, horn_closure, link_passages)
from certify_sim import evaluate_clean  # noqa: E402

SEEDS = (20261027, 1, 7, 42)


def test_horn_closure_derives_head_only_from_all_premises():
    rules = [Rule("D", ("L1", "L2"))]
    assert horn_closure({"L1"}, rules) == {"L1"}
    assert horn_closure({"L1", "L2"}, rules) == {"L1", "L2", "D"}


def test_entity_check_blocks_wrong_entity_link():
    reqs = [Requirement("r0", "Ada", "f1")]
    passages = [Passage("w", "Bo", "f1", False)]
    assert link_passages(reqs, passages, entity_check=True)["r0"] is None
    assert link_passages(reqs, passages, entity_check=False)["r0"] is not None


def test_entity_naming_passage_does_not_link():
    reqs = [Requirement("r0", "Ada", "f1")]
    passages = [Passage("e", "Ada", None, False)]
    assert link_passages(reqs, passages, entity_check=True)["r0"] is None


def test_certify_answers_when_closure_covers_all():
    reqs = [Requirement("r0", "Ada", "L1"),
            Requirement("r1", "Ada", "L2"),
            Requirement("rD", "Ada", "D")]
    rules = [Rule("D", ("L1", "L2"))]
    passages = [Passage("a", "Ada", "L1", False),
                Passage("b", "Ada", "L2", False)]
    d = certify(reqs, rules, passages)
    assert d.decision == "ANSWER"
    assert d.open_requirements == [] and d.missing_leaf_facts == []


def test_certify_abstains_naming_the_missing_leaf():
    reqs = [Requirement("r0", "Ada", "L1"),
            Requirement("r1", "Ada", "L2"),
            Requirement("rD", "Ada", "D")]
    rules = [Rule("D", ("L1", "L2"))]
    passages = [Passage("a", "Ada", "L1", False)]  # L2 missing
    d = certify(reqs, rules, passages)
    assert d.decision == "ABSTAIN"
    assert d.missing_leaf_facts == ["L2"]
    assert "r1" in d.open_requirements and "rD" in d.open_requirements


def test_simulation_claims_hold_on_every_seed():
    for seed in SEEDS:
        r = evaluate_clean(seed)
        assert r["cert_decision"] == 1.0, (seed, r)
        assert r["twin_abstain_rate"] == 1.0, (seed, r)
        assert r["twin_naming_exact"] == 1.0, (seed, r)
        assert r["holistic_decision"] == 0.5, (seed, r)
        assert r["allpass_decision"] == 0.5, (seed, r)
        assert r["noentity_decision"] < r["cert_decision"], (seed, r)
        assert r["entity_check_gain_pts"] > 0, (seed, r)
        assert r["cert_answer"] > r["allpass_answer"], (seed, r)


def main():
    tests = [v for k, v in sorted(globals().items())
             if k.startswith("test_")]
    for t in tests:
        t()
        print("PASS", t.__name__)
    print("certify: %d/%d tests passed" % (len(tests), len(tests)))


if __name__ == "__main__":
    main()
