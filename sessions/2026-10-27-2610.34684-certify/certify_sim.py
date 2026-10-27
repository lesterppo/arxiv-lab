"""Certify simulation (arXiv 2610.34684) — deterministic, stdlib only.

Each base question has direct requirements (need one gold passage each)
and optionally one derived requirement (head :- two leaf premises; no
passage states the head directly, so Horn closure must derive it).
Its unanswerable twin removes exactly one gold leaf passage while keeping
entity-naming distractors, so the twin's context still names every entity
but lacks one required fact — the paper's central trap.

Simulated answer generation (disclosed regime, not the paper's numbers):
* a per-question "hard" flag (p=0.20, seeded) makes even a certificate
  answer wrong — intrinsic generation error the certificate cannot remove;
* a per-question "misleading" flag (p=0.30, seeded) is a distractor
  carrying a conflicting value for a required fact; it corrupts any
  answer built from all retrieved passages but never enters a certificate
  (it links to no requirement).
Answer accuracy is therefore reported as a regime outcome; the structural
claims under test are the ORDERINGS and the soundness/completeness
properties, which do not depend on these rates.

Decision correctness: answerable -> ANSWER (+ correct answer) is correct;
twin -> ABSTAIN is correct; anything else is wrong.

Run: python3 certify_sim.py  (prints the metrics table for all seeds)
"""

import os
import random
import sys

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "src"))

from arxiv_lab.retrieval.certify import (  # noqa: E402
    Passage, Requirement, Rule, certify)

SEEDS = (20261027, 1, 7, 42)
N_PAIRS = 100
HARD_P = 0.20
MISLEADING_P = 0.30
WRONG_ENTITY_TWIN_P = 0.25
ENTITIES = ["Ada", "Bo", "Cy", "Di", "Eli", "Fay", "Gus", "Hal"]


def build_question(rng, qid):
    """Return (requirements, rules, gold, distractors, missing_leaf)."""
    ents = rng.sample(ENTITIES, 3)  # first two are the question's entities
    qents, other = ents[:2], ents[2]
    n_direct = rng.randint(1, 3)
    reqs, rules, gold = [], [], []
    for i in range(n_direct):
        fact = "q%d:direct%d" % (qid, i)
        reqs.append(Requirement("r%d" % i, qents[i % 2], fact))
        gold.append(Passage("g%d" % i, qents[i % 2], fact, False))
    derived = None
    if rng.random() < 0.5:
        leaf1, leaf2 = "q%d:leaf0" % qid, "q%d:leaf1" % qid
        head = "q%d:derived" % qid
        # The leaf premises are requirements too: they link directly, and
        # the derived requirement is satisfied only via Horn closure over
        # them (no passage ever states the head directly). Without the
        # leaf requirements the closure would have no linked facts to run
        # on — that was a generator bug in the first draft (cert_decision
        # came out ~0.75 because derived questions always abstained).
        reqs.append(Requirement("rL0", qents[0], leaf1))
        reqs.append(Requirement("rL1", qents[1], leaf2))
        reqs.append(Requirement("rD", qents[0], head))
        rules.append(Rule(head, (leaf1, leaf2)))
        gold.append(Passage("gL0", qents[0], leaf1, False))
        gold.append(Passage("gL1", qents[1], leaf2, False))
        derived = (leaf1, leaf2, head)
    distractors = [
        Passage("dE0", qents[0], None, False),  # names entity, no fact
        Passage("dE1", qents[1], None, False),
    ]
    # One wrong-entity passage for a sampled gold fact (same fact-relation,
    # different entity): harmless under the entity check, fatal without it.
    if gold:
        g = rng.choice(gold)
        distractors.append(Passage("dW", other, g.fact, False))
    return reqs, rules, gold, distractors, derived


def make_case(rng, qid):
    """One (answerable, twin) pair plus per-question flags."""
    reqs, rules, gold, distractors, derived = build_question(rng, qid)
    hard = rng.random() < HARD_P
    misleading = rng.random() < MISLEADING_P
    passages = list(gold) + list(distractors)
    if misleading:
        # Conflicting value for a required entity's fact; links nowhere.
        passages.append(Passage("dM", reqs[0].entity,
                                "conflict:" + reqs[0].fact, True))
    rng.shuffle(passages)
    answerable = (reqs, rules, passages, None)
    # Twin: remove one gold leaf; with prob WRONG_ENTITY_TWIN_P keep a
    # wrong-entity counterpart for it (the no-entity-check trap).
    removed = rng.choice(gold)
    twin_passages = [p for p in passages if p.pid != removed.pid]
    if rng.random() < WRONG_ENTITY_TWIN_P:
        twin_passages.append(Passage("dWT", "Zed", removed.fact, False))
    rng.shuffle(twin_passages)
    twin = (reqs, rules, twin_passages, removed.fact)
    return answerable, twin, hard, misleading


def holistic_answers(reqs, passages):
    """Holistic sufficiency judge: answers iff every required entity is
    named anywhere in the retrieved passages (facts unchecked)."""
    named = {p.entity for p in passages}
    return all(r.entity in named for r in reqs)


def evaluate_clean(seed):
    """Full evaluation; returns a metrics dict (fractions)."""
    rng = random.Random(seed)
    c = {"cert_dec": 0, "hol_dec": 0, "all_dec": 0, "noec_dec": 0,
         "cert_ans": 0, "cert_ans_n": 0, "all_ans": 0, "all_ans_n": 0,
         "hol_ans": 0, "hol_ans_n": 0, "twin_abstain": 0, "twin_naming": 0,
         "cases": 0, "twins": 0}
    for qid in range(N_PAIRS):
        ans, twin, hard, misleading = make_case(rng, qid)
        for reqs, rules, passages, removed in (ans, twin):
            is_twin = removed is not None
            c["cases"] += 1
            # Certify, entity-checked. Answer accuracy is scored over
            # answerable cases only (the paper's certificate-vs-all
            # comparison); twins are scored via decision + abstention.
            d = certify(reqs, rules, passages, entity_check=True)
            if d.decision == "ANSWER" and not is_twin:
                c["cert_ans_n"] += 1
                c["cert_ans"] += int(not hard)
            c["cert_dec"] += int((d.decision == "ANSWER") == (not is_twin))
            if is_twin:
                c["twins"] += 1
                c["twin_abstain"] += int(d.decision == "ABSTAIN")
                c["twin_naming"] += int(
                    d.decision == "ABSTAIN"
                    and d.missing_leaf_facts == [removed])
            # Certify WITHOUT entity check (ablation)
            d2 = certify(reqs, rules, passages, entity_check=False)
            c["noec_dec"] += int((d2.decision == "ANSWER") == (not is_twin))
            # Holistic entity-coverage judge, answering from all passages
            if holistic_answers(reqs, passages):
                c["hol_ans_n"] += 1
                c["hol_ans"] += int(
                    (not is_twin) and (not hard) and (not misleading))
            c["hol_dec"] += int(
                holistic_answers(reqs, passages) == (not is_twin))
            # All-passages answerer: always answers from all passages;
            # answer accuracy scored over answerable cases only, like
            # the certificate's, so the two rates are comparable.
            if not is_twin:
                c["all_ans_n"] += 1
                c["all_ans"] += int((not hard) and (not misleading))
            c["all_dec"] += int(not is_twin)
    n = c["cases"]
    return {
        "seed": seed,
        "cert_decision": c["cert_dec"] / n,
        "holistic_decision": c["hol_dec"] / n,
        "allpass_decision": c["all_dec"] / n,
        "noentity_decision": c["noec_dec"] / n,
        "entity_check_gain_pts": 100 * (c["cert_dec"] - c["noec_dec"]) / n,
        "cert_answer": c["cert_ans"] / c["cert_ans_n"] if c["cert_ans_n"] else 0,
        "allpass_answer": c["all_ans"] / c["all_ans_n"] if c["all_ans_n"] else 0,
        "holistic_answer": c["hol_ans"] / c["hol_ans_n"] if c["hol_ans_n"] else 0,
        "twin_abstain_rate": c["twin_abstain"] / c["twins"] if c["twins"] else 0,
        "twin_naming_exact": c["twin_naming"] / c["twins"] if c["twins"] else 0,
        "cases": n,
    }


def main():
    print("seed | cert_dec hol_dec all_dec noent_dec | cert_ans all_ans "
          "| twin_abst twin_name | ent_gain_pts")
    for seed in SEEDS:
        r = evaluate_clean(seed)
        print("%d | %.3f %.3f %.3f %.3f | %.3f %.3f | %.3f %.3f | %.1f"
              % (r["seed"], r["cert_decision"], r["holistic_decision"],
                 r["allpass_decision"], r["noentity_decision"],
                 r["cert_answer"], r["allpass_answer"],
                 r["twin_abstain_rate"], r["twin_naming_exact"],
                 r["entity_check_gain_pts"]))


if __name__ == "__main__":
    main()
