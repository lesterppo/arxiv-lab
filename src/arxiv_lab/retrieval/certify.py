"""Certify: requirement-directed search with Horn-closure certificates.

arXiv:2610.34684 — "Knowing When the Evidence Is Enough:
Requirement-Directed Search with Horn-Closure Certificates" (Certify).

Paper mechanism implemented here (stdlib only, no model calls):

* A per-query requirement hypergraph states which facts jointly yield
  another fact (Horn rules: ``head :- premise_1, ..., premise_k``).
* Each requirement is linked to passages only after an entity check:
  a passage links to a requirement only when it is about the requirement's
  entity AND supports the requirement's fact. A passage that merely names
  the entity, or supports the same fact-relation for a different entity,
  does not link.
* Horn closure over the linked facts yields a certificate: the minimal
  sufficient passages and their derivation — or an abstention naming the
  open requirements (the leaf facts still missing).
* Given correct retrieval, requirements and links, the decision is sound
  (it never answers an unanswerable question) and complete up to the
  entity check (it never abstains on an answerable question).

Paper claims: on pooled corpora with unanswerable twins, Certify decides
significantly more accurately than search agents, graph retrieval and
holistic sufficiency judges; entity checks add 4-13 points of decision
accuracy where gold evidence exists; answers from its certificate are
80% correct against 57% from all retrieved passages.
"""

__all__ = [
    "CertifyDecision",
    "Passage",
    "Requirement",
    "Rule",
    "certify",
    "horn_closure",
    "link_passages",
]

from collections import namedtuple

Requirement = namedtuple("Requirement", ["req_id", "entity", "fact"])
Rule = namedtuple("Rule", ["head", "premises"])
Passage = namedtuple("Passage", ["pid", "entity", "fact", "misleading"])
CertifyDecision = namedtuple(
    "CertifyDecision",
    ["decision", "certificate", "covered_facts", "open_requirements",
     "missing_leaf_facts"],
)


def link_passages(requirements, passages, entity_check=True):
    """Link each requirement to at most one supporting passage.

    With ``entity_check`` (the paper's default), a passage links only when
    its entity matches the requirement's entity and its fact matches the
    requirement's fact. Without the entity check, any passage supporting
    the fact links — the ablation the paper's +4..13 point entity-check
    gain is measured against. Returns {req_id: Passage or None}.
    """
    links = {}
    for req in requirements:
        hit = None
        for p in passages:
            if p.fact is None:
                continue  # names an entity but states no usable fact
            if p.fact != req.fact:
                continue
            if entity_check and p.entity != req.entity:
                continue
            hit = p
            break
        links[req.req_id] = hit
    return links


def horn_closure(linked_facts, rules):
    """Least fixpoint of the Horn rules over the linked facts."""
    covered = set(linked_facts)
    changed = True
    while changed:
        changed = False
        for rule in rules:
            if rule.head not in covered and all(
                    p in covered for p in rule.premises):
                covered.add(rule.head)
                changed = True
    return covered


def certify(requirements, rules, passages, entity_check=True):
    """Decide ANSWER vs ABSTAIN for one question.

    ANSWER iff every requirement's fact is in the Horn closure of the
    linked facts; the certificate is the linked passages that support a
    covered leaf fact (plus, implicitly, the rules that derive the rest).
    ABSTAIN otherwise, naming the open requirements and the missing leaf
    facts (leaf = a fact that is no rule's head).
    """
    links = link_passages(requirements, passages, entity_check=entity_check)
    linked_facts = {p.fact for p in links.values() if p is not None}
    covered = horn_closure(linked_facts, rules)
    open_reqs = [r for r in requirements if r.fact not in covered]
    heads = {r.head for r in rules}
    leaf_facts = {r.fact for r in requirements if r.fact not in heads}
    missing_leaf = sorted(f for f in leaf_facts if f not in covered)
    if open_reqs:
        return CertifyDecision(
            decision="ABSTAIN",
            certificate=[],
            covered_facts=sorted(covered),
            open_requirements=[r.req_id for r in open_reqs],
            missing_leaf_facts=missing_leaf,
        )
    certificate = [links[r.req_id] for r in requirements
                   if links[r.req_id] is not None]
    return CertifyDecision(
        decision="ANSWER",
        certificate=certificate,
        covered_facts=sorted(covered),
        open_requirements=[],
        missing_leaf_facts=[],
    )
