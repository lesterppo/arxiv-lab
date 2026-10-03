"""
Mem++ (arXiv:2610.02002) — OrgMemBench-style versioned-decision test.

Setup mirrors the paper's scenario: revised decisions arrive as NEW
documents (not edits). Questions ask what held AT A GIVEN TIME.
  - Mem++: stores every version whole; read-time selection filters
    docs to date <= as_of, then fuses lexical+semantic ranks.
  - Baseline (write-time distillation): keeps only the latest distilled
    fact per topic — the standard practice the paper argues against.

Claim under test: on backdated (as-of) questions, read-time selection
beats write-time distillation; on current questions they tie.
Run: python3 tests/test_memory_mempp_cpu.py
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "src"))
from arxiv_lab.memory.mempp import MemStore, retrieve


# ── versioned organizational record ──────────────────────────────────────────
DOCS = [
    ("d1", "Q1 engineering budget is set at 50k USD.", "2026-01-10", "Alice"),
    ("d2", "Q1 engineering budget revised to 70k USD.", "2026-03-05", "Bob"),
    ("d3", "Q1 engineering budget revised to 60k USD.", "2026-06-20", "Alice"),
    ("d4", "The data retention policy is 30 days.", "2026-02-01", "Carol"),
    ("d5", "The data retention policy is updated to 90 days.", "2026-05-11", "Dave"),
    ("d6", "Office reopening planned for September.", "2026-04-02", "Erin"),
]

# (question, as_of, expected_answer_substring)
QUESTIONS = [
    ("What is the Q1 engineering budget?", "2026-02-01", "50k"),
    ("What is the Q1 engineering budget?", "2026-04-01", "70k"),
    ("What is the Q1 engineering budget?", "2026-07-01", "60k"),
    ("What is the data retention policy?", "2026-03-01", "30 days"),
    ("What is the data retention policy?", "2026-06-01", "90 days"),
    ("When is the office reopening?", "2026-08-01", "September"),
]


def build_mempp():
    store = MemStore()
    for doc_id, text, date, author in DOCS:
        store.add(doc_id, text, date, author)
    return store


class DistillBaseline:
    """Write-time distillation: per topic keep ONLY the latest fact."""

    def __init__(self):
        self.facts = {}

    @staticmethod
    def _topic(q):
        q = q.lower()
        if "budget" in q:
            return "budget"
        if "retention" in q:
            return "retention"
        return "office"

    def add(self, text, date):
        self.facts[self._topic(text)] = (date, text)  # overwrite!

    def answer(self, question, as_of=None):
        # as_of is IGNORED — the superseded versions are gone (the paper's
        # exact complaint: write-time distillation fixes what can be answered
        # before any question is asked)
        return self.facts.get(self._topic(question), ("", ""))[1]


def answer_with_mempp(store, question, as_of, k=3):
    hits = retrieve(store, question, as_of=as_of, k=k)
    return " | ".join(h["text"] for h in hits)


def main():
    store = build_mempp()
    assert len(store) == len(DOCS), "nothing may be dropped at write time"
    # write path calls no model: add() is pure python (by construction)

    base = DistillBaseline()
    for _, text, date, _ in DOCS:
        base.add(text, date)

    mempp_ok = base_ok = 0
    print(f"{'question':42s} {'as_of':10s} {'mem++':6s} {'baseline':8s}")
    for q, as_of, expected in QUESTIONS:
        m_ans = answer_with_mempp(store, q, as_of)
        b_ans = base.answer(q, as_of)
        m_hit = expected in m_ans
        b_hit = expected in b_ans
        mempp_ok += m_hit
        base_ok += b_hit
        print(f"{q[:40]:42s} {as_of:10s} {'OK' if m_hit else 'MISS':6s} "
              f"{'OK' if b_hit else 'MISS':8s}")

    # time-filter check: a Feb question must NEVER see June docs
    hits = retrieve(store, "Q1 engineering budget", as_of="2026-02-01", k=5)
    assert all(h["date"] <= "2026-02-01" for h in hits), "time filter violated"
    assert any("50k" in h["text"] for h in hits), "v1 must be retrievable"

    # whole-document storage: no distillation
    assert all(len(h["text"].split()) > 5 for h in hits), "docs stored whole"

    print(f"\nMem++ accuracy: {mempp_ok}/{len(QUESTIONS)} | "
          f"baseline: {base_ok}/{len(QUESTIONS)}")
    assert mempp_ok == len(QUESTIONS), "Mem++ must answer every as-of question"
    assert base_ok < mempp_ok, "baseline must lose on backdated questions"
    print("ALL MEM++ CHECKS PASSED: read-time selection beats write-time "
          "distillation on versioned questions.")


if __name__ == "__main__":
    main()
