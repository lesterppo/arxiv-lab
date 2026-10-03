"""
Mem++: Non-Destructive Memory for Long-Term Organizational LLM Agents.

Paper: arXiv:2610.02002 — "Mem++: Non-Destructive Memory for Long-Term
Organizational LLM Agents".

Core claim: most memory systems compress the record at WRITE time (distill
each document into facts/notes/graph edges), fixing what can be answered
before any question is asked. When a revised decision arrives as a NEW
document (not an edit), answering "what held at time T?" requires the
superseded versions — which write-time distillation has destroyed.

Mem++ shifts from write-time distillation to READ-TIME selection:
  - write: store every document WHOLE with its date and author.
    No generative model is called at write time. Nothing is overwritten.
  - read: given a query (optionally with an as-of time), retrieve only
    documents dated <= as_of, fuse lexical (BM25) and semantic rankings
    (TF-IDF cosine by default; inject any embedding fn), and return whole
    documents — leaving the final choice to the answering model.

This module implements the store + read-time selection. The "semantic"
channel defaults to TF-IDF cosine (stdlib-only); pass embed_fn (e.g.
sentence-transformers) for true dense retrieval.
"""

import math
from collections import Counter


def _tok(text):
    return [w.strip(".,;:!?()[]{}\"'").lower() for w in text.split()
            if w.strip(".,;:!?()[]{}\"'")]


class MemStore:
    """Non-destructive document store. Write path calls no model."""

    def __init__(self):
        self.docs = []  # list of dicts: id, text, date (ISO), author, tokens

    def add(self, doc_id, text, date, author):
        """Store a document whole. date: ISO 'YYYY-MM-DD' (string compare)."""
        self.docs.append({"id": doc_id, "text": text, "date": date,
                          "author": author, "tokens": _tok(text)})
        return doc_id

    def __len__(self):
        return len(self.docs)


def _bm25_scores(query_toks, docs, k1=1.5, b=0.75):
    n = len(docs)
    df = Counter()
    for d in docs:
        for t in set(d["tokens"]):
            df[t] += 1
    avgdl = sum(len(d["tokens"]) for d in docs) / max(n, 1)
    scores = []
    for d in docs:
        tf = Counter(d["tokens"])
        dl = len(d["tokens"])
        s = 0.0
        for t in query_toks:
            if t not in tf:
                continue
            idf = math.log((n - df[t] + 0.5) / (df[t] + 0.5) + 1.0)
            s += idf * (tf[t] * (k1 + 1)) / (tf[t] + k1 * (1 - b + b * dl / avgdl))
        scores.append(s)
    return scores


def _tfidf_cosine_scores(query_toks, docs):
    n = len(docs)
    df = Counter()
    for d in docs:
        for t in set(d["tokens"]):
            df[t] += 1
    idf = {t: math.log((n + 1) / (c + 1)) + 1.0 for t, c in df.items()}
    qtf = Counter(query_toks)
    qv = {t: (1 + math.log(c)) * idf.get(t, 0.0) for t, c in qtf.items()
          if t in idf}
    qn = math.sqrt(sum(v * v for v in qv.values())) or 1.0
    scores = []
    for d in docs:
        tf = Counter(d["tokens"])
        dv = {t: (1 + math.log(c)) * idf[t] for t, c in tf.items() if t in idf}
        dot = sum(qv.get(t, 0.0) * v for t, v in dv.items())
        dn = math.sqrt(sum(v * v for v in dv.values())) or 1.0
        scores.append(dot / (qn * dn))
    return scores


def _embed_scores(query, docs, embed_fn):
    qv = embed_fn(query)
    dvs = [embed_fn(d["text"]) for d in docs]

    def cos(a, b):
        dot = sum(x * y for x, y in zip(a, b))
        na = math.sqrt(sum(x * x for x in a)) or 1.0
        nb = math.sqrt(sum(x * x for x in b)) or 1.0
        return dot / (na * nb)

    return [cos(qv, dv) for dv in dvs]


def _rrf(rank_lists, k=60):
    fused = Counter()
    for ranks in rank_lists:
        for rank, idx in enumerate(ranks):
            fused[idx] += 1.0 / (k + rank + 1)
    return fused


def retrieve(store, query, as_of=None, k=5, embed_fn=None):
    """Read-time selection.

    1. Time filter: only docs with date <= as_of (None = no filter).
    2. Rank filtered docs by BM25 (lexical) and by semantic channel
       (TF-IDF cosine, or embed_fn cosine when provided).
    3. Fuse with Reciprocal Rank Fusion; return whole docs (newest first
       on ties) — the answering model makes the final choice.
    """
    cands = [d for d in store.docs
             if as_of is None or d["date"] <= as_of]
    if not cands:
        return []
    qt = _tok(query)
    lex = _bm25_scores(qt, cands)
    sem = (_embed_scores(query, cands, embed_fn) if embed_fn
           else _tfidf_cosine_scores(qt, cands))
    rank_lex = sorted(range(len(cands)), key=lambda i: -lex[i])
    rank_sem = sorted(range(len(cands)), key=lambda i: -sem[i])
    fused = _rrf([rank_lex, rank_sem])
    # higher fused score first; newest date first on ties
    order = sorted(range(len(cands)),
                   key=lambda i: (-fused[i], _neg_date(cands[i]["date"])))
    out = []
    for i in order[:k]:
        d = cands[i]
        out.append({"id": d["id"], "text": d["text"], "date": d["date"],
                    "author": d["author"], "lex": lex[i], "sem": sem[i],
                    "rrf": fused[i]})
    return out


def _neg_date(d):
    # for newest-first tie-break via tuple sort
    return tuple(-ord(c) for c in d)
