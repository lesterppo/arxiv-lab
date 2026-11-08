"""CORE: Coarse-to-Fine Search with Context Management (CPU proxy).

Replication of arXiv 2611.07859 (Vuong et al., 2026) -- a training-free
deep-research harness with three stages:

  1. Coarse retrieval over passages.
     Paper: dense retriever. CPU proxy here: BM25 (documented divergence).
  2. Snippet extraction via token-level similarities.
     Paper: reuse token-level similarities from a late-interaction reranker
     (ColBERT-style) to return only the most relevant snippet per passage.
     CPU proxy here: TF-IDF cosine between the query and each sentence
     window of a retrieved passage (documented divergence).
  3. Context management across turns.
     Paper: expand the context around sources already retrieved, and
     suppress repeated snippets. Implemented here as ContextManager:
     a turn-2 candidate snippet that near-duplicates an already-returned
     snippet (Jaccard >= threshold) is suppressed (reused from context);
     if the already-returned snippet lacks the answer-bearing sentence,
     the context is expanded by that adjacent sentence instead.

Stdlib + numpy only. Deterministic (no randomness except where a seeded
Random is passed in by the caller).
"""

import math
import re

import numpy as np

TOKEN_RE = re.compile(r"[a-z0-9]+")
STOPWORDS = {
    "which", "is", "the", "a", "an", "in", "on", "at", "to", "of", "for",
    "who", "with", "by", "that", "it", "as", "are", "was", "were", "be",
    "been", "has", "have", "had", "do", "does", "did", "and", "or", "but",
    "from", "into", "their", "they", "this", "these", "those", "affiliated",
}


def tokenize(text):
    """Lowercase alphanumeric tokens."""
    return TOKEN_RE.findall(text.lower())


def content_tokens(text):
    """Query/content tokens with stopwords removed (retrieval + snippet scoring)."""
    return [t for t in tokenize(text) if t not in STOPWORDS]


def count_tokens(text):
    """Whitespace token count (the paper's cost metric, approximated)."""
    return len(text.split())


class BM25:
    """Classic BM25 coarse retriever (CPU proxy for the paper's dense retriever)."""

    def __init__(self, passages, k1=1.5, b=0.75):
        self.passages = list(passages)
        self.docs = [content_tokens(p) for p in passages]
        self.k1 = k1
        self.b = b
        self.N = len(self.docs)
        self.avgdl = sum(len(d) for d in self.docs) / max(self.N, 1)
        df = {}
        for d in self.docs:
            for t in set(d):
                df[t] = df.get(t, 0) + 1
        self.idf = {t: math.log(1 + (self.N - c + 0.5) / (c + 0.5))
                    for t, c in df.items()}

    def score(self, query_tokens, doc_idx):
        doc = self.docs[doc_idx]
        if not doc:
            return 0.0
        tf = {}
        for t in doc:
            tf[t] = tf.get(t, 0) + 1
        dl = len(doc)
        s = 0.0
        for t in set(query_tokens):
            if t not in tf:
                continue
            f = tf[t]
            idf = self.idf.get(t, 0.0)
            s += idf * f * (self.k1 + 1) / (
                f + self.k1 * (1 - self.b + self.b * dl / self.avgdl))
        return s

    def search(self, query, k=5):
        """Return [(doc_idx, score)] sorted by score desc, top-k."""
        qt = content_tokens(query)
        scored = [(i, self.score(qt, i)) for i in range(self.N)]
        scored.sort(key=lambda x: (-x[1], x[0]))
        return scored[:k]


class SnippetExtractor:
    """Token-similarity snippet extraction (CPU proxy for the late-interaction
    reranker's token similarities): TF-IDF cosine between the query and each
    sentence window of a passage; return the best window."""

    def __init__(self, all_sentences):
        self.sent_token_lists = [content_tokens(s) for s in all_sentences]
        vocab = {}
        for toks in self.sent_token_lists:
            for t in toks:
                if t not in vocab:
                    vocab[t] = len(vocab)
        self.vocab = vocab
        n = len(self.sent_token_lists)
        df = np.zeros(len(vocab))
        for toks in self.sent_token_lists:
            for t in set(toks):
                df[vocab[t]] += 1
        # BM25-style positive idf
        self.idf = np.log(1 + (n - df + 0.5) / (df + 0.5))

    def _vector(self, tokens):
        v = np.zeros(len(self.vocab))
        for t in tokens:
            idx = self.vocab.get(t)
            if idx is not None:
                v[idx] += 1.0
        v = v * self.idf
        norm = np.linalg.norm(v)
        return v / norm if norm > 0 else v

    def best_window(self, sentences, query, width=2):
        """Best consecutive `width`-sentence window by TF-IDF cosine.
        Returns (indices, text)."""
        qt = content_tokens(query)
        qv = self._vector(qt)
        best_idx, best_score = None, -1.0
        for start in range(0, max(1, len(sentences) - width + 1)):
            idxs = list(range(start, min(start + width, len(sentences))))
            toks = []
            for i in idxs:
                toks.extend(content_tokens(sentences[i]))
            wv = self._vector(toks)
            score = float(np.dot(qv, wv))
            if score > best_score:
                best_score, best_idx = score, idxs
        text = " ".join(sentences[i] for i in best_idx)
        return best_idx, text

    def first_window(self, sentences, width=2):
        """Ablation: first `width` sentences (naive snippet baseline)."""
        idxs = list(range(min(width, len(sentences))))
        return idxs, " ".join(sentences[i] for i in idxs)


def window_text(sentences, indices):
    return " ".join(sentences[i] for i in indices)


def jaccard(a_tokens, b_tokens):
    """Jaccard overlap between two token sets."""
    a, b = set(a_tokens), set(b_tokens)
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


class ContextManager:
    """Cross-turn context management (paper stage 3).

    Tracks snippets already returned for each passage in a trajectory.
    Turn-2 decision per passage:
      - fresh: passage not seen before -> return the fresh snippet.
      - suppress: fresh snippet near-duplicates an already-returned snippet
        (Jaccard >= threshold), or the already-returned snippet already
        contains the answer-bearing sentence -> add nothing (reuse context).
      - expand: snippet is new but the already-returned snippet lacks the
        answer-bearing sentence -> add only that adjacent sentence.
    """

    def __init__(self, jaccard_threshold=0.8):
        self.threshold = jaccard_threshold
        self.seen = {}  # passage_id -> snippet text

    def register(self, passage_id, snippet_text):
        self.seen[passage_id] = snippet_text

    def was_seen(self, passage_id):
        return passage_id in self.seen

    def is_duplicate(self, snippet_text):
        for prev in self.seen.values():
            if jaccard(content_tokens(snippet_text),
                       content_tokens(prev)) >= self.threshold:
                return True
        return False

    def turn2_decision(self, passage_id, fresh_text, answer_sentence):
        """Return (action, text). action in {fresh, suppress, expand}."""
        if passage_id not in self.seen:
            return "fresh", fresh_text
        prev = self.seen[passage_id]
        if jaccard(content_tokens(fresh_text), content_tokens(prev)) >= self.threshold:
            return "suppress", ""
        if answer_sentence in prev:
            return "suppress", ""
        return "expand", answer_sentence
