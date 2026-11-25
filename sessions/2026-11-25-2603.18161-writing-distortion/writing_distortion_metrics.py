"""Metrics for testing the claim of arXiv:2603.18161v2
"How LLMs Distort Our Written Language".

One-line claim under test: LLM edits shift essay semantics and style
more than human edits, even for grammar-only edits.

This module is CPU-only (stdlib + optionally numpy) and makes no
network calls. It provides the distortion measures used by the
session's CPU test and baseline analysis:

- unigram_jsd: Jensen-Shannon divergence of unigram distributions
  (semantic-shift proxy on surface word choice).
- pronoun_rate / adjective_rate: style proxies, per 100 words.
  adjective_rate is a PROXY based on suffixes plus a small lexicon;
  it is NOT LIWC and must not be reported as a LIWC score.
- keyword_stance: keyword-count stance on self-driving cars, using
  the session's fixed PRO/ANTI wordlists (copied verbatim).
- pct_change: percentage change helper.
- mean_shift_alignment: homogenization measure over shift vectors
  (mean cosine similarity of per-item shifts to the mean shift
  direction); identical shifts -> ~1, dispersed shifts -> lower.

Adjective lexicon used by adjective_rate (in addition to the suffix
list): a small hand-picked set of common adjectives that do not end
in one of the listed suffixes, e.g. "safe", "dangerous", "good",
"bad", "big", "small", "new", "old", "important", "different",
"public", "personal", "human", "electric", "automatic", "future",
"modern", "major", "main", "high", "low", "large", "great", "best",
"worst", "easy", "hard", "fast", "slow", "strong", "weak", "free",
"open", "full", "real", "true", "clear", "common", "general",
"local", "national", "social", "economic", "political", "legal",
"medical", "physical", "mental", "daily", "weekly", "annual",
"central", "rural", "urban", "global", "severe", "serious",
"significant", "potential", "possible", "likely", "certain",
"ready", "able" is covered by suffix, "aware", "careful".
Words are counted at most once each (suffix match OR lexicon hit).
"""

import math
import re

try:
    import numpy as _np
except Exception:  # numpy absent: pure-python fallback paths are used
    _np = None

__all__ = [
    "tokenize",
    "unigram_jsd",
    "pronoun_rate",
    "adjective_rate",
    "keyword_stance",
    "pct_change",
    "mean_shift_alignment",
    "PRONOUNS",
    "ADJECTIVE_SUFFIXES",
    "ADJECTIVE_LEXICON",
    "PRO_WORDS",
    "ANTI_WORDS",
]

_WORD_RE = re.compile(r"[a-z0-9]+")


def tokenize(text):
    """Lowercase alphanumeric words of *text* (list of str)."""
    if not text:
        return []
    return _WORD_RE.findall(text.lower())


PRONOUNS = {
    "i", "me", "my", "mine", "we", "us", "our", "ours", "you", "your",
    "yours", "he", "him", "his", "she", "her", "hers", "it", "its",
    "they", "them", "their", "theirs", "this", "that", "these",
    "those", "who", "whom", "whose", "myself", "ourselves",
    "yourself", "yourselves", "himself", "herself", "itself",
    "themselves", "someone", "anyone", "everyone", "nobody",
    "somebody", "everybody",
}

ADJECTIVE_SUFFIXES = (
    "ous", "ive", "ful", "less", "able", "ible", "al", "ic",
    "ary", "ory", "ant", "ent",
)

# Small hand-picked lexicon of common adjectives not caught by the
# suffix list. This is a PROXY, NOT LIWC.
ADJECTIVE_LEXICON = {
    "safe", "dangerous", "good", "bad", "big", "small", "new", "old",
    "important", "different", "public", "personal", "human",
    "electric", "automatic", "future", "modern", "major", "main",
    "high", "low", "large", "great", "best", "worst", "easy", "hard",
    "fast", "slow", "strong", "weak", "free", "open", "full", "real",
    "true", "clear", "common", "general", "local", "national",
    "social", "economic", "political", "legal", "medical", "physical",
    "mental", "daily", "weekly", "annual", "central", "rural",
    "urban", "global", "severe", "serious", "significant",
    "potential", "possible", "likely", "certain", "ready", "aware",
    "careful", "happy", "sad", "young", "elderly", "disabled",
    "remote", "huge", "tiny", "heavy", "light", "dark", "bright",
    "quiet", "busy", "rich", "poor", "cheap", "costly", "safe",
}

# Keyword stance wordlists -- copied verbatim from the session spec;
# do not alter.
PRO_WORDS = [
    "safer", "safety", "reduce accidents", "fewer accidents",
    "reduce crashes", "save lives", "saves lives", "efficient",
    "improve traffic", "reduce traffic", "cost saving", "save time",
    "save money", "greater access", "convenient", "reliable",
    "reduce congestion", "less pollution", "benefit", "benefits",
    "improve", "improves",
]
ANTI_WORDS = [
    "dangerous", "kill", "killed", "death", "deaths", "accident",
    "accidents", "crash", "crashes", "risk", "risks", "risky",
    "unsafe", "unethical", "job loss", "jobs", "unemployment",
    "expensive", "hack", "hacking", "privacy", "liability",
    "unpredictable", "fail", "failure", "malfunction", "error",
    "errors",
]


def _counts(words):
    counts = {}
    for w in words:
        counts[w] = counts.get(w, 0) + 1
    return counts


def unigram_jsd(text_a, text_b):
    """Jensen-Shannon divergence of unigram distributions, log base 2.

    Range 0..1; 0 for identical texts. Small add-alpha smoothing
    (alpha=1e-6 per vocabulary type) so unseen words get nonzero
    mass; deterministic.
    """
    wa = tokenize(text_a)
    wb = tokenize(text_b)
    ca = _counts(wa)
    cb = _counts(wb)
    vocab = set(ca) | set(cb)
    if not vocab:
        return 0.0
    alpha = 1e-6
    na = len(wa) + alpha * len(vocab)
    nb = len(wb) + alpha * len(vocab)
    jsd = 0.0
    for w in vocab:
        p = (ca.get(w, 0) + alpha) / na
        q = (cb.get(w, 0) + alpha) / nb
        m = 0.5 * (p + q)
        jsd += 0.5 * p * math.log2(p / m) + 0.5 * q * math.log2(q / m)
    # Clamp tiny floating point excursions.
    return min(1.0, max(0.0, jsd))


def pronoun_rate(text):
    """Pronouns per 100 words. Returns (rate, count, n_words)."""
    words = tokenize(text)
    n = len(words)
    count = sum(1 for w in words if w in PRONOUNS)
    rate = (100.0 * count / n) if n else 0.0
    return (rate, count, n)


def adjective_rate(text):
    """Adjective PROXY per 100 words. Returns (rate, count, n_words).

    A word counts if it has length >= 5 and ends in one of
    ADJECTIVE_SUFFIXES, or if it is in ADJECTIVE_LEXICON.
    This is a suffix+lexicon proxy, NOT LIWC.
    """
    words = tokenize(text)
    n = len(words)
    count = 0
    for w in words:
        if len(w) >= 5 and w.endswith(ADJECTIVE_SUFFIXES):
            count += 1
        elif w in ADJECTIVE_LEXICON:
            count += 1
    rate = (100.0 * count / n) if n else 0.0
    return (rate, count, n)


def _count_phrase(text_lower, phrase):
    """Whole-word/phrase occurrences of *phrase* in lowercased text."""
    pattern = r"\b" + re.escape(phrase) + r"\b"
    return len(re.findall(pattern, text_lower))


def keyword_stance(text):
    """Keyword stance on self-driving cars.

    Returns {"pro": int, "anti": int, "score": int, "label": str}.
    Whole-word/phrase occurrences, case-insensitive; multiword
    entries counted as phrases. label: "pro" if pro-anti >= 3,
    "anti" if <= -3, else "mixed".
    """
    lowered = (text or "").lower()
    pro = sum(_count_phrase(lowered, p) for p in PRO_WORDS)
    anti = sum(_count_phrase(lowered, a) for a in ANTI_WORDS)
    score = pro - anti
    if score >= 3:
        label = "pro"
    elif score <= -3:
        label = "anti"
    else:
        label = "mixed"
    return {"pro": pro, "anti": anti, "score": score, "label": label}


def pct_change(before, after):
    """(after-before)/before*100; guards before == 0.

    Returns 0.0 when both are 0, +inf when before is 0 and after
    is nonzero.
    """
    if before == 0:
        return 0.0 if after == 0 else float("inf")
    return (after - before) / before * 100.0


def _normalize(vec):
    norm = math.sqrt(sum(x * x for x in vec))
    if norm == 0:
        return None
    return [x / norm for x in vec]


def mean_shift_alignment(embeddings_a, embeddings_b):
    """Homogenization measure over shift vectors, in [-1, 1].

    Shifts are b - a per pair, L2-normalized; the mean direction is
    the (normalized) mean of the normalized shifts. Returns the mean
    cosine similarity of each normalized shift to the mean
    direction. Identical shifts -> ~1.0; opposing shifts cancel the
    mean direction -> 0.0. Uses numpy when available, else pure
    python. Raises ValueError on empty input or mismatched lengths.
    """
    if len(embeddings_a) != len(embeddings_b):
        raise ValueError("embeddings_a and embeddings_b must have equal length")
    if not embeddings_a:
        raise ValueError("empty embedding lists")
    shifts = []
    for a, b in zip(embeddings_a, embeddings_b):
        if len(a) != len(b):
            raise ValueError("paired vectors must have equal length")
        shifts.append([float(y) - float(x) for x, y in zip(a, b)])
    if _np is not None:
        arr = _np.asarray(shifts, dtype=float)
        norms = _np.linalg.norm(arr, axis=1)
        if (norms == 0).any():
            raise ValueError("zero shift vector cannot be normalized")
        unit = arr / norms[:, None]
        mean_dir = unit.mean(axis=0)
        mean_norm = float(_np.linalg.norm(mean_dir))
        if mean_norm == 0:
            return 0.0
        mean_dir = mean_dir / mean_norm
        cosines = unit @ mean_dir
        return float(min(1.0, max(-1.0, float(cosines.mean()))))
    unit = []
    for s in shifts:
        n = _normalize(s)
        if n is None:
            raise ValueError("zero shift vector cannot be normalized")
        unit.append(n)
    dim = len(unit[0])
    mean_dir = [sum(u[d] for u in unit) / len(unit) for d in range(dim)]
    mean_dir = _normalize(mean_dir)
    if mean_dir is None:
        return 0.0
    cosines = [sum(u[d] * mean_dir[d] for d in range(dim)) for u in unit]
    return min(1.0, max(-1.0, sum(cosines) / len(cosines)))
