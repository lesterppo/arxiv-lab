"""Kontoyiannis entropy-rate estimator h_k (arXiv:2610.01493).

Paper claim implemented: the non-parametric Kontoyiannis entropy rate
estimator, computed entirely from raw text via match-length statistics with
*no model of any kind*, is a superior training-data filter against
iterative fine-tuning collapse (model collapse): output diversity narrows as
rare patterns are progressively lost, a signature most visible as
phrase-level repetition.

The estimator (Kontoyiannis et al. 1998): for a sequence x_1..x_n, let L_i
be the length of the longest match of the substring starting at position i
with any substring starting at an earlier position. Then::

    h_hat = 1 / ( (1/n) * sum_i L_i / log2(i+1) )

Low h_hat  -> repetitive / low-diversity text (collapse).
High h_hat -> diverse / high-entropy text.

Computed here at the WORD level (tokens = whitespace-split words), which
matches the paper's phrase-level repetition framing while keeping the
alphabet finite. Match search uses C-speed substring checks with binary
search on the match length, so it runs on CPU in seconds.

Also provides the paper's headline diversity metrics (``unique_trigrams``,
``repetition_rate``) and the synthetic text generators used to validate the
estimator (``bernoulli_words``, ``repetitive_words``, ``vocab_words``).

Pure stdlib. No numpy needed.
"""

import math


def _longest_match_len(words, i, cap=200):
    """Longest prefix of words[i:] occurring at any start j < i (word-level).

    Uses binary search on length with C-speed ``in`` checks over the joined
    string. Returns the match length in WORDS.
    """
    if i == 0:
        return 1
    sep = "\x00"
    prefix = sep.join(words[:i]) + sep
    # binary search longest L in [1, cap] with words[i:i+L] in prefix
    lo, hi, best = 1, min(cap, len(words) - i), 0
    while lo <= hi:
        mid = (lo + hi) // 2
        if (sep + sep.join(words[i:i + mid]) + sep) in (sep + prefix):
            best = mid
            lo = mid + 1
        else:
            hi = mid - 1
    return max(best, 1)


def entropy_rate(words, cap=200, sample_stride=1):
    """Kontoyiannis h_hat for a word list, in bits per word.

    ``sample_stride`` subsamples positions for speed (statistically unbiased
    for large n). Returns 0.0 for sequences shorter than 16 words.
    """
    n = len(words)
    if n < 16:
        return 0.0
    idx = range(0, n, sample_stride)
    s = 0.0
    m = 0
    for i in idx:
        L = _longest_match_len(words, i, cap)
        s += L / math.log2(i + 2)  # 1-indexed log(i+1)
        m += 1
    return m / s if s > 0 else 0.0


def text_entropy_rate(text, **kw):
    """Kontoyiannis h_hat for a raw text string (whitespace tokenized)."""
    return entropy_rate(text.split(), **kw)


def unique_trigrams(words):
    """Number of distinct word trigrams (the paper's diversity metric)."""
    return len({tuple(words[i:i + 3]) for i in range(len(words) - 2)})


def repetition_rate(words):
    """Fraction of word trigrams occurring more than once (phrase-level
    repetition, the paper's collapse signature)."""
    if len(words) < 4:
        return 0.0
    tris = [tuple(words[i:i + 3]) for i in range(len(words) - 2)]
    return 1.0 - len(set(tris)) / len(tris)


def bernoulli_words(p, n, rng):
    """i.i.d. Bernoulli(p) word sequence over the alphabet {a, b}."""
    return ["a" if rng.random() < p else "b" for _ in range(n)]


def repetitive_words(period, n):
    """Perfectly periodic word sequence (entropy rate ~ 0)."""
    base = [f"w{k}" for k in range(period)]
    return [base[i % period] for i in range(n)]


def vocab_words(vocab_size, n, rng, zipf=1.2):
    """Zipf-distributed word sequence (natural-text-like diversity)."""
    vocab = [f"w{k}" for k in range(vocab_size)]
    weights = [1.0 / ((k + 1) ** zipf) for k in range(vocab_size)]
    return rng.choices(vocab, weights=weights, k=n)
