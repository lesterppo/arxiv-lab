"""Kontoyiannis entropy-rate text filter (arXiv:2610.01493).

Model-free text-diversity scoring from raw-text match-length statistics:
calibrated entropy-rate proxy, collapse tracker, and diversity filter for
iterative fine-tuning pipelines.

Pure stdlib. No numpy needed.
"""

from arxiv_lab.text.entropy import (
    entropy_rate,
    text_entropy_rate,
    unique_trigrams,
    repetition_rate,
    bernoulli_words,
    repetitive_words,
    vocab_words,
)

__all__ = [
    "entropy_rate",
    "text_entropy_rate",
    "unique_trigrams",
    "repetition_rate",
    "bernoulli_words",
    "repetitive_words",
    "vocab_words",
]
