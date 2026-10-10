"""Model backends for arxiv-lab experiments (stdlib only).

``openai_compat`` provides an OpenAI-compatible ``/chat/completions`` client
usable against NVIDIA NIM, DeepSeek, OpenAI, Ollama, and vLLM endpoints.
API keys come ONLY from explicit arguments or environment variables and are
never printed or logged.

``systemone`` provides a typed-decision client for Ollama >= 0.35
``/v1/systemone`` (Cloudflare Clef / clef-flash decision models): noul /
choice / score answers with calibrated probabilities, no text parsing.
"""

from arxiv_lab.backends.openai_compat import (
    OpenAICompatClient,
    chat,
    ENV_KEYS,
    PRESETS,
)
from arxiv_lab.backends.systemone import (
    SystemoneClient,
    decide,
    build_request,
    parse_answers,
    score_value,
    QUESTION_TYPES,
    DEFAULT_MODEL,
)

__all__ = [
    "OpenAICompatClient", "chat", "ENV_KEYS", "PRESETS",
    "SystemoneClient", "decide", "build_request", "parse_answers",
    "score_value", "QUESTION_TYPES", "DEFAULT_MODEL",
]
