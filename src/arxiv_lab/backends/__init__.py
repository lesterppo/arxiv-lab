"""Model backends for arxiv-lab experiments (stdlib only).

``openai_compat`` provides an OpenAI-compatible ``/chat/completions`` client
usable against NVIDIA NIM, DeepSeek, OpenAI, Ollama, and vLLM endpoints.
API keys come ONLY from explicit arguments or environment variables and are
never printed or logged.
"""

from arxiv_lab.backends.openai_compat import (
    OpenAICompatClient,
    chat,
    ENV_KEYS,
    PRESETS,
)

__all__ = ["OpenAICompatClient", "chat", "ENV_KEYS", "PRESETS"]
