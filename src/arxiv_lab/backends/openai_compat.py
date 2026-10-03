"""OpenAI-compatible chat client over stdlib urllib (no dependencies).

Lets arxiv-lab experiments call any OpenAI-compatible chat endpoint —
NVIDIA NIM, DeepSeek, OpenAI, Ollama, vLLM — with one interface.

Authentication rule (strict): the API key comes ONLY from an explicit
argument or an environment variable (``NVIDIA_API_KEY``,
``OPENAI_API_KEY``, ``DEEPSEEK_API_KEY``). Keys are stored on the instance
only, and are NEVER printed, logged, or included in exception messages.

Pure stdlib.
"""

import json
import os
import urllib.error
import urllib.request

#: Env vars searched (in order) for an API key when none is passed explicitly.
ENV_KEYS = ("NVIDIA_API_KEY", "OPENAI_API_KEY", "DEEPSEEK_API_KEY")

#: (base_url, key env var) presets. ``key_var=None`` means no key needed.
PRESETS = {
    "openai": ("https://api.openai.com/v1", "OPENAI_API_KEY"),
    "nvidia": ("https://integrate.api.nvidia.com/v1", "NVIDIA_API_KEY"),
    "deepseek": ("https://api.deepseek.com/v1", "DEEPSEEK_API_KEY"),
    "ollama": ("http://localhost:11434/v1", None),
}


def _first_env(names):
    for var in names:
        val = os.environ.get(var)
        if val:
            return val
    return None


class OpenAICompatClient:
    """Minimal ``/chat/completions`` client for OpenAI-compatible APIs."""

    def __init__(self, api_key=None, base_url=None, timeout=60):
        """``api_key``: explicit key, else the first set var in ``ENV_KEYS``.
        ``base_url``: explicit base URL, else ``OPENAI_BASE_URL`` env var,
        else the OpenAI default. Trailing slashes are stripped."""
        self.timeout = timeout
        # The key lives only on this instance and is never logged.
        self._api_key = api_key if api_key else _first_env(ENV_KEYS)
        self.base_url = (base_url
                         or os.environ.get("OPENAI_BASE_URL")
                         or "https://api.openai.com/v1").rstrip("/")

    @classmethod
    def for_preset(cls, name, api_key=None, timeout=60):
        """Build a client for a known preset: 'openai', 'nvidia',
        'deepseek', or 'ollama'. The preset's key env var is used when no
        explicit ``api_key`` is given."""
        try:
            base_url, key_var = PRESETS[name]
        except KeyError:
            raise ValueError(
                f"unknown preset {name!r}; choose from {sorted(PRESETS)}")
        key = api_key if api_key else (os.environ.get(key_var) if key_var else None)
        return cls(api_key=key, base_url=base_url, timeout=timeout)

    @property
    def has_key(self):
        """Whether an API key is configured (the key itself is never exposed)."""
        return bool(self._api_key)

    def chat(self, model, messages, temperature=0.7, max_tokens=256):
        """POST ``{base_url}/chat/completions``; return the assistant message
        content as ``str``.

        ``messages``: list of ``{"role": ..., "content": ...}`` dicts.
        Raises ``RuntimeError`` on HTTP or response-shape errors. Error
        messages never contain the API key.
        """
        url = self.base_url + "/chat/completions"
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = "Bearer " + self._api_key
        req = urllib.request.Request(
            url, data=json.dumps(payload).encode("utf-8"), headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            # Read a short body for diagnostics; the key is never in it.
            try:
                body = e.read().decode("utf-8", "replace")[:500]
            except Exception:
                body = ""
            raise RuntimeError(
                f"chat completions failed: HTTP {e.code} {body}") from None
        except urllib.error.URLError as e:
            raise RuntimeError(f"chat completions failed: {e.reason}") from None
        try:
            content = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as e:
            raise RuntimeError(
                "chat completions: unexpected response shape") from e
        return content if isinstance(content, str) else str(content)


def chat(model, messages, temperature=0.7, max_tokens=256,
         api_key=None, base_url=None, timeout=60):
    """One-shot convenience wrapper: build a client and call ``chat``."""
    client = OpenAICompatClient(api_key=api_key, base_url=base_url,
                                timeout=timeout)
    return client.chat(model, messages, temperature=temperature,
                       max_tokens=max_tokens)
