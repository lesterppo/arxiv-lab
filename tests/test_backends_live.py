"""Live smoke test for arxiv_lab.backends (OpenAI-compatible chat).

SKIPPED unless at least one API-key env var is present. When a key is set,
does exactly ONE tiny chat call per configured backend and asserts a
non-empty reply. Never prints keys.

Backends probed (key env var -> preset -> cheap model):
  NVIDIA_API_KEY -> nvidia   (e.g. a free small instruct model)
  DEEPSEEK_API_KEY -> deepseek
  OPENAI_API_KEY -> openai

Run:  python3 tests/test_backends_live.py
Set e.g. NVIDIA_API_KEY=... to enable. Costs are negligible (max_tokens=8).
"""

import os
import sys

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from arxiv_lab.backends import OpenAICompatClient

# (env var, preset, tiny/cheap model id)
BACKENDS = [
    ("NVIDIA_API_KEY", "nvidia", "nvidia/llama-3.1-8b-instruct"),
    ("DEEPSEEK_API_KEY", "deepseek", "deepseek-chat"),
    ("OPENAI_API_KEY", "openai", "gpt-4o-mini"),
]

TINY_MESSAGES = [{"role": "user", "content": "Reply with exactly the word: ok"}]


def main():
    configured = [(var, preset, model) for var, preset, model in BACKENDS
                  if os.environ.get(var)]
    if not configured:
        print("SKIPPED: no API key env var set "
              "(NVIDIA_API_KEY, DEEPSEEK_API_KEY, OPENAI_API_KEY)")
        return
    for var, preset, model in configured:
        client = OpenAICompatClient.for_preset(preset, timeout=60)
        assert client.has_key, f"{preset}: expected a key from {var}"
        reply = client.chat(model, TINY_MESSAGES, temperature=0.0,
                            max_tokens=8)
        assert isinstance(reply, str) and reply.strip(), \
            f"{preset}: empty reply"
        print(f"  {preset} ({model}): reply={reply.strip()[:60]!r} OK")
    print("ALL LIVE BACKEND CHECKS PASSED")


if __name__ == "__main__":
    main()
