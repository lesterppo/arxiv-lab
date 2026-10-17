#!/usr/bin/env python3
"""Run the CLM deep-test harness against a Colab-deployed Ollama model (VM-side).

The harness (deep_clm_task.py) is backend-agnostic: it takes a chat_fn.
This driver wires chat_fn to the VM-local Ollama OpenAI-compatible endpoint
(http://127.0.0.1:11434/v1/chat/completions) and runs both arms
(baseline truncation vs CLM live-file editing).

Stdlib only. No credentials. Safe to commit.

Usage:
    python3 colab_run_clm.py [--model gpt-oss:20b] [--out /content/clm_colab_results.json]
"""
import argparse
import json
import os
import sys
import time
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from deep_clm_task import run_arm, DocStore, BUDGET_TOKENS

OLLAMA = "http://127.0.0.1:11434"


def make_chat(model):
    def chat(system, user):
        body = json.dumps({
            "model": model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": 0.2,
            "stream": False,
        }).encode()
        req = urllib.request.Request(
            OLLAMA + "/v1/chat/completions", data=body,
            headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=900) as r:
            return json.load(r)["choices"][0]["message"]["content"]
    return chat


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gpt-oss:20b")
    ap.add_argument("--out", default="/content/clm_colab_results.json")
    args = ap.parse_args()

    chat = make_chat(args.model)
    results = {
        "model": args.model,
        "deployed": "colab-t4-ollama",
        "budget_tokens": BUDGET_TOKENS,
        "arms": {},
    }
    for arm in ("baseline", "clm"):
        print("=== arm %s ===" % arm, flush=True)
        t0 = time.time()
        res = run_arm(chat, arm, docstore=DocStore(), verbose=True)
        res["wall_secs"] = round(time.time() - t0, 1)
        results["arms"][arm] = res
        print("=== arm %s done: %s ===" % (
            arm, {k: res.get(k) for k in ("accuracy", "turns", "searches",
                                          "bash_edits", "peak_tokens",
                                          "error")}), flush=True)
        with open(args.out, "w") as f:
            json.dump(results, f, indent=1)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
