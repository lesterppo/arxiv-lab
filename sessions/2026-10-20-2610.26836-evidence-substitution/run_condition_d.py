"""Condition-D model-backed run (own-output pre-seeded, gpt-oss-20b via NIM)."""
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
for _p in ("/tmp/arxiv_20261020",
           os.path.expanduser("~/workspace/goals/arxiv-daily-read-and-test/hidden_files")):
    if _p not in sys.path:
        sys.path.append(_p)

import aqsim
import harness
import nim_client

ITEM_SEEDS = [5100 + i for i in range(15)]


def call_model(messages, tools=None):
    if len(messages) > 11:
        messages = [messages[0]] + messages[-10:]
    text, tcalls = nim_client.chat(messages, tools=tools, max_tokens=768)
    time.sleep(1.0)
    return text, tcalls


def main():
    items = [aqsim.make_question(s) for s in ITEM_SEEDS]
    results = []
    for it in items:
        try:
            res = harness.run_condition(it, "D", call_model)
            res["cls"] = harness.classify(res, it)
        except Exception as e:
            res = {"seed": it["seed"], "condition": "D",
                   "error": f"{type(e).__name__}: {e}",
                   "correct": False, "cls": "error"}
        results.append(res)
        print(f"[D] seed={it['seed']} pred={res.get('pred')} ans={it['answer']} "
              f"correct={res.get('correct')} cls={res.get('cls')} "
              f"tools={[n for n, _ in res.get('tool_log', [])]}", flush=True)
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "results_d.json"), "w") as f:
        json.dump(results, f, indent=1)
    acc = sum(1 for r in results if r.get("correct")) / len(results)
    subs = sum(1 for r in results if r.get("cls") == "substitution")
    print(f"cond D: acc={acc:.3f} substitution_errors={subs}/{len(results)}")


if __name__ == "__main__":
    main()
