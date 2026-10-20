"""Model-backed evidence-substitution run (gpt-oss-20b via NVIDIA NIM).

15 items x 3 conditions (A inline / B retrieval / C named-authority).
Writes results.json + summary to stdout. The NIM client lives in /tmp
(surrogate auth, never committed).
"""
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

N_ITEMS = 15
ITEM_SEEDS = [5100 + i for i in range(N_ITEMS)]


def call_model(messages, tools=None):
    # trim runaway histories (keep system + last 10)
    if len(messages) > 11:
        messages = [messages[0]] + messages[-10:]
    text, tcalls = nim_client.chat(messages, tools=tools, max_tokens=768)
    time.sleep(1.0)
    return text, tcalls


def main():
    items = [aqsim.make_question(s) for s in ITEM_SEEDS]
    # resume: skip (cond, seed) pairs already completed in a prior log
    done = set()
    log_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "model_run.log")
    if os.path.exists(log_path):
        import re
        for line in open(log_path):
            m = re.match(r"\[(.)\] seed=(\d+)", line)
            if m:
                done.add((m.group(1), int(m.group(2))))
    if done:
        print(f"resuming: skipping {len(done)} completed runs", flush=True)
    results = []
    for cond in ("A", "B", "C"):
        for it in items:
            if (cond, it["seed"]) in done:
                continue
            try:
                res = harness.run_condition(it, cond, call_model)
                res["cls"] = harness.classify(res, it)
            except Exception as e:
                res = {"seed": it["seed"], "condition": cond,
                       "error": f"{type(e).__name__}: {e}",
                       "correct": False, "cls": "error"}
            results.append(res)
            ok = res.get("correct")
            print(f"[{cond}] seed={it['seed']} pred={res.get('pred')} "
                  f"ans={it['answer']} correct={ok} cls={res.get('cls')}",
                  flush=True)
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "results.json")
    # merge: reconstruct minimal records for runs completed in a prior
    # (killed) invocation from the log lines
    import re
    seen = {(r.get("condition"), r.get("seed")) for r in results}
    for line in open(log_path):
        m = re.match(r"\[(.)\] seed=(\d+) pred=(\S+) ans=(\S+) correct=(\S+) cls=(\S+)",
                     line)
        if m and (m.group(1), int(m.group(2))) not in seen:
            results.append({"seed": int(m.group(2)), "condition": m.group(1),
                            "pred": m.group(3), "answer": m.group(4),
                            "correct": m.group(5) == "True",
                            "cls": m.group(6), "recovered_from_log": True})
    with open(out, "w") as f:
        json.dump(results, f, indent=1)
    # summary
    for cond in ("A", "B", "C"):
        rs = [r for r in results if r.get("condition") == cond]
        acc = sum(1 for r in rs if r.get("correct")) / len(rs)
        subs = sum(1 for r in rs if r.get("cls") == "substitution")
        wrong = sum(1 for r in rs if not r.get("correct"))
        print(f"cond {cond}: acc={acc:.3f} ({sum(1 for r in rs if r.get('correct'))}/{len(rs)}) "
              f"errors={wrong} substitution={subs}")
    print("wrote", out)


if __name__ == "__main__":
    main()
