"""Python side of the GrepMem Python-vs-Rust parity sweep.

Modes (mirroring the `grepmem` binary subcommands, same JSON keys):

    parity_gen.py rng <seed> <n>
        # first n random() values + raw getrandbits(32) words
    parity_gen.py gen <seed> <n_lines> <n_questions>
        # one generated session in the line-based session format
    parity_gen.py eval <session_file> <policy>
        #   [--max-lines N] [--entity-only] [--attr-only]
        # evaluate one policy on a session file, print eval JSON

The eval JSON has the same shape as the Rust binary's `eval` output,
so tests/parity_sweep.py can compare them field by field. Floats are
compared with a 1e-9 tolerance there; everything else is exact.
Not part of the Rust build; kept under tests/ as the parity oracle.
"""
import json
import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..",
                                "..", "..", "src"))

from arxiv_lab.agent import grepmem


def session_to_text(sess, seed, n_lines, n_questions):
    out = ["# grepmem-session-v1 seed=%d n_lines=%d n_questions=%d"
           % (seed, n_lines, n_questions)]
    for ln in sess["log"].lines:
        out.append("L " + ln)
    # Recover the fact step from the position (step = round(pos*(n-1))).
    for q in sess["questions"]:
        step = int(round(q["position"] * max(n_lines - 1, 1)))
        out.append("Q\t%s\t%s\t%s\t%d\t%d\t%s" % (
            q["entity"], q["attr"], q["answer"], step,
            1 if q["paraphrased"] else 0, q["text"]))
    return "\n".join(out) + "\n"


def session_from_text(text):
    log = grepmem.AppendOnlyLog()
    questions = []
    for raw in text.splitlines():
        if raw.startswith("#") or not raw:
            continue
        if raw.startswith("L "):
            log.append(raw[2:])
        elif raw.startswith("Q\t"):
            parts = raw[2:].split("\t")
            assert len(parts) == 6, raw
            step = int(parts[3])
            questions.append({
                "text": parts[5],
                "entity": parts[0],
                "attr": parts[1],
                "answer": parts[2],
                "step": step,
                "position": step / max(len(log.lines) - 1, 1),
                "paraphrased": parts[4] == "1",
            })
    return {"log": log, "questions": questions}


def eval_to_json(res, per_question):
    return json.dumps({
        "policy": res["policy"],
        "accuracy": res["accuracy"],
        "n": res["n"],
        "correct": sum(1 for q in per_question if q["correct"]),
        "mean_context_lines": res["mean_context_lines"],
        "total_ctx_lines": sum(q["ctx_lines"] for q in per_question),
        "by_position": res["by_position"],
        "by_position_n": [res["by_position_n"][k] for k in ("early", "mid", "late")],
        "paraphrase_accuracy": res["paraphrase_accuracy"],
        "paraphrase_n": res["paraphrase_n"],
        "per_question": per_question,
    })


def rng_kind(seed, n):
    r = random.Random(seed)
    randoms = [r.random() for _ in range(n)]
    r2 = random.Random(seed)
    u32s = [r2.getrandbits(32) for _ in range(n)]
    print(json.dumps({"kind": "rng", "seed": seed,
                      "random": randoms, "u32": u32s}))


def gen_kind(seed, n_lines, n_questions):
    sess = grepmem.generate_session(seed, n_lines=n_lines,
                                    n_questions=n_questions)
    sys.stdout.write(session_to_text(sess, seed, n_lines, n_questions))


def eval_kind(session_file, policy, max_lines=40, entity_only=False,
              attr_only=False):
    with open(session_file) as f:
        sess = session_from_text(f.read())
    log = sess["log"]
    per_question = []
    correct = 0
    total_ctx = 0
    pos_ok = {"early": 0, "mid": 0, "late": 0}
    pos_n = {"early": 0, "mid": 0, "late": 0}
    para_ok = para_n = 0
    for q in sess["questions"]:
        if policy == "grepmem":
            kw = {"max_lines": max_lines, "entity_only": entity_only,
                  "attr_only": attr_only}
            ctx = grepmem.policy_grepmem(log, q, **kw)
        elif policy == "full":
            ctx = grepmem.policy_full(log, q)
        elif policy == "summary":
            ctx = grepmem.policy_summary(log, q)
        elif policy == "recency":
            ctx = grepmem.policy_recency(log, q)
        elif policy == "random_mask":
            ctx = grepmem.policy_random_mask(log, q)
        else:
            raise SystemExit("unknown policy: %s" % policy)
        ans = grepmem.answer_question(q, ctx)
        ok = ans == q["answer"]
        correct += ok
        total_ctx += len(ctx)
        pos = ("early" if q["position"] < 0.33
               else ("mid" if q["position"] < 0.67 else "late"))
        pos_ok[pos] += ok
        pos_n[pos] += 1
        if q["paraphrased"]:
            para_ok += ok
            para_n += 1
        per_question.append({
            "text": q["text"], "entity": q["entity"], "attr": q["attr"],
            "expected": q["answer"], "answer": ans, "correct": ok,
            "position": q["position"], "paraphrased": q["paraphrased"],
            "ctx_lines": len(ctx), "ctx": ctx,
        })
    n = len(sess["questions"])
    res = {
        "policy": policy,
        "accuracy": correct / n if n else 0.0,
        "n": n,
        "mean_context_lines": total_ctx / n if n else 0.0,
        "by_position": {k: (pos_ok[k] / pos_n[k] if pos_n[k] else 0.0)
                        for k in pos_ok},
        "by_position_n": pos_n,
        "paraphrase_accuracy": para_ok / para_n if para_n else 0.0,
        "paraphrase_n": para_n,
    }
    print(eval_to_json(res, per_question))


if __name__ == "__main__":
    kind = sys.argv[1]
    if kind == "rng":
        rng_kind(int(sys.argv[2]), int(sys.argv[3]))
    elif kind == "gen":
        gen_kind(int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]))
    elif kind == "eval":
        # eval <session_file> <policy> [--max-lines N]
        #      [--entity-only] [--attr-only]
        rest = sys.argv[4:]
        ml, eo, ao = 40, False, False
        i = 0
        while i < len(rest):
            if rest[i] == "--max-lines":
                ml = int(rest[i + 1])
                i += 2
            elif rest[i] == "--entity-only":
                eo = True
                i += 1
            elif rest[i] == "--attr-only":
                ao = True
                i += 1
            else:
                raise SystemExit("unknown flag: %s" % rest[i])
        eval_kind(sys.argv[2], sys.argv[3], ml, eo, ao)
    else:
        raise SystemExit("unknown kind: %s" % kind)
