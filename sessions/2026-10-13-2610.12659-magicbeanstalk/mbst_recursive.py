"""MagicBeanStalk recursive calibration (arXiv:2610.12659), phase 3.

Implements the paper's actual "recursive data synthesis" loop: for each
hard (all-failed) problem, iterate with the teacher -- if a variant comes
out all-correct ask for a HARDER one, if all-failed ask for an EASIER one,
keeping the same reasoning operations -- until a variant yields MIXED
rollout outcomes (non-zero advantage) or 3 rounds are spent.

Stdlib only. Writes /content/mbst_recursive.json. Run VM-side.
"""

import json
import re
import time
import urllib.request

OLLAMA = "http://localhost:11434"
MODEL = "qwen3.5:9b"
G = 3
MAX_ROUNDS = 3
TIMEOUT = 300

HARD = [
    ("What is the greatest common divisor of 84 and 126?", 42),
    ("A palindrome number between 200 and 300 divisible by 11?", 242),
]

SOLVE_PROMPT = (
    "Solve this math problem. Show brief working, then put your final "
    "answer on its own line exactly as: ANSWER: <integer>\n\nProblem: {q}"
)

TEACH_EASIER = (
    "You are a math problem generator. The student FAILED this problem on "
    "every attempt:\n\nProblem: {q} (answer: {a})\n\n"
    "Generate 1 NEW problem requiring the SAME reasoning operations but "
    "EASIER (smaller numbers, fewer steps), single integer answer. Reply in "
    "EXACTLY this format:\nQ: <problem text> A: <integer>"
)

TEACH_HARDER = (
    "You are a math problem generator. The student solved this problem too "
    "easily:\n\nProblem: {q} (answer: {a})\n\n"
    "Generate 1 NEW problem requiring the SAME reasoning operations but "
    "HARDER (larger numbers, one more step), single integer answer. Reply in "
    "EXACTLY this format:\nQ: <problem text> A: <integer>"
)


def generate(prompt, temperature=0.9, num_predict=256):
    body = json.dumps({
        "model": MODEL, "prompt": prompt, "stream": False, "think": False,
        "options": {"temperature": temperature, "num_predict": num_predict},
    }).encode()
    req = urllib.request.Request(
        OLLAMA + "/api/generate", data=body,
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
        return json.loads(r.read())["response"]


def extract_answer(text):
    m = re.findall(r"ANSWER:\s*(-?\d+)", text)
    if m:
        return int(m[-1])
    m = re.findall(r"-?\d+", text)
    return int(m[-1]) if m else None


def parse_variant(text):
    m = re.search(r"Q:\s*(.*?)\s*A:\s*(-?\d+)", text, re.S)
    if not m:
        return None, None
    return " ".join(m.group(1).split()), int(m.group(2))


def rollouts(question, answer):
    oks = []
    for _ in range(G):
        try:
            got = extract_answer(generate(SOLVE_PROMPT.format(q=question),
                                          temperature=0.7, num_predict=128))
            oks.append(got == answer)
        except Exception as e:
            print("rollout error:", e, flush=True)
            oks.append(False)
    return oks


def main():
    t0 = time.time()
    report = []
    for hq, ha in HARD:
        entry = {"hard_problem": hq[:80], "rounds": []}
        prompt = TEACH_EASIER.format(q=hq, a=ha)
        calibrated = None
        for rnd in range(1, MAX_ROUNDS + 1):
            try:
                text = generate(prompt)
            except Exception as e:
                print("teacher error:", e, flush=True)
                break
            vq, va = parse_variant(text)
            if vq is None:
                print(f"round {rnd}: unparseable teacher output", flush=True)
                break
            oks = rollouts(vq, va)
            s = sum(oks)
            mixed = 0 < s < G
            entry["rounds"].append({"round": rnd, "variant": vq[:120],
                                    "answer": va, "successes": s,
                                    "mixed": mixed})
            print(f"hard='{hq[:40]}...' round={rnd} successes={s}/{G} "
                  f"mixed={mixed}", flush=True)
            if mixed:
                calibrated = entry["rounds"][-1]
                break
            # Recursive step: steer difficulty toward the learnable zone.
            prompt = (TEACH_HARDER if s == G else TEACH_EASIER).format(
                q=vq, a=va)
        entry["calibrated"] = calibrated is not None
        entry["rounds_used"] = len(entry["rounds"])
        report.append(entry)

    result = {
        "model": MODEL,
        "rollouts_per_variant": G,
        "max_rounds": MAX_ROUNDS,
        "n_hard_problems": len(HARD),
        "n_calibrated": sum(e["calibrated"] for e in report),
        "report": report,
        "elapsed_s": round(time.time() - t0, 1),
    }
    with open("/content/mbst_recursive.json", "w") as f:
        json.dump(result, f, indent=1)
    print(json.dumps({k: v for k, v in result.items()
                      if k != "report"}, indent=1), flush=True)
    print("WROTE /content/mbst_recursive.json", flush=True)


if __name__ == "__main__":
    main()
