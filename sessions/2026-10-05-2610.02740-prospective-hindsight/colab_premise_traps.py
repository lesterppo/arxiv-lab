#!/usr/bin/env python3
"""Premise validation for 2610.02740 "Prospective Hindsight" on a deployed Ollama endpoint.

Asks ~24 verifiable math questions; each prompt demands:
  CONFIDENCE: <int 0-100>   (prospective prediction of success before feedback)
  ANSWER: <final answer>    (retrospective outcome after grading)
Surprise per sample = |confidence/100 - correctness|.\nHARD SET (v2): harder computation + classic cognitive-reflection traps with integer answers,\ndesigned so a 9B model is genuinely uncertain on several items (v1 ceilinged at 14/14).
Also reports the paper's headline single-turn mode: OVERCONFIDENT FAILURES
(fraction of wrong answers given with confidence >= 70).
Stdlib only (urllib). Results JSON path given as argv[1]; API base URL as argv[2].
"""
import json, re, sys, time, urllib.request

OUT_PATH = sys.argv[1]
BASE_URL = sys.argv[2].rstrip("/")

# (question, correct integer answer) — unambiguous integers, invented for this test
PROBLEMS = [
    # Trap-only probe (v3): the 8 cognitive-reflection items from v2 that never
    # got a clean shot (tunnel died). Short run to finally measure
    # overconfident failures, if any.
    ("How many months have 28 days?", 12),
    ("A bat and ball cost $1.10 in total. The bat costs $1.00 more than the ball. How many cents does the ball cost?", 5),
    ("You have 3 apples and you take away 2. How many apples do you have?", 2),
    ("How many times does the digit 9 appear when writing out the integers from 1 to 100?", 20),
    ("What is the 10th prime number?", 29),
    ("A lily pad doubles in area every day and covers the whole lake on day 48. On which day was half the lake covered?", 47),
    ("A farmer has 17 sheep. All but 9 die. How many sheep remain?", 9),
    ("If 5 machines make 5 widgets in 5 minutes, how many minutes do 100 machines need to make 100 widgets?", 5),
]
PROMPT_TMPL = (
    "First write CONFIDENCE: <integer 0-100> estimating the probability you will answer this correctly.\n"
    "Then solve the problem and write ANSWER: <final answer as a single integer>.\n"
    "Problem: {q}"
)

CONF_RE = re.compile(r"CONFIDENCE\s*:\s*(\d{1,3})")
ANS_RE = re.compile(r"ANSWER\s*:\s*(-?\d[\d,]*)")


def generate(prompt, timeout=120):
    body = json.dumps({
        "model": "qwen3.5:9b",
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.0},
    }).encode()
    req = urllib.request.Request(BASE_URL + "/api/generate", data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)["response"]


def main():
    results = []
    for q, correct in PROBLEMS:
        resp = ""
        conf, ans = None, None
        try:
            resp = generate(PROMPT_TMPL.format(q=q))
            m = CONF_RE.search(resp)
            conf = int(m.group(1)) if m else None
            m2 = ANS_RE.search(resp)
            if m2:
                try:
                    ans = int(m2.group(1).replace(",", ""))
                except ValueError:
                    ans = None
        except Exception as e:
            resp = f"<ERROR: {type(e).__name__}: {e}>"
        correct_flag = (ans == correct) if ans is not None else False
        surprise = None
        if conf is not None:
            surprise = abs(conf / 100.0 - (1.0 if correct_flag else 0.0))
        results.append({
            "question": q, "expected": correct,
            "confidence": conf, "parsed_answer": ans,
            "correct": correct_flag, "surprise": surprise,
            "raw": resp[:800],
        })
        time.sleep(1)
    graded = [r for r in results if r["confidence"] is not None]
    acc = sum(r["correct"] for r in graded) / len(graded)
    confs = [r["confidence"] for r in graded]
    mean_conf = sum(confs) / len(confs)
    # ECE: 10 equal-width bins
    ece = 0.0
    for b in range(10):
        lo, hi = b / 10.0, (b + 1) / 10.0
        inb = [r for r in graded if lo <= r["confidence"] / 100.0 < (hi if b < 9 else 1.0001)]
        if inb:
            bin_acc = sum(r["correct"] for r in inb) / len(inb)
            bin_conf = sum(r["confidence"] for r in inb) / len(inb) / 100.0
            ece += len(inb) / len(graded) * abs(bin_acc - bin_conf)
    wrong = [r for r in graded if not r["correct"]]
    overconf_fail = (sum(1 for r in wrong if r["confidence"] >= 70) / len(wrong)) if wrong else None
    surprises = sorted(r["surprise"] for r in graded)
    mean_surprise = sum(surprises) / len(surprises)
    q1 = surprises[len(surprises) // 4]
    med = surprises[len(surprises) // 2]
    q3 = surprises[3 * len(surprises) // 4]
    summary = {
        "model": "qwen3.5:9b", "n_questions": len(PROBLEMS),
        "n_graded": len(graded),
        "accuracy": round(acc, 4),
        "ECE_10bin": round(ece, 4),
        "mean_confidence": round(mean_conf, 2),
        "overconfident_failure_rate": round(overconf_fail, 4) if overconf_fail is not None else None,
        "mean_surprise": round(mean_surprise, 4),
        "surprise_quartiles": {"q1": round(q1, 4), "median": round(med, 4), "q3": round(q3, 4)},
    }
    with open(OUT_PATH, "w") as f:
        json.dump({"summary": summary, "samples": results}, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
