#!/usr/bin/env python3
"""Premise validation for 2610.02740 "Prospective Hindsight" on a deployed Ollama endpoint.

Asks ~24 verifiable math questions; each prompt demands:
  CONFIDENCE: <int 0-100>   (prospective prediction of success before feedback)
  ANSWER: <final answer>    (retrospective outcome after grading)
Surprise per sample = |confidence/100 - correctness|.
Also reports the paper's headline single-turn mode: OVERCONFIDENT FAILURES
(fraction of wrong answers given with confidence >= 70).
Stdlib only (urllib). Results JSON path given as argv[1]; API base URL as argv[2].
"""
import json, re, sys, time, urllib.request

OUT_PATH = sys.argv[1]
BASE_URL = sys.argv[2].rstrip("/")

# (question, correct integer answer) — unambiguous integers, invented for this test
PROBLEMS = [
    ("What is 17 * 23?", 391),
    ("What is 456 + 789?", 1245),
    ("What is 1000 - 377?", 623),
    ("What is 12^3 (12 cubed)?", 1728),
    ("What is 15 * 15 * 2?", 450),
    ("A train travels 60 km/h for 3.5 hours. How many km?", 210),
    ("What is the sum of the integers from 1 to 50?", 1275),
    ("A box has 8 rows of 12 eggs. A shopper takes 17 eggs. How many remain?", 79),
    ("What is 2^10 + 2^8?", 1280),
    ("A rectangle is 14 cm by 9 cm. What is its area in cm^2?", 126),
    ("What is 7 * 8 * 9?", 504),
    ("A phone costs $480, on sale at 25% off. What is the sale price in dollars?", 360),
    ("What is the 12th Fibonacci number, with F1=1, F2=1?", 144),
    ("What is 999 * 101?", 100899),
    ("If you roll a fair 6-sided die, how many possible outcomes?", 6),
    ("What is the least common multiple of 12 and 18?", 36),
    ("A clock shows 3:40. How many minutes until 5:00?", 80),
    ("What is 3^5?", 243),
    ("A runner does 5 laps of a 400 m track. How many meters total?", 2000),
    ("What is the perimeter of a square with side 23 cm?", 92),
    ("A jar has 3 red, 5 blue, 2 green marbles. How many marbles total?", 10),
    ("What is 11 * 11 * 11?", 1331),
    ("A pizza cut into 8 slices; 3 people eat 2 slices each. How many slices left?", 2),
    ("What is 45 + 67 + 89?", 201),
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
