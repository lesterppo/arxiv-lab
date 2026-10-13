"""MagicBeanStalk live test (arXiv:2610.12659) on Qwen3.5-9B (Ollama, Colab T4).

Paper claim under test: for a fixed target policy and rollout budget, the
bulk of a pooled RLVR math corpus falls into the "zero-advantage trap"
(problems solved in EVERY rollout or failed in EVERY rollout -> zero
advantage -> no learning signal), and recursive data synthesis (teacher
generates calibrated variants of hard problems requiring the same
reasoning operations) produces problems with MIXED outcomes, i.e. a
usable learning signal.

Phase 1: G rollouts per problem on N inline verifiable math problems;
  measure the zero-advantage fraction.
Phase 2: for all-failed ("hard") problems, ask the model-as-teacher for 2
  easier variants requiring the same operations; roll out on each variant;
  measure the mixed-outcome (non-zero-advantage) fraction.

Stdlib only. Writes /content/mbst_results.json. Run VM-side (localhost
Ollama); no tunnel needed.
"""

import json
import re
import time
import urllib.request

OLLAMA = "http://localhost:11434"
MODEL = "qwen3.5:9b"
N_ROLLOUTS = 4
VARIANT_ROLLOUTS = 3
TIMEOUT = 300

# 20 inline verifiable problems (integer answers), graduated difficulty.
PROBLEMS = [
    ("A bakery sells 24 loaves per day. How many loaves in 17 days?", 408),
    ("What is 137 + 286 - 154?", 269),
    ("A rectangle is 15 cm by 9 cm. What is its area in square cm?", 135),
    ("What is 48 divided by 6, multiplied by 11?", 88),
    ("A train travels 320 km in 4 hours. How far in 7 hours at the same speed?", 560),
    ("What is the sum of the first 20 positive integers?", 210),
    ("A shopkeeper buys 12 pens at $3 each and sells them at $5 each. Profit?", 24),
    ("What is 7 squared plus 5 cubed?", 174),
    ("If 3 workers build a wall in 12 days, how many days for 4 workers?", 9),
    ("What is the greatest common divisor of 84 and 126?", 42),
    ("A number doubled plus 14 equals 92. What is the number?", 39),
    ("What is the least common multiple of 12 and 18?", 36),
    ("The product of two consecutive integers is 156. What is the larger?", 13),
    ("What remainder when 7^5 is divided by 10?", 7),
    ("A palindrome number between 200 and 300 divisible by 11?", 242),
    ("How many positive divisors does 72 have?", 12),
    ("What is 15% of 240 plus 25% of 160?", 76),
    ("The sum of three consecutive even numbers is 102. Largest?", 36),
    ("What is the 10th prime number?", 29),
    ("A clock shows 3:00. How many degrees between the hands?", 90),
]

SOLVE_PROMPT = (
    "Solve this math problem. Show brief working, then put your final "
    "answer on its own line exactly as: ANSWER: <integer>\n\nProblem: {q}"
)

TEACHER_PROMPT = (
    "You are a math problem generator. The student FAILED this problem on "
    "every attempt:\n\nProblem: {q} (answer: {a})\n\n"
    "Generate 2 NEW problems that require the SAME reasoning operations "
    "but are slightly easier (smaller numbers, fewer steps). Each must have "
    "a single integer answer. Reply in EXACTLY this format:\n"
    "Q1: <problem text> A1: <integer>\n"
    "Q2: <problem text> A2: <integer>"
)


def generate(prompt, temperature=0.7):
    body = json.dumps({
        "model": MODEL,
        "prompt": prompt,
        "stream": False,
        # qwen3.5:9b is a hybrid thinking model: with thinking enabled it
        # spends the whole token budget in the `thinking` field and returns
        # an empty `response`. Disable thinking for direct short answers.
        "think": False,
        "options": {"temperature": temperature, "num_predict": 256},
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


def rollouts(question, answer, g):
    oks = []
    for _ in range(g):
        try:
            got = extract_answer(generate(SOLVE_PROMPT.format(q=question)))
            oks.append(got == answer)
        except Exception as e:
            print("rollout error:", e, flush=True)
            oks.append(False)
    return oks


def main():
    t0 = time.time()
    phase1 = []
    for q, a in PROBLEMS:
        oks = rollouts(q, a, N_ROLLOUTS)
        s = sum(oks)
        phase1.append({"q": q, "answer": a, "successes": s,
                       "g": N_ROLLOUTS,
                       "trap": s == 0 or s == N_ROLLOUTS})
        print(f"P1 [{len(phase1):2d}/20] successes={s}/{N_ROLLOUTS} "
              f"trap={phase1[-1]['trap']}", flush=True)

    zero_frac = sum(p["trap"] for p in phase1) / len(phase1)
    all_ok = sum(p["successes"] == N_ROLLOUTS for p in phase1)
    all_bad = sum(p["successes"] == 0 for p in phase1)

    # Phase 2: recursive synthesis on the hard (all-failed) problems.
    variants = []
    hard = [p for p in phase1 if p["successes"] == 0]
    for h in hard:
        try:
            text = generate(TEACHER_PROMPT.format(q=h["q"], a=h["answer"]),
                            temperature=0.9)
        except Exception as e:
            print("teacher error:", e, flush=True)
            continue
        pairs = re.findall(r"Q\d:\s*(.*?)\s*A\d:\s*(-?\d+)", text, re.S)
        for vq, va in pairs[:2]:
            vq = " ".join(vq.split())
            oks = rollouts(vq, int(va), VARIANT_ROLLOUTS)
            s = sum(oks)
            variants.append({"from": h["q"][:60], "q": vq[:120],
                             "answer": int(va), "successes": s,
                             "g": VARIANT_ROLLOUTS,
                             "mixed": 0 < s < VARIANT_ROLLOUTS})
            print(f"P2 variant successes={s}/{VARIANT_ROLLOUTS} "
                  f"mixed={variants[-1]['mixed']}", flush=True)

    mixed_frac = (sum(v["mixed"] for v in variants) / len(variants)
                  if variants else None)

    result = {
        "model": MODEL,
        "phase1": {
            "n_problems": len(phase1),
            "rollouts_per_problem": N_ROLLOUTS,
            "zero_advantage_fraction": zero_frac,
            "all_correct": all_ok,
            "all_failed": all_bad,
            "per_problem_successes": [p["successes"] for p in phase1],
        },
        "phase2": {
            "n_hard_problems": len(hard),
            "n_variants": len(variants),
            "mixed_outcome_fraction": mixed_frac,
            "variant_successes": [v["successes"] for v in variants],
        },
        "elapsed_s": round(time.time() - t0, 1),
    }
    with open("/content/mbst_results.json", "w") as f:
        json.dump(result, f, indent=1)
    print(json.dumps(result["phase1"], indent=1), flush=True)
    print(json.dumps(result["phase2"], indent=1), flush=True)
    print("WROTE /content/mbst_results.json", flush=True)


if __name__ == "__main__":
    main()
