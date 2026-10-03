"""
Data generation for the Finetuning-with-Sampling replication (arXiv:2610.02140).

Pure stdlib — runs anywhere (this VM for prep, Colab for the live test).

Design
------
* Toy reasoning task: two-step arithmetic word problems with a verifiable
  \\boxed{} answer.
* OFF-policy expert traces: a rigid, verbose, bracketed "PROTOCOL 7" style
  the base model would essentially never produce on its own (deliberately
  low likelihood under the reference model, correct by construction).
* Held-out test problems: same distribution, different seed.
* Forgetting probes: (a) general-knowledge QA items the base model should
  answer (filtered dynamically at run time to items the base gets right),
  (b) plain general-domain sentences for an NLL probe.

No secrets, no network, no model weights needed.
"""
import random
import re

BOXED_RE = re.compile(r"\\boxed\{\s*(-?\d+)\s*\}")


def extract_boxed(text):
    m = BOXED_RE.findall(text)
    return int(m[-1]) if m else None


def gen_problems(n, seed):
    """Two-step arithmetic, medium difficulty. Returns [(question, answer)]."""
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        pat = rng.choice([0, 1])
        if pat == 0:
            a, b = rng.randint(2, 9), rng.randint(2, 9)
            c = rng.randint(2, 6)
            ans = (a + b) * c
            q = f"What is ({a} + {b}) times {c}? Put the final answer in \\boxed{{}}."
            steps = (f"{a} + {b}", a + b, f"{a + b} * {c}", ans)
        else:
            a, b = rng.randint(2, 9), rng.randint(2, 9)
            c = rng.randint(2, 20)
            ans = a * b - c
            q = f"What is {a} times {b} minus {c}? Put the final answer in \\boxed{{}}."
            steps = (f"{a} * {b}", a * b, f"{a * b} - {c}", ans)
        out.append({"question": q, "answer": ans, "steps": steps})
    return out


def expert_trace(problem):
    """Rigid off-policy expert trace: correct by construction, stylistically
    alien to a chat model's natural output."""
    s1, v1, s2, v2 = problem["steps"]
    return (
        "REASONING TRACE \u2014 PROTOCOL 7\n"
        f"[PARSE] operands extracted from query\n"
        f"[STEP 1] compute {s1} = {v1}\n"
        f"[STEP 2] compute {s2} = {v2}\n"
        f"[CHECK] recompute {s2} = {v2} \u2713 consistent\n"
        f"[ANSWER] \\boxed{{{v2}}}"
    )


def is_correct_trace(text, answer):
    return extract_boxed(text) == answer


# ---------------------------------------------------------------------------
# Forgetting probes
# ---------------------------------------------------------------------------

# (prompt, expected substring). Filtered at run time to items the BASE model
# answers correctly, so the probe measures forgetting, not base inability.
FORGET_QA = [
    ("What is the capital of France? Answer with one word.", "paris"),
    ("How many days are in a week? Answer with a number.", "7"),
    ("What color is the sky on a clear day? Answer with one word.", "blue"),
    ("What is 2 + 2? Answer with a number.", "4"),
    ("Which planet is known as the Red Planet? Answer with one word.", "mars"),
    ("How many legs does a spider have? Answer with a number.", "8"),
    ("What is the opposite of hot? Answer with one word.", "cold"),
    ("How many months are in a year? Answer with a number.", "12"),
    ("What do bees make? Answer with one word.", "honey"),
    ("What is the first letter of the alphabet? Answer with one letter.", "a"),
    ("How many wheels does a bicycle have? Answer with a number.", "2"),
    ("What is ice made of? Answer with one word.", "water"),
    ("Which animal says moo? Answer with one word.", "cow"),
    ("What season comes after winter? Answer with one word.", "spring"),
    ("How many fingers are on one hand? Answer with a number.", "5"),
    ("What is the capital of Japan? Answer with one word.", "tokyo"),
    ("What gas do humans need to breathe? Answer with one word.", "oxygen"),
    ("How many sides does a triangle have? Answer with a number.", "3"),
    ("What is the largest ocean on Earth? Answer with one word.", "pacific"),
    ("What do you call a baby dog? Answer with one word.", "puppy"),
]

GENERAL_SENTENCES = [
    "The cat sat on the warm mat by the window.",
    "She poured a cup of coffee and read the newspaper.",
    "The train arrived at the station five minutes late.",
    "Children played football in the park after school.",
    "He opened the door and greeted his old friend.",
    "The baker sold fresh bread every morning.",
    "Rain fell softly on the empty street.",
    "They watched the sunset from the top of the hill.",
    "The dog barked loudly at the passing car.",
    "She wrote a letter to her grandmother.",
    "The library closes at nine o'clock tonight.",
    "Birds sang in the trees at dawn.",
    "He fixed the broken chair with a screwdriver.",
    "The market was full of colorful fruits.",
    "Water boils at one hundred degrees Celsius.",
    "She planted flowers in the small garden.",
    "The airplane flew above the thick clouds.",
    "He bought a new bicycle for his birthday.",
    "The museum displayed ancient Roman coins.",
    "They enjoyed a picnic near the quiet lake.",
    "The baby slept peacefully in the crib.",
    "Snow covered the rooftops of the village.",
    "She learned to swim during the summer holidays.",
    "The chef prepared a delicious meal for the guests.",
    "Leaves fell from the trees in autumn.",
    "He listened to music while walking home.",
    "The teacher explained the lesson clearly.",
    "A rainbow appeared after the heavy rain.",
    "They traveled to the coast by bus.",
    "The old bridge crossed the wide river.",
]


def qa_correct(generation, expected):
    return expected.lower() in generation.lower()


if __name__ == "__main__":
    # smoke test: problems, expert traces, boxed extraction
    probs = gen_problems(5, 0)
    for p in probs:
        t = expert_trace(p)
        assert is_correct_trace(t, p["answer"]), f"expert trace wrong:\n{t}"
        print(p["question"], "->", p["answer"])
    print("expert trace example:\n" + expert_trace(probs[0]))
    print(f"QA items: {len(FORGET_QA)}, sentences: {len(GENERAL_SENTENCES)}")
    print("data_gen OK")
