"""
Data generation for the PG-SFT replication (arXiv:2610.00949).

Pure stdlib — runs anywhere (this VM for prep, Colab for the live test).

Design
------
* Toy "agent trajectory" task: two-step arithmetic word problems with a
  verifiable \\boxed{} answer, rendered as multi-turn agent trajectories:
      USER -> ASSISTANT(reasoning) -> TOOL(calc) -> OBSERVATION ->
      ASSISTANT(reasoning) -> TOOL(calc) -> OBSERVATION -> ASSISTANT(final)
  Only ASSISTANT and TOOL turns are supervised (USER/OBSERVATION masked).
* Half the trajectories carry an extra low-information "double-check"
  reasoning turn, so turn-level information gain varies across turns.
* Forgetting probes: (a) general-knowledge QA items (filtered at run time
  to items the base model gets right), (b) plain general-domain sentences
  for an NLL probe and for the KL-drift measurement.

No secrets, no network, no model weights needed.
"""
import random
import re

BOXED_RE = re.compile(r"\\boxed\{\s*(-?\d+)\s*\}")


def extract_boxed(text):
    m = BOXED_RE.findall(text)
    return int(m[-1]) if m else None


def gen_traj_problems(n, seed):
    """Two-step arithmetic problems. Returns list of dicts with the full
    turn list; each turn: {"role": "user"|"assistant"|"tool"|"observation",
    "text": str}. Only assistant/tool turns are supervised."""
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        pat = rng.choice([0, 1])
        if pat == 0:
            a, b = rng.randint(2, 9), rng.randint(2, 9)
            c = rng.randint(2, 6)
            ans = (a + b) * c
            q = (f"What is ({a} + {b}) times {c}? "
                 f"Put the final answer in \\boxed{{}}.")
            e1, v1 = f"{a} + {b}", a + b
            e2, v2 = f"{a + b} * {c}", ans
        else:
            a, b = rng.randint(2, 9), rng.randint(2, 9)
            c = rng.randint(2, 20)
            ans = a * b - c
            q = (f"What is {a} times {b} minus {c}? "
                 f"Put the final answer in \\boxed{{}}.")
            e1, v1 = f"{a} * {b}", a * b
            e2, v2 = f"{a * b} - {c}", ans
        turns = [
            {"role": "user", "text": q},
            {"role": "assistant",
             "text": f"I'll solve this step by step. First I need {e1}."},
        ]
        # ~half the trajectories get a low-information filler turn
        if rng.random() < 0.5:
            turns.append({"role": "assistant",
                          "text": "Let me double-check my plan before "
                                  "continuing. Yes, the plan looks right."})
        turns += [
            {"role": "tool", "text": f"<calc>{e1}</calc>"},
            {"role": "observation", "text": f"{v1}"},
            {"role": "assistant",
             "text": f"Got {v1}. Now I need {e2}."},
            {"role": "tool", "text": f"<calc>{e2}</calc>"},
            {"role": "observation", "text": f"{v2}"},
            {"role": "assistant",
             "text": f"The final answer is \\boxed{{{v2}}}."},
        ]
        out.append({"question": q, "answer": ans, "turns": turns})
    return out


def render_trajectory(traj):
    """Full training text with role prefixes."""
    prefix = {"user": "USER", "assistant": "ASSISTANT",
              "tool": "TOOL", "observation": "OBSERVATION"}
    return "\n".join(f"{prefix[t['role']]}: {t['text']}"
                     for t in traj["turns"]) + "\n"


def supervised_turns(traj):
    """Indices of turns that receive supervision (assistant + tool)."""
    return [i for i, t in enumerate(traj["turns"])
            if t["role"] in ("assistant", "tool")]


# ---- forgetting probes (general capability) ----
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
    trajs = gen_traj_problems(4, 0)
    for t in trajs:
        print("Q:", t["question"], "->", t["answer"],
              "| supervised turns:", len(supervised_turns(t)))
    print("---- example ----")
    print(render_trajectory(trajs[0]))
    assert extract_boxed(render_trajectory(trajs[0])) == trajs[0]["answer"]
    print("OK: boxed answer recoverable from trajectory")
