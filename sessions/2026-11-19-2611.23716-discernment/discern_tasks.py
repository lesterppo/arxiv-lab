"""Synthetic shop-assistant task suite for the Discernment training track
(arXiv:2611.23716, "Learning through Discernment").

The paper studies a small model with a LoRA adapter, fine-tuned each
night on the attempts a judge accepts, over 127 lives on WebShop and
tau^2-bench. This module is the bounded-scale analogue of WebShop: a
user request with constraints, a small catalog, exactly one correct
item, and a templated user reaction whose clarity varies (clear
verdict / polite give-up / ambiguous) -- the paper's trap is that a
polite give-up reads as success to a naive judge.

Pure Python, no model, no network: importable and testable on CPU.
Deterministic under a seeded random.Random.
"""
import random
import re

COLORS = ["red", "blue", "green", "black", "white", "navy"]
MATERIALS = ["cotton", "wool", "linen", "denim", "silk"]
SIZES = ["S", "M", "L", "XL"]
CATEGORIES = ["t-shirt", "hoodie", "jacket", "sweater", "shirt"]

FEWSHOT = """You are a shop assistant. The user gives a request with constraints.
Pick the ONE catalog item that satisfies ALL constraints and reply in exactly
this format:
Choice: Item <number>
Reason: <one short sentence>

Example 1:
Request: a blue cotton t-shirt, size M, under $30
Catalog:
Item 1: blue cotton t-shirt, size M, $22
Item 2: red cotton t-shirt, size M, $22
Item 3: blue wool t-shirt, size M, $22
Item 4: blue cotton t-shirt, size L, $22
Choice: Item 1
Reason: It matches the color, material, size, and price.

Example 2:
Request: a black wool sweater, size L, under $50
Catalog:
Item 1: black wool sweater, size M, $40
Item 2: black wool sweater, size L, $45
Item 3: black cotton sweater, size L, $45
Item 4: navy wool sweater, size L, $45
Choice: Item 2
Reason: It matches the color, material, size, and price.

Now your turn:
"""

REACTIONS_SUCCESS_CLEAR = [
    "Perfect, that's exactly what I wanted. Thank you!",
    "Yes! That's the right one. Great job.",
    "Exactly right -- color, size, everything. Thanks!",
]
REACTIONS_SUCCESS_AMBIG = [
    "Thanks!",
    "OK, got it.",
    "Thanks for your help.",
]
REACTIONS_FAILURE_CLEAR = [
    "No, that's the wrong size. This doesn't work.",
    "That's not what I asked for -- the color is wrong.",
    "No, this one is over my budget. Wrong pick.",
]
REACTIONS_FAILURE_POLITE = [
    "Thanks anyway, I'll manage with this one.",
    "No worries, thank you for trying!",
    "It's fine, thanks for your help anyway.",
]
REACTIONS_FAILURE_AMBIG = [
    "Hmm, not sure about this one.",
    "Let me think about it.",
]


def generate_catalog(rng, n_items=6):
    """Build a catalog with exactly one item matching all constraints.

    Returns (spec, items, correct_index) where spec is a dict of
    constraints and items is a list of dicts. The correct item
    satisfies every constraint; 1-2 near-misses satisfy all but one;
    the rest are random.
    """
    spec = {
        "category": rng.choice(CATEGORIES),
        "color": rng.choice(COLORS),
        "material": rng.choice(MATERIALS),
        "size": rng.choice(SIZES),
        "max_price": rng.choice([20, 25, 30, 40, 50]),
    }
    correct = {
        "category": spec["category"],
        "color": spec["color"],
        "material": spec["material"],
        "size": spec["size"],
        "price": spec["max_price"] - rng.choice([1, 3, 5, 8]),
    }
    items = [correct]
    # Near misses: violate exactly one constraint each.
    n_miss = rng.choice([1, 2])
    violated = rng.sample(["color", "material", "size", "price"], n_miss)
    for v in violated:
        bad = dict(correct)
        if v == "color":
            bad["color"] = rng.choice([c for c in COLORS if c != spec["color"]])
        elif v == "material":
            bad["material"] = rng.choice(
                [m for m in MATERIALS if m != spec["material"]])
        elif v == "size":
            bad["size"] = rng.choice([s for s in SIZES if s != spec["size"]])
        else:
            bad["price"] = spec["max_price"] + rng.choice([2, 5, 10])
        items.append(bad)
    while len(items) < n_items:
        # Redraw until the filler violates at least one constraint, so
        # the correct item stays unique by construction.
        for _ in range(50):
            cand = {
                "category": rng.choice(CATEGORIES),
                "color": rng.choice(COLORS),
                "material": rng.choice(MATERIALS),
                "size": rng.choice(SIZES),
                "price": rng.choice([12, 18, 22, 27, 33, 45, 55]),
            }
            if (cand["category"] != spec["category"]
                    or cand["color"] != spec["color"]
                    or cand["material"] != spec["material"]
                    or cand["size"] != spec["size"]
                    or cand["price"] > spec["max_price"]):
                break
        items.append(cand)
    rng.shuffle(items)
    correct_index = items.index(correct)
    return spec, items, correct_index


def spec_text(spec):
    return (f"a {spec['color']} {spec['material']} {spec['category']}, "
            f"size {spec['size']}, under ${spec['max_price']}")


def item_text(item):
    return (f"{item['color']} {item['material']} {item['category']}, "
            f"size {item['size']}, ${item['price']}")


def make_task(rng, task_id):
    spec, items, correct_index = generate_catalog(rng)
    lines = [f"Item {i + 1}: {item_text(it)}" for i, it in enumerate(items)]
    prompt = (FEWSHOT + f"Request: {spec_text(spec)}\nCatalog:\n"
              + "\n".join(lines)
              # End on "Choice:" -- a base model left to continue a bare
              # catalog list just writes more catalog items (Item 7, 8, ...).
              # Starting the answer line forces the choice format the
              # few-shot examples demonstrate.
              + "\nChoice:")
    return {
        "task_id": task_id,
        "spec": spec,
        "items": items,
        "correct_index": correct_index,  # 0-based
        "prompt": prompt,
    }


def parse_choice(text):
    """Extract the chosen 1-based item number, or None."""
    if not text:
        return None
    m = re.search(r"[Cc]hoice:\s*Item\s*(\d+)", text)
    if m:
        return int(m.group(1))
    m = re.search(r"[Ii]tem\s*(\d+)", text)
    if m:
        return int(m.group(1))
    m = re.search(r"\b([1-9])\b", text)
    return int(m.group(1)) if m else None


def make_reaction(rng, correct):
    """Templated user reaction. Returns (text, clarity) where clarity is
    one of: clear_success, clear_failure, polite_giveup, ambiguous.
    Only the clear ones carry a verdict the user can be said to have
    given; polite give-ups thank the agent for a FAILURE (the trap).
    """
    if correct:
        if rng.random() < 0.7:
            return rng.choice(REACTIONS_SUCCESS_CLEAR), "clear_success"
        return rng.choice(REACTIONS_SUCCESS_AMBIG), "ambiguous"
    r = rng.random()
    if r < 0.5:
        return rng.choice(REACTIONS_FAILURE_CLEAR), "clear_failure"
    if r < 0.8:
        return rng.choice(REACTIONS_FAILURE_POLITE), "polite_giveup"
    return rng.choice(REACTIONS_FAILURE_AMBIG), "ambiguous"


JUDGE_SELF_PROMPT = """You judge whether a shop assistant's attempt succeeded.
Read the attempt and the user's reaction, then answer with exactly one word:
YES if the attempt succeeded, NO if it failed.

Attempt:
{attempt}

User reaction: {reaction}

Answer:"""

JUDGE_DISCERN_PROMPT = """You judge whether a shop assistant's attempt succeeded.
FIRST ask: did the user give a clear verdict about the outcome?
- If the user only thanked the assistant, said something vague, or gave
  no verdict, answer UNCLEAR.
- If the user clearly said the attempt was right, answer YES.
- If the user clearly said the attempt was wrong, answer NO.
Answer with exactly one word: YES, NO, or UNCLEAR.

Attempt:
{attempt}

User reaction: {reaction}

Answer:"""


def parse_judge(text):
    """First YES/NO/UNCLEAR token in the judge output, or None."""
    if not text:
        return None
    m = re.search(r"\b(UNCLEAR|YES|NO)\b", text.upper())
    return m.group(1) if m else None


def spearman(xs, ys):
    """Spearman rank correlation (no scipy). Returns None if undefined."""
    n = len(xs)
    if n < 3:
        return None

    def ranks(v):
        order = sorted(range(n), key=lambda i: v[i])
        r = [0.0] * n
        i = 0
        while i < n:
            j = i
            while j + 1 < n and v[order[j + 1]] == v[order[i]]:
                j += 1
            avg = (i + j) / 2.0 + 1.0
            for k in range(i, j + 1):
                r[order[k]] = avg
            i = j + 1
        return r

    rx, ry = ranks(xs), ranks(ys)
    mx, my = sum(rx) / n, sum(ry) / n
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = (sum((a - mx) ** 2 for a in rx)
           * sum((b - my) ** 2 for b in ry)) ** 0.5
    return num / den if den else None
