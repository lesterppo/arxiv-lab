"""CPU check for the Discernment task suite (arXiv:2611.23716).

Verifies, with no model and no network:
- every generated task has exactly one correct item;
- parse_choice recovers the choice from well-formed and sloppy text;
- reaction clarity labels match the outcome branch (polite give-ups
  only ever follow failures -- the trap);
- the judge prompts render and parse_judge reads YES/NO/UNCLEAR;
- Spearman behaves (perfect +1/-1 on monotone data, None on n<3).

Run: python3 cpu_discern_check.py
"""
import random
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from discern_tasks import (
    JUDGE_DISCERN_PROMPT, JUDGE_SELF_PROMPT, generate_catalog, make_reaction,
    make_task, parse_choice, parse_judge, spearman,
)


def check(cond, msg):
    if not cond:
        raise AssertionError(msg)
    print("  ok: %s" % msg)


def satisfies(item, spec):
    return (item["category"] == spec["category"]
            and item["color"] == spec["color"]
            and item["material"] == spec["material"]
            and item["size"] == spec["size"]
            and item["price"] <= spec["max_price"])


def main():
    print("== task generation ==")
    rng = random.Random(42)
    for i in range(500):
        spec, items, ci = generate_catalog(rng)
        n_ok = sum(1 for it in items if satisfies(it, spec))
        assert n_ok == 1, f"task {i}: {n_ok} items satisfy the spec"
        assert satisfies(items[ci], spec), f"task {i}: correct item fails spec"
    print("  ok: 500 catalogs, exactly one correct item each")
    tasks = [make_task(random.Random(1000 + i), i) for i in range(200)]
    for t in tasks:
        assert satisfies(t["items"][t["correct_index"]], t["spec"])
        assert t["prompt"].startswith("You are a shop assistant.")
        assert "Choice: Item" in t["prompt"]  # few-shot format present
        assert t["prompt"].endswith("Choice:"), "prompt ends on Choice:"
    print("  ok: 200 tasks well-formed, few-shot format present")

    print("== parse_choice ==")
    check(parse_choice("Choice: Item 3\nReason: matches.") == 3, "canonical")
    check(parse_choice("choice: item 5") == 5, "lowercase")
    check(parse_choice("I pick Item 2, it looks right.") == 2, "sloppy")
    check(parse_choice("no choice here at all") is None, "no number -> None")
    check(parse_choice("") is None, "empty -> None")

    print("== reactions ==")
    rng = random.Random(7)
    counts = {}
    for _ in range(2000):
        correct = rng.random() < 0.5
        _, clarity = make_reaction(rng, correct)
        counts[(correct, clarity)] = counts.get((correct, clarity), 0) + 1
        if clarity == "polite_giveup":
            assert not correct, "polite give-up must follow a failure"
        if clarity in ("clear_success", "clear_failure"):
            assert (clarity == "clear_success") == correct, \
                "clear verdict must match the true outcome"
    print("  reaction counts (correct, clarity): %s" % counts)
    check(counts[(True, "clear_success")] > counts[(True, "ambiguous")],
          "successes are mostly clear verdicts")
    check(counts[(False, "polite_giveup")] > 0, "polite give-ups occur")

    print("== judge prompts ==")
    attempt = "Choice: Item 2\nReason: It matches."
    p = JUDGE_SELF_PROMPT.format(attempt=attempt, reaction="Perfect!")
    check("YES" in p and "NO" in p, "self judge prompt renders")
    p2 = JUDGE_DISCERN_PROMPT.format(attempt=attempt,
                                     reaction="Thanks anyway, I'll manage.")
    check("UNCLEAR" in p2, "discern judge prompt renders")
    check(parse_judge("YES") == "YES", "judge YES")
    check(parse_judge("The answer is NO.") == "NO", "judge NO in sentence")
    check(parse_judge("UNCLEAR -- no verdict given.") == "UNCLEAR",
          "judge UNCLEAR")
    check(parse_judge("maybe") is None, "judge no token -> None")

    print("== spearman ==")
    check(abs(spearman([1, 2, 3, 4], [1, 2, 3, 4]) - 1.0) < 1e-9,
          "perfect positive")
    check(abs(spearman([1, 2, 3, 4], [4, 3, 2, 1]) + 1.0) < 1e-9,
          "perfect negative")
    check(spearman([1, 2], [1, 2]) is None, "n<3 -> None")
    print("ALL DISCERN CPU CHECKS PASSED")


if __name__ == "__main__":
    main()
