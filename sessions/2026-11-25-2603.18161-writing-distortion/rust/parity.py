"""Python-vs-Rust parity sweep for the writing_distortion Rust port.

Generates random cases (fixed seed), computes each metric with the
Python module and with the Rust `parity` bin (one process, JSON
lines over stdin/stdout), and compares within 1e-9 for floats,
exactly for counts/labels. Exits non-zero on any mismatch.

Usage (from rust/):
    cargo build --release
    python3 parity.py [--cases N] [--seed S] [--binary ./target/release/parity]

Writes parity_results.json next to this script.
Stdlib only. No network.
"""

import json
import math
import os
import random
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SESSION = os.path.dirname(HERE)
sys.path.insert(0, SESSION)
import writing_distortion_metrics as m

TOL = 1e-9

FILLER = [
    "car", "cars", "road", "roads", "city", "people", "drive", "driving",
    "traffic", "time", "day", "night", "street", "travel", "future",
    "technology", "system", "systems", "human", "humans", "world",
    "life", "way", "ways", "thing", "things", "place", "work", "home",
    "self", "autonomous", "vehicle", "vehicles", "computer", "sensor",
    "sensors", "data", "speed", "control", "safety2", "x9", "abc123",
]
PRONOUNS = sorted(m.PRONOUNS)
ADJ_SUFFIX = [
    "beautiful", "wonderful", "dangerous", "careful", "helpful",
    "useless", "possible", "capable", "national", "electric",
    "historic", "dramatic", "important", "different", "constant",
    "brilliant", "furious", "famous", "anxious", "curious",
]
ADJ_LEX = ["safe", "good", "bad", "big", "small", "new", "old", "great",
           "fast", "slow", "happy", "sad", "young", "huge", "tiny",
           "cheap", "costly", "local", "global", "daily"]
PRO = m.PRO_WORDS
ANTI = m.ANTI_WORDS
VOCAB = FILLER + PRONOUNS + ADJ_SUFFIX + ADJ_LEX + PRO + ANTI


def rand_text(rng, n_words):
    words = []
    for _ in range(n_words):
        r = rng.random()
        if r < 0.12:
            # Multiword phrase from the stance lists (split into words
            # so the phrase appears verbatim in the text).
            words.extend(rng.choice(PRO + ANTI).split())
        elif r < 0.18:
            words.append(rng.choice(PRO + ANTI))
        else:
            words.append(rng.choice(VOCAB))
    text = " ".join(words)
    # Random casing / punctuation: exercises lowercasing + tokenization.
    if rng.random() < 0.3:
        text = text.title()
    if rng.random() < 0.3:
        text = text.upper()
    if rng.random() < 0.4:
        text = text.replace(" ", rng.choice([". ", ", ", "! ", "; ", "  "]), 3)
    return text


def rand_vec(rng, dim):
    return [round(rng.uniform(-5, 5), 3) for _ in range(dim)]


def gen_cases(rng, n_cases):
    """Yield (op, python_expected, rust_request_dict)."""
    counts = {"jsd": 0, "pronoun": 0, "adjective": 0, "stance": 0,
              "pct": 0, "align": 0}
    # Long-text soak cases are generated separately and counted here.
    plan = [("jsd", 1300), ("pronoun", 850), ("adjective", 850),
            ("stance", 1300), ("pct", 450), ("align", 850)]
    total = sum(n for _, n in plan)
    scale = n_cases / total if n_cases else 1.0
    for op, n in plan:
        for _ in range(max(1, int(n * scale))):
            if op == "jsd":
                a = rand_text(rng, rng.randint(0, 40))
                b = rand_text(rng, rng.randint(0, 40)) if rng.random() < 0.9 else a
                yield op, m.unigram_jsd(a, b), {"op": "jsd", "a": a, "b": b}
            elif op == "pronoun":
                t = rand_text(rng, rng.randint(0, 40))
                r, c, nn = m.pronoun_rate(t)
                yield op, (r, c, nn), {"op": "pronoun", "text": t}
            elif op == "adjective":
                t = rand_text(rng, rng.randint(0, 40))
                r, c, nn = m.adjective_rate(t)
                yield op, (r, c, nn), {"op": "adjective", "text": t}
            elif op == "stance":
                t = rand_text(rng, rng.randint(0, 50))
                s = m.keyword_stance(t)
                yield op, (s["pro"], s["anti"], s["score"], s["label"]), {"op": "stance", "text": t}
            elif op == "pct":
                before = rng.choice([0.0, 0.0, 1.0, 10.0, 100.0, -50.0, 0.5])
                after = rng.choice([0.0, 5.0, -5.0, 50.0, 100.0, 200.0])
                yield op, m.pct_change(before, after), {"op": "pct", "before": before, "after": after}
            elif op == "align":
                dim = rng.randint(1, 6)
                k = rng.randint(1, 6)
                a = [rand_vec(rng, dim) for _ in range(k)]
                b = [[x + rng.uniform(-2, 2) for x in row] for row in a]
                yield op, m.mean_shift_alignment(a, b), {"op": "align", "a": a, "b": b}
            counts[op] += 1
    # Structured alignment edge cases.
    structured = [
        ([[0.0, 0.0], [1.0, 2.0], [3.0, 1.0]], [[1.0, 0.0], [2.0, 2.0], [4.0, 1.0]]),
        ([[0.0, 0.0], [0.0, 0.0]], [[1.0, 0.0], [0.0, 1.0]]),
        ([[0.0, 0.0], [0.0, 0.0]], [[1.0, 0.0], [-1.0, 0.0]]),
        ([[1.0, 1.0]], [[4.0, 5.0]]),
        ([[0.0, 0.0, 0.0]], [[0.0, 3.0, 4.0]]),
        ([[2.0, -1.0, 0.5, 3.0]] * 3, [[3.0, 0.0, 1.5, 4.0]] * 3),
    ]
    for a, b in structured:
        yield "align", m.mean_shift_alignment(a, b), {"op": "align", "a": a, "b": b}
        counts["align"] += 1
    # Long-text soak: 200 texts of 2000+ words each (jsd + stance).
    for _ in range(200):
        a = rand_text(rng, rng.randint(2000, 3000))
        b = rand_text(rng, rng.randint(2000, 3000))
        yield "jsd", m.unigram_jsd(a, b), {"op": "jsd", "a": a, "b": b}
        counts["jsd"] += 1
        yield "stance", (lambda s: (s["pro"], s["anti"], s["score"], s["label"]))(m.keyword_stance(a)), {"op": "stance", "text": a}
        counts["stance"] += 1
    return counts


def close_float(x, y):
    if isinstance(x, float) and math.isinf(x):
        return isinstance(y, float) and math.isinf(y) and (x > 0) == (y > 0)
    if isinstance(y, float) and math.isinf(y):
        return False
    return abs(float(x) - float(y)) <= TOL


def compare(op, expected, got):
    """Return None on match, else a mismatch description string."""
    if "error" in got:
        return "rust error: %s (expected %r)" % (got["error"], expected)
    if op == "jsd":
        if not close_float(expected, got["value"]):
            return "jsd py=%r rust=%r" % (expected, got["value"])
    elif op in ("pronoun", "adjective"):
        r, c, nn = expected
        if not (close_float(r, got["rate"]) and c == got["count"] and nn == got["n"]):
            return "%s py=%r rust=%r" % (op, expected, got)
    elif op == "stance":
        pro, anti, score, label = expected
        if not (pro == got["pro"] and anti == got["anti"]
                and score == got["score"] and label == got["label"]):
            return "stance py=%r rust=%r" % (expected, got)
    elif op == "pct":
        if not close_float(expected, got["value"]):
            return "pct py=%r rust=%r" % (expected, got["value"])
    elif op == "align":
        if not close_float(expected, got["value"]):
            return "align py=%r rust=%r" % (expected, got["value"])
    return None


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", type=int, default=5600)
    ap.add_argument("--seed", type=int, default=20261125)
    ap.add_argument("--binary", default=os.path.join(HERE, "target", "release", "parity"))
    args = ap.parse_args()

    rng = random.Random(args.seed)
    cases = list(gen_cases(rng, args.cases))
    # Error cases: Rust must report an error where Python raises.
    error_cases = [
        ("align", [[1.0, 1.0]], [[1.0, 1.0]]),          # zero shift
        ("align", [[0.0, 0.0]], [[1.0, 0.0], [0.0, 1.0]]),  # len mismatch
        ("align", [[0.0, 0.0]], [[1.0]]),               # dim mismatch
    ]
    err_reqs = []
    for _, a, b in error_cases:
        try:
            m.mean_shift_alignment(a, b)
            err_reqs.append(({"op": "align", "a": a, "b": b}, None))
        except ValueError as e:
            err_reqs.append(({"op": "align", "a": a, "b": b}, str(e)))

    req_lines = "".join(json.dumps(r) + "\n" for _, _, r in cases)
    req_lines += "".join(json.dumps(r) + "\n" for r, _ in err_reqs)
    proc = subprocess.run(
        [args.binary], input=req_lines, capture_output=True, text=True, timeout=300)
    if proc.returncode != 0:
        print("parity binary failed rc=%d stderr=%s" % (proc.returncode, proc.stderr[:2000]))
        return 1
    out_lines = proc.stdout.strip().split("\n") if proc.stdout.strip() else []
    if len(out_lines) != len(cases) + len(err_reqs):
        print("line count mismatch: sent %d got %d" % (len(cases) + len(err_reqs), len(out_lines)))
        return 1

    per_op = {}
    mismatches = []
    max_abs_diff = 0.0
    max_abs_diff_op = None
    for (op, expected, _), line in zip(cases, out_lines):
        got = json.loads(line)
        per_op[op] = per_op.get(op, 0) + 1
        bad = compare(op, expected, got)
        if bad:
            mismatches.append((op, bad))
            if len(mismatches) <= 5:
                print("MISMATCH", op, bad)
        elif op in ("jsd", "pct", "align") and "value" in got:
            d = abs(float(expected) - float(got["value"]))
            if d > max_abs_diff:
                max_abs_diff, max_abs_diff_op = d, op
    err_ok = 0
    for (req, _py_err), line in zip(err_reqs, out_lines[len(cases):]):
        got = json.loads(line)
        if "error" in got:
            err_ok += 1
        else:
            mismatches.append(("align-error", "expected error, got %r" % (got,)))

    summary = {
        "seed": args.seed,
        "total_cases": len(cases),
        "per_op": per_op,
        "error_cases": len(err_reqs),
        "error_cases_matched": err_ok,
        "mismatches": len(mismatches),
        "mismatch_examples": [list(x) for x in mismatches[:5]],
        "max_abs_float_diff": max_abs_diff,
        "max_abs_float_diff_op": max_abs_diff_op,
        "float_tolerance": TOL,
    }
    with open(os.path.join(HERE, "parity_results.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))
    return 0 if not mismatches and err_ok == len(err_reqs) else 1


if __name__ == "__main__":
    sys.exit(main())
