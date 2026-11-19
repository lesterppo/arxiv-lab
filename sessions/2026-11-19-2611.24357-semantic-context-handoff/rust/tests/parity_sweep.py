#!/usr/bin/env python3
"""Run the Python-vs-Rust parity sweep and report mismatches.

For each case, runs the Python oracle (tests/parity_gen.py) and the
Rust binary (`semantic-context-handoff parity ...`), parses both JSON
outputs and compares. Every compared value is an integer or a bool, so
the comparison is exact -- 0 mismatches required. Exits non-zero on any
mismatch.
"""
import json
import os
import subprocess
import sys

RUST_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = ["python3", f"{RUST_DIR}/tests/parity_gen.py"]
RS = [f"{RUST_DIR}/target/release/semantic-context-handoff", "parity"]

# Calibrated params, in CLI order: steps facts_per_step delayed_frac
# max_delay min_fact_tokens max_fact_tokens state_retention compression
# state_overhead recall lookup_overhead fp_per_lookup
# pass_step_fraction.
CAL = ["12", "3", "0.25", "4", "10", "30", "0.85", "0.3", "40",
       "0.9", "5", "0.2", "0.8"]
NO_DELAY = ["12", "3", "0.0", "4", "10", "30", "0.85", "0.3", "40",
            "0.9", "5", "0.2", "0.8"]
FULL_RETENTION = ["12", "3", "0.25", "4", "10", "30", "1.0", "0.3", "40",
                  "0.9", "5", "0.2", "0.8"]
NO_RECALL = ["12", "3", "0.25", "4", "10", "30", "0.85", "0.3", "40",
             "0.0", "5", "0.2", "0.8"]
PERFECT_RECALL = ["12", "3", "0.25", "4", "10", "30", "0.85", "0.3", "40",
                  "1.0", "5", "0.2", "0.8"]
TINY = ["1", "1", "0.25", "4", "10", "30", "0.85", "0.3", "40",
        "0.9", "5", "0.2", "0.8"]
SHORT = ["5", "1", "0.35", "1", "10", "30", "0.55", "0.5", "40",
         "0.7", "5", "0.4", "0.8"]
HEAVY_DELAY = ["12", "3", "0.55", "4", "10", "30", "0.55", "0.3", "40",
               "0.7", "5", "0.2", "0.8"]
BIG = ["20", "5", "0.3", "6", "5", "50", "0.75", "0.25", "25",
       "0.8", "3", "0.3", "0.9"]

CASES = [
    ("rng", ["20261119", "20"], "rng seed=20261119 x20 draws"),
    ("rng", ["0", "20"], "rng seed=0 x20 draws"),
    ("rng", ["1", "20"], "rng seed=1 x20 draws"),
    ("rng", ["42", "20"], "rng seed=42 x20 draws"),
    ("rng", ["9223372036854775807", "20"], "rng seed=2^63-1 x20 draws"),
    ("task", ["20261119"] + CAL, "task calibrated seed=20261119"),
    ("task", ["7"] + CAL, "task calibrated seed=7"),
    ("task", ["123"] + CAL, "task calibrated seed=123"),
    ("task", ["0"] + CAL, "task calibrated seed=0"),
    ("task", ["99"] + NO_DELAY, "task no-delay seed=99"),
    ("task", ["99"] + FULL_RETENTION, "task full-retention seed=99"),
    ("task", ["99"] + NO_RECALL, "task recall=0 seed=99"),
    ("task", ["99"] + PERFECT_RECALL, "task recall=1 seed=99"),
    ("task", ["5"] + TINY, "task steps=1 facts=1 seed=5"),
    ("task", ["5"] + SHORT, "task short/max_delay=1 seed=5"),
    ("task", ["77"] + HEAVY_DELAY, "task heavy-delay seed=77"),
    ("task", ["77"] + BIG, "task big steps=20 facts=5 seed=77"),
    ("experiment", ["20261119", "200"] + CAL,
     "experiment calibrated seed=20261119 x200"),
    ("experiment", ["7", "200"] + CAL, "experiment calibrated seed=7 x200"),
    ("experiment", ["99", "200"] + NO_DELAY,
     "experiment no-delay seed=99 x200"),
    ("experiment", ["99", "200"] + NO_RECALL,
     "experiment recall=0 seed=99 x200"),
    ("experiment", ["5", "200"] + TINY, "experiment steps=1 x200"),
    ("experiment", ["77", "200"] + HEAVY_DELAY,
     "experiment heavy-delay seed=77 x200"),
    ("experiment", ["77", "100"] + BIG, "experiment big seed=77 x100"),
]


def deep_diff(a, b, path="$"):
    """Return a list of (path, a, b) differences."""
    diffs = []
    if type(a) is not type(b):
        # JSON ints vs bools: bool is a subclass of int in Python, but
        # json.loads keeps them distinct; a bool/int mismatch is real.
        diffs.append((path, a, b))
    elif isinstance(a, dict):
        if set(a) != set(b):
            diffs.append((path + " keys", sorted(a), sorted(b)))
        else:
            for k in a:
                diffs.extend(deep_diff(a[k], b[k], path + "." + str(k)))
    elif isinstance(a, list):
        if len(a) != len(b):
            diffs.append((path + " len", len(a), len(b)))
        else:
            for i, (x, y) in enumerate(zip(a, b)):
                diffs.extend(deep_diff(x, y, "%s[%d]" % (path, i)))
    elif a != b:
        diffs.append((path, a, b))
    return diffs


def main():
    n_fail = 0
    for kind, args, label in CASES:
        py = subprocess.run(PY + [kind] + args, capture_output=True,
                            text=True)
        rs = subprocess.run(RS + [kind] + args, capture_output=True,
                            text=True)
        if py.returncode != 0:
            print("PY-FAIL %s: %s" % (label, py.stderr.strip()[-300:]))
            n_fail += 1
            continue
        if rs.returncode != 0:
            print("RS-FAIL %s: %s" % (label, rs.stderr.strip()[-300:]))
            n_fail += 1
            continue
        try:
            a = json.loads(py.stdout)
            b = json.loads(rs.stdout)
        except json.JSONDecodeError as e:
            print("JSON-FAIL %s: %s" % (label, e))
            n_fail += 1
            continue
        diffs = deep_diff(a, b)
        if diffs:
            n_fail += 1
            print("MISMATCH %s (%d diffs):" % (label, len(diffs)))
            for path, x, y in diffs[:5]:
                print("   %s: python=%r rust=%r" % (path, x, y))
        else:
            print("  ok: %s" % label)
    print("parity sweep: %d/%d cases match, %d mismatches"
          % (len(CASES) - n_fail, len(CASES), n_fail))
    sys.exit(1 if n_fail else 0)


if __name__ == "__main__":
    main()
