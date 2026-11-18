#!/usr/bin/env python3
"""Run the Python-vs-Rust parity sweep and report mismatches.

For each case, runs the Python oracle (tests/parity_gen.py) and the
Rust binary (`anytime-stopping parity ...`), parses both JSON outputs
and compares. Every compared value is an integer or a bool, so the
comparison is exact -- 0 mismatches required. Exits non-zero on any
mismatch.
"""
import json
import os
import subprocess
import sys

RUST_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = ["python3", f"{RUST_DIR}/tests/parity_gen.py"]
RS = [f"{RUST_DIR}/target/release/anytime-stopping", "parity"]

# Calibrated params, in CLI order: budget p_solvable p_fix
# p_change_after_self_pass p_change_after_self_fail p_break
# p_self_false_pass p_self_false_fail.
CAL = ["30", "0.65", "0.03", "0.15", "0.90", "0.08", "0.01", "0.15"]
EDGE_ZERO = ["30", "0.0", "0.03", "0.15", "0.90", "0.08", "0.01", "0.15"]
EDGE_ONE = ["30", "1.0", "1.0", "0.15", "0.90", "0.0", "0.01", "0.15"]
SHORT = ["5", "0.65", "0.03", "0.15", "0.90", "0.08", "0.01", "0.15"]
TINY = ["1", "0.65", "0.03", "0.15", "0.90", "0.08", "0.01", "0.15"]
NOISY = ["30", "0.65", "0.03", "0.45", "0.90", "0.30", "0.05", "0.30"]
BREAKY = ["30", "0.5", "0.1", "0.45", "0.5", "0.9", "0.2", "0.4"]

CASES = [
    ("rng", ["20261118", "20"], "rng seed=20261118 x20 draws"),
    ("rng", ["0", "20"], "rng seed=0 x20 draws"),
    ("rng", ["1", "20"], "rng seed=1 x20 draws"),
    ("rng", ["42", "20"], "rng seed=42 x20 draws"),
    ("rng", ["9223372036854775807", "20"], "rng seed=2^63-1 x20 draws"),
    ("trajectory", ["20261118"] + CAL, "trajectory calibrated seed=20261118"),
    ("trajectory", ["7"] + CAL, "trajectory calibrated seed=7"),
    ("trajectory", ["123"] + CAL, "trajectory calibrated seed=123"),
    ("trajectory", ["0"] + CAL, "trajectory calibrated seed=0"),
    ("trajectory", ["99"] + EDGE_ZERO, "trajectory unsolvable seed=99"),
    ("trajectory", ["99"] + EDGE_ONE, "trajectory always-repair seed=99"),
    ("trajectory", ["5"] + SHORT, "trajectory budget=5 seed=5"),
    ("trajectory", ["5"] + TINY, "trajectory budget=1 seed=5"),
    ("trajectory", ["77"] + NOISY, "trajectory noisy self-tests seed=77"),
    ("trajectory", ["77"] + BREAKY, "trajectory breaky regime seed=77"),
    ("trajectory", ["20261118"] + NOISY, "trajectory noisy seed=20261118"),
    ("experiment", ["20261118", "200"] + CAL,
     "experiment calibrated seed=20261118 x200"),
    ("experiment", ["7", "200"] + CAL, "experiment calibrated seed=7 x200"),
    ("experiment", ["99", "200"] + EDGE_ZERO,
     "experiment unsolvable seed=99 x200"),
    ("experiment", ["99", "200"] + EDGE_ONE,
     "experiment always-repair seed=99 x200"),
    ("experiment", ["5", "200"] + SHORT, "experiment budget=5 x200"),
    ("experiment", ["77", "200"] + NOISY,
     "experiment noisy self-tests seed=77 x200"),
    ("experiment", ["77", "200"] + BREAKY,
     "experiment breaky regime seed=77 x200"),
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
