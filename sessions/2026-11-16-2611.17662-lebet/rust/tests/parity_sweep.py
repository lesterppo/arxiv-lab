#!/usr/bin/env python3
"""Run the Python-vs-Rust parity sweep and report mismatches.

For each case, runs the Python oracle (tests/parity_gen.py) and the
Rust binary (`lebet parity ...`), parses both JSON outputs and compares
with == (floats compare exactly after repr round-trip, so any numeric
difference is a real divergence). Exits non-zero on any mismatch.
"""
import json
import subprocess
import sys

RUST_DIR = "/home/hatch/workspace/arxiv-lab-build/sessions/2026-11-16-2611.17662-lebet/rust"
PY = ["python3", f"{RUST_DIR}/tests/parity_gen.py"]
RS = [f"{RUST_DIR}/target/release/lebet", "parity"]

CASES = [
    # (kind, args, count-label)
    ("rng", ["20261116", "20"], "rng seed=20261116 x20 draws"),
    ("rng", ["0", "20"], "rng seed=0 x20 draws"),
    ("rng", ["1", "20"], "rng seed=1 x20 draws"),
    ("rng", ["42", "20"], "rng seed=42 x20 draws"),
    ("rng", ["9223372036854775807", "20"], "rng seed=2^63-1 x20 draws"),
    ("trajs", ["20261116", "50", "50"], "trajs seed=20261116 50+50"),
    ("trajs", ["7", "30", "30"], "trajs seed=7 30+30"),
    ("trajs", ["12345", "30", "30"], "trajs seed=12345 30+30"),
    ("mats", ["20261116", "50", "50"], "mats seed=20261116 50+50"),
    ("mats", ["7", "30", "30"], "mats seed=7 30+30"),
    ("mutate", ["20261116", "1000"], "mutate seed=20261116 x1000"),
    ("mutate", ["99", "1000"], "mutate seed=99 x1000"),
    ("mutate", ["5", "1000"], "mutate seed=5 x1000"),
    ("wealth", ["20261116", "50", "50"], "wealth seed=20261116 50+50"),
    ("wealth", ["7", "30", "30"], "wealth seed=7 30+30"),
    ("experiment", ["20261116"], "experiment seed=20261116"),
    ("experiment", ["42"], "experiment seed=42"),
]


def deep_diff(a, b, path="$"):
    """Return a list of (path, a, b) for the first few differences."""
    diffs = []
    if type(a) is not type(b):
        # int vs float: JSON ints load as int, floats as float; a Rust
        # integral float (e.g. 1.0) loads as float while Python may emit
        # an int for counts -- compare numerically then.
        if isinstance(a, (int, float)) and isinstance(b, (int, float)) and a == b:
            return diffs
        diffs.append((path, a, b))
        return diffs
    if isinstance(a, dict):
        if set(a) != set(b):
            diffs.append((path + " keys", sorted(a), sorted(b)))
        for k in a:
            if k in b:
                diffs.extend(deep_diff(a[k], b[k], f"{path}.{k}"))
    elif isinstance(a, list):
        if len(a) != len(b):
            diffs.append((path + " len", len(a), len(b)))
        for i, (x, y) in enumerate(zip(a, b)):
            diffs.extend(deep_diff(x, y, f"{path}[{i}]"))
            if len(diffs) >= 5:
                return diffs
    else:
        if a != b:
            diffs.append((path, a, b))
    return diffs


def main():
    mismatches = 0
    for kind, args, label in CASES:
        py = subprocess.run(PY + [kind] + args, capture_output=True, text=True)
        rs = subprocess.run(RS + [kind] + args, capture_output=True, text=True)
        if py.returncode != 0:
            print(f"FAIL {label}: python error: {py.stderr[:300]}")
            mismatches += 1
            continue
        if rs.returncode != 0:
            print(f"FAIL {label}: rust error: {rs.stderr[:300]}")
            mismatches += 1
            continue
        a = json.loads(py.stdout)
        b = json.loads(rs.stdout)
        diffs = deep_diff(a, b)
        if diffs:
            mismatches += 1
            print(f"MISMATCH {label}: {len(diffs)} diff(s) (first 5):")
            for path, x, y in diffs[:5]:
                print(f"    {path}: python={x!r} rust={y!r}")
        else:
            print(f"OK {label}")
    print(f"\nparity sweep: {len(CASES)} cases, {mismatches} mismatched")
    sys.exit(1 if mismatches else 0)


if __name__ == "__main__":
    main()
