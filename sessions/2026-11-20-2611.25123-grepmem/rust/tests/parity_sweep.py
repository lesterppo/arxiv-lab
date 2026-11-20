#!/usr/bin/env python3
"""Run the GrepMem Python-vs-Rust parity sweep and report mismatches.

Three case kinds:
- rng:  first n random() values + raw u32 words for a seed (JSON compare)
- gen:  a generated session printed in the session file format
         (byte-for-byte stdout compare — the generators must be
         identical because PyRandom reproduces CPython's random.Random)
- eval: one policy evaluated on a session file (JSON compare: exact
         context lines, answers, counts; floats within 1e-9)

Exits non-zero on any mismatch. Builds the Rust binary first if it is
missing (CARGO_TARGET_DIR keeps target/ out of the session dir).
"""
import json
import os
import subprocess
import sys

RUST_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TARGET_DIR = os.environ.get("CARGO_TARGET_DIR", "/tmp/grepmem-rs-target")
RS_BIN = os.environ.get(
    "GREPMEM_RS_BIN", os.path.join(TARGET_DIR, "release", "grepmem"))
PY = ["python3", os.path.join(RUST_DIR, "tests", "parity_gen.py")]

RNG_CASES = [(0, 20), (1, 20), (42, 20), (20261119, 20)]

GEN_CASES = [
    (0, 500, 20), (1, 500, 20), (7, 200, 20), (42, 1000, 20),
    (123, 2000, 20), (20261119, 500, 20), (5, 30, 5), (9, 1000, 1),
    (3, 100, 10),
]

# (seed, n_lines, n_questions) sessions used for eval cases.
EVAL_SESSIONS = [
    (0, 500, 20), (7, 200, 20), (42, 1000, 20), (5, 30, 5), (9, 1000, 1),
]

# (policy, extra argv) per eval case.
EVAL_POLICIES = [
    ("full", []),
    ("grepmem", ["--max-lines", "40"]),
    ("grepmem", ["--max-lines", "5"]),
    ("grepmem", ["--max-lines", "40", "--entity-only"]),
    ("grepmem", ["--max-lines", "40", "--attr-only"]),
    ("summary", []),
    ("recency", []),
    ("random_mask", []),
]


def ensure_binary():
    if os.path.exists(RS_BIN):
        return
    print("building Rust binary ...")
    env = dict(os.environ)
    env["PATH"] = os.path.expanduser("~/.cargo/bin") + ":" + env["PATH"]
    env["CARGO_TARGET_DIR"] = TARGET_DIR
    subprocess.run(["cargo", "build", "--release", "--offline"],
                   cwd=RUST_DIR, env=env, check=True)


def deep_diff(a, b, path="$"):
    """Return a list of (path, a, b) differences. Numbers compare with
    1e-9 tolerance (int/float interchangeable); bools must match bools;
    strings and lists compare exactly."""
    diffs = []
    if isinstance(a, bool) or isinstance(b, bool):
        if a is not b:
            diffs.append((path, a, b))
    elif isinstance(a, (int, float)) and isinstance(b, (int, float)):
        if abs(a - b) > 1e-9:
            diffs.append((path, a, b))
    elif type(a) is not type(b):
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


def run(cmd):
    return subprocess.run(cmd, capture_output=True, text=True)


def main():
    ensure_binary()
    n_ok = n_fail = 0
    tmp = "/tmp/grepmem-parity"
    os.makedirs(tmp, exist_ok=True)

    for seed, n in RNG_CASES:
        label = "rng seed=%d x%d" % (seed, n)
        py = run(PY + ["rng", str(seed), str(n)])
        rs = run([RS_BIN, "rng", str(seed), str(n)])
        if py.returncode != 0 or rs.returncode != 0:
            print("FAIL %s: py rc=%d rs rc=%d %s" % (
                label, py.returncode, rs.returncode,
                (py.stderr or rs.stderr).strip()[-200:]))
            n_fail += 1
            continue
        diffs = deep_diff(json.loads(py.stdout), json.loads(rs.stdout))
        if diffs:
            print("MISMATCH %s: %s" % (label, diffs[:3]))
            n_fail += 1
        else:
            n_ok += 1

    for seed, n_lines, n_q in GEN_CASES:
        label = "gen seed=%d lines=%d q=%d" % (seed, n_lines, n_q)
        py = run(PY + ["gen", str(seed), str(n_lines), str(n_q)])
        rs = run([RS_BIN, "gen", str(seed), str(n_lines), str(n_q)])
        if py.returncode != 0 or rs.returncode != 0:
            print("FAIL %s: py rc=%d rs rc=%d %s" % (
                label, py.returncode, rs.returncode,
                (py.stderr or rs.stderr).strip()[-200:]))
            n_fail += 1
            continue
        if py.stdout != rs.stdout:
            # Show the first differing line.
            pl, rl = py.stdout.splitlines(), rs.stdout.splitlines()
            first = next((i for i, (x, y) in enumerate(zip(pl, rl)) if x != y),
                         min(len(pl), len(rl)))
            print("MISMATCH %s: first diff at line %d:\n  py: %r\n  rs: %r"
                  % (label, first,
                     pl[first] if first < len(pl) else "<eof>",
                     rl[first] if first < len(rl) else "<eof>"))
            n_fail += 1
        else:
            n_ok += 1

    for seed, n_lines, n_q in EVAL_SESSIONS:
        sess_file = os.path.join(
            tmp, "sess_%d_%d_%d.txt" % (seed, n_lines, n_q))
        gen = run(PY + ["gen", str(seed), str(n_lines), str(n_q)])
        if gen.returncode != 0:
            print("FAIL gen session %s" % sess_file)
            n_fail += 1
            continue
        with open(sess_file, "w") as f:
            f.write(gen.stdout)
        for policy, extra in EVAL_POLICIES:
            label = ("eval seed=%d lines=%d q=%d %s %s"
                     % (seed, n_lines, n_q, policy, " ".join(extra)))
            py = run(PY + ["eval", sess_file, policy] + extra)
            rs = run([RS_BIN, "eval", "--session", sess_file,
                      "--policy", policy] + extra)
            if py.returncode != 0 or rs.returncode != 0:
                print("FAIL %s: py rc=%d rs rc=%d %s" % (
                    label, py.returncode, rs.returncode,
                    (py.stderr or rs.stderr).strip()[-200:]))
                n_fail += 1
                continue
            diffs = deep_diff(json.loads(py.stdout), json.loads(rs.stdout))
            if diffs:
                print("MISMATCH %s (%d diffs):" % (label, len(diffs)))
                for path, x, y in diffs[:5]:
                    print("   %s: python=%r rust=%r" % (path, x, y))
                n_fail += 1
            else:
                n_ok += 1

    total = n_ok + n_fail
    print("parity sweep: %d/%d cases match, %d mismatches"
          % (n_ok, total, n_fail))
    sys.exit(1 if n_fail else 0)


if __name__ == "__main__":
    main()
