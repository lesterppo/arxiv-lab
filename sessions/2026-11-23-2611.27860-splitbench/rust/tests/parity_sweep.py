#!/usr/bin/env python3
"""Python-vs-Rust parity sweep for the SplitBench Rust port.

Three case kinds, all compared exactly (0 mismatches required):
- values-equal: 5,000 random JSON pairs + edge values (tolerance
  boundary, bool/int confusion, nested structures) via the
  `values-equal-batch` subcommand; the exhaustive small universe
  (all pairs over 19 values) is covered too.
- grade: 2,000 random output/expected pairs (valid + invalid
  payloads: wrong length, non-list) via `grade-batch`; verdict and
  n_correct must match Python exactly.
- classify: 500 random sandbox probe logs via `classify-batch`;
  all four classification fields must match Python exactly.

Also runs the `verify` binary (which replays the Python test's
results.json) and asserts exit code 0.

Exits non-zero on any mismatch. Builds the Rust binary first if it
is missing (CARGO_TARGET_DIR keeps target/ out of the session dir).
"""
import json
import os
import random
import subprocess
import sys

RUST_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SESSION_DIR = os.path.dirname(RUST_DIR)
sys.path.insert(0, os.path.join(SESSION_DIR, "..", "..", "src"))

from arxiv_lab.agent.splitbench import (  # noqa: E402
    classify_attempt, grade_outputs, validate_outputs, values_equal,
)

TARGET_DIR = os.environ.get("CARGO_TARGET_DIR", "/tmp/splitbench-rs-target")
RS_BIN = os.environ.get(
    "SPLITBENCH_RS_BIN", os.path.join(TARGET_DIR, "release", "splitbench"))


def ensure_binary():
    if os.path.exists(RS_BIN):
        return
    print("building Rust binary ...")
    env = dict(os.environ)
    env["PATH"] = os.path.expanduser("~/.cargo/bin") + ":" + env["PATH"]
    env["CARGO_TARGET_DIR"] = TARGET_DIR
    subprocess.run(["cargo", "build", "--release", "--offline"],
                   cwd=RUST_DIR, env=env, check=True)


def run_batch(subcommand, lines):
    proc = subprocess.run(
        [RS_BIN, subcommand], input="\n".join(lines) + "\n",
        capture_output=True, text=True, check=True)
    return proc.stdout.strip().split("\n") if proc.stdout.strip() else []


def random_json(rng, depth=0):
    roll = rng.randrange(7 if depth < 2 else 5)
    if roll == 0:
        return None
    if roll == 1:
        return rng.choice([True, False])
    if roll == 2:
        return rng.randint(-10, 10)
    if roll == 3:
        return round(rng.uniform(-10, 10), 6)
    if roll == 4:
        return rng.choice(["a", "b", "abc", "hello", "x", "y z", "", "0"])
    if roll == 5:
        return [random_json(rng, depth + 1) for _ in range(rng.randrange(4))]
    keys = ["k0", "k1", "k2"]
    rng.shuffle(keys)
    return {k: random_json(rng, depth + 1)
            for k in keys[:rng.randrange(3)]}


def main():
    ensure_binary()
    rng = random.Random(20261123)
    mismatches = 0

    # ---- values-equal ----
    cases = []
    for _ in range(5000):
        cases.append((random_json(rng), random_json(rng)))
    # Edge values: tolerance boundary, bool/int, nested.
    edges = [
        (1.0, 1.0 + 5e-10), (1.0, 1.0 + 2e-9), (True, 1), (False, 0),
        (1, 1.0), ([1, 2.0], [1, 2]), ({"a": 1}, {"a": 1.0}),
        (None, 0), ("0", 0), ([1, [2, {"x": True}]], [1, [2, {"x": True}]]),
        ({"a": [1, 2]}, {"a": [1, 3]}), (0.1 + 0.2, 0.3),
    ]
    cases.extend(edges)
    # Exhaustive small universe.
    universe = [None, False, True, -2, -1, 0, 1, 2, 0.5, "", "a", "b", "0",
                [0], [1], [0, 1], {"a": 1}, {"a": 2}]
    for a in universe:
        for b in universe:
            cases.append((a, b))
    lines = [json.dumps(g) + "\t" + json.dumps(e) for g, e in cases]
    out = run_batch("values-equal-batch", lines)
    assert len(out) == len(cases), f"batch short: {len(out)} vs {len(cases)}"
    for (g, e), line in zip(cases, out):
        want = "true" if values_equal(g, e) else "false"
        if line != want:
            mismatches += 1
            if mismatches <= 5:
                print(f"MISMATCH values-equal {g!r} vs {e!r}: "
                      f"rust={line} python={want}")
    print(f"values-equal: {len(cases)} cases, {mismatches} mismatches")

    # ---- grade ----
    gcases = []
    for _ in range(2000):
        k = rng.randrange(1, 6)
        expected = [random_json(rng) for _ in range(k)]
        if rng.random() < 0.5:
            outputs = list(expected)
        else:
            outputs = [random_json(rng) for _ in range(k)]
        gcases.append((outputs, expected))
    # Invalid payloads.
    gcases.append(({"a": 1}, [1]))          # not a list
    gcases.append(([1, 2], [1, 2, 3]))      # wrong length
    gcases.append((42, [1]))                # not a list
    gcases.append(([], []))                  # empty is valid
    glines = [json.dumps(o) + "\t" + json.dumps(e) for o, e in gcases]
    gout = run_batch("grade-batch", glines)
    assert len(gout) == len(gcases)
    for (o, e), line in zip(gcases, gout):
        if not isinstance(o, list) or len(o) != len(e):
            want = "INVALID"
            if not line.startswith("INVALID"):
                mismatches += 1
                print(f"MISMATCH grade-invalid {o!r}: rust={line}")
            continue
        verdict, n = grade_outputs(o, e)
        want = ("PASS" if verdict == "PASS" else "FAIL") + f" {n}"
        if line != want:
            mismatches += 1
            if mismatches <= 10:
                print(f"MISMATCH grade {o!r} vs {e!r}: rust={line} python={want}")
    print(f"grade: {len(gcases)} cases, total mismatches now {mismatches}")

    # ---- classify ----
    ccases = []
    for _ in range(500):
        n = rng.randrange(0, 6)
        probes = []
        for _ in range(n):
            probes.append({
                "path": rng.choice(["expected.json", "../expected.json",
                                    "test_task.py", "runner.py",
                                    "outputs.json", "<grade_native_verdict>"]),
                "found": rng.random() < 0.3,
                "kind": rng.choice(["answers", "tests", "verdict",
                                    "outputs", "code"]),
            })
        ccases.append(probes)
    clines = []
    for probes in ccases:
        clines.append(";".join(
            f"{p['path']}|{1 if p['found'] else 0}|{p['kind']}"
            for p in probes))
    cout = run_batch("classify-batch", clines)
    assert len(cout) == len(ccases)
    for probes, line in zip(ccases, cout):
        c = classify_attempt(probes)
        want = (f"{1 if c['answers_reached'] else 0} "
                f"{1 if c['verdict_tampered'] else 0} "
                f"{c['n_probes']} {c['n_failed']}")
        if line != want:
            mismatches += 1
            if mismatches <= 5:
                print(f"MISMATCH classify {probes!r}: rust={line} python={want}")
    print(f"classify: {len(ccases)} cases, total mismatches now {mismatches}")

    # ---- verify binary ----
    proc = subprocess.run(
        [RS_BIN, "verify", "--results",
         os.path.join(SESSION_DIR, "results.json")],
        capture_output=True, text=True)
    print(proc.stdout.strip().split("\n")[-1])
    if proc.returncode != 0:
        print(proc.stdout)
        print(proc.stderr)
        mismatches += 1

    print(f"TOTAL cases: {len(cases) + len(gcases) + len(ccases)}, "
          f"mismatches: {mismatches}")
    if mismatches:
        sys.exit(1)
    print("PARITY SWEEP PASSED: 0 mismatches")


if __name__ == "__main__":
    main()
