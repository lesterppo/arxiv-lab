"""Python-vs-Rust parity sweep for RECEIPT (arXiv 2611.01930).

Generates random (guarded, mode, policy) cases, feeds them to the Rust
`receipt parity` binary, and compares each output to Python's
simulate_outcome. 0 mismatches required for a parity VERIFIED claim.
"""
import random
import subprocess
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
from arxiv_lab.agent.receipt import simulate_outcome  # noqa: E402

BIN = os.path.join(os.path.dirname(__file__), "rust", "target", "debug", "receipt")
MODES = ["closed", "open", "receipt"]


def fmt(outcome):
    status, done, timed_out, false_success = outcome
    return "%s,%d,%d,%d" % (-1 if status is None else status,
                            int(done), int(timed_out), int(false_success))


def main():
    rng = random.Random(20261103)
    cases = []
    # Exhaustive cells first, then random cases.
    for guarded in (0, 1):
        for mode in MODES:
            for policy in (0, 1):
                cases.append((guarded, mode, policy))
    for _ in range(200):
        cases.append((rng.randint(0, 1), rng.choice(MODES), rng.randint(0, 1)))
    inp = "\n".join("%d,%s,%d" % c for c in cases) + "\n"
    proc = subprocess.run([BIN, "parity"], input=inp, capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    lines = proc.stdout.strip().split("\n")
    assert len(lines) == len(cases), (len(lines), len(cases))
    mismatches = 0
    for case, line in zip(cases, lines):
        guarded, mode, policy = case
        expected = fmt(simulate_outcome(bool(guarded), mode, bool(policy)))
        if line.strip() != expected:
            mismatches += 1
            print("MISMATCH %r: python=%s rust=%s" % (case, expected, line))
    print("parity sweep: %d cases, %d mismatches" % (len(cases), mismatches))
    assert mismatches == 0
    print("PARITY VERIFIED")


if __name__ == "__main__":
    main()
