"""Python-vs-Rust parity sweep for signed manifests (arXiv 2611.07483).

Generates explicit cases (requirement-id list, format, threat, victim,
inject flag), feeds them to the Rust `signed_manifest parity` binary,
and compares each output line to Python's run_case_explicit.
0 mismatches required for a parity VERIFIED claim.
"""
import os
import random
import subprocess
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
from arxiv_lab.multiagent.signed_manifest import run_case_explicit  # noqa: E402

BIN = os.path.join(os.path.dirname(__file__), "rust", "target", "debug",
                   "signed_manifest")
IDS = ["env", "email", "spend", "payment", "logging"]
FORMATS = ["free_text", "json_unsigned", "signed_manifest"]
THREATS = ["none", "drop", "alter", "ignore", "mutate", "inject"]


def main():
    rng = random.Random(20261106)
    cases = []
    # Exhaustive: every single requirement x format x threat, inject on/off.
    for rid in IDS:
        for fmt in FORMATS:
            for threat in THREATS:
                for inject in (0, 1):
                    cases.append(([rid], fmt, threat, 0, inject))
    # Random cases: subsets of size 1-5, random format/threat/victim/inject.
    for _ in range(300):
        k = rng.randint(1, 5)
        ids = rng.sample(IDS, k)
        cases.append((ids, rng.choice(FORMATS), rng.choice(THREATS),
                      rng.randint(0, 4), rng.randint(0, 1)))
    inp = "\n".join("%s|%s|%s|%d|%d" % (",".join(c[0]), c[1], c[2], c[3], c[4])
                    for c in cases) + "\n"
    proc = subprocess.run([BIN, "parity"], input=inp, capture_output=True,
                          text=True)
    assert proc.returncode == 0, proc.stderr
    lines = proc.stdout.strip().split("\n")
    assert len(lines) == len(cases), (len(lines), len(cases))
    mismatches = 0
    for case, line in zip(cases, lines):
        ids, fmt, threat, victim, inject = case
        expected = run_case_explicit(ids, fmt, threat, victim, bool(inject))
        if line.strip() != expected:
            mismatches += 1
            print("MISMATCH %r:\n  python=%s\n  rust  =%s"
                  % (case, expected, line))
    print("parity sweep: %d cases, %d mismatches" % (len(cases), mismatches))
    assert mismatches == 0
    print("PARITY VERIFIED")


if __name__ == "__main__":
    main()
