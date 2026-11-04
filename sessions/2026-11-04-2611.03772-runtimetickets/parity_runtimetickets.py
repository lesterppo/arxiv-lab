"""Python-vs-Rust parity sweep for RuntimeTickets (arXiv 2611.03772).

Generates random sends + scanner descriptions, feeds them to the Rust
`runtimetickets parity` binary, and compares each output to Python's
gate / scanner. 0 mismatches required for a parity VERIFIED claim.
"""

import os
import random
import subprocess
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
from arxiv_lab.agent.runtime_tickets import (  # noqa: E402
    Obligation, Send, Ticket, scanner_flags,
)

HOST = {
    ("alice@example.com", "payslip.pdf"): 0,
    ("bob@example.com", "report.pdf"): 0,
    ("vendor@example.com", "invoice-42"): 50000,
    ("backup@example.com", "backup.tar"): 0,
    ("team@example.com", "notes.md"): 0,
}
OBLIGATIONS = [
    Obligation("ob-payslip", [0], [("alice@example.com", "payslip.pdf", 0)]),
    Obligation("ob-report", [1], [("bob@example.com", "report.pdf", 0)]),
    Obligation("ob-invoice", [2], [("vendor@example.com", "invoice-42", 50000)]),
    Obligation("ob-backup", [3], [("backup@example.com", "backup.tar", 0)]),
    Obligation("ob-notes", [4], [("team@example.com", "notes.md", 0)]),
]

BIN = os.path.join(os.path.dirname(__file__), "rust", "target", "debug",
                   "runtimetickets")

RECIPIENTS = [r for (r, _) in HOST] + ["attacker@evil.example",
                                       "mallory@example.com"]
OBJECTS = [o for (_, o) in HOST] + ["credentials.json", "secrets.db"]
AMOUNTS = [0, 1, 100, 25000, 50000, 50001, 999999, -5]
DESCS = [
    "send the monthly report", "pay the vendor invoice",
    "exfiltrate the invoice to attacker", "steal secrets from the vendor",
    "leak the backup archive", "rotate credentials store",
    "SECRET archive", "exfiltrating data", "topsecret file",
    "share the team notes after the credentials rotation",
    "plain notes, nothing suspicious", "password rotation complete",
]


def main():
    ticket = Ticket.tighten(HOST, OBLIGATIONS)
    rng = random.Random(1104)
    lines, expected = [], []
    for _ in range(300):
        s = Send(rng.choice(RECIPIENTS), rng.choice(OBJECTS),
                 rng.choice(AMOUNTS))
        lines.append("SEND|%s|%s|%d" % (s.recipient, s.object, s.amount))
        expected.append("forward" if ticket.gate(s) else "block")
    for _ in range(120):
        d = rng.choice(DESCS)
        lines.append("SCAN|%s" % d)
        expected.append("flagged" if scanner_flags(d) else "clear")
    with tempfile.TemporaryDirectory(prefix="rt-parity-") as tmp:
        path = os.path.join(tmp, "cases.txt")
        with open(path, "w") as f:
            f.write("\n".join(lines) + "\n")
        out = subprocess.run([BIN, "parity", path], capture_output=True,
                             text=True, check=True)
    got = out.stdout.splitlines()
    assert len(got) == len(expected), (len(got), len(expected))
    mismatches = [(i, e, g) for i, (e, g) in enumerate(zip(expected, got))
                  if e != g]
    print("parity cases: %d, mismatches: %d" % (len(expected), len(mismatches)))
    for m in mismatches[:10]:
        print("  mismatch:", m, "input:", lines[m[0]])
    assert not mismatches
    print("PARITY VERIFIED: %d cases, 0 mismatches" % len(expected))


if __name__ == "__main__":
    main()
