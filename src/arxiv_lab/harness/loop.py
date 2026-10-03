"""Signature-level loop detection (Mingbird, arXiv:2610.02001).

Failure form addressed: tool demonstrations loop — the model repeats the
same (tool, args) action or a short action cycle, burning the step budget.

The mechanism: fingerprint every action as (tool, canonical-arg-signature);
if the same signature repeats ``repeat_k`` times consecutively, or a short
signature cycle repeats, flag the loop so the harness can halt. Pure stdlib.
"""

import hashlib
import json


class LoopDetector:
    """Detect exact-repeat streaks and short repeated action cycles."""

    def __init__(self, repeat_k: int = 3, max_cycle: int = 4):
        self.repeat_k = repeat_k
        self.max_cycle = max_cycle
        self.history = []

    @staticmethod
    def signature(tool: str, args: dict) -> str:
        """Stable fingerprint of one action."""
        canon = json.dumps(args, sort_keys=True, separators=(",", ":"))
        return hashlib.sha1(f"{tool}|{canon}".encode()).hexdigest()[:12]

    def observe(self, tool: str, args: dict):
        """Record one action. Returns a loop description string, or None."""
        sig = self.signature(tool, args)
        self.history.append(sig)
        h = self.history
        # exact-repeat streak
        if len(h) >= self.repeat_k and len(set(h[-self.repeat_k:])) == 1:
            return f"loop: signature {sig} repeated {self.repeat_k}x"
        # cycle of length 2..max_cycle repeated twice
        for cyc in range(2, self.max_cycle + 1):
            if len(h) >= 2 * cyc and h[-2 * cyc:-cyc] == h[-cyc:]:
                return f"loop: {cyc}-step cycle repeated"
        return None

    def reset(self):
        """Clear the observed history (e.g. between tasks)."""
        self.history = []
