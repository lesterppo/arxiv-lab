"""RuntimeTickets: source-grounded security obligations for agent skills.

Paper: arXiv 2611.03772 "RuntimeTickets: Source-Grounded Security
Obligations for Agent Skills" (submitted 2 Nov 2026).

One-line claim: a skill's security requirements should be enforced at
runtime, not by reading its description or by granting tool permissions
-- a ticket derived from the task's delivery contract cites the skill
lines each obligation answers, tightens the host's approvals into
fine-grained allowed actions (never loosening them), a runtime gate
checks the recipient, object and amount of every outbound send before
forwarding it, trusted monitors record the effects, and a deterministic
decoder verifies each obligation from the ticket + trace. Neither
enforcement nor decoding involves a language model, so text injected by
a skill can change a proposal but never the rule that decides a send.

This module implements that mechanism literally (stdlib only, no model,
no network). The obligation-proposal step is out of scope by design:
in the paper a language model proposes obligations from the delivery
contract; here obligations are given as data, exactly as a proposed
ticket would present them, and everything downstream of the proposal
-- citation grounding, tightening, gating, monitoring, decoding -- is
what is measured. A send's amount is an integer in minor units
(cents for payments, 0 for pure data sends); a ticket's allowed set
is (recipient, object, amount_cap) triples.
"""

import re

# A scanner-style baseline: the paper's regex description scanner
# rejects 20/45 scenarios at a 7.2% false-positive rate -- it reads
# words, not execution. Ours flags a skill description containing any
# of these words, which is exactly the level of evidence a scanner has.
_SCANNER_RE = re.compile(
    r"\b(exfiltrat\w*|attacker|secret\w*|credential\w*|password\w*"
    r"|steal\w*|leak\w*)",
    re.IGNORECASE)


class Send:
    """One outbound send attempt: (recipient, object, amount)."""

    __slots__ = ("recipient", "object", "amount")

    def __init__(self, recipient, object_, amount):
        self.recipient = recipient
        self.object = object_
        self.amount = int(amount)

    def __repr__(self):
        return "Send(%r, %r, %d)" % (self.recipient, self.object,
                                     self.amount)

    def __eq__(self, other):
        return (isinstance(other, Send)
                and self.recipient == other.recipient
                and self.object == other.object
                and self.amount == other.amount)

    def __hash__(self):
        return hash((self.recipient, self.object, self.amount))


class Obligation:
    """One security obligation: id, cited skill lines, allowed sends.

    `cited_lines` are 0-based indices into the skill's source lines;
    `allowed` is a tuple of (recipient, object, amount_cap) triples
    this obligation authorizes. An obligation is grounded only if
    every cited line exists in the skill source.
    """

    __slots__ = ("oid", "cited_lines", "allowed")

    def __init__(self, oid, cited_lines, allowed):
        self.oid = oid
        self.cited_lines = tuple(int(i) for i in cited_lines)
        self.allowed = tuple(
            (r, o, int(c)) for (r, o, c) in allowed)

    def is_grounded(self, n_skill_lines):
        """True iff every cited line index exists in the skill source."""
        return all(0 <= i < n_skill_lines for i in self.cited_lines)


class Ticket:
    """A ticket: the union of its obligations' allowed sends.

    Built only through `tighten`, which enforces the paper's
    tightening rule against the host's approvals: a ticket may only
    narrow an approval (same recipient/object, cap no higher) and
    may never introduce a new (recipient, object) pair.
    """

    __slots__ = ("obligations", "allowed")

    def __init__(self, obligations, allowed):
        self.obligations = tuple(obligations)
        self.allowed = allowed  # dict (recipient, object) -> cap

    @staticmethod
    def tighten(host_approvals, obligations):
        """Build a ticket from host approvals + proposed obligations.

        `host_approvals`: dict (recipient, object) -> cap.
        Raises ValueError on any loosening: a new pair, or a cap
        above the host's cap for that pair. Obligations whose allowed
        set is empty are dropped (they authorize nothing).
        """
        allowed = {}
        kept = []
        for ob in obligations:
            for (recipient, object_, cap) in ob.allowed:
                if (recipient, object_) not in host_approvals:
                    raise ValueError(
                        "ticket loosens host approvals: new pair %r"
                        % ((recipient, object_),))
                if cap > host_approvals[(recipient, object_)]:
                    raise ValueError(
                        "ticket loosens host approvals: cap %d > host %d"
                        " for %r" % (cap,
                                     host_approvals[(recipient, object_)],
                                     (recipient, object_)))
                key = (recipient, object_)
                if key not in allowed or cap < allowed[key]:
                    allowed[key] = cap
            kept.append(ob)
        return Ticket(kept, allowed)

    def gate(self, send):
        """True iff this send is forwarded: pair allowed, amount in cap.

        This is the runtime gate, and it reads nothing but the ticket:
        skill description text, including injected instructions, cannot
        change its decision.
        """
        cap = self.allowed.get((send.recipient, send.object))
        return cap is not None and send.amount <= cap


def gate_decision(ticket, send):
    """Gate decision as a string, for parity with the Rust port."""
    return "forward" if ticket.gate(send) else "block"


def run_trace(ticket, sends):
    """Apply the gate to each attempted send.

    Returns a list of (send, forwarded) pairs -- the trace a trusted
    monitor would record. Effects are recorded only for forwarded
    sends: a blocked send has no effect, by construction.
    """
    return [(s, ticket.gate(s)) for s in sends]


def decode(ticket, trace, obligations_grounded):
    """Deterministic decoder: verify the ticket + trace, no model.

    Returns (verdict, violations) where verdict is "verified" iff:
      - every obligation is grounded (citations exist), and
      - every forwarded send is within the ticket, and
      - no blocked send has a recorded effect (in this model, effects
        exist only for forwarded sends, so a trace entry marked
        forwarded-but-outside-ticket, or an effect on a blocked send,
        is a violation -- see `decode_tampered` in the test).
    `trace` entries are (send, forwarded, effect_recorded) triples.
    """
    violations = []
    if not obligations_grounded:
        violations.append("ungrounded_obligation")
    for (send, forwarded, effect) in trace:
        if forwarded and not ticket.gate(send):
            violations.append("forwarded_outside_ticket:%r" % (send,))
        if (not forwarded) and effect:
            violations.append("effect_on_blocked_send:%r" % (send,))
    return ("verified" if not violations else "violated", violations)


def decode_gate_trace(ticket, trace, n_skill_lines):
    """Decode a gate-produced trace: (send, forwarded) pairs.

    Effects are recorded exactly for forwarded sends, so the only
    possible violations are ungrounded obligations.
    """
    grounded = all(ob.is_grounded(n_skill_lines)
                   for ob in ticket.obligations)
    full = [(s, f, f) for (s, f) in trace]
    return decode(ticket, full, grounded)


def scanner_flags(description):
    """Baseline: regex description scanner (words, not execution)."""
    return bool(_SCANNER_RE.search(description))


def coarse_allowlist_allows(send, host_approvals):
    """Baseline: AgentBound-style coarse allow-list.

    Allows any send whose action type is "send" (all sends here are)
    to any recipient/object, subject only to a single global cap:
    the largest host cap. It ignores per-pair approvals, so it
    overlooks recipient/object swaps that stay under the global cap.
    """
    if not host_approvals:
        return False
    return send.amount <= max(host_approvals.values())


def oracle_allows(ticket, send):
    """Independent oracle for the gate: pair in ticket, amount in cap."""
    cap = ticket.allowed.get((send.recipient, send.object))
    return cap is not None and send.amount <= cap
