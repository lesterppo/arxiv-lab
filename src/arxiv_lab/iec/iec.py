"""Intent-execution correspondence (IEC) across harness hops.

Paper: arXiv 2610.04375 "Do Tool Calls Execute as Intended? Measuring and
Repairing Intent-Execution Correspondence in LLM Agents".

One-line claim: a tool call emitted by an LLM traverses several hops
(quoting layers, JSON re-serialization, proxies, truncation) and any hop can
silently change what actually executes; observing what each hop received
*without executing* names the first mutating hop, and delivering the call in
a hop-unalterable armored form (or refusing it) stops mutated calls from
executing.

Pure stdlib. Deterministic given an explicit RNG seed.
"""

import base64
import hashlib
import json
import random

__all__ = [
    "Call",
    "ToolContract",
    "ContractViolation",
    "Hop",
    "IdentityHop",
    "JsonReserializeHop",
    "ShellQuoteHop",
    "UrlDecodeHop",
    "TruncateHop",
    "EnvExpandHop",
    "HopReceipt",
    "observe",
    "first_mutating_hop",
    "armor",
    "deliver",
    "Refusal",
    "ExecutedCall",
    "NaiveTrajectoryJudge",
]


class ContractViolation(Exception):
    """A received text does not parse under the tool contract."""


class Call:
    """A tool call as the LLM emitted it: tool name + structured args."""

    def __init__(self, tool, args):
        self.tool = tool
        self.args = dict(args)

    def canonical(self):
        """Canonical wire form: sorted-keys compact JSON of the full call."""
        return json.dumps(
            {"tool": self.tool, "args": self.args},
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        )

    def __eq__(self, other):
        return (
            isinstance(other, Call)
            and self.tool == other.tool
            and self.args == other.args
        )

    def __repr__(self):
        return f"Call(tool={self.tool!r}, args={self.args!r})"


class ToolContract:
    """The tool's contract: names the tool and validates/parses received text.

    The receiver's own parser decides whether a hop changed the call: two
    wire forms denote the same action iff they parse to equal (tool, args).
    """

    def __init__(self, tool, required_args):
        self.tool = tool
        self.required_args = tuple(required_args)

    def parse(self, text):
        """Parse received text with the receiver's own parser."""
        try:
            obj = json.loads(text)
        except (ValueError, TypeError) as e:
            raise ContractViolation(f"not JSON: {e}")
        if not isinstance(obj, dict):
            raise ContractViolation("top level is not an object")
        if obj.get("tool") != self.tool:
            raise ContractViolation(f"tool mismatch: {obj.get('tool')!r}")
        args = obj.get("args")
        if not isinstance(args, dict):
            raise ContractViolation("args is not an object")
        for key in self.required_args:
            if key not in args:
                raise ContractViolation(f"missing arg: {key}")
        return Call(self.tool, args)


# ---------------------------------------------------------------------------
# Hops: pipeline stages a call traverses between the LLM and the tool.
# ---------------------------------------------------------------------------

class Hop:
    """One pipeline stage. `receive` transforms the wire text (faithfully or
    not). Hops never execute anything."""

    def __init__(self, name):
        self.name = name

    def receive(self, text):
        raise NotImplementedError

    def __repr__(self):
        return f"{type(self).__name__}({self.name!r})"


class IdentityHop(Hop):
    """Faithful hop: delivers the wire text unchanged."""

    def receive(self, text):
        return text


class JsonReserializeHop(Hop):
    """Cosmetic hop: parses and re-dumps JSON with different formatting.

    Changes the raw text but not what the contract parser sees, so it is
    never named as a mutating hop (models e.g. a logging layer that
    pretty-prints the payload).
    """

    def receive(self, text):
        try:
            obj = json.loads(text)
        except ValueError:
            return text  # non-JSON payload passes through untouched
        return json.dumps(obj, indent=2, sort_keys=True)


class ShellQuoteHop(Hop):
    """A shell-quoting layer that unescapes backslashes one time too many.

    Models the paper's observed class: Claude Code's Bash tool changed 12%
    of calls carrying code/escapes/long text; backslash changes are the
    dominant mutation.
    """

    def receive(self, text):
        try:
            obj = json.loads(text)
        except ValueError:
            return text

        def mangle(value):
            if isinstance(value, str):
                # one spurious unescape pass over string args
                return value.replace("\\\\", "\\")
            if isinstance(value, dict):
                return {k: mangle(v) for k, v in value.items()}
            if isinstance(value, list):
                return [mangle(v) for v in value]
            return value

        return json.dumps(mangle(obj), separators=(",", ":"))


class UrlDecodeHop(Hop):
    """A proxy hop that percent-decodes the raw wire text.

    Models middleboxes that "helpfully" normalize the payload (the same
    family as a tunnel swallowing POST bodies): literal %XX sequences in
    code or text args get rewritten before the tool ever sees them.
    """

    def receive(self, text):
        out = []
        i = 0
        hexdigits = set("0123456789abcdefABCDEF")
        while i < len(text):
            ch = text[i]
            if (
                ch == "%"
                and i + 2 < len(text)
                and text[i + 1] in hexdigits
                and text[i + 2] in hexdigits
            ):
                out.append(chr(int(text[i + 1 : i + 3], 16)))
                i += 3
            else:
                out.append(ch)
                i += 1
        return "".join(out)


class TruncateHop(Hop):
    """An output/payload cap that silently cuts long wire texts."""

    def __init__(self, name, max_len):
        super().__init__(name)
        self.max_len = max_len

    def receive(self, text):
        return text[: self.max_len]


class EnvExpandHop(Hop):
    """A hop that expands `$NAME`-style sequences in string args."""

    def __init__(self, name, env):
        super().__init__(name)
        self.env = dict(env)

    def receive(self, text):
        try:
            obj = json.loads(text)
        except ValueError:
            return text

        def expand(value):
            if isinstance(value, str):
                for key, val in self.env.items():
                    value = value.replace("$" + key, val)
                return value
            if isinstance(value, dict):
                return {k: expand(v) for k, v in value.items()}
            if isinstance(value, list):
                return [expand(v) for v in value]
            return value

        return json.dumps(expand(obj), separators=(",", ":"))


# ---------------------------------------------------------------------------
# Observation: what did each hop receive, without executing anything.
# ---------------------------------------------------------------------------

class HopReceipt:
    def __init__(self, hop_name, received_text, parsed, changed):
        self.hop_name = hop_name
        self.received_text = received_text
        self.parsed = parsed  # Call or None if unparsable
        self.changed = changed  # parsed form differs from the emitted call

    def __repr__(self):
        return (
            f"HopReceipt(hop={self.hop_name!r}, changed={self.changed}, "
            f"parsable={self.parsed is not None})"
        )


def observe(call, contract, hops, tool_name=None):
    """Feed `call` through `hops` without executing; record per-hop receipts.

    Each receipt parses what the stage *received* with the receiver's own
    parser and compares it to the emitted call under the contract. A final
    terminal receipt records what the tool itself received (so damage done
    by the last hop is visible too).
    """
    receipts = []
    text = call.canonical()
    for hop in hops:
        received = text
        try:
            parsed = contract.parse(received)
            changed = parsed != call
        except ContractViolation:
            parsed = None
            changed = True
        receipts.append(HopReceipt(hop.name, received, parsed, changed))
        text = hop.receive(received)
    terminal = tool_name or contract.tool
    try:
        parsed = contract.parse(text)
        changed = parsed != call
    except ContractViolation:
        parsed = None
        changed = True
    receipts.append(HopReceipt(terminal, text, parsed, changed))
    return receipts


def first_mutating_hop(receipts):
    """Name the first hop that changed the call (the paper's protocol).

    Receipt ``i`` shows what stage ``i`` received; the mutation was done by
    the *previous* stage (the emitter if ``i == 0``). Returns
    ``(index, name)`` of the mutating stage, or None if nothing changed.
    """
    for i, receipt in enumerate(receipts):
        if receipt.changed:
            if i == 0:
                return 0, "emitter"
            return i - 1, receipts[i - 1].hop_name
    return None


# ---------------------------------------------------------------------------
# IntAct: armored delivery — hop-unalterable form, or refusal.
# ---------------------------------------------------------------------------

ARMOR_PREFIX = "INTACT1:"


def armor(call):
    """Encode the call in a form text-mangling hops cannot silently alter.

    base64's alphabet carries no backslashes, quotes, or %-sequences, so the
    ShellQuote/UrlDecode hop classes cannot change its meaning; a checksum
    turns truncation into a detectable refusal instead of a silent mutation.
    """
    body = base64.b64encode(call.canonical().encode("utf-8")).decode("ascii")
    digest = hashlib.sha256(call.canonical().encode("utf-8")).hexdigest()[:16]
    return f"{ARMOR_PREFIX}{digest}:{body}"


class Refusal:
    """The armored call could not be delivered intact: refuse, never execute."""

    def __init__(self, reason, hop_name=None):
        self.reason = reason
        self.hop_name = hop_name

    def __repr__(self):
        return f"Refusal(reason={self.reason!r}, hop={self.hop_name!r})"


class ExecutedCall:
    """A call that passed armor verification and was handed to the tool."""

    def __init__(self, call):
        self.call = call


def deliver(armored, contract, hops):
    """Push an armored call through `hops`; return ExecutedCall or Refusal.

    Verification happens at the tool side, after the last hop: format check,
    checksum check, then contract parse. Any failure refuses the call — a
    mutated call is never executed.
    """
    text = armored
    for hop in hops:
        text = hop.receive(text)
    if not text.startswith(ARMOR_PREFIX):
        return Refusal("armor prefix missing", None)
    try:
        _prefix, digest, body = text.split(":", 2)
    except ValueError:
        return Refusal("armor structure broken", None)
    try:
        raw = base64.b64decode(body.encode("ascii")).decode("utf-8")
    except Exception:
        return Refusal("base64 undecodable", None)
    if hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16] != digest:
        return Refusal("checksum mismatch", None)
    try:
        call = contract.parse(raw)
    except ContractViolation as e:
        return Refusal(f"contract violation after de-armor: {e}", None)
    return ExecutedCall(call)


# ---------------------------------------------------------------------------
# Naive trajectory judge: reads the emitted call + the result, not the hops.
# ---------------------------------------------------------------------------

class NaiveTrajectoryJudge:
    """Attributes every failed call to the LLM (the paper's finding: 95.1% of
    production failures were attributed to the LLM although the path caused
    more than half of them)."""

    def attribute(self, emitted_call, outcome):
        # outcome: "ok" | "wrong-action" | "error"
        if outcome == "ok":
            return "none"
        return "llm"  # never "path": the judge cannot see the hops


# ---------------------------------------------------------------------------
# Synthetic workload: chains of dependent tool calls with trap arg classes.
# ---------------------------------------------------------------------------

ARG_CLASSES = ("plain", "code", "longtext", "envlike", "pct")


def _arg_for(rng, cls, i):
    if cls == "plain":
        return {"cmd": f"run task-{i}", "note": "simple command"}
    if cls == "code":
        # code-carrying: backslashes, quotes, escapes
        return {
            "cmd": "python3 -c \"import re; print(re.sub('\\\\\\\\d+', '#', 'abc123'))\"",
            "path": f"C:\\\\Users\\\\agent\\\\task-{i}.py",
        }
    if cls == "longtext":
        return {"cmd": "summarize", "text": ("lorem ipsum dolor sit amet " * 120) + f"#{i}"}
    if cls == "envlike":
        return {"cmd": f"deploy $APP_NAME --env $ENV_{i} --flag", "note": "uses $VARS"}
    if cls == "pct":
        return {"cmd": f"curl https://example.com/a%2Fb?q=100%25&n={i}", "note": "100% sure"}
    raise AssertionError(cls)


def make_workload(n, seed=0):
    """n tool calls cycling through the trap arg classes (deterministic)."""
    rng = random.Random(seed)
    calls = []
    for i in range(n):
        cls = ARG_CLASSES[i % len(ARG_CLASSES)]
        calls.append((cls, Call("bash", _arg_for(rng, cls, i))))
    return calls


def production_hops(env=None, truncate_at=2000):
    """A 4-hop pipeline modeled on the paper's measured harness paths."""
    return [
        JsonReserializeHop("logger-pretty"),
        ShellQuoteHop("shell-quoting"),
        UrlDecodeHop("proxy-normalize"),
        TruncateHop("payload-cap", truncate_at),
    ]
