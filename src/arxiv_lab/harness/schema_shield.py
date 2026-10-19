"""Interface-drift repair layer for tool-calling agents (arXiv:2610.23342).

The paper studies *interface drift*: after deployment, tool descriptions,
argument schemas, or the candidate tool list change while tool semantics
stay fixed, breaking function-calling agents. Their method, SchemaShield-Lite,
is a lightweight, training-free repair layer combining:

1. schema validation of each emitted tool call,
2. candidate-set retry for missing tool names,
3. one-shot constrained repair of invalid calls.

This module implements that layer in pure stdlib against a synthetic tool
registry, plus a deterministic *drift simulator* so the core claim — "a
training-free repair layer recovers drift robustness without retraining" —
can be tested on CPU with measured numbers.

Public API
----------
- ``ToolDef``        — tool name, description, parameter schema.
- ``ToolCall``       — an emitted call: (tool_name, arguments).
- ``validate``       — schema-validate a call against the registry.
- ``SchemaShield``   — the repair layer; ``invoke(call)`` returns a
                       ``RepairResult`` describing what happened.
- ``DriftSimulator`` — applies rename / distractor / schema-edit /
                       description-rewrite drift to a registry and replays
                       stale (pre-drift) call sequences through it.

RepairResult fields: ok (bool), call (final ToolCall or None), stage
("pass" | "name_retry" | "arg_repair" | "failed"), notes (list of str).
"""

import difflib
import re


# ── Tool registry types ─────────────────────────────────────────────────────

class ToolDef:
    """A tool's interface: name, description, and parameter schema.

    ``params`` maps arg name -> dict with keys ``type`` ("string", "integer",
    "number", "boolean"), ``required`` (bool), and optional ``default``.
    """

    def __init__(self, name, description, params):
        self.name = name
        self.description = description
        self.params = params

    def __repr__(self):
        return "ToolDef(%r)" % self.name


class ToolCall:
    """An emitted tool call: tool name + argument dict."""

    def __init__(self, tool_name, arguments):
        self.tool_name = tool_name
        self.arguments = dict(arguments)

    def __repr__(self):
        return "ToolCall(%r, %r)" % (self.tool_name, self.arguments)

    def __eq__(self, other):
        return (isinstance(other, ToolCall)
                and self.tool_name == other.tool_name
                and self.arguments == other.arguments)


# ── Validation ──────────────────────────────────────────────────────────────

_TYPE_CHECKS = {
    "string": lambda v: isinstance(v, str),
    "integer": lambda v: isinstance(v, bool) is False and isinstance(v, int),
    "number": lambda v: isinstance(v, bool) is False
    and isinstance(v, (int, float)),
    "boolean": lambda v: isinstance(v, bool),
}


def validate(call, registry):
    """Validate ``call`` against ``registry`` (name -> ToolDef).

    Returns a list of error strings; empty means valid.
    """
    errors = []
    tool = registry.get(call.tool_name)
    if tool is None:
        return ["unknown tool %r" % call.tool_name]
    for pname, spec in tool.params.items():
        if spec.get("required") and pname not in call.arguments:
            errors.append("missing required arg %r" % pname)
    for aname, value in call.arguments.items():
        spec = tool.params.get(aname)
        if spec is None:
            errors.append("unknown arg %r" % aname)
            continue
        check = _TYPE_CHECKS.get(spec.get("type", "string"))
        if check is not None and not check(value):
            errors.append("arg %r: expected %s, got %r"
                          % (aname, spec.get("type"), value))
    return errors


# ── SchemaShield repair layer ───────────────────────────────────────────────

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokens(text):
    return set(_TOKEN_RE.findall(text.lower()))


def _strip_version(name):
    return re.sub(r"[_-]v?\d+(\.\d+)*$", "", name.lower())


def _name_score(stale, candidate):
    """Similarity between a stale tool name and a candidate registry name."""
    s_tok = _tokens(_strip_version(stale))
    c_tok = _tokens(_strip_version(candidate))
    jacc = (len(s_tok & c_tok) / len(s_tok | c_tok)) if (s_tok | c_tok) else 0.0
    seq = difflib.SequenceMatcher(None, stale.lower(), candidate.lower()).ratio()
    return 0.6 * jacc + 0.4 * seq


def _schema_fit(arguments, tool):
    """Fraction of call args that map to known params of ``tool`` (0..1)."""
    if not arguments:
        return 0.5
    known = sum(1 for a in arguments if a in tool.params)
    return known / len(arguments)


def _coerce(value, ptype):
    """Best-effort type coercion; returns (ok, new_value)."""
    if ptype == "integer" and isinstance(value, str) and value.strip().lstrip("+-").isdigit():
        return True, int(value.strip())
    if ptype == "number" and isinstance(value, str):
        try:
            return True, float(value.strip())
        except ValueError:
            return False, value
    if ptype == "boolean" and isinstance(value, str):
        low = value.strip().lower()
        if low in ("true", "yes", "1"):
            return True, True
        if low in ("false", "no", "0"):
            return True, False
        return False, value
    if ptype == "string" and isinstance(value, (int, float)):
        return True, str(value)
    if ptype == "string" and isinstance(value, bool):
        return True, str(value)
    return False, value


def _map_arg_name(aname, tool):
    """Map a stale arg name onto a param of ``tool``; None if unmappable."""
    if aname in tool.params:
        return aname
    low = aname.lower()
    for pname in tool.params:
        if pname.lower() == low:
            return pname
    # prefix match either direction, unique: stale abbreviation -> full new
    # name (q -> query) or full stale name -> abbreviated new name (query -> q)
    cands = [p for p in tool.params
             if p.lower().startswith(low) or low.startswith(p.lower())]
    if len(cands) == 1:
        return cands[0]
    # token overlap fallback
    a_tok = _tokens(aname)
    best, best_s = None, 0.0
    for pname in tool.params:
        p_tok = _tokens(pname)
        s = len(a_tok & p_tok) / len(a_tok | p_tok) if (a_tok | p_tok) else 0.0
        if s > best_s:
            best, best_s = pname, s
    return best if best_s >= 0.5 else None


class RepairResult:
    def __init__(self, ok, call, stage, notes):
        self.ok = ok
        self.call = call
        self.stage = stage
        self.notes = notes

    def __repr__(self):
        return "RepairResult(ok=%r, stage=%r, call=%r)" % (
            self.ok, self.stage, self.call)


class SchemaShield:
    """Training-free interface-drift repair layer.

    ``invoke`` pipeline:
      1. validate -> if valid, pass through unchanged;
      2. unknown tool name -> candidate-set retry (name match, tie-broken by
         schema fit against the call's arguments), then re-validate;
      3. invalid args -> exactly ONE constrained repair attempt (arg-name
         mapping, type coercion, dropping unknown args, filling defaults),
         then re-validate. Anything still invalid fails.
    """

    NAME_THRESHOLD = 0.45

    def __init__(self, registry):
        self.registry = dict(registry)

    # -- stage 1 ------------------------------------------------------
    def _retry_name(self, stale_name, arguments):
        scored = []
        for cname, tool in self.registry.items():
            s = _name_score(stale_name, cname)
            if s >= self.NAME_THRESHOLD:
                scored.append((0.7 * s + 0.3 * _schema_fit(arguments, tool),
                               cname))
        if not scored:
            return None
        scored.sort(reverse=True)
        return scored[0][1]

    # -- stage 2 ------------------------------------------------------
    def _repair_args(self, arguments, tool):
        fixed = {}
        notes = []
        for aname, value in arguments.items():
            pname = _map_arg_name(aname, tool)
            if pname is None:
                notes.append("dropped unmappable arg %r" % aname)
                continue
            spec = tool.params[pname]
            check = _TYPE_CHECKS.get(spec.get("type", "string"))
            if check is not None and not check(value):
                okc, new_v = _coerce(value, spec.get("type"))
                if okc:
                    notes.append("coerced arg %r %r -> %r" % (pname, value, new_v))
                    value = new_v
                else:
                    return None, ["uncoercible arg %r" % pname]
            if pname != aname:
                notes.append("renamed arg %r -> %r" % (aname, pname))
            fixed[pname] = value
        for pname, spec in tool.params.items():
            if spec.get("required") and pname not in fixed:
                if "default" in spec:
                    fixed[pname] = spec["default"]
                    notes.append("filled default for %r" % pname)
                else:
                    return None, ["missing required arg %r" % pname]
        return fixed, notes

    # -- pipeline -----------------------------------------------------
    def invoke(self, call):
        notes = []
        if not validate(call, self.registry):
            return RepairResult(True, call, "pass", ["valid as emitted"])

        # Stage 1: candidate-set retry for missing tool names.
        if call.tool_name not in self.registry:
            guess = self._retry_name(call.tool_name, call.arguments)
            if guess is None:
                return RepairResult(False, None, "failed",
                                    ["unknown tool %r; no candidate match"
                                     % call.tool_name])
            notes.append("name retry: %r -> %r" % (call.tool_name, guess))
            call = ToolCall(guess, call.arguments)

        tool = self.registry.get(call.tool_name)
        if tool is None:  # pragma: no cover - guarded above
            return RepairResult(False, None, "failed", notes)

        if not validate(call, self.registry):
            return RepairResult(True, call, "name_retry", notes)

        # Stage 2: one-shot constrained repair of arguments.
        fixed, rnotes = self._repair_args(call.arguments, tool)
        notes.extend(rnotes)
        if fixed is None:
            return RepairResult(False, None, "failed", notes)
        repaired = ToolCall(call.tool_name, fixed)
        if validate(repaired, self.registry):
            return RepairResult(False, None, "failed",
                                notes + ["repair still invalid"])
        stage = "name_retry+arg_repair" if any(
            n.startswith("name retry") for n in notes) else "arg_repair"
        return RepairResult(True, repaired, stage, notes)


# ── Drift simulator ─────────────────────────────────────────────────────────

class DriftSimulator:
    """Applies interface drift to a registry and replays stale call streams.

    ``drift`` is a dict with optional keys:
      renames:      {old_name: new_name}
      distractors:  [ToolDef, ...] appended to the candidate list
      schema_edits: {tool_name: {old_arg: new_arg, ...}} (arg renames)
      type_edits:   {tool_name: {arg: new_type}}
      desc_rewrite: {tool_name: new_description}  (kept for record only)
    """

    def __init__(self, registry, drift):
        self.old_registry = dict(registry)
        self.drift = drift
        self.new_registry = self._apply()

    def _apply(self):
        reg = {}
        renames = self.drift.get("renames", {})
        schema_edits = self.drift.get("schema_edits", {})
        type_edits = self.drift.get("type_edits", {})
        desc_rewrite = self.drift.get("desc_rewrite", {})
        for name, tool in self.old_registry.items():
            new_name = renames.get(name, name)
            params = {}
            for pname, spec in tool.params.items():
                np = dict(spec)
                if name in type_edits and pname in type_edits[name]:
                    np["type"] = type_edits[name][pname]
                params[schema_edits.get(name, {}).get(pname, pname)] = np
            reg[new_name] = ToolDef(
                new_name, desc_rewrite.get(name, tool.description), params)
        for d in self.drift.get("distractors", []):
            reg[d.name] = d
        return reg

    def replay_stale(self, stale_calls, shield=None):
        """Replay pre-drift calls against the drifted registry.

        Returns (baseline_ok_count, shield_ok_count, details) where each
        detail is (call, baseline_ok, result).
        """
        details = []
        b_ok = s_ok = 0
        for call in stale_calls:
            baseline_ok = not validate(call, self.new_registry)
            b_ok += baseline_ok
            if shield is None:
                details.append((call, baseline_ok, None))
                continue
            res = shield.invoke(call)
            exec_ok = res.ok and not validate(res.call, self.new_registry)
            s_ok += exec_ok
            details.append((call, baseline_ok, res))
        return b_ok, s_ok, details
