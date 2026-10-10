"""Typed-decision client for Ollama >= 0.35 ``/v1/systemone`` (stdlib only).

Integration module (not an arXiv reproduction): lets local agents use a
Cloudflare Clef / clef-flash decision model as a *decision backend*.
Instead of free-text chat, the model answers typed questions about a state
string and returns calibrated values — no text parsing needed:

- ``noul``  -> probability the statement holds (float in [0, 1])
- ``choice`` -> winning candidate + full probability distribution
- ``score``  -> scalar score (float)

Request shape (verified live 2026-10-05 / 2026-10-07 against clef-flash:latest
on a T4)::

    POST {base_url}/v1/systemone
    {"model": <tag>, "state": "...",
     "questions": {"<name>": {"type": "noul|choice|score",
                             "instructions": "...",
                             "criteria": ...}}}

Quirks encoded here (all verified against the live server):
- ``questions`` must be an *object* (dict), never a list.
- ``noul`` takes NO criteria.
- ``choice`` requires ``criteria`` as an *object* (never a list) with 2-26
  ``candidate -> description`` entries.
- ``score`` requires ``criteria`` as an *array* (never an object): the scale
  anchors, e.g. ``[0, 0.5, 1]`` or
  ``[{"score": 0, "description": "..."}, ...]``. The server returns a
  probability distribution over the array positions plus ``score`` = the
  expected position index; :func:`score_value` maps that back onto numeric
  criteria.
- ``/api/generate`` is NOT meaningful for this decision model — use
  ``decide`` below.

Pure stdlib. No API key needed (talks to a local Ollama server). No network
at import time.
"""

import json
import urllib.error
import urllib.request

#: Supported question types.
QUESTION_TYPES = ("noul", "choice", "score")

#: Bounds for ``choice`` criteria entries (server-side requirement).
_CRITERIA_MIN = 2
_CRITERIA_MAX = 26

#: Default Ollama base URL and model tag.
DEFAULT_BASE_URL = "http://localhost:11434"
DEFAULT_MODEL = "clef-flash:latest"


def build_request(model, state, questions):
    """Validate and build the ``/v1/systemone`` request body (dict).

    ``state``: the situation string the questions are about.
    ``questions``: ``{name: {"type": ..., "instructions": ...,
    "criteria": ...}}`` with per-type criteria rules:

    - ``noul``: no ``criteria`` allowed.
    - ``choice``: ``criteria`` required, a dict of 2-26
      ``candidate -> description`` entries (never a list).
    - ``score``: ``criteria`` required, a non-empty *list* of scale anchors
      (numbers, strings, or ``{"score": <number>, "description": ...}``
      dicts — never an object).

    Raises ``ValueError`` on any shape violation, ``TypeError`` on wrong
    Python types. Never touches the network.
    """
    if not isinstance(model, str) or not model:
        raise ValueError("model must be a non-empty string")
    if not isinstance(state, str) or not state.strip():
        raise ValueError("state must be a non-empty string")
    if not isinstance(questions, dict) or not questions:
        raise ValueError("questions must be a non-empty dict "
                         "(an object, not a list)")
    for name, q in questions.items():
        if not isinstance(q, dict):
            raise TypeError(f"question {name!r} must be a dict")
        qtype = q.get("type")
        if qtype not in QUESTION_TYPES:
            raise ValueError(
                f"question {name!r}: type must be one of {QUESTION_TYPES}, "
                f"got {qtype!r}")
        if not isinstance(q.get("instructions"), str) or not q["instructions"].strip():
            raise ValueError(
                f"question {name!r}: 'instructions' must be a non-empty string")
        criteria = q.get("criteria")
        if qtype == "noul":
            if criteria is not None:
                raise ValueError(
                    f"question {name!r}: 'noul' takes no criteria")
        elif qtype == "choice":
            if not isinstance(criteria, dict):
                raise ValueError(
                    f"question {name!r}: 'criteria' must be a dict of "
                    f"candidate -> description (never a list)")
            n = len(criteria)
            if not (_CRITERIA_MIN <= n <= _CRITERIA_MAX):
                raise ValueError(
                    f"question {name!r}: 'criteria' needs "
                    f"{_CRITERIA_MIN}-{_CRITERIA_MAX} entries, got {n}")
            for cand, desc in criteria.items():
                if not isinstance(desc, str) or not desc.strip():
                    raise ValueError(
                        f"question {name!r}: criteria[{cand!r}] description "
                        f"must be a non-empty string")
        elif qtype == "score":
            if not isinstance(criteria, list) or not criteria:
                raise ValueError(
                    f"question {name!r}: 'score' criteria must be a non-empty "
                    f"list of scale anchors (never an object)")
    return {"model": model, "state": state, "questions": questions}


def parse_answers(payload):
    """Normalize a ``/v1/systemone`` response dict into typed answers.

    Returns ``{name: {"type": ..., ...}}`` where each answer is one of::

        {"type": "noul",   "p": float}
        {"type": "choice", "choice": str, "probabilities": {str: float},
         "confidence": float | None}
        {"type": "score",  "score": float, "legend": {str: Any},
         "probabilities": {str: float}, "confidence": float | None}

    For ``score``, ``score`` is the expected *position index* over the
    criteria array; ``legend`` maps each position to its criterion. Use
    :func:`score_value` to map it back onto numeric criteria.

    Raises ``RuntimeError`` on unexpected response shapes.
    """
    try:
        answers = payload["answers"]
    except (KeyError, TypeError) as e:
        raise RuntimeError("systemone: response has no 'answers'") from e
    if not isinstance(answers, dict) or not answers:
        raise RuntimeError("systemone: 'answers' must be a non-empty object")
    out = {}
    for name, a in answers.items():
        if not isinstance(a, dict):
            raise RuntimeError(f"systemone: answer {name!r} is not an object")
        atype = a.get("type")
        if atype == "noul":
            p = a.get("noul")
            if not isinstance(p, (int, float)) or isinstance(p, bool):
                raise RuntimeError(
                    f"systemone: answer {name!r} 'noul' is not a number")
            out[name] = {"type": "noul", "p": float(p)}
        elif atype == "choice":
            choice = a.get("choice")
            probs = a.get("probabilities")
            if not isinstance(choice, str) or not isinstance(probs, dict):
                raise RuntimeError(
                    f"systemone: answer {name!r} has bad 'choice'/"
                    f"'probabilities'")
            conf = a.get("confidence")
            out[name] = {
                "type": "choice",
                "choice": choice,
                "probabilities": {k: float(v) for k, v in probs.items()},
                "confidence": float(conf) if isinstance(
                    conf, (int, float)) and not isinstance(conf, bool)
                else None,
            }
        elif atype == "score":
            s = a.get("score")
            if not isinstance(s, (int, float)) or isinstance(s, bool):
                raise RuntimeError(
                    f"systemone: answer {name!r} 'score' is not a number")
            legend = a.get("legend")
            probs = a.get("probabilities")
            if not isinstance(legend, dict) or not isinstance(probs, dict):
                raise RuntimeError(
                    f"systemone: answer {name!r} has bad 'legend'/"
                    f"'probabilities'")
            conf = a.get("confidence")
            out[name] = {
                "type": "score",
                "score": float(s),
                "legend": legend,
                "probabilities": {k: float(v) for k, v in probs.items()},
                "confidence": float(conf) if isinstance(
                    conf, (int, float)) and not isinstance(conf, bool)
                else None,
            }
        else:
            raise RuntimeError(
                f"systemone: answer {name!r} has unknown type {atype!r}")
    return out


def _criterion_number(value):
    """Numeric value of one score criterion, or None if not numeric.

    Accepts a bare number or a dict carrying it under ``"score"`` or
    ``"value"`` (e.g. ``{"score": 0.5, "description": "..."}``).
    """
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, dict):
        for key in ("score", "value"):
            cand = value.get(key)
            if isinstance(cand, (int, float)) and not isinstance(cand, bool):
                return float(cand)
    return None


def score_value(answer):
    """Map a parsed ``score`` answer onto its numeric criteria.

    Returns ``sum_i p_i * v_i`` over the criteria positions, where ``v_i``
    is the numeric criterion (bare number or ``{"score"|"value": number}``).
    Returns ``None`` when the criteria aren't numeric (e.g. plain strings).

    Raises ``ValueError`` if ``answer`` is not a parsed score answer.
    """
    if not isinstance(answer, dict) or answer.get("type") != "score":
        raise ValueError("score_value needs a parsed 'score' answer")
    legend = answer.get("legend") or {}
    probs = answer.get("probabilities") or {}
    total = 0.0
    for idx, prob in probs.items():
        num = _criterion_number(legend.get(idx))
        if num is None:
            return None
        total += float(prob) * num
    return total


class SystemoneClient:
    """Typed-decision client for a Clef model served by Ollama >= 0.35."""

    def __init__(self, model=DEFAULT_MODEL, base_url=DEFAULT_BASE_URL,
                 timeout=600):
        """``model``: Ollama tag (e.g. ``clef-flash:latest``).
        ``base_url``: Ollama server root, trailing slashes stripped.
        ``timeout``: seconds per request (decision inference is fast, but
        the first call after model load can take a while)."""
        self.model = model
        self.base_url = (base_url or DEFAULT_BASE_URL).rstrip("/")
        self.timeout = timeout

    def decide(self, state, questions):
        """POST ``/v1/systemone`` and return normalized typed answers.

        ``state``/``questions`` are validated by :func:`build_request`
        before anything is sent. Raises ``ValueError``/``TypeError`` on bad
        input shape, ``RuntimeError`` on HTTP or response-shape errors.
        """
        body = build_request(self.model, state, questions)
        url = self.base_url + "/v1/systemone"
        req = urllib.request.Request(
            url, data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                raw = resp.read().decode("utf-8")
        except urllib.error.HTTPError as e:
            try:
                detail = e.read().decode("utf-8", "replace")[:500]
            except Exception:
                detail = ""
            raise RuntimeError(
                f"systemone decide failed: HTTP {e.code} {detail}") from None
        except urllib.error.URLError as e:
            raise RuntimeError(f"systemone decide failed: {e.reason}") from None
        except TimeoutError:
            # urlopen raises bare TimeoutError (not wrapped in URLError).
            raise RuntimeError(
                f"systemone decide timed out after {self.timeout}s") from None
        try:
            payload = json.loads(raw)
        except ValueError as e:
            raise RuntimeError(
                f"systemone decide: invalid JSON response: {e}") from None
        return parse_answers(payload)


def decide(state, questions, model=DEFAULT_MODEL, base_url=DEFAULT_BASE_URL,
           timeout=600):
    """One-shot convenience wrapper: build a client and call ``decide``."""
    return SystemoneClient(model=model, base_url=base_url,
                           timeout=timeout).decide(state, questions)
