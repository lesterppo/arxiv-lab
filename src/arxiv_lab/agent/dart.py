"""DART — Diverse Alternative Reasoning Traces for hard math problems.

Paper: arXiv 2611.34928 (2026-11-27). Core mechanism, as implemented here:

1. Start from a REJECTED trace (greedy sample whose final answer is wrong).
2. Decompose it into reasoning steps.
3. Rank the steps by position-weighted uncertainty: a step's score is
   ``uncertainty * ((i + 1) / n) ** alpha`` — later steps count more,
   because an early misstep poisons everything after it while a late
   misstep is cheap to repair. (The paper estimates uncertainty from the
   generator's token logprobs; the NIM endpoint used for the live run does
   not expose logprobs, so the live experiment uses the position-only
   proxy ``uncertainty = 1`` for every step and says so in the session
   README. The ranking function itself accepts per-step uncertainties
   whenever they are available.)
4. For the top-ranked step, generate deliberately distinct alternative
   candidates with REFERENCE-ANSWER GUIDANCE (the miner knows the gold
   answer — DART is a training-data mining method, not a test-time solver)
   plus a META-THOUGHT: a 1–2 sentence reflection on what went wrong in
   the original step. Then complete the trace from the revised step.
5. Compare against plain Monte-Carlo sampling at a MATCHED trace budget:
   MC generates k fresh traces; DART spends the same k traces on
   1 revision + (k - 1) completions. Recovery = fraction of greedy-unsolved
   problems solved by at least one trace.

Only deterministic, model-free logic lives in this module (step splitting,
ranking, answer extraction/normalisation, budget accounting). The live
model calls live in the session driver, not here.
"""

from __future__ import annotations

import math
import re

# ---------------------------------------------------------------------------
# Step decomposition
# ---------------------------------------------------------------------------

_STEP_HEAD = re.compile(r"^\s*(?:step\s+)?\d+\s*[.):]\s+", re.IGNORECASE)


def split_steps(trace: str) -> list[str]:
    """Split a reasoning trace into steps.

    Primary split: blank-line-separated blocks (the usual layout of a
    model-written solution). A block that is a single short line starting
    with a step number (``1. ...`` / ``Step 2: ...``) is kept as its own
    step; consecutive short numbered lines are NOT merged — each numbered
    line is one step. Long unnumbered blocks stay whole.
    """
    trace = (trace or "").strip()
    if not trace:
        return []
    blocks = [b.strip() for b in re.split(r"\n\s*\n", trace) if b.strip()]
    steps: list[str] = []
    for block in blocks:
        lines = [ln.strip() for ln in block.splitlines() if ln.strip()]
        if len(lines) > 1 and all(_STEP_HEAD.match(ln) for ln in lines):
            steps.extend(lines)
        else:
            steps.append(block)
    return steps


# ---------------------------------------------------------------------------
# Position-weighted uncertainty ranking
# ---------------------------------------------------------------------------

def position_weight(index: int, n_steps: int, alpha: float = 1.0) -> float:
    """Weight of step ``index`` (0-based) in a trace of ``n_steps`` steps.

    Later steps weigh more: ``((i + 1) / n) ** alpha``. ``alpha = 1`` is
    the linear default; larger alpha concentrates the mass on the tail.
    """
    if n_steps <= 0:
        return 0.0
    return ((index + 1) / n_steps) ** alpha


def rank_steps(steps: list[str], uncertainties: list[float] | None = None,
               alpha: float = 1.0) -> list[int]:
    """Rank step indices best-first by position-weighted uncertainty.

    ``uncertainties[i]`` is the model's uncertainty for step ``i``
    (e.g. mean token entropy); when omitted every step gets 1.0 and the
    ranking is purely positional (latest step first). Ties break toward
    the later step, then toward the earlier index for determinism.
    """
    n = len(steps)
    if n == 0:
        return []
    unc = uncertainties if uncertainties is not None else [1.0] * n
    if len(unc) != n:
        raise ValueError(f"uncertainties has {len(unc)} entries for {n} steps")
    scored = [(unc[i] * position_weight(i, n, alpha), -i, i) for i in range(n)]
    scored.sort(key=lambda t: (-t[0], t[1]))
    return [i for _, _, i in scored]


# ---------------------------------------------------------------------------
# Answer extraction / normalisation
# ---------------------------------------------------------------------------

_BOXED = re.compile(r"\\boxed\{", re.IGNORECASE)


def extract_boxed_answer(text: str) -> str | None:
    """Return the contents of the LAST ``\\boxed{...}`` in ``text``.

    Braces are matched, so nested ``\\frac{a}{b}`` inside the box works.
    """
    text = text or ""
    matches = list(_BOXED.finditer(text))
    if not matches:
        return None
    start = matches[-1].end()
    depth = 1
    i = start
    while i < len(text) and depth > 0:
        ch = text[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
        i += 1
    if depth != 0:
        return None
    return text[start:i - 1].strip()


def normalize_answer(ans: str) -> str:
    """Normalise a math answer string for comparison.

    Strips LaTeX spacing/boxing commands, collapses whitespace, and
    rewrites ``\\frac{a}{b}`` as ``(a)/(b)`` so ``\\frac12``-style and
    ``(1)/(2)``-style spellings compare equal. This is a string-level
    normaliser, not a symbolic checker: ``p-q`` and ``-q+p`` do NOT match.
    """
    s = (ans or "").strip()
    s = s.replace("$", "")
    s = re.sub(r"\\(left|right|big|Big|bigg|Bigg)\b", "", s)
    s = re.sub(r"\\[,;:\s]", "", s)
    s = re.sub(r"\s+", "", s)
    # \frac{a}{b} -> (a)/(b), applied innermost-first
    frac = re.compile(r"\\(?:d|t)?frac\{([^{}]*)\}\{([^{}]*)\}")
    while frac.search(s):
        s = frac.sub(r"(\1)/(\2)", s)
    s = s.replace("\\pi", "pi").replace("\\infty", "inf")
    return s


def _as_float(s: str) -> float | None:
    try:
        return float(s)
    except (TypeError, ValueError):
        return None


def _strip_assignment(s: str) -> str:
    """If ``s`` is a bare assignment ``x = <expr>``, return ``<expr>``.

    MATH-500 gold answers are sometimes written ``x=5`` while a solver
    correctly answers ``5``. The bare value is the comparable part.
    """
    m = re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*\s*=\s*(.+)", s)
    return m.group(1).strip() if m else s


_ANSWER_IS = re.compile(
    r"(?:the\s+)?(?:final\s+)?answer\s+is\s*:?\s*\$?(.+?)\$?\s*[.\n]?\s*$",
    re.IGNORECASE | re.DOTALL)


def extract_answer_fallback(text: str) -> str | None:
    """Last-resort answer extraction when no ``\\boxed{}`` is present.

    Looks for a trailing "the answer is X" (with X taken up to the end
    of the line / sentence) or a trailing bare ``$...$`` math span.
    Returns None when nothing plausible is found. Conservative by
    design: it only fires on explicit answer phrasing, never on an
    arbitrary last line.
    """
    text = (text or "").strip()
    if not text:
        return None
    tail = text[-500:]
    m = _ANSWER_IS.search(tail)
    if m:
        cand = m.group(1).strip().strip("$").strip()
        cand = re.split(r"\n", cand)[0].strip()
        if cand:
            return cand
    m2 = re.search(r"\$([^$]{1,80})\$\s*$", tail)
    if m2:
        return m2.group(1).strip()
    return None


def extract_answer(text: str) -> tuple[str | None, str]:
    """Extract an answer, returning (answer, method).

    ``method`` is ``"boxed"``, ``"fallback"`` or ``"none"`` — recorded
    so the live run can say how many traces were scored off the
    fallback path instead of the ``\\boxed{}`` contract.
    """
    boxed = extract_boxed_answer(text)
    if boxed is not None:
        return boxed, "boxed"
    fb = extract_answer_fallback(text)
    if fb is not None:
        return fb, "fallback"
    return None, "none"


def answers_equal(pred: str | None, gold: str | None,
                  tol: float = 1e-9) -> bool:
    """True when ``pred`` matches ``gold`` after normalisation.

    Pure numbers compare with a tolerance; anything else compares as
    normalised strings. ``None`` never matches. A gold of the form
    ``x = <expr>`` also matches a prediction of just ``<expr>``.
    """
    if pred is None or gold is None:
        return False
    p, g = normalize_answer(pred), normalize_answer(gold)
    if p == g:
        return True
    g2 = normalize_answer(_strip_assignment(gold))
    if p == g2:
        return True
    pf, gf = _as_float(p), _as_float(g2)
    if pf is not None and gf is not None:
        return math.isclose(pf, gf, rel_tol=0.0, abs_tol=tol)
    return False


# ---------------------------------------------------------------------------
# Trace-budget accounting + recovery
# ---------------------------------------------------------------------------

class TraceBudget:
    """Matched trace budget for one problem: MC vs DART.

    Both arms may spend exactly ``k`` freshly generated traces. MC spends
    all k on independent samples; DART spends 1 on a step revision and
    k - 1 on completions from the revised step. The original rejected
    (greedy) trace is screening output, not part of either arm's budget.
    """

    def __init__(self, k: int):
        if k < 2:
            raise ValueError("k must be >= 2 (1 revision + >=1 completion)")
        self.k = k
        self.mc_used = 0
        self.dart_used = 0

    def mc_plan(self) -> list[str]:
        return ["sample"] * self.k

    def dart_plan(self) -> list[str]:
        return ["revise"] + ["complete"] * (self.k - 1)

    def record(self, arm: str, n: int = 1) -> None:
        if arm == "mc":
            self.mc_used += n
        elif arm == "dart":
            self.dart_used += n
        else:
            raise ValueError(f"unknown arm {arm!r}")

    def balanced(self) -> bool:
        return self.mc_used == self.dart_used == self.k


def recovery_rate(solved: list[bool]) -> float:
    """Fraction of greedy-unsolved problems solved by >= 1 trace."""
    if not solved:
        return 0.0
    return sum(1 for s in solved if s) / len(solved)
