"""CRoS: Causality Reward on Scrambled CoTs.

Paper: arXiv 2610.42054 "CRoS: Causality Reward on Scrambled CoTs
Improves Accuracy and Reasoning Faithfulness" (submitted 30 Oct 2026).

One-line claim: in RL with verifiable rewards, outcome-only rewards do
not guarantee that a chain-of-thought (CoT) causally contributes to the
final answer; perturbing a CoT by SCRAMBLING its step order (instead of
deleting steps or injecting errors) measures true causality, because
scrambling preserves every step's content exactly while breaking the
causal order -- so a causally-linked CoT loses answer likelihood under
scrambling, while an order-independent CoT does not.

Operationalization note: the abstract does not print CRoS's exact
reward formula, so this module implements the abstract's stated
mechanism literally -- scramble = a permutation of the step list,
content preservation = exact multiset equality, causality = the drop
in answer score under scrambling -- and validates the claimed
PROPERTIES (preservation, chained-vs-independent discrimination,
deletion-as-baseline contrast) rather than bit-parity with the paper's
code. Stdlib only; no model, no network.
"""

import random
import re

_STEP_RE = re.compile(r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.+?)\s*$")
_INT_RE = re.compile(r"^-?\d+$")


def scramble_steps(steps, seed=0):
    """Return a permuted copy of ``steps`` (deterministic for ``seed``).

    The permutation changes order only: it never adds, removes, or
    edits a step. ``seed=0`` uses the plain reversal, which is the
    maximal order disruption for a short chain; other seeds use a
    seeded shuffle.
    """
    out = list(steps)
    if seed == 0:
        out.reverse()
    else:
        random.Random(seed).shuffle(out)
    return out


def content_preserved(original, scrambled):
    """True iff ``scrambled`` is a permutation of ``original``.

    Exact multiset equality on the step strings: no step's content may
    change, which is precisely what deletion / error-injection
    perturbations fail to satisfy.
    """
    return sorted(original) == sorted(scrambled)


def deletion_changes_content(original, deleted):
    """True iff deleting steps changes the step multiset (it always does)."""
    return sorted(original) != sorted(deleted)


def causality_score(score_original, score_scrambled):
    """Answer-score drop caused by scrambling: orig minus scrambled."""
    return score_original - score_scrambled


def eval_sequential(steps):
    """Evaluate ``name = expr`` steps in order; exprs may use earlier names.

    Returns (env, failed) where ``failed`` is True if some step
    referenced a name not yet defined (a broken causal chain). Only
    integers and + - * over defined names are allowed.
    """
    env = {}
    for step in steps:
        m = _STEP_RE.match(step)
        if not m:
            return env, True
        name, expr = m.group(1), m.group(2)
        try:
            value = _eval_expr(expr, env)
        except (KeyError, ValueError):
            return env, True
        env[name] = value
    return env, False


def _eval_expr(expr, env):
    """Evaluate a tiny integer expression over ``env`` (no builtins)."""
    tokens = re.findall(r"[A-Za-z_][A-Za-z0-9_]*|-?\d+|[+\-*()]", expr)
    if not tokens or "".join(tokens).replace(" ", "") != expr.replace(" ", ""):
        raise ValueError("unsupported expression: %r" % expr)
    out = []
    for tok in tokens:
        if _INT_RE.match(tok):
            out.append(tok)
        elif tok in "+-*()":
            out.append(tok)
        elif tok in env:
            out.append(str(env[tok]))
        else:
            raise KeyError(tok)
    return eval("".join(out), {"__builtins__": {}}, {})


def eval_independent_sum(steps):
    """Evaluate constant-only steps; the sum is order-invariant by design.

    Each step must be ``name = <int>``. Returns (total, failed).
    """
    total = 0
    for step in steps:
        m = _STEP_RE.match(step)
        if not m or not _INT_RE.match(m.group(2)):
            return total, True
        total += int(m.group(2))
    return total, False


# -- deterministic toy problems ------------------------------------------------

def chained_problems():
    """Six chained arithmetic problems: each step uses the previous one."""
    problems = []
    for i, (a, b, c) in enumerate(
        [(3, 4, 5), (2, 7, 3), (5, 2, 8), (4, 4, 4), (6, 3, 2), (7, 1, 6)]
    ):
        steps = [
            "x = %d" % a,
            "y = x + %d" % b,
            "z = y * %d" % c,
            "ans = z - x",
        ]
        env, failed = eval_sequential(steps)
        problems.append({"id": "chain-%d" % i, "steps": steps,
                         "gold": env["ans"], "failed": failed})
    return problems


def independent_problems():
    """Six order-independent problems: constant facts, answer = their sum."""
    problems = []
    for i, vals in enumerate(
        [(3, 4, 5), (2, 7, 3), (5, 2, 8), (4, 4, 4), (6, 3, 2), (7, 1, 6)]
    ):
        steps = ["v%d = %d" % (j, v) for j, v in enumerate(vals)]
        total, failed = eval_independent_sum(steps)
        problems.append({"id": "indep-%d" % i, "steps": steps,
                         "gold": total, "failed": failed})
    return problems


def run_experiment():
    """Scramble every toy problem; measure the causality signal per type.

    Score is 1 iff evaluation under the given order recovers the gold
    answer, else 0. Returns a dict of measured numbers.
    """
    out = {"chained": [], "independent": []}
    for prob in chained_problems():
        env, failed = eval_sequential(prob["steps"])
        s_orig = 0 if failed else int(env.get("ans") == prob["gold"])
        env_s, failed_s = eval_sequential(scramble_steps(prob["steps"]))
        s_scram = 0 if failed_s else int(env_s.get("ans") == prob["gold"])
        out["chained"].append({
            "id": prob["id"],
            "preserved": content_preserved(prob["steps"],
                                           scramble_steps(prob["steps"])),
            "orig": s_orig, "scrambled": s_scram,
            "causality": causality_score(s_orig, s_scram),
            "scrambled_failed": failed_s,
        })
    for prob in independent_problems():
        total, failed = eval_independent_sum(prob["steps"])
        s_orig = 0 if failed else int(total == prob["gold"])
        total_s, failed_s = eval_independent_sum(
            scramble_steps(prob["steps"]))
        s_scram = 0 if failed_s else int(total_s == prob["gold"])
        out["independent"].append({
            "id": prob["id"],
            "preserved": content_preserved(prob["steps"],
                                           scramble_steps(prob["steps"])),
            "orig": s_orig, "scrambled": s_scram,
            "causality": causality_score(s_orig, s_scram),
            "scrambled_failed": failed_s,
        })
    for kind in ("chained", "independent"):
        rows = out[kind]
        out[kind + "_mean_causality"] = (
            sum(r["causality"] for r in rows) / len(rows))
        out[kind + "_scrambled_failed"] = sum(
            r["scrambled_failed"] for r in rows)
    return out
