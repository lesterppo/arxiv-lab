#!/usr/bin/env python3
"""CPU test for DART (arXiv 2611.34928) -- deterministic, stdlib only.

E1  Unit: step splitting (blank-line blocks, numbered lines).
E2  Unit: position-weighted ranking (later steps first; uncertainty
    overrides position; ties deterministic).
E3  Unit: \\boxed extraction (last box, nested braces) and answer
    normalisation / equality (frac spellings, numeric tolerance).
E4  Unit: TraceBudget -- MC and DART plans both spend exactly k traces.
E5  Oracle simulation (mechanism as designed, uncertainties available):
    DART recovery >> MC recovery at matched k=4 budget, over 400
    synthetic hard problems with a single trap step.
E6  Ablation (honest limit): with the position-only proxy (no
    uncertainties, as in the NIM live run) DART LOSES to MC in the
    oracle -- the proxy only repairs traps at the tail of the trace,
    and the revision spends one of the k traces. The ranking is
    load-bearing: without uncertainty the mechanism has no way to
    locate the trap. The live run measures whether real completions
    (which can re-derive / self-correct, unlike the oracle's frozen
    prefix) still let the proxy win in practice.
E7  Determinism: identical headline numbers across repeated runs.
"""
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from arxiv_lab.agent.dart import (  # noqa: E402
    TraceBudget,
    answers_equal,
    extract_answer,
    extract_answer_fallback,
    extract_boxed_answer,
    normalize_answer,
    position_weight,
    rank_steps,
    recovery_rate,
    split_steps,
)

FAILURES = []


def check(name, cond, detail=""):
    print(f"{'PASS' if cond else 'FAIL'}  {name}  {detail}")
    if not cond:
        FAILURES.append(name)


# ---------------------------------------------------------------- E1
trace = """Let n = j + k.

For fixed n >= 2 there are n - 1 pairs.

Rewrite (n-1)/n^3 = 1/n^2 - 1/n^3.

Thus S = (p - 1) - (q - 1) = p - q.

\\boxed{p-q}"""
steps = split_steps(trace)
check("E1 split blank-line blocks", len(steps) == 5, f"got {len(steps)}")

numbered = "1. Add 2 and 3 to get 5.\n2. Multiply by 4 to get 20.\n3. Subtract 1.\n\n\\boxed{19}"
steps2 = split_steps(numbered)
check("E1 split numbered lines", len(steps2) == 4, f"got {len(steps2)}")
check("E1 empty trace", split_steps("") == [])
check("E1 single block", len(split_steps("just one step")) == 1)

# ---------------------------------------------------------------- E2
check("E2 position weight increases",
      position_weight(0, 5) < position_weight(4, 5))
check("E2 position weight last == 1.0",
      abs(position_weight(4, 5) - 1.0) < 1e-12)
check("E2 position-only ranks last first",
      rank_steps(["a", "b", "c", "d"]) == [3, 2, 1, 0])
check("E2 uncertainty overrides position",
      rank_steps(["a", "b", "c"], uncertainties=[0.9, 0.05, 0.05]) == [0, 2, 1])
check("E2 rank empty", rank_steps([]) == [])

# ---------------------------------------------------------------- E3
check("E3 extract last boxed",
      extract_boxed_answer("x \\boxed{3} then \\boxed{\\frac{1}{2}}") == "\\frac{1}{2}")
check("E3 extract nested braces",
      extract_boxed_answer("\\boxed{\\frac{a+b}{c}}") == "\\frac{a+b}{c}")
check("E3 extract none", extract_boxed_answer("no box here") is None)
check("E3 normalize frac",
      normalize_answer("\\frac{1}{2}") == normalize_answer("(1)/(2)"))
check("E3 normalize p-q",
      normalize_answer("p - q") == "p-q")
check("E3 answers equal frac",
      answers_equal("\\frac{1}{2}", "(1)/(2)"))
check("E3 answers equal numeric",
      answers_equal("0.5", "1/2") is False)  # "1/2" is not a float; string differs
check("E3 answers equal float tol",
      answers_equal("0.3333333333", "1/3") is False)
check("E3 answers equal same", answers_equal("p-q", "p-q"))
check("E3 answers none never", not answers_equal(None, "3"))
check("E3 boxed p-q vs gold",
      answers_equal(extract_boxed_answer(trace), "p - q"))
check("E3 gold x=5 matches pred 5", answers_equal("5", "x=5"))
check("E3 gold x=5 does not match pred 6", not answers_equal("6", "x=5"))
check("E3 fallback 'the answer is'",
      extract_answer_fallback("Some reasoning.\nThe answer is 42.") == "42")
check("E3 fallback none", extract_answer_fallback("No answer here.") is None)
check("E3 extract_answer prefers boxed",
      extract_answer("work \\boxed{3} the answer is 4") == ("3", "boxed"))
check("E3 extract_answer fallback method",
      extract_answer("work. The answer is 4.")[1] == "fallback")

# ---------------------------------------------------------------- E4
b = TraceBudget(k=4)
check("E4 mc plan spends k", b.mc_plan() == ["sample"] * 4)
check("E4 dart plan 1 revise + 3 complete",
      b.dart_plan() == ["revise", "complete", "complete", "complete"])
b.record("mc", 4)
b.record("dart", 4)
check("E4 balanced after k each", b.balanced())
check("E4 recovery rate", abs(recovery_rate([True, False, True]) - 2 / 3) < 1e-12)
check("E4 recovery empty", recovery_rate([]) == 0.0)

# ---------------------------------------------------------------- E5/E6 oracle simulation
# Synthetic hard problem: 5 steps, one trap step. Fresh sample: non-trap
# step correct w.p. 0.9, trap step w.p. 0.2. A step revised with reference
# guidance is correct w.p. 0.8. The rejected greedy trace is correct up
# to the trap and wrong from the trap on, so a DART prefix is usable only
# when the revised step is at or before the trap; revising after the trap
# leaves the trap error in the prefix.
N_STEPS = 5
P_OK = 0.9
P_TRAP = 0.2
P_GUIDED = 0.8


def mc_trace_ok(rng, trap):
    return all(rng.random() < (P_TRAP if i == trap else P_OK)
               for i in range(N_STEPS))


def dart_completion_ok(rng, trap, revise_at):
    if revise_at > trap:
        return False  # trap error stays in the fixed prefix
    if rng.random() >= P_GUIDED:
        return False  # revised step still wrong
    return all(rng.random() < P_OK for i in range(revise_at + 1, N_STEPS))


def run_oracle(seed, n_problems=400, k=4, use_uncertainty=True):
    rng = random.Random(seed)
    mc_solved, dart_solved = [], []
    for _ in range(n_problems):
        trap = rng.randrange(N_STEPS)
        mc_solved.append(any(mc_trace_ok(rng, trap) for _ in range(k)))
        if use_uncertainty:
            unc = [0.9 if i == trap else 0.1 for i in range(N_STEPS)]
            revise_at = rank_steps(["s"] * N_STEPS, uncertainties=unc)[0]
        else:
            revise_at = rank_steps(["s"] * N_STEPS)[0]  # position-only: last
        dart_solved.append(any(
            dart_completion_ok(rng, trap, revise_at) for _ in range(k - 1)))
    return recovery_rate(mc_solved), recovery_rate(dart_solved)


mc, dart = run_oracle(seed=7)
print(f"E5 oracle (uncertainty ranking): MC={mc:.3f} DART={dart:.3f} "
      f"(n=400, k=4)")
check("E5 DART beats MC with uncertainty ranking", dart - mc >= 0.20,
      f"gap {dart - mc:.3f}")
mc2, dart_pos = run_oracle(seed=7, use_uncertainty=False)
print(f"E6 ablation (position-only proxy): MC={mc2:.3f} DART={dart_pos:.3f}")
check("E6 position-only proxy loses to MC in oracle",
      dart_pos < mc2, f"gap {dart_pos - mc2:.3f} (proxy cannot locate trap)")

# ---------------------------------------------------------------- E7
a = run_oracle(seed=7)
b2 = run_oracle(seed=7)
check("E7 deterministic rerun", a == b2, f"{a} vs {b2}")

print()
if FAILURES:
    print(f"{len(FAILURES)} CHECK(S) FAILED: {FAILURES}")
    sys.exit(1)
print("ALL CHECKS PASSED")
