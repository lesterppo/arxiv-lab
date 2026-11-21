"""CPU test for Budget Internalization (arXiv:2611.26172) helpers.

Tests the pure logic the Colab driver depends on -- no torch, no GPU:
  T1  extract_answer: '#### <number>' parsing (last wins, commas, $, sign,
      decimals; None when absent or non-numeric).
  T2  Arm-split rule (mirrors the driver): a correct trace goes to the
      "truncated" arm iff its reasoning exceeds the budget B, else to the
      "natural" arm; truncated completions end with a forced '#### answer'.
  T3  Internalization signature criterion: given (model, budget, accuracy)
      rows, the signature holds iff the truncated arm beats base near B
      and falls below base at 4*B, while the natural arm does not fall
      below base at 4*B. Checked on synthetic rows (paper-like and null).
  T4  Eval-budget grid sanity: budgets straddle B (one below, B itself,
      two above) so the signature is measurable.

Usage: python3 test_budget_internalization_cpu.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from colab_budget_internalization import extract_answer  # noqa: E402

BUDGET = 256


def check(name, cond, detail=""):
    print(f"[{'PASS' if cond else 'FAIL'}] {name} {detail}")
    if not cond:
        raise AssertionError(f"{name} {detail}")


def split_arms(traces, budget):
    """Mirror of the driver's Phase-1 split. traces: (reasoning, gold, n_tok)."""
    trunc, nat = [], []
    for reasoning, gold, n_tok in traces:
        ans = f"#### {int(gold) if float(gold).is_integer() else gold}"
        if n_tok > budget:
            trunc.append(reasoning + "\n" + ans)
        else:
            nat.append(reasoning + "\n" + ans)
    return trunc, nat


def internalization_signature(rows, budget):
    """Return (truncated_internalizes, natural_internalizes) from eval rows.

    rows: list of dicts {model, budget, accuracy}. Internalization for an
    arm = beats base at `budget` AND falls below base at 4*budget.
    """
    acc = {(r["model"], r["budget"]): r["accuracy"] for r in rows}
    out = {}
    for arm in ("truncated", "natural"):
        near = acc.get((arm, budget), 0) > acc.get(("base", budget), 0)
        far = acc.get((arm, 4 * budget), 0) < acc.get(("base", 4 * budget), 0)
        out[arm] = bool(near and far)
    return out


def main():
    # T1
    check("T1 plain", extract_answer("blah\n#### 42") == 42.0)
    check("T1 commas+$/last wins",
          extract_answer("#### 1,234\n#### $5,678") == 5678.0)
    check("T1 negative decimal",
          extract_answer("#### -3.5") == -3.5)
    check("T1 absent", extract_answer("no answer here") is None)
    check("T1 non-numeric", extract_answer("#### abc") is None)
    check("T1 empty", extract_answer("") is None)

    # T2
    traces = [("long reasoning " * 40, 42, 300),
              ("short", 7, 50),
              ("exact budget", 9, 256)]
    trunc, nat = split_arms(traces, BUDGET)
    check("T2 long trace truncated", len(trunc) == 1
          and trunc[0].endswith("#### 42"))
    check("T2 short+natural", len(nat) == 2)
    check("T2 at-budget is natural (not >B)",
          any(t.endswith("#### 9") for t in nat))

    # T3 — paper-like rows: truncated internalizes, natural does not.
    paper_rows = [
        {"model": "base", "budget": 128, "accuracy": 0.40},
        {"model": "base", "budget": 256, "accuracy": 0.55},
        {"model": "base", "budget": 512, "accuracy": 0.62},
        {"model": "base", "budget": 1024, "accuracy": 0.68},
        {"model": "truncated", "budget": 128, "accuracy": 0.45},
        {"model": "truncated", "budget": 256, "accuracy": 0.66},
        {"model": "truncated", "budget": 512, "accuracy": 0.60},
        {"model": "truncated", "budget": 1024, "accuracy": 0.58},
        {"model": "natural", "budget": 128, "accuracy": 0.42},
        {"model": "natural", "budget": 256, "accuracy": 0.57},
        {"model": "natural", "budget": 512, "accuracy": 0.63},
        {"model": "natural", "budget": 1024, "accuracy": 0.68},
    ]
    sig = internalization_signature(paper_rows, BUDGET)
    check("T3 paper-like: truncated internalizes",
          sig["truncated"] is True, str(sig))
    check("T3 paper-like: natural does not",
          sig["natural"] is False, str(sig))
    null_rows = [{**r, "accuracy": 0.50} for r in paper_rows]
    sig0 = internalization_signature(null_rows, BUDGET)
    check("T3 null rows: neither internalizes",
          sig0 == {"truncated": False, "natural": False}, str(sig0))

    # T4
    budgets = [128, 256, 512, 1024]
    check("T4 grid straddles B",
          min(budgets) < BUDGET and BUDGET in budgets
          and max(budgets) >= 4 * BUDGET)
    print("ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
