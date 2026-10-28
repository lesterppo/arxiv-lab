"""CPU tests for arxiv_lab.harness.context_attrition (arXiv:2610.36114).

Claims under test (Context Is a Pipeline, Not a Window: fixed-length
slicing of tool-result JSON deletes end-positioned fields unevenly):
A. Every cut result becomes broken JSON under the naive slice: strict
   JSON validity among cut results is 0.000 in every arm and seed
   (paper: every one of 3,827 cut results became broken JSON).
B. Field survival splits by evidence position: among cut results, the
   early-positioned evidence field survives (salvageable) in ~all cases
   while late-positioned evidence and tail records survive in ~none
   (paper: 99.3% vs 4.7% field reach; the simulation is starker because
   evidence position is deterministic, so only the direction and the
   split — not the magnitudes — are asserted).
C. The cut's cost appears first as extra work: with the targeted
   re-query fallback, final accuracy recovers to ~1.0 under every
   policy, but the naive slice needs strictly more tool calls and
   yields strictly less (solved-per-call) than the field-aware policy
   on the late and records_only arms (paper: +43% calls, yield
   0.61 -> 0.31; magnitudes are regime-dependent, direction asserted).
D. The remedy: field-aware truncation at the SAME byte budget keeps
   valid JSON, evidence survival, and first-attempt accuracy at ~1.0
   in every arm.
E. Calibration: the fixed budget cuts ~1/3 of early/late results
   (paper: 3,827/11,189 = 34.2%); the records_only arm cuts more
   because its results are larger (128 faulty tail records).

Seeded and deterministic; stdlib only. The reader is a deterministic
rule (strict parse + tolerant salvage + guaranteed re-query), not an
LLM, and the corpus is synthetic — hence SUPPORTED, not VERIFIED.

Run:  python3 tests/test_harness_context_attrition_cpu.py
"""

import json
import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from arxiv_lab.harness.context_attrition import (  # noqa: E402
    BUDGET_CHARS,
    CANONICAL_SEED,
    ROBUSTNESS_SEEDS,
    field_aware_truncate,
    generate_tool_result,
    naive_slice,
    run_condition,
    run_experiment,
    serialize,
)

N_TASKS = 300
SEEDS = (CANONICAL_SEED,) + ROBUSTNESS_SEEDS


def _cell(rows, policy, position):
    for r in rows:
        if r["policy"] == policy and r["evidence_position"] == position:
            return r
    raise KeyError((policy, position))


def _check_seed(seed):
    rows = run_experiment(n_tasks=N_TASKS, seed=seed)
    for position in ("early", "late", "records_only"):
        naive = _cell(rows, "naive_slice", position)
        aware = _cell(rows, "field_aware", position)
        control = _cell(rows, "no_cut", position)
        # A: every cut result is broken JSON under the naive slice.
        assert naive["json_valid_among_cut"] == 0.0, (seed, position, naive)
        assert naive["json_valid_rate"] < 1.0, (seed, position, naive)
        # D: field-aware output is always valid JSON with full survival.
        assert aware["json_valid_rate"] == 1.0, (seed, position, aware)
        assert aware["evidence_survival"] >= 0.99, (seed, position, aware)
        assert aware["first_accuracy"] >= 0.99, (seed, position, aware)
        assert aware["calls_per_task"] == 1.0, (seed, position, aware)
        # Control sanity: no cut, perfect first attempt.
        assert control["cut_rate"] == 0.0, (seed, position, control)
        assert control["first_accuracy"] == 1.0, (seed, position, control)
        # C: recovery via re-query, at strictly higher cost, where the
        # evidence is lost under the naive slice.
        assert naive["final_accuracy"] >= 0.99, (seed, position, naive)
        if position in ("late", "records_only"):
            assert naive["calls_per_task"] > aware["calls_per_task"], (
                seed, position, naive, aware)
            assert naive["yield"] < aware["yield"], (
                seed, position, naive, aware)
    # B: survival split among cut results.
    early = _cell(rows, "naive_slice", "early")
    late = _cell(rows, "naive_slice", "late")
    tail = _cell(rows, "naive_slice", "records_only")
    assert early["evidence_survival_among_cut"] >= 0.99, (seed, early)
    assert late["evidence_survival_among_cut"] <= 0.01, (seed, late)
    assert tail["evidence_survival_among_cut"] <= 0.01, (seed, tail)
    assert early["first_accuracy_among_cut"] >= 0.99, (seed, early)
    assert late["first_accuracy_among_cut"] <= 0.01, (seed, late)
    # E: calibration band for the field-position arms.
    assert 0.25 <= early["cut_rate"] <= 0.45, (seed, early)
    assert 0.25 <= late["cut_rate"] <= 0.45, (seed, late)
    assert tail["cut_rate"] > late["cut_rate"], (seed, tail, late)
    # Window nearly empty even while a third of results are cut.
    for r in rows:
        assert r["max_prompt_fraction_of_window"] < 0.15, (seed, r)
    return rows


def test_determinism():
    a = run_experiment(n_tasks=60, seed=CANONICAL_SEED)
    b = run_experiment(n_tasks=60, seed=CANONICAL_SEED)
    assert a == b, "same seed must reproduce identical rows"


def test_field_aware_respects_budget_and_validity():
    rng = random.Random(99)
    for position in ("early", "late", "records_only"):
        for _ in range(30):
            result, _ = generate_tool_result(rng, position)
            text = serialize(result)
            out = field_aware_truncate(text)
            assert len(out) <= BUDGET_CHARS, (position, len(out))
            json.loads(out)  # must be valid JSON
            # Naive slice of an over-budget result is invalid JSON.
            if len(text) > BUDGET_CHARS:
                try:
                    json.loads(naive_slice(text))
                except json.JSONDecodeError:
                    pass
                else:
                    raise AssertionError("cut naive slice parsed as valid")


def main():
    test_determinism()
    test_field_aware_respects_budget_and_validity()
    for seed in SEEDS:
        rows = _check_seed(seed)
        naive_late = _cell(rows, "naive_slice", "late")
        naive_tail = _cell(rows, "naive_slice", "records_only")
        print(
            "seed %d: cut=%.3f valid|cut=0 surv|cut early=%.3f late=%.3f "
            "tail=%.3f; late calls=%.2f yield=%.3f; tail calls=%.2f "
            "yield=%.3f" % (
                seed,
                naive_late["cut_rate"],
                _cell(rows, "naive_slice", "early")[
                    "evidence_survival_among_cut"],
                naive_late["evidence_survival_among_cut"],
                naive_tail["evidence_survival_among_cut"],
                naive_late["calls_per_task"],
                naive_late["yield"],
                naive_tail["calls_per_task"],
                naive_tail["yield"],
            )
        )
    print("test_harness_context_attrition_cpu: ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
