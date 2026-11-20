"""CPU test for the RLCDR driver pure helpers (arXiv:2611.25135) — stdlib only.

Extracts the marked PURE HELPERS block from colab_rlcdr_driver.py and
asserts the reward/calibration math: answer extraction, debiasing over
option orders, temperature fitting, ECE and AUC.

Run: python3 tests/test_rlcdr_cpu.py
"""
import os
import sys

DRIVER = os.path.join(os.path.dirname(__file__), "..", "sessions",
                      "2026-11-20-2611.25135-rlcdr", "colab_rlcdr_driver.py")


def load_pure_helpers():
    src = open(DRIVER).read()
    start = src.index("# --- PURE HELPERS (stdlib) ---")
    end = src.index("# --- END PURE HELPERS ---")
    ns: dict = {}
    exec(compile(src[start:end], "rlcdr_pure", "exec"), ns)
    return ns


def main():
    h = load_pure_helpers()
    extract_answer = h["extract_answer"]
    answers_equal = h["answers_equal"]
    debias_p_yes = h["debias_p_yes"]
    calibrate_p = h["calibrate_p"]
    fit_temperature = h["fit_temperature"]
    ece_score = h["ece_score"]
    auc_score = h["auc_score"]
    decision_prompt = h["decision_prompt"]

    # --- answer extraction ---
    assert extract_answer("work ... \\boxed{42}") == "42"
    assert extract_answer("Answer: $7$") == "7"
    assert extract_answer("no answer here") is None
    assert answers_equal("42", "42")
    assert answers_equal("\\frac{1}{2}", "\\frac{1}{2}")
    assert answers_equal("0.5", "\\frac{1}{2}")
    assert not answers_equal("43", "42")
    assert not answers_equal(None, "42")

    # --- debiasing: symmetric orders cancel position bias ---
    assert abs(debias_p_yes(0.9, 0.1) - 0.5) < 1e-9
    assert abs(debias_p_yes(0.7, 0.7) - 0.7) < 1e-9
    assert debias_p_yes(None, 0.5) is None

    # --- calibration: T=1 is a fixed point; overconfident -> T>1 ---
    assert abs(calibrate_p(0.8, 1.0) - 0.8) < 1e-9
    assert calibrate_p(0.9, 2.0) < 0.9  # temperature softens
    assert calibrate_p(0.1, 2.0) > 0.1
    # Frequency-matched data (9/10 of the p=0.9 group positive, 1/10 of
    # the p=0.1 group) -> T near 1. (Perfectly separated labels would
    # drive T to the grid floor, which is correct NLL behaviour, not a bug.)
    ps = [0.9] * 10 + [0.1] * 10
    ys = [1] * 9 + [0] + [1] + [0] * 9
    t, _ = fit_temperature(ps, ys)
    assert 0.8 <= t <= 1.3, t
    # Overconfident judge (says 0.99, right 60%) -> T > 1.
    ps2 = [0.99] * 6 + [0.01] * 4
    ys2 = [1, 1, 1, 1, 0, 0, 0, 0, 0, 0]
    t2, _ = fit_temperature(ps2, ys2)
    assert t2 > 1.5, t2

    # --- ECE: perfect calibration -> ~0; constant 0.5 on 50/50 -> ~0 too ---
    assert ece_score([0.9] * 9 + [0.1], [1] * 9 + [0]) < 0.11
    assert ece_score([0.5] * 10, [1, 0] * 5) < 1e-9
    assert ece_score([0.99] * 10, [0] * 10) > 0.9

    # --- AUC: perfect separation -> 1.0; random -> ~0.5; single class -> None ---
    assert auc_score([0.9, 0.8, 0.2, 0.1], [1, 1, 0, 0]) == 1.0
    assert auc_score([0.5, 0.5, 0.5, 0.5], [1, 1, 0, 0]) == 0.5
    assert auc_score([0.9, 0.8], [1, 1]) is None

    # --- decision prompt: both orders, exemplars, truncation ---
    p1 = decision_prompt("What is 2+2?", "2+2=4. Answer: 4")
    p2 = decision_prompt("What is 2+2?", "2+2=4. Answer: 4", swapped=True)
    assert "A) Yes" in p1 and "B) No" in p1
    assert "A) No" in p2 and "B) Yes" in p2
    assert "Answer: A" in p1 and "Answer: B" in p1  # exemplars present
    assert p1.endswith("Answer: ")
    long_sol = "x" * 5000
    assert len(decision_prompt("Q?", long_sol)) < len(long_sol)

    print("ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
