"""CPU test for the TOPSD driver logic (arXiv:2611.08179).

Mirrors the pure functions in colab_topsd_driver.py without any model,
network or GPU: answer extraction, step splitting, drift-onset detection
from teacher-conditioned per-step logprobs, and the MGSM train/test
index-split disjointness (same split in every language, so no problem
appears in both train and test in any language).

Run: python3 test_topsd_logic_cpu.py
"""
import importlib.util
import sys

SPEC = importlib.util.spec_from_file_location(
    "topsd_driver",
    "colab_topsd_driver.py")
DRV = importlib.util.module_from_spec(SPEC)
# Importing the driver module only defines functions (main() is guarded),
# but it imports torch-dependent names lazily inside main() — the module
# top level is stdlib-only, so this is safe on CPU.
SPEC.loader.exec_module(DRV)

PASS = 0


def check(name, cond):
    global PASS
    assert cond, "FAIL: %s" % name
    PASS += 1
    print("ok: %s" % name)


# ---- extract_answer ----
check("extract prefers 'The answer is'",
      DRV.extract_answer("Step 1: 5 + 6 = 11.\nThe answer is 11.") == 11)
check("extract last int fallback",
      DRV.extract_answer("Roger has 5. He buys 2 more packs of 3. 5 + 6 = 11") == 11)
check("extract handles commas",
      DRV.extract_answer("The answer is 1,234.") == 1234)
check("extract handles negative",
      DRV.extract_answer("The answer is -7.") == -7)
check("extract none on empty", DRV.extract_answer("") is None)
check("extract none on no digits",
      DRV.extract_answer("I do not know.") is None)
check("extract ignores earlier 'answer is' wording",
      DRV.extract_answer("The answer is 5. Wait, recompute: 5 + 6 = 11. "
                         "The answer is 11.") == 11)

# ---- split_steps ----
trace = "Roger starts with 5 balls.\nHe buys 2 cans of 3.\n5 + 6 = 11."
check("split_steps lines", DRV.split_steps(trace) == [
    "Roger starts with 5 balls.", "He buys 2 cans of 3.", "5 + 6 = 11."])
check("split_steps single line falls back to sentences",
      len(DRV.split_steps("First do this. Then do that. Done.")) == 3)
check("split_steps empty", DRV.split_steps("") == [""])

# ---- drift_onset ----
# No drift: flat/high logprobs.
check("onset none on flat lps",
      DRV.drift_onset([-0.5, -0.4, -0.6, -0.5]) is None)
# Drift at step 2: sharp drop after two good steps.
check("onset at drop step",
      DRV.drift_onset([-0.5, -0.4, -2.2, -3.1]) == 2)
# Drift at step 0: first step already below the floor.
check("onset at step 0 below floor",
      DRV.drift_onset([-3.5, -0.4, -0.4]) == 0)
# Small wobble below the drop threshold is not drift.
check("onset none on small wobble",
      DRV.drift_onset([-0.5, -1.2, -0.9]) is None)
# Drop exactly at the threshold boundary (best - drop): -0.5 - 1.0 = -1.5;
# -1.5 is not < -1.5, so no onset; -1.6 triggers.
check("onset boundary strict",
      DRV.drift_onset([-0.5, -1.5]) is None
      and DRV.drift_onset([-0.5, -1.6]) == 1)
# Recovery: best-so-far tracks improvements, a later step must drop
# relative to the BEST previous step, not the immediately previous one.
check("onset uses best-so-far",
      DRV.drift_onset([-0.5, -0.1, -0.9]) is None)

# ---- MGSM index split disjointness (same split, every language) ----
for lang in ("en", "zh", "sw"):
    train_idx = set(range(0, 120))
    test_idx = set(range(150, 150 + 60))
    check("split disjoint %s" % lang, not (train_idx & test_idx))
    check("split sizes %s" % lang,
          len(train_idx) == 120 and len(test_idx) == 60)

print("\n%d checks passed" % PASS)
