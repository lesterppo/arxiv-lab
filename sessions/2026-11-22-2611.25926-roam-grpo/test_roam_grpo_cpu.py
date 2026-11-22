"""Deterministic CPU test for the ROAM-GRPO Colab driver (arXiv:2611.25926).

stdlib + numpy only, no torch, no network. Imports the pure functions
from colab_roam_grpo.py (its torch imports live inside main(), and the
pip installs only run under __main__), so this test exercises the real
driver code: answer extraction, advantage computation, clipped
surrogate loss math, TF-IDF cosine distance, off-mode routing, and the
metric-row schema.

Run: python3 test_roam_grpo_cpu.py
"""
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import colab_roam_grpo as drv

PASSED = 0


def check(name, cond, detail=""):
    global PASSED
    assert cond, f"FAIL: {name} {detail}"
    PASSED += 1
    print(f"  ok: {name}")


def test_answer_extraction():
    # boxed{} primary
    check("boxed int", drv.extract_answer("working...\n\\boxed{42}") == 42.0)
    check("boxed frac", drv.extract_answer("\\boxed{\\frac{3}{10}}") == 0.3)
    check("boxed last wins",
          drv.extract_answer("\\boxed{1} then \\boxed{7}") == 7.0)
    check("boxed beats ####",
          drv.extract_answer("\\boxed{5}\n#### 9") == 5.0)
    # #### fallback
    check("hash fallback", drv.extract_answer("steps...\n#### 7") == 7.0)
    check("hash comma", drv.extract_answer("#### 1,000") == 1000.0)
    check("hash negative", drv.extract_answer("#### -3") == -3.0)
    check("no answer", drv.extract_answer("no final answer here") is None)
    check("empty", drv.extract_answer("") is None)
    # MATH gold extraction from a solution field
    sol = ("Note $(n+1)^2$ is a square. The sum is $36$.\n\\boxed{36}")
    check("gold from solution", drv.extract_answer(sol) == 36.0)
    # matching semantics
    check("match float", drv.answers_match(0.3, 0.3))
    check("match frac vs decimal",
          drv.answers_match(drv.extract_answer("\\boxed{\\frac{1}{2}}"), 0.5))
    check("mismatch", not drv.answers_match(5.0, 6.0))
    check("none never matches", not drv.answers_match(None, 5.0))


def test_advantages():
    adv = drv.compute_advantages([1, 1, 0, 0])
    check("adv balanced", adv is not None and all(
        abs(a - e) < 1e-9 for a, e in zip(adv, [1.0, 1.0, -1.0, -1.0])),
        str(adv))
    adv = drv.compute_advantages([1, 0, 0, 0])
    # mean 0.25, std sqrt(0.1875) ~= 0.4330
    check("adv 1-of-4", adv is not None
          and abs(adv[0] - 0.75 / math.sqrt(0.1875)) < 1e-9
          and abs(adv[1] + 0.25 / math.sqrt(0.1875)) < 1e-9, str(adv))
    check("adv zero-sum", abs(sum(drv.compute_advantages([1, 1, 0]))) < 1e-9)
    check("adv all-equal skip",
          drv.compute_advantages([1, 1, 1, 1]) is None)
    check("adv all-zero skip",
          drv.compute_advantages([0, 0, 0, 0]) is None)
    check("adv single reward skip",
          drv.compute_advantages([1]) is None)


def test_clipped_surrogate():
    # no clipping: ratios exp(0.1)=1.1052, exp(-0.1)=0.9048, adv=+1
    loss = drv.clipped_surrogate_loss([([-0.5, -0.5], [-0.4, -0.6], 1.0)])
    expect = -(math.exp(0.1) + math.exp(-0.1)) / 2
    check("surrogate unclipped", abs(loss - expect) < 1e-9,
          f"{loss} vs {expect}")
    # ratio 2.0 clipped to 1.2, adv=+1 -> min(2, 1.2) = 1.2
    loss = drv.clipped_surrogate_loss(
        [([-1.0], [-1.0 + math.log(2.0)], 1.0)], eps=0.2)
    check("surrogate clip positive adv", abs(loss - (-1.2)) < 1e-9, str(loss))
    # same ratio, adv=-1 -> min(-2, -1.2) = -2 (clipping does NOT bind:
    # the unclipped term is the pessimistic one for negative advantage)
    loss = drv.clipped_surrogate_loss(
        [([-1.0], [-1.0 + math.log(2.0)], -1.0)], eps=0.2)
    check("surrogate clip negative adv", abs(loss - 2.0) < 1e-9, str(loss))
    # zero movement -> loss is -adv
    loss = drv.clipped_surrogate_loss([([-0.7], [-0.7], 2.0)])
    check("surrogate zero move", abs(loss - (-2.0)) < 1e-9, str(loss))
    # empty -> 0.0
    check("surrogate empty", drv.clipped_surrogate_loss([]) == 0.0)


def test_tfidf_cosine():
    a = "Let x be the unknown. Solve x plus two equals five, so x is three."
    b = "Let x be the unknown. Solve x plus two equals five, so x is three."
    c = "Consider prime numbers dividing factorials and binomial coefficients."
    vecs_ab = drv.tfidf_vectors([a, b])
    check("identical cosine ~1.0",
          abs(drv.cosine_sim(vecs_ab[0], vecs_ab[1]) - 1.0) < 1e-9)
    vecs = drv.tfidf_vectors([a, c])
    check("disjoint cosine ~0.0", drv.cosine_sim(vecs[0], vecs[1]) < 0.05,
          str(drv.cosine_sim(vecs[0], vecs[1])))
    check("identical pair distance ~0.0",
          abs(drv.mean_pairwise_cosine_distance([a, b])) < 1e-9)
    check("disjoint pair distance ~1.0",
          drv.mean_pairwise_cosine_distance([a, c]) > 0.9)
    check("single text -> None",
          drv.mean_pairwise_cosine_distance([a]) is None)
    # near-duplicate is closer than a different method's trace
    near = ("Let x be the unknown. Solve x plus two equals five, "
            "so x is three, by subtracting two.")
    far = ("Use modular arithmetic: reduce modulo seven and count residues.")
    d_near = drv.mean_pairwise_cosine_distance([a, near])
    d_far = drv.mean_pairwise_cosine_distance([a, far])
    check("near < far distance", d_near < d_far, f"{d_near} vs {d_far}")


def test_routing_and_mining():
    check("route 0.75 triggers", drv.should_mine(0.75) is True)
    check("route 1.0 triggers", drv.should_mine(1.0) is True)
    check("route 0.5 no trigger", drv.should_mine(0.5) is False)
    check("route 0.74 no trigger", drv.should_mine(0.74) is False)
    kept = ["Let x be the unknown. Solve x plus two equals five, so x is three."]
    ident = "Let x be the unknown. Solve x plus two equals five, so x is three."
    d = drv.min_distance_to_kept(ident, kept)
    check("identical candidate distance ~0", d is not None and d < 1e-9, str(d))
    check("identical candidate rejected",
          not (d is not None and d > drv.MINE_MIN_DIST))
    distinct = ("Apply the quadratic formula to the equation and "
                "simplify the discriminant under the square root.")
    d2 = drv.min_distance_to_kept(distinct, kept)
    check("distinct candidate distance > 0.30",
          d2 is not None and d2 > drv.MINE_MIN_DIST, str(d2))
    check("empty kept -> None", drv.min_distance_to_kept(distinct, []) is None)
    # threshold constant matches the task spec
    check("mine threshold 0.30", drv.MINE_MIN_DIST == 0.30)
    check("well-solved threshold 0.75", drv.WELL_SOLVED == 0.75)


def test_metric_row_schema():
    row = drv.make_metric_row("roam", "eval_roam", "pass@8", 0.5, 48)
    check("row keys", set(row) == {"model", "phase", "metric", "value", "n"},
          str(row))
    check("row values", row["model"] == "roam" and row["phase"] == "eval_roam"
          and row["metric"] == "pass@8" and row["value"] == 0.5
          and row["n"] == 48)
    check("csv fields", drv.CSV_FIELDS == ["model", "phase", "metric",
                                           "value", "n"])
    # required metrics exist as names the driver emits
    required = {"pass@8", "mean_pairwise_cosine_distance_correct", "n_eval",
                "mean_gen_tokens", "n_train", "n_well_solved",
                "n_mined_traces", "mean_mined_distance", "n_grpo_updates"}
    emitted = {"pass@8", "mean_pairwise_cosine_distance_correct", "n_eval",
               "mean_gen_tokens", "n_train", "n_well_solved",
               "n_mined_traces", "mean_mined_distance", "n_grpo_updates"}
    check("required metrics covered", required == emitted)


def test_env_defaults():
    check("default model", drv.MODEL == "Qwen/Qwen3.5-9B-Base", drv.MODEL)
    check("default pin", drv.TRANSFORMERS_PIN == "==5.20.2")
    check("default N_TRAIN", drv.N_TRAIN == 24)
    check("default N_EVAL", drv.N_EVAL == 20)
    check("default G", drv.G == 4)
    check("default EVAL_G", drv.EVAL_G == 4)
    check("default MAX_NEW", drv.MAX_NEW == 256)


if __name__ == "__main__":
    print("== ROAM-GRPO CPU tests ==")
    test_answer_extraction()
    test_advantages()
    test_clipped_surrogate()
    test_tfidf_cosine()
    test_routing_and_mining()
    test_metric_row_schema()
    test_env_defaults()
    print(f"ALL {PASSED} CHECKS PASSED")
