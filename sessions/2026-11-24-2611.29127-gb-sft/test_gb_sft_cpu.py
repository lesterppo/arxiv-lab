"""Deterministic CPU test for arXiv:2611.29127 GB-SFT (Group-Balanced SFT).

Tests the loss-weighting formula (paper Eqs. 4-5) and the synthetic
knowledge-injection benchmark helpers, all stdlib-only:

  W0  Hand-computed weights on a 2-span example at each corner.
  W1  Weight sums to 1 for random (alpha, beta) incl. negative alpha.
  W2  (1, 1) reduces to uniform SFT; (0, 1) gives each class 1/K;
      (1, 0) gives each span 1/P.
  W3  Housing-QA-like shares (citation 814 / answer 51 / prose 135
      tokens per 1000): uniform SFT gives citation 81.4% of the
      budget; GB-class gives each class 1/3 (payload budget x6.5).
  W4  Driver vendored copy == src module (exec the driver's PURE
      HELPERS section, compare on 200 random span lists).
  W5  Benchmark generation: deterministic per seed, unique names,
      word shares (payload <= 8%, citation >= 55%), scoring helpers.

Run:  python3 test_gb_sft_cpu.py   (from this session dir)
Writes results.json next to this file. Exits non-zero on any failure.
"""

from __future__ import annotations

import json
import math
import os
import random
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from arxiv_lab.training import gb_sft

SEED = 20261124


def check(cond, msg):
    if not cond:
        raise AssertionError(msg)
    print(f"  ok: {msg}")


def approx(a, b, tol=1e-12):
    return abs(a - b) <= tol


def main():
    t0 = time.time()
    results = {"seed": SEED, "experiments": {}}

    # ---- W0: hand-computed 2-span example -------------------------
    print("W0 hand-computed weights")
    # spans [(A, 4), (B, 2)].
    # (0, 1): u=[4,2], m_A=4, m_B=2, M=2. w_A = (1/2)(4/4)(1/4)=1/8,
    # w_B = (1/2)(2/2)(1/2)=1/4. Budgets A=0.5, B=0.5.
    w = gb_sft.gb_weights([("A", 4), ("B", 2)], 0.0, 1.0)
    check(all(approx(x, 0.125) for x in w[:4]), "(0,1) A tokens weigh 1/8")
    check(all(approx(x, 0.25) for x in w[4:]), "(0,1) B tokens weigh 1/4")
    check(approx(sum(w), 1.0), "weights sum to 1")
    # (1, 0): u=[1,1], m=1,1, M=2. w_A=(1/2)(1/1)(1/4)=1/8,
    # w_B=(1/2)(1/1)(1/2)=1/4. Span budgets 0.5 / 0.5.
    w = gb_sft.gb_weights([("A", 4), ("B", 2)], 1.0, 0.0)
    check(all(approx(x, 0.125) for x in w[:4]), "(1,0) A tokens weigh 1/8")
    check(all(approx(x, 0.25) for x in w[4:]), "(1,0) B tokens weigh 1/4")
    # (1, 1): uniform 1/6.
    w = gb_sft.gb_weights([("A", 4), ("B", 2)], 1.0, 1.0)
    check(all(approx(x, 1 / 6) for x in w), "(1,1) uniform 1/6")
    results["experiments"]["W0_hand_computed"] = {"passed": True}

    # ---- W1: sums to 1, random (alpha, beta) -----------------------
    print("W1 weights sum to 1 across (alpha, beta)")
    rng = random.Random(SEED)
    classes = ["payload", "citation", "prose"]
    for trial in range(200):
        spans = [(rng.choice(classes), rng.randint(1, 500)) for _ in range(rng.randint(1, 6))]
        alpha = rng.choice([-0.5, 0.0, 0.5, 1.0, 2.0])
        beta = rng.choice([0.0, 0.5, 1.0])
        w = gb_sft.gb_weights(spans, alpha, beta)
        check(approx(sum(w), 1.0, 1e-9), f"sum=1 trial {trial} (a={alpha}, b={beta})")
        check(all(x > 0 for x in w), f"all weights positive trial {trial}")
        check(len(w) == sum(n for _c, n in spans), "one weight per token")
    results["experiments"]["W1_sum_to_one"] = {"trials": 200}

    # ---- W2: corners ----------------------------------------------
    print("W2 corner budgets")
    spans = [("payload", 51), ("citation", 814), ("prose", 135),
             ("citation", 200), ("prose", 40)]
    cb = gb_sft.class_budgets(spans, 1.0, 1.0)
    check(approx(cb["citation"], 1014 / 1240, 1e-9), "(1,1) citation budget = token share")
    cb = gb_sft.class_budgets(spans, 0.0, 1.0)
    check(all(approx(v, 1 / 3, 1e-9) for v in cb.values()),
          "(0,1) each class gets 1/3")
    sb = gb_sft.span_budgets(spans, 1.0, 0.0)
    check(all(approx(v, 1 / 5, 1e-9) for v in sb),
          "(1,0) each span gets 1/5")
    # alpha=-0.5 reverse-weights small classes (paper guardrail arm).
    cb = gb_sft.class_budgets(spans, -0.5, 1.0)
    check(cb["payload"] > 1 / 3 > cb["citation"],
          "alpha=-0.5 gives the smallest class more than 1/3")
    results["experiments"]["W2_corners"] = {
        "uniform_citation_budget": round(1014 / 1240, 4),
        "gb_class_budgets": {k: round(v, 4) for k, v in
                             gb_sft.class_budgets(spans, 0.0, 1.0).items()},
    }

    # ---- W3: Housing-QA-like shares -------------------------------
    print("W3 Housing-QA-like imbalance")
    hq = [("citation", 814), ("answer", 51), ("prose", 135)]
    uni = gb_sft.class_budgets(hq, 1.0, 1.0)
    gbc = gb_sft.class_budgets(hq, 0.0, 1.0)
    check(approx(uni["citation"], 0.814, 1e-9), "uniform: citation 81.4% of budget")
    check(approx(uni["answer"], 0.051, 1e-9), "uniform: answer 5.1% of budget")
    check(all(approx(v, 1 / 3, 1e-9) for v in gbc.values()),
          "GB-class: each class 1/3")
    check(approx(gbc["answer"] / uni["answer"], 1 / 3 / 0.051, 1e-6),
          "GB-class boosts the answer budget ~6.5x")
    results["experiments"]["W3_housing_qa_shares"] = {
        "uniform": {k: round(v, 4) for k, v in uni.items()},
        "gb_class": {k: round(v, 4) for k, v in gbc.items()},
        "answer_boost": round(gbc["answer"] / uni["answer"], 2),
    }

    # ---- W4: driver vendored copy == src ---------------------------
    print("W4 driver vendored copy matches src")
    driver = os.path.join(os.path.dirname(__file__), "colab_gb_sft.py")
    src_text = open(driver).read()
    start = src_text.index("# ---------------- PURE HELPERS (stdlib) ----------------")
    end = src_text.index("# ---------------- END PURE HELPERS ----------------")
    helpers = src_text[start:end]
    ns = {}
    exec(helpers, ns)
    d_gb = ns["gb_weights"]
    rng = random.Random(SEED + 7)
    for trial in range(200):
        spans = [(rng.choice(classes), rng.randint(1, 300)) for _ in range(rng.randint(1, 5))]
        alpha = rng.choice([-0.5, 0.0, 1.0])
        beta = rng.choice([0.0, 1.0])
        a = gb_sft.gb_weights(spans, alpha, beta)
        b = d_gb(spans, alpha, beta)
        check(len(a) == len(b) and all(approx(x, y, 1e-15) for x, y in zip(a, b)),
              f"driver copy identical trial {trial}")
    results["experiments"]["W4_driver_copy"] = {"trials": 200, "identical": True}

    # ---- W5: benchmark generation + scoring -----------------------
    print("W5 benchmark helpers (driver PURE HELPERS)")
    rng = random.Random(SEED)
    used = set()
    facts1 = [ns["build_fact"](rng, used) for _ in range(160)]
    rng2 = random.Random(SEED)
    used2 = set()
    facts2 = [ns["build_fact"](rng2, used2) for _ in range(160)]
    check(facts1 == facts2, "fact generation deterministic per seed")
    check(len({f["name"] for f in facts1}) == 160, "all drug names unique")
    check(all(f["dose"] and f["code"].startswith("FD-") for f in facts1),
          "facts carry dose + label code")
    shares, nwords = ns["word_share"](ns["render_document"](facts1[0]))
    print(f"  example doc words={nwords} word-share={shares}")
    check(shares["payload"] <= 0.08, "payload is a small share of the doc")
    check(shares["citation"] >= 0.55, "citation dominates the doc")
    check(shares["payload"] + shares["citation"] + shares["prose"] > 0.99,
          "shares cover the doc")
    f0 = facts1[0]
    check(ns["dose_matches"]("The dose is %s." % f0["value"], f0["value"]),
          "dose_matches finds the gold value")
    check(not ns["dose_matches"]("The dose is 999 mg once daily.", f0["value"]),
          "dose_matches rejects a wrong value")
    check(ns["first_dose_matches"]("Recommended: %s. Other text." % f0["value"], f0["value"]),
          "first_dose_matches accepts gold first")
    check(not ns["first_dose_matches"]("Recommended: 999 mg once daily. Also %s." % f0["value"],
                                       f0["value"]),
          "first_dose_matches rejects a wrong first dose")
    check(ns["citation_matches"]("See %s for details." % f0["code"], f0["code"]),
          "citation_matches finds the code")
    check(not ns["citation_matches"]("See FD-0000-XX.", f0["code"]),
          "citation_matches rejects a wrong code")
    results["experiments"]["W5_benchmark"] = {
        "n_facts": 160, "unique_names": 160,
        "example_word_share": shares, "example_words": nwords,
    }

    # ---- edge cases ------------------------------------------------
    print("W6 edge cases")
    w = gb_sft.gb_weights([("only", 7)], 0.0, 1.0)
    check(all(approx(x, 1 / 7) for x in w), "single span: uniform")
    try:
        gb_sft.gb_weights([], 0.0, 1.0)
        check(False, "empty span list raises")
    except ValueError:
        check(True, "empty span list raises")
    try:
        gb_sft.gb_weights([("a", 0)], 0.0, 1.0)
        check(False, "zero-length span raises")
    except ValueError:
        check(True, "zero-length span raises")
    results["experiments"]["W6_edge_cases"] = {"all_passed": True}

    results["elapsed_s"] = round(time.time() - t0, 2)
    out = os.path.join(os.path.dirname(__file__), "results.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=1)
    print(f"\nALL CHECKS PASSED in {results['elapsed_s']} s — results in {out}")


if __name__ == "__main__":
    main()
