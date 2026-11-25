"""Deterministic CPU test for writing_distortion_metrics.py.

Runnable as: python3 test_writing_distortion_cpu.py
from the session directory. Prints PASS/FAIL per check and exits
non-zero on any failure. stdlib + numpy only, no network.
"""

import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import writing_distortion_metrics as m

RESULTS = []


def check(name, ok, detail=""):
    RESULTS.append((name, bool(ok), detail))
    print(("PASS" if ok else "FAIL") + "  " + name + (("  -- " + str(detail)) if detail else ""))


def main():
    # 1. JSD of a text with itself is exactly 0.
    t = "Self driving cars are safer and more efficient than human drivers."
    check("jsd_identical_is_zero", m.unigram_jsd(t, t) == 0.0,
          "got %r" % m.unigram_jsd(t, t))

    # 2. JSD in [0, 1] and > 0 for different texts.
    a = "Self driving cars may seem too futuristic to believe."
    b = "Cooking rice requires water heat and patience every day."
    j = m.unigram_jsd(a, b)
    check("jsd_range_and_positive", 0.0 < j <= 1.0, "jsd=%.4f" % j)

    # 3. Determinism: repeat call identical.
    j1 = m.unigram_jsd(a, b)
    j2 = m.unigram_jsd(a, b)
    check("jsd_deterministic", j1 == j2, "%.6f vs %.6f" % (j1, j2))

    # 4. pronoun_rate on a hand-computed example.
    # "I am happy. We are here." -> 6 words, pronouns i, we -> 2/6*100.
    rate, count, n = m.pronoun_rate("I am happy. We are here.")
    check("pronoun_rate_hand_computed",
          count == 2 and n == 6 and abs(rate - 100.0 * 2 / 6) < 1e-9,
          "rate=%.4f count=%d n=%d" % (rate, count, n))

    # 5. pronoun_rate empty text guard.
    rate, count, n = m.pronoun_rate("")
    check("pronoun_rate_empty", (rate, count, n) == (0.0, 0, 0),
          "got %r" % ((rate, count, n),))

    # 6. adjective_rate: suffix words counted, plain words not.
    rate, count, n = m.adjective_rate("beautiful wonderful dangerous cat dog run")
    # beautiful/ful, wonderful/ful, dangerous/ous -> 3 of 6 words.
    check("adjective_rate_suffix_proxy", count == 3 and abs(rate - 50.0) < 1e-9,
          "rate=%.4f count=%d n=%d" % (rate, count, n))

    # 7. adjective_rate is a proxy, sane range on real essays.
    with open("sample16.json", encoding="utf-8") as f:
        sample = json.load(f)
    essays = sample["essays"]
    ok = True
    for e in essays:
        r, _, _ = m.adjective_rate(e["draft1"])
        if not (0.0 <= r <= 100.0):
            ok = False
    check("adjective_rate_real_range_0_100", ok, "n=%d" % len(essays))

    # 8. keyword_stance matches the recorded label AND counts for all
    # 16 sample essays on draft1.
    bad = []
    for e in essays:
        got = m.keyword_stance(e["draft1"])
        rec = e["keyword_stance"]
        if got["label"] != rec["label"] or got["pro"] != rec["pro"] or got["anti"] != rec["anti"]:
            bad.append((e["essay_id"], got, rec))
    check("keyword_stance_matches_sample16_all16", not bad,
          "mismatches=%s" % (bad,) if bad else "16/16 match")

    # 9. keyword_stance: essay 17 is pro (known sample).
    e17 = next(e for e in essays if e["essay_id"] == "17")
    got17 = m.keyword_stance(e17["draft1"])
    check("keyword_stance_essay17_pro",
          got17["label"] == "pro" and got17["score"] >= 3,
          "got %r" % (got17,))

    # 10. pct_change correctness incl. zero guard.
    ok = (m.pct_change(100, 150) == 50.0 and m.pct_change(100, 50) == -50.0
          and m.pct_change(100, 100) == 0.0 and m.pct_change(0, 0) == 0.0
          and math.isinf(m.pct_change(0, 5)))
    check("pct_change_values", ok, "50/-50/0/0/inf expected")

    # 11. mean_shift_alignment: identical shifts -> ~1.0.
    a_vecs = [[0.0, 0.0], [1.0, 2.0], [3.0, 1.0]]
    b_vecs = [[1.0, 0.0], [2.0, 2.0], [4.0, 1.0]]
    val = m.mean_shift_alignment(a_vecs, b_vecs)
    check("mean_shift_identical_shifts_near_1", abs(val - 1.0) < 1e-9,
          "got %.6f" % val)

    # 12. mean_shift_alignment: orthogonal shifts -> ~0.7071.
    val = m.mean_shift_alignment([[0.0, 0.0], [0.0, 0.0]],
                                 [[1.0, 0.0], [0.0, 1.0]])
    check("mean_shift_orthogonal_shifts", abs(val - math.sqrt(0.5)) < 1e-9,
          "got %.6f" % val)

    # 13. mean_shift_alignment: opposing shifts cancel -> low (0.0).
    val = m.mean_shift_alignment([[0.0, 0.0], [0.0, 0.0]],
                                 [[1.0, 0.0], [-1.0, 0.0]])
    check("mean_shift_opposing_shifts_low", val < 0.1,
          "got %.6f" % val)

    # 14. Human D1->D2 JSD in [0, 1] for all 16 sample essays.
    ok = True
    worst = 0.0
    for e in essays:
        j = m.unigram_jsd(e["draft1"], e["draft2"])
        worst = max(worst, j)
        if not (0.0 <= j <= 1.0):
            ok = False
    check("human_jsd_range_all16", ok, "max jsd=%.4f" % worst)

    # 15. Frontier grammar JSD in [0, 1] for all 16 x 3 models, and
    # pronoun/adjective rates in sane ranges on frontier texts.
    ok = True
    for e in essays:
        for model in ("GPT-5-MINI", "GEMINI-2.5-FLASH", "CLAUDE-HAIKU-4-5-20251001"):
            cell = e["frontier_with_feedback"]["grammar"][model]
            ai = cell["ai_text"] if isinstance(cell, dict) else cell
            j = m.unigram_jsd(e["draft1"], ai)
            pr, _, _ = m.pronoun_rate(ai)
            ar, _, _ = m.adjective_rate(ai)
            if not (0.0 <= j <= 1.0 and 0.0 <= pr <= 100.0 and 0.0 <= ar <= 100.0):
                ok = False
    check("frontier_grammar_jsd_and_rates_sane", ok, "16x3 grammar cells")

    # 16. Determinism on real text (repeat JSD + stance identical).
    e0 = essays[0]
    ok = (m.unigram_jsd(e0["draft1"], e0["draft2"])
          == m.unigram_jsd(e0["draft1"], e0["draft2"])
          and m.keyword_stance(e0["draft1"]) == m.keyword_stance(e0["draft1"]))
    check("determinism_on_real_text", ok)

    n_pass = sum(1 for _, ok, _ in RESULTS if ok)
    print("\n%d/%d checks passed" % (n_pass, len(RESULTS)))
    return 0 if n_pass == len(RESULTS) else 1


if __name__ == "__main__":
    sys.exit(main())
