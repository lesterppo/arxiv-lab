"""Local baseline analysis for arXiv:2603.18161v2 (writing distortion).

Reads essays_llm.csv + sample16.json and computes, for the SAME
16 essays:
  (a) human baseline D1->D2: mean unigram JSD, mean pronoun-rate
      % change, mean adjective-rate % change, keyword-stance flip
      count (label changes D1->D2);
  (b) frontier baselines per model x mode (grammar/minimal/general,
      "With Feedback" rows only): same metrics D1->ai_text, plus
      stance flip counts.

Prints a readable table to stdout and writes baselines.json
(means + per-essay values + n=16). stdlib + numpy only, no network.

Usage: python3 analyze_baselines.py   (from the session directory)
"""

import csv
import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import writing_distortion_metrics as m

HERE = os.path.dirname(os.path.abspath(__file__))
MODELS = ["GPT-5-MINI", "GEMINI-2.5-FLASH", "CLAUDE-HAIKU-4-5-20251001"]
MODES = ["grammar", "minimal", "general"]


def _metrics(before, after):
    """Per-pair distortion metrics dict."""
    jsd = m.unigram_jsd(before, after)
    pr_b, _, _ = m.pronoun_rate(before)
    pr_a, _, _ = m.pronoun_rate(after)
    ar_b, _, _ = m.adjective_rate(before)
    ar_a, _, _ = m.adjective_rate(after)
    st_b = m.keyword_stance(before)
    st_a = m.keyword_stance(after)
    return {
        "jsd": jsd,
        "pronoun_pct_change": m.pct_change(pr_b, pr_a),
        "adjective_pct_change": m.pct_change(ar_b, ar_a),
        "pronoun_before": pr_b,
        "pronoun_after": pr_a,
        "adjective_before": ar_b,
        "adjective_after": ar_a,
        "stance_before": st_b["label"],
        "stance_after": st_a["label"],
        "stance_flip": 1 if st_b["label"] != st_a["label"] else 0,
    }


def _mean_finite(values):
    vals = [v for v in values if math.isfinite(v)]
    if not vals:
        return None
    return sum(vals) / len(vals)


def _summarize(per_essay):
    return {
        "n": len(per_essay),
        "mean_jsd": _mean_finite([p["jsd"] for p in per_essay]),
        "mean_pronoun_pct_change": _mean_finite(
            [p["pronoun_pct_change"] for p in per_essay]),
        "mean_adjective_pct_change": _mean_finite(
            [p["adjective_pct_change"] for p in per_essay]),
        "stance_flips": sum(p["stance_flip"] for p in per_essay),
    }


def main():
    with open(os.path.join(HERE, "sample16.json"), encoding="utf-8") as f:
        sample = json.load(f)
    essays = sample["essays"]
    ids = [e["essay_id"] for e in essays]

    # Frontier "With Feedback" rows from the CSV, keyed (essay_id, model, mode).
    frontier = {}
    with open(os.path.join(HERE, "essays_llm.csv"), encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["essay_id"] in set(ids) and row["feedback"] == "With Feedback":
                frontier[(row["essay_id"], row["llm_model"], row["mode"])] = row["ai_text"]
    missing = [(eid, mo, md) for eid in ids for mo in MODELS for md in MODES
               if (eid, mo, md) not in frontier]
    if missing:
        print("WARNING: %d frontier cells missing from CSV: %s"
              % (len(missing), missing[:5]), file=sys.stderr)

    results = {"n": len(essays), "essay_ids": ids}

    # (a) Human baseline D1 -> D2.
    human_per = []
    for e in essays:
        rec = _metrics(e["draft1"], e["draft2"])
        rec["essay_id"] = e["essay_id"]
        human_per.append(rec)
    results["human_d1_to_d2"] = {
        "summary": _summarize(human_per),
        "per_essay": human_per,
    }

    # (b) Frontier baselines per model x mode.
    results["frontier_with_feedback"] = {}
    for model in MODELS:
        for mode in MODES:
            per = []
            for e in essays:
                ai = frontier.get((e["essay_id"], model, mode))
                if ai is None:
                    continue
                rec = _metrics(e["draft1"], ai)
                rec["essay_id"] = e["essay_id"]
                per.append(rec)
            results["frontier_with_feedback"]["%s/%s" % (model, mode)] = {
                "summary": _summarize(per),
                "per_essay": per,
            }

    # Print table.
    def fmt(x):
        return ("%.4f" % x) if x is not None else "n/a"

    print("Writing-distortion baselines (n=%d essays, arXiv:2603.18161v2)" % len(essays))
    print("Human = draft1 -> draft2 (expert-feedback revision).")
    print("Frontier = draft1 -> ai_text, 'With Feedback' rows only.")
    print("JSD = unigram Jensen-Shannon divergence (0..1); pronoun/adjective")
    print("  = mean %% change in rate per 100 words; flips = stance-label changes.")
    print("")
    print("%-38s %8s %12s %12s %6s" % ("condition", "mean JSD", "pronoun %", "adjective %", "flips"))
    print("-" * 80)
    h = results["human_d1_to_d2"]["summary"]
    print("%-38s %8s %12s %12s %6d" % (
        "HUMAN D1->D2", fmt(h["mean_jsd"]),
        fmt(h["mean_pronoun_pct_change"]), fmt(h["mean_adjective_pct_change"]),
        h["stance_flips"]))
    for model in MODELS:
        for mode in MODES:
            key = "%s/%s" % (model, mode)
            s = results["frontier_with_feedback"][key]["summary"]
            print("%-38s %8s %12s %12s %6d" % (
                key, fmt(s["mean_jsd"]),
                fmt(s["mean_pronoun_pct_change"]),
                fmt(s["mean_adjective_pct_change"]), s["stance_flips"]))

    out_path = os.path.join(HERE, "baselines.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=1, ensure_ascii=False)
    print("")
    print("Wrote %s" % out_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
