"""CPU test for GrepMem (arXiv:2611.25123) — deterministic, stdlib only.

Reproduces the paper's core claim at bounded scale: an append-only
plain-text log + regex grep recall (GrepMem) is lossless — accuracy near
the full-history ceiling at a fraction of the token cost — while lossy
baselines (summarization, recency truncation, random masking) collapse.

Run: python3 tests/test_grepmem_cpu.py
"""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from arxiv_lab.agent import grepmem


def main() -> dict:
    out: dict = {}

    # --- Experiment 1: main comparison, 5 seeds x 20 questions, 500 lines.
    suite = grepmem.run_suite(seeds=range(5), n_lines=500, n_questions=20)
    out["main_suite"] = suite
    r = suite["results"]
    print("== main suite: 5 seeds x 20 questions, 500-line streams ==")
    for name in ("full", "grepmem", "summary", "recency", "random_mask"):
        res = r[name]
        print(f"  {name:12s} acc={res['accuracy']:.3f} "
              f"ctx_lines={res['mean_context_lines']:7.1f} "
              f"para_acc={res['paraphrase_accuracy']:.3f} (n={res['paraphrase_n']})")

    # Paper's directional claims, as assertions on the measured numbers:
    # (a) GrepMem near the FULL ceiling (paper: 95-97%, essentially lossless).
    assert r["grepmem"]["accuracy"] >= 0.90, r["grepmem"]
    assert r["full"]["accuracy"] - r["grepmem"]["accuracy"] <= 0.05, r
    # (b) Lossy baselines collapse (paper: at most 21% on ContextBench).
    best_lossy = max(r["summary"]["accuracy"], r["recency"]["accuracy"],
                     r["random_mask"]["accuracy"])
    assert best_lossy <= 0.60, {"best_lossy": best_lossy}
    assert r["grepmem"]["accuracy"] - best_lossy >= 0.25, r
    # (c) GrepMem stays on the accuracy-token-cost Pareto frontier.
    assert r["grepmem"]["mean_context_lines"] <= 0.15 * r["full"]["mean_context_lines"], r
    # (d) Early facts survive: GrepMem early-position accuracy near ceiling.
    assert r["grepmem"]["by_position"]["early"] >= 0.85, r["grepmem"]
    assert r["recency"]["by_position"]["early"] <= 0.40, r["recency"]

    # --- Experiment 2: scaling — stream length 200 -> 2000 lines.
    print("== scaling: stream length sweep ==")
    scaling = {}
    for n_lines in (200, 500, 1000, 2000):
        s = grepmem.run_suite(seeds=range(5), n_lines=n_lines, n_questions=20)
        scaling[n_lines] = {k: round(v["accuracy"], 4)
                            for k, v in s["results"].items()}
        print(f"  lines={n_lines:5d} " +
              "  ".join(f"{k}={v:.3f}" for k, v in scaling[n_lines].items()))
    out["scaling"] = scaling
    assert scaling[2000]["grepmem"] >= scaling[200]["grepmem"] - 0.05, scaling
    assert scaling[2000]["recency"] <= scaling[200]["recency"] - 0.15, scaling

    # --- Experiment 3: grep budget sweep (accuracy vs retrieved lines).
    print("== grep budget sweep (max_lines) ==")
    budget = {}
    sessions = [grepmem.generate_session(s, n_lines=500, n_questions=20)
                for s in range(5)]
    for k in (5, 10, 20, 40):
        res = grepmem.evaluate_policy("grepmem", sessions, max_lines=k)
        budget[k] = {"accuracy": round(res["accuracy"], 4),
                     "mean_context_lines": round(res["mean_context_lines"], 1)}
        print(f"  max_lines={k:3d} acc={res['accuracy']:.3f} "
              f"ctx={res['mean_context_lines']:.1f}")
    out["grep_budget"] = budget
    assert budget[40]["accuracy"] >= 0.90, budget
    assert budget[5]["accuracy"] <= budget[40]["accuracy"], budget

    # --- Experiment 4: vocabulary mismatch (honest limit).
    # Attribute-only grep on paraphrased questions: the query vocabulary
    # ("funding") does not match the log vocabulary ("budget"), so grep
    # recall collapses. Reported, not asserted to pass — it is the
    # mechanism's known failure mode (paper's grep needs the agent to
    # issue good patterns; entity tokens save the full policy).
    print("== vocabulary mismatch: attribute-only grep, paraphrased Qs ==")
    mismatch = {}
    for label, kw in (("entity+attr", {}),
                      ("entity_only", {"entity_only": True}),
                      ("attr_only", {"attr_only": True})):
        res = grepmem.evaluate_policy("grepmem", sessions,
                                      max_lines=40, **kw)
        mismatch[label] = {"accuracy": round(res["accuracy"], 4),
                           "paraphrase_accuracy": round(res["paraphrase_accuracy"], 4)}
        print(f"  {label:12s} acc={res['accuracy']:.3f} "
              f"para_acc={res['paraphrase_accuracy']:.3f}")
    out["vocabulary_mismatch"] = mismatch

    # --- Experiment 5: long-stream recall (single fact, 1000-line stream).
    print("== long-stream recall: 1 buried fact, 1000 lines ==")
    recall = {}
    for seed in range(5):
        sess = grepmem.generate_session(seed, n_lines=1000, n_questions=1)
        for name in ("full", "grepmem", "recency", "summary"):
            kw = {"max_lines": 40} if name == "grepmem" else {}
            res = grepmem.evaluate_policy(name, [sess], **kw)
            recall.setdefault(name, []).append(res["accuracy"])
    recall = {k: round(sum(v) / len(v), 3) for k, v in recall.items()}
    print("  " + "  ".join(f"{k}={v:.3f}" for k, v in recall.items()))
    out["long_stream_recall"] = recall
    assert recall["grepmem"] >= 0.80, recall

    print("ALL CHECKS PASSED")
    return out


if __name__ == "__main__":
    results = main()
    out_path = os.path.join(os.path.dirname(__file__), "..", "sessions",
                            "2026-11-20-2611.25123-grepmem", "results.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=1)
    print(f"results -> {out_path}")
