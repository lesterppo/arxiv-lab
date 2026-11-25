#!/usr/bin/env python3
"""CPU-side analysis of the writing-distortion Colab results.

Reads writing_distortion_results.json (+ sample16.json) and computes, with
the session metrics module (writing_distortion_metrics.py, stdlib+numpy):
  per model x mode (and human + frontier baselines): mean unigram JSD,
  mean pronoun/adjective rate % change, keyword-stance flips, LLM-judge
  stance flips (parsed only), judge/keyword agreement, mean word counts;
  peer-review leg: parsed SCOREs + clarity/significance vs
  reproducibility/scalability mention counts.
Also verifies the Colab cosine distances from the stored embeddings.
Usage: python3 analyze_qwen.py [results.json] -> prints table, writes
analysis.json next to the session dir files.
"""
import json, os, re, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from writing_distortion_metrics import (
    unigram_jsd, pronoun_rate, adjective_rate, keyword_stance, pct_change)

RESULTS_PATH = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "writing_distortion_results.json")
SAMPLE_PATH = os.path.join(HERE, "sample16.json")

results = json.load(open(RESULTS_PATH))
sample = json.load(open(SAMPLE_PATH))
ESSAYS = {e["essay_id"]: e for e in sample["essays"]}
MODES = ["grammar", "minimal", "general"]
FRONTIER = ["GPT-5-MINI", "GEMINI-2.5-FLASH", "CLAUDE-HAIKU-4-5-20251001"]

def rates(text):
    pr, _, nw = pronoun_rate(text)
    ar, _, _ = adjective_rate(text)
    return pr, ar, nw

def summarize(pairs):
    """pairs: list of (before_text, after_text). Returns means + flips."""
    jsds, prs, ars, kflips, nws_b, nws_a = [], [], [], 0, [], []
    for b, a in pairs:
        jsds.append(unigram_jsd(b, a))
        pb, ab, nb = rates(b)
        pa, aa, na = rates(a)
        prs.append(pct_change(pb, pa))
        ars.append(pct_change(ab, aa))
        nws_b.append(nb); nws_a.append(na)
        if keyword_stance(b)["label"] != keyword_stance(a)["label"]:
            kflips += 1
    n = len(pairs)
    return {"n": n, "mean_jsd": round(sum(jsds) / n, 4),
            "mean_pronoun_pct": round(sum(prs) / n, 2),
            "mean_adjective_pct": round(sum(ars) / n, 2),
            "keyword_flips": kflips,
            "mean_words_before": round(sum(nws_b) / n, 1),
            "mean_words_after": round(sum(nws_a) / n, 1)}

def judge_stance(key):
    e = results.get("judge", {}).get(key, {})
    p = e.get("parsed")
    return p.get("stance") if isinstance(p, dict) else None

def judge_flips(prefix_fn, n_expected=16):
    """prefix_fn(essay_id) -> judge key for the 'after' text."""
    flips, counted, missing = 0, 0, []
    for eid in ESSAYS:
        s0 = judge_stance(f"d1:{eid}")
        s1 = judge_stance(prefix_fn(eid))
        if s0 is None or s1 is None:
            missing.append(eid)
            continue
        counted += 1
        if s0 != s1:
            flips += 1
    return {"flips": flips, "counted": counted, "missing_essay_ids": missing}

# --- judge vs keyword agreement (computed below from judge keys) ---

out = {"conditions": {}, "judge": {}, "peer_reviews": {}, "cosine_check": {}}

# --- human baseline ---
pairs = [(e["draft1"], e["draft2"]) for e in ESSAYS.values()]
out["conditions"]["human_d2"] = summarize(pairs)
out["judge"]["human_d2"] = judge_flips(lambda eid: f"d2:{eid}")

# --- frontier baselines ---
for fm in FRONTIER:
    for mode in MODES:
        pairs = [(ESSAYS[eid]["draft1"], ESSAYS[eid]["frontier_with_feedback"][mode][fm])
                 for eid in ESSAYS]
        out["conditions"][f"frontier_{fm}_{mode}"] = summarize(pairs)
        out["judge"][f"frontier_{fm}_{mode}"] = judge_flips(
            lambda eid, fm=fm, mode=mode: f"frontier:{fm}:{mode}:{eid}")

# --- Qwen generations ---
for slug, modes in results.get("generations", {}).items():
    for mode in MODES:
        d = modes.get(mode, {})
        pairs = [(ESSAYS[eid]["draft1"], d[eid]) for eid in ESSAYS if eid in d]
        if not pairs:
            continue
        out["conditions"][f"gen_{slug}_{mode}"] = summarize(pairs)
        out["judge"][f"gen_{slug}_{mode}"] = judge_flips(
            lambda eid, slug=slug, mode=mode: f"gen:{slug}:{mode}:{eid}")

# --- judge vs keyword agreement on generated + frontier + human texts ---
def text_for_key(key):
    kind, *rest = key.split(":")
    if kind == "d1":
        return ESSAYS[rest[0]]["draft1"]
    if kind == "d2":
        return ESSAYS[rest[0]]["draft2"]
    if kind == "frontier":
        fm, mode, eid = rest
        return ESSAYS[eid]["frontier_with_feedback"][mode][fm]
    if kind == "gen":
        slug, mode, eid = rest
        return results["generations"][slug][mode][eid]
    return None

agree, total = 0, 0
agree_by_kind = {}
for key, e in results.get("judge", {}).items():
    p = e.get("parsed")
    if not isinstance(p, dict) or "stance" not in p:
        continue
    t = text_for_key(key)
    if t is None:
        continue
    k = keyword_stance(t)["label"]
    j = {"for": "pro", "against": "anti", "neutral": "mixed"}.get(p["stance"])
    kind = key.split(":")[0]
    a, tt = agree_by_kind.get(kind, (0, 0))
    agree_by_kind[kind] = (a + (1 if j == k else 0), tt + 1)
    total += 1
    agree += (1 if j == k else 0)
out["judge"]["keyword_agreement"] = {
    "agree": agree, "total": total,
    "rate": round(agree / total, 3) if total else None,
    "by_kind": {k: {"agree": a, "total": t, "rate": round(a / t, 3) if t else None}
                for k, (a, t) in agree_by_kind.items()}}

# --- peer-review leg ---
CLAR = ["clarity", "clear", "significance", "significant", "important", "importance"]
REPRO = ["reproducib", "reproducibility", "scalab", "scale", "scaling"]
for slug, revs in results.get("peer_reviews", {}).items():
    scores = {}
    clar_n, repro_n = 0, 0
    for k, txt in revs.items():
        m = re.search(r"SCORE:\s*(\d+)", txt)
        scores[k] = int(m.group(1)) if m else None
        low = txt.lower()
        clar_n += sum(low.count(w) for w in CLAR)
        repro_n += sum(low.count(w) for w in REPRO)
    vals = [v for v in scores.values() if v is not None]
    out["peer_reviews"][slug] = {
        "scores": scores,
        "mean_score": round(sum(vals) / len(vals), 2) if vals else None,
        "n_scored": len(vals),
        "clarity_significance_mentions": clar_n,
        "reproducibility_scalability_mentions": repro_n}

# --- verify Colab cosine distances from stored embeddings ---
E = {k: np.array(v) for k, v in results.get("embeddings", {}).items()}
if E:
    def cos_dist(a, b):
        return float(1.0 - float(np.dot(E[a], E[b])))
    for label, pair_fn in [
            ("human_d2", lambda eid: (f"d1:{eid}", f"d2:{eid}")),
    ]:
        ds = [cos_dist(*pair_fn(eid)) for eid in ESSAYS]
        out["cosine_check"][label] = round(sum(ds) / len(ds), 4)
    for slug, modes in results.get("generations", {}).items():
        for mode in MODES:
            pairs = [(f"d1:{eid}", f"gen:{slug}:{mode}:{eid}") for eid in ESSAYS]
            if all(p[1] in E for p in pairs):
                ds = [cos_dist(*p) for p in pairs]
                out["cosine_check"][f"gen_{slug}_{mode}"] = round(sum(ds) / len(ds), 4)
    out["cosine_check"]["colab_values"] = {
        k: v["mean_cos_dist"] for k, v in results.get("cosine", {}).items()}

with open(os.path.join(HERE, "analysis.json"), "w") as f:
    json.dump(out, f, indent=1)

print(f"{'condition':38s} {'JSD':>7s} {'pron%':>7s} {'adj%':>7s} {'kwFlip':>6s} {'jFlips':>12s} {'words':>11s}")
for name, s in out["conditions"].items():
    j = out["judge"].get(name, {})
    jf = f"{j.get('flips', '?')}/{j.get('counted', '?')}" if j else "-"
    print(f"{name:38s} {s['mean_jsd']:7.4f} {s['mean_pronoun_pct']:7.2f} {s['mean_adjective_pct']:7.2f} "
          f"{s['keyword_flips']:6d} {jf:>12s} {s['mean_words_before']:5.0f}->{s['mean_words_after']:<5.0f}")
print("judge/keyword agreement:", out["judge"]["keyword_agreement"])
print("peer reviews:", json.dumps(out["peer_reviews"], indent=1))
print("cosine check (local recompute vs colab):", json.dumps(out["cosine_check"], indent=1))
print("wrote analysis.json")
