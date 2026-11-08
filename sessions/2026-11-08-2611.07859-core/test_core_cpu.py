"""Deterministic CPU test for CORE (arXiv 2611.07859).

E1  single-turn, 60 two-hop Qs, k=5 passages:
    gold-evidence recall (both hops present) + deterministic extractive-reader
    accuracy for FULL (whole passages) vs CORE (token-sim snippets) vs
    NAIVE-SNIPPET (first sentence of each passage), plus mean context tokens.
    E1b single-hop control accuracy (30 Qs, FULL vs CORE).
E2  multi-turn, 30 two-turn trajectories: cumulative context tokens for
    FULL vs CORE (no CM) vs CORE+CM, and the CM suppression rate.
E3  snippet-quality ablation on the 60 two-hop Qs: token-similarity snippet
    vs first-sentence vs random window, on evidence recall.

Assertions (per task spec; a failure is reported honestly, not tuned away):
  A1  CORE recall within 5pp of FULL.
  A2  CORE mean tokens <= 50% of FULL.
  A3  CORE recall beats NAIVE-SNIPPET by >= 10pp.
  A4  CORE+CM cumulative tokens < CORE without CM on E2.

Run:  python3 test_core_cpu.py
Writes: results.json (same directory). Exit 1 if any assertion fails.
"""

import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from corpus import build_corpus, build_questions
from core import (
    BM25, ContextManager, SnippetExtractor, content_tokens, count_tokens,
    jaccard, tokenize, window_text,
)

K = 5
SNIPPET_WIDTH = 2
JACCARD_THRESHOLD = 0.8


# ---------------------------------------------------------------- reader
def _find_all(hay_tokens, needle_tokens):
    """Start indices where needle token sequence occurs in haystack."""
    if not needle_tokens or len(needle_tokens) > len(hay_tokens):
        return []
    out = []
    for i in range(len(hay_tokens) - len(needle_tokens) + 1):
        if hay_tokens[i:i + len(needle_tokens)] == needle_tokens:
            out.append(i)
    return out


def reader_answer(context, full_name, last_name, institutions):
    """Deterministic extractive reader: pick the institution string whose
    closest mention is nearest (in tokens) to the researcher's full name
    (fallback: last name). Score 1/(1+min_distance); absent -> 0."""
    ctx = tokenize(context)
    name_pos = _find_all(ctx, tokenize(full_name))
    if not name_pos:
        name_pos = _find_all(ctx, tokenize(last_name))
    if not name_pos:
        return None
    best, best_score = None, -1.0
    for inst in institutions:
        ipos = _find_all(ctx, tokenize(inst))
        if not ipos:
            continue
        d = min(abs(a - b) for a in ipos for b in name_pos)
        score = 1.0 / (1.0 + d)
        if score > best_score:  # strict > keeps first on ties (deterministic)
            best, best_score = inst, score
    return best


def recall_both(context, entity):
    """Both gold evidence sentences present in the returned context?"""
    return (entity["gold_hop1"] in context) and (entity["gold_hop2"] in context)


# ---------------------------------------------------------------- E1
def run_e1(bm25, extractor, entities, two_hop, institutions):
    rows = []
    for q in two_hop:
        e = q["entity"]
        ranked = bm25.search(q["text"], k=K)
        passages = [(entities[i]["id"], entities[i]) for i, _ in ranked]
        full_ctx = " ".join(p["passage"] for _, p in passages)
        core_snips, naive_snips = [], []
        for _, p in passages:
            _, s = extractor.best_window(p["sentences"], q["text"], SNIPPET_WIDTH)
            core_snips.append(s)
            # NAIVE-SNIPPET per spec: the FIRST SENTENCE only (width 1),
            # not the first 2-sentence window.
            _, s2 = extractor.first_window(p["sentences"], 1)
            naive_snips.append(s2)
        core_ctx = " ".join(core_snips)
        naive_ctx = " ".join(naive_snips)
        row = {"qid": q["qid"], "entity": e["id"]}
        for name, ctx in (("FULL", full_ctx), ("CORE", core_ctx),
                          ("NAIVE", naive_ctx)):
            row[name] = {
                "recall_both": recall_both(ctx, e),
                "answer": reader_answer(ctx, e["name"], e["last"], institutions),
                "tokens": count_tokens(ctx),
            }
            row[name]["correct"] = row[name]["answer"] == e["institution"]
        rows.append(row)

    def agg(name):
        n = len(rows)
        return {
            "recall_both": sum(r[name]["recall_both"] for r in rows) / n,
            "accuracy": sum(r[name]["correct"] for r in rows) / n,
            "mean_tokens": sum(r[name]["tokens"] for r in rows) / n,
        }

    return {"FULL": agg("FULL"), "CORE": agg("CORE"), "NAIVE": agg("NAIVE"),
            "n": len(rows)}


def run_e1b(bm25, extractor, entities, single_hop, institutions):
    out = {}
    for name in ("FULL", "CORE"):
        correct, toks = 0, 0
        for q in single_hop:
            e = q["entity"]
            ranked = bm25.search(q["text"], k=K)
            if name == "FULL":
                ctx = " ".join(entities[i]["passage"] for i, _ in ranked)
            else:
                snips = []
                for i, _ in ranked:
                    _, s = extractor.best_window(
                        entities[i]["sentences"], q["text"], SNIPPET_WIDTH)
                    snips.append(s)
                ctx = " ".join(snips)
            if reader_answer(ctx, e["name"], e["last"], institutions) == e["institution"]:
                correct += 1
            toks += count_tokens(ctx)
        out[name] = {"accuracy": correct / len(single_hop),
                     "mean_tokens": toks / len(single_hop)}
    out["n"] = len(single_hop)
    return out


# ---------------------------------------------------------------- E2
def run_e2(bm25, extractor, entities, two_hop):
    """30 two-turn trajectories. Turn 1: award+year+field -> researcher.
    Turn 2: institution of that researcher. Compare cumulative tokens
    FULL vs CORE (no CM) vs CORE+CM; report CM suppression rate."""
    trajs = two_hop[:30]
    full_cum, core_cum, cm_cum = 0, 0, 0
    n_seen_slots, n_suppressed, n_expanded = 0, 0, 0
    cm_evidence = 0  # turn-2 gold hop2 sentence present in cumulative CM context
    core_evidence = 0
    for q in trajs:
        e = q["entity"]
        t1 = (f"Which researcher won the {e['award']} in {e['year']} "
              f"in the field of {e['field']}?")
        t2 = (f"Which institution is {e['name']}, the {e['field']} "
              f"researcher, affiliated with?")
        r1 = bm25.search(t1, k=K)
        r2 = bm25.search(t2, k=K)

        full_cum += sum(count_tokens(entities[i]["passage"]) for i, _ in r1)
        full_cum += sum(count_tokens(entities[i]["passage"]) for i, _ in r2)

        cm = ContextManager(jaccard_threshold=JACCARD_THRESHOLD)
        t1_snips = {}
        t1_tokens = 0
        for i, _ in r1:
            _, s = extractor.best_window(entities[i]["sentences"], t1, SNIPPET_WIDTH)
            t1_snips[entities[i]["id"]] = s
            cm.register(entities[i]["id"], s)
            t1_tokens += count_tokens(s)
        core_cum += t1_tokens
        cm_cum += t1_tokens

        t2_snips, t2_adds = [], []
        for i, _ in r2:
            p = entities[i]
            _, fresh = extractor.best_window(p["sentences"], t2, SNIPPET_WIDTH)
            t2_snips.append(fresh)
            action, text = cm.turn2_decision(p["id"], fresh, p["gold_hop2"])
            if cm.was_seen(p["id"]):
                n_seen_slots += 1
                if action == "suppress":
                    n_suppressed += 1
                elif action == "expand":
                    n_expanded += 1
            if action in ("fresh", "expand"):
                t2_adds.append(text)
        core_cum += sum(count_tokens(s) for s in t2_snips)
        cm_cum += sum(count_tokens(s) for s in t2_adds)

        cm_ctx = " ".join(list(t1_snips.values()) + t2_adds)
        core_ctx = " ".join(list(t1_snips.values()) + t2_snips)
        if e["gold_hop2"] in cm_ctx:
            cm_evidence += 1
        if e["gold_hop2"] in core_ctx:
            core_evidence += 1

    n = len(trajs)
    return {
        "n": n,
        "FULL_cum_tokens": full_cum / n,
        "CORE_noCM_cum_tokens": core_cum / n,
        "CORE_CM_cum_tokens": cm_cum / n,
        "CM_token_ratio_vs_FULL": (cm_cum / n) / (full_cum / n),
        "suppression_rate": (n_suppressed / n_seen_slots) if n_seen_slots else 0.0,
        "n_seen_slots": n_seen_slots,
        "n_suppressed": n_suppressed,
        "n_expanded": n_expanded,
        "turn2_evidence_coverage_CORE_noCM": core_evidence / n,
        "turn2_evidence_coverage_CORE_CM": cm_evidence / n,
    }


# ---------------------------------------------------------------- E3
def run_e3(bm25, extractor, entities, two_hop):
    """Snippet strategy ablation on evidence recall (both hops)."""
    rng = random.Random(7)
    strategies = {"token_sim": 0, "first_sentence": 0, "random_window": 0}
    n = len(two_hop)
    for q in two_hop:
        e = q["entity"]
        ranked = bm25.search(q["text"], k=K)
        for name in strategies:
            parts = []
            for i, _ in ranked:
                sents = entities[i]["sentences"]
                if name == "token_sim":
                    _, s = extractor.best_window(sents, q["text"], SNIPPET_WIDTH)
                elif name == "first_sentence":
                    # first sentence only (width 1), per spec
                    _, s = extractor.first_window(sents, 1)
                else:
                    start = rng.randrange(0, len(sents) - SNIPPET_WIDTH + 1)
                    s = window_text(sents, list(range(start, start + SNIPPET_WIDTH)))
                parts.append(s)
            if recall_both(" ".join(parts), e):
                strategies[name] += 1
    return {k: v / n for k, v in strategies.items()} | {"n": n}


# ---------------------------------------------------------------- main
def main():
    entities = build_corpus()
    two_hop, single_hop = build_questions(entities)
    institutions = sorted({e["institution"] for e in entities})
    all_sentences = [s for e in entities for s in e["sentences"]]
    bm25 = BM25([e["passage"] for e in entities])
    extractor = SnippetExtractor(all_sentences)

    e1 = run_e1(bm25, extractor, entities, two_hop, institutions)
    e1b = run_e1b(bm25, extractor, entities, single_hop, institutions)
    e2 = run_e2(bm25, extractor, entities, two_hop)
    e3 = run_e3(bm25, extractor, entities, two_hop)

    # ---- verdict per task guidance
    # Parent-review cap (2026-11-08): the numbers meet the mechanical
    # VERIFIED bar, but the verdict is capped at SUPPORTED. The corpus
    # fixes both gold sentences at positions 0-1 of every passage, so
    # the test cannot separate token-similarity scoring from positional
    # bias (a first-2-sentences baseline would also score recall 1.000),
    # and the ranking components are BM25/TF-IDF proxies, not the paper's
    # dense retriever + late-interaction reranker. That is a bounded-scale
    # directional reproduction (SUPPORTED), not an end-to-end verification.
    rec_gap = e1["FULL"]["recall_both"] - e1["CORE"]["recall_both"]
    acc_gap = e1["FULL"]["accuracy"] - e1["CORE"]["accuracy"]
    tok_ratio = e1["CORE"]["mean_tokens"] / e1["FULL"]["mean_tokens"]
    if (abs(rec_gap) <= 0.01 and abs(acc_gap) <= 0.01 and tok_ratio <= 0.5
            and e2["CORE_CM_cum_tokens"] < e2["CORE_noCM_cum_tokens"]):
        verdict = "SUPPORTED"
    elif rec_gap <= 0.05 and acc_gap <= 0.05 and tok_ratio <= 0.5:
        verdict = "SUPPORTED"
    elif rec_gap > 0.05 or acc_gap > 0.05 or tok_ratio > 0.5:
        verdict = "NEGATIVE"
    else:
        verdict = "PARTIAL"

    results = {
        "paper": "arXiv 2611.07859 CORE: Coarse-to-Fine Search with Context "
                 "Management for Deep Research",
        "config": {"k": K, "snippet_width_sentences": SNIPPET_WIDTH,
                   "jaccard_threshold": JACCARD_THRESHOLD,
                   "n_entities": len(entities),
                   "retriever": "BM25 (CPU proxy for dense retriever)",
                   "snippet_scorer": "TF-IDF cosine (CPU proxy for "
                                     "late-interaction reranker token sims)"},
        "E1_two_hop": e1,
        "E1b_single_hop_control": e1b,
        "E2_multi_turn": e2,
        "E3_snippet_ablation": e3,
        "verdict": verdict,
    }

    print("=" * 70)
    print("CORE (arXiv 2611.07859) — CPU replication results")
    print("=" * 70)
    print(f"E1 two-hop (n={e1['n']}, k={K}):")
    for m in ("FULL", "CORE", "NAIVE"):
        d = e1[m]
        print(f"  {m:6s} recall_both={d['recall_both']:.3f} "
              f"accuracy={d['accuracy']:.3f} mean_tokens={d['mean_tokens']:.1f}")
    print(f"  CORE recall gap vs FULL: {rec_gap:+.3f} | "
          f"accuracy gap: {acc_gap:+.3f} | token ratio: {tok_ratio:.3f}")
    print(f"E1b single-hop control (n={e1b['n']}): "
          f"FULL acc={e1b['FULL']['accuracy']:.3f} "
          f"({e1b['FULL']['mean_tokens']:.1f} tok) | "
          f"CORE acc={e1b['CORE']['accuracy']:.3f} "
          f"({e1b['CORE']['mean_tokens']:.1f} tok)")
    print(f"E2 multi-turn (n={e2['n']} trajectories):")
    print(f"  FULL cum tokens={e2['FULL_cum_tokens']:.1f} | "
          f"CORE no-CM={e2['CORE_noCM_cum_tokens']:.1f} | "
          f"CORE+CM={e2['CORE_CM_cum_tokens']:.1f} "
          f"(ratio vs FULL {e2['CM_token_ratio_vs_FULL']:.3f})")
    print(f"  suppression rate={e2['suppression_rate']:.3f} "
          f"({e2['n_suppressed']}/{e2['n_seen_slots']} seen slots; "
          f"expanded {e2['n_expanded']})")
    print(f"  turn-2 evidence coverage: CORE no-CM "
          f"{e2['turn2_evidence_coverage_CORE_noCM']:.3f} | CORE+CM "
          f"{e2['turn2_evidence_coverage_CORE_CM']:.3f}")
    print(f"E3 snippet ablation (evidence recall, n={e3['n']}): "
          f"token_sim={e3['token_sim']:.3f} | "
          f"first_sentence={e3['first_sentence']:.3f} | "
          f"random_window={e3['random_window']:.3f}")
    print("-" * 70)
    checks = [
        ("A1 CORE recall within 5pp of FULL", rec_gap <= 0.05),
        ("A2 CORE mean tokens <= 50% of FULL", tok_ratio <= 0.5),
        ("A3 CORE recall beats NAIVE by >= 10pp",
         e1["CORE"]["recall_both"] - e1["NAIVE"]["recall_both"] >= 0.10),
        ("A4 CORE+CM cum tokens < CORE no-CM (E2)",
         e2["CORE_CM_cum_tokens"] < e2["CORE_noCM_cum_tokens"]),
    ]
    all_pass = True
    for label, ok in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {label}")
        all_pass = all_pass and ok
    print(f"VERDICT: {verdict}")
    print("=" * 70)

    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"wrote {out}")
    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
