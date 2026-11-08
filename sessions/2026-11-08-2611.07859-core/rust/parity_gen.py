"""Generate Python-vs-Rust parity cases for the CORE Rust leg.

Builds N random cases (query + passages + k) with the Python
implementation (`core.py` in the parent session directory) and writes
`parity_cases.json` next to this script. The Rust `parity` binary reads
the same file, recomputes each case, and compares (BM25 top-k ids exact
+ scores within 1e-6 relative; best snippet-window indices exact;
ContextManager turn-2 decision exact).

Run from the rust/ directory:
    python3 parity_gen.py
"""

import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core import BM25, ContextManager, SnippetExtractor  # noqa: E402

SEED = 20261108
N_CASES = 250

WORDS = [
    "quantum", "photon", "qubit", "coral", "plankton", "magnetar", "exoplanet",
    "parser", "lexicon", "dialogue", "gripper", "quadruped", "swarm", "enzyme",
    "protein", "carbon", "cloud", "glacier", "memory", "neural", "reward",
    "ceramic", "battery", "polymer", "prime", "sieve", "residue", "researcher",
    "institute", "university", "college", "award", "medal", "prize", "fellowship",
    "discovery", "laboratory", "experiment", "theory", "survey", "circuit",
    "feedback", "dynamics", "genomics", "ecology", "optics", "lattice",
    "the", "is", "which", "a", "an", "of", "for", "with", "in", "on",
]


def random_sentence(rng):
    n = rng.randint(4, 10)
    words = [rng.choice(WORDS) for _ in range(n)]
    return " ".join(words).capitalize() + "."


def main():
    rng = random.Random(SEED)
    cases = []
    for _ in range(N_CASES):
        n_pass = rng.randint(3, 8)
        passages, all_sents, per_pass_sents = [], [], []
        for _ in range(n_pass):
            sents = [random_sentence(rng) for _ in range(rng.randint(4, 8))]
            per_pass_sents.append(sents)
            all_sents.extend(sents)
            passages.append(" ".join(sents))
        query = " ".join(rng.choice(WORDS) for _ in range(rng.randint(3, 7)))
        k = rng.randint(1, 8)

        bm25 = BM25(passages)
        topk = [[i, s] for i, s in bm25.search(query, k=k)]
        ext = SnippetExtractor(all_sents)
        bw_idx, _ = ext.best_window(per_pass_sents[0], query, 2)

        registered = rng.choice(all_sents)
        fresh = rng.choice(all_sents)
        answer = rng.choice(all_sents)
        cm = ContextManager(0.8)
        cm.register("p0", registered)
        action, text = cm.turn2_decision("p0", fresh, answer)

        cases.append({
            "query": query,
            "passages": passages,
            "sentences": per_pass_sents[0],
            "all_sentences": all_sents,
            "k": k,
            "expected": {
                "bm25_topk": topk,
                "best_window": bw_idx,
                "suppression": {
                    "registered": registered,
                    "fresh": fresh,
                    "answer_sentence": answer,
                    "action": action,
                    "text": text,
                },
            },
        })

    out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "parity_cases.json")
    with open(out, "w") as f:
        json.dump({"cases": cases}, f)
    print(f"wrote {out} ({len(cases)} cases)")


if __name__ == "__main__":
    main()
