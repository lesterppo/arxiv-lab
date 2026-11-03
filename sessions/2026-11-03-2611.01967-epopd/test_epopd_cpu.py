"""CPU properties for the EP-OPD matched-pair design (arXiv 2611.01967).

No model, no network: checks the synthetic pair generator that the
Colab driver uses -- the paper's matched-document intervention
requires pairs that differ ONLY in the answer-critical span.
"""
import importlib.util
import os
import sys

HERE = os.path.dirname(__file__)


def load_driver():
    spec = importlib.util.spec_from_file_location(
        "colab_epopd_driver", os.path.join(HERE, "colab_epopd_driver.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    mod = load_driver()
    pairs = mod.build_pairs(220)
    assert len(pairs) == 220
    # Disjoint names; every pair differs only in the critical value.
    names = [p["name"] for p in pairs]
    assert len(set(names)) == 220
    for p in pairs:
        assert p["doc_orig"] != p["doc_cf"]
        assert p["gold_orig"] != p["gold_cf"]
        assert p["doc_cf"].replace(p["gold_cf"], p["gold_orig"]) == p["doc_orig"]
        assert p["gold_orig"] in p["doc_orig"]
        assert p["gold_cf"] in p["doc_cf"]
        assert p["gold_orig"] not in p["doc_cf"]
        assert p["gold_cf"] not in p["doc_orig"]
        # Prompt format is stable: the doc is the only varying input.
        assert mod.prompt(p["doc_orig"], p["q"]).startswith("Document: ")
        assert p["q"] in mod.prompt(p["doc_orig"], p["q"])
    train, ev = pairs[:160], pairs[160:]
    assert len(train) == 160 and len(ev) == 60
    assert not (set(p["name"] for p in train) & set(p["name"] for p in ev))
    # Determinism: rebuilding gives identical pairs.
    assert mod.build_pairs(220) == pairs
    print("EP-OPD CPU properties PASSED: 220 pairs, matched-intervention "
          "invariant holds on all, train/eval disjoint (160/60)")


if __name__ == "__main__":
    main()
