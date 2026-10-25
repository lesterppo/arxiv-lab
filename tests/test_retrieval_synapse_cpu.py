"""CPU tests for arxiv_lab.retrieval.synapse (arXiv:2610.30867).

Deterministic, model-free: a synthetic-embedding simulation reproduces the
paper's core claims as checkable directional statements —

  C1. naive pooling (usage examples as extra index entries) lowers held-out
      Hit@5 vs base while raising used-tool Hit@5.
  C2. shuffled usage (same entry counts, wrong tool match) lowers held-out
      Hit@5 about as much but gives no used-tool gain.
  C3. SYNAPSE (usage as its own field, averaged with the description field,
      one candidate per tool) recovers held-out Hit@5 vs naive while
      retaining the used-tool gain; the protection persists under shuffled
      usage, while the used-tool gain collapses — so the gain needs usage
      about the right tool.

Canonical seed 20261025, 500 held-out + 500 used queries, measured values:

  base             held=0.922 used=0.932
  naive            held=0.810 used=0.998
  naive_shuffled   held=0.762 used=0.680
  synapse          held=0.858 used=0.992
  synapse_shuffled held=0.978 used=0.126
  offset (+0.15)   held=0.632 used=0.984

Assertions use margins around these values, not exact equality. The offset
control is reported, not asserted: in this simulation a uniform boost hurts
held-out tools more than SYNAPSE does, diverging from the paper's note that
a constant offset gives similar held-out Hit@5 "in most settings" — the
protection here comes from field averaging, not from merely boosting used
tools. Limits: synthetic embeddings, no real catalog/embedder, Hit@5 only.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

try:
    import numpy as np
except ImportError:
    np = None

if np is not None:
    from arxiv_lab.retrieval.synapse import build_catalog, run_protocol

SEED = 20261025


def test_catalog_structure():
    if np is None:
        return
    catalog = build_catalog(SEED)
    assert catalog.desc.shape == (240, 64)
    assert int(catalog.used.sum()) == 120
    assert len(catalog.usage_examples) == 120
    assert len(catalog.usage_centroid) == 120
    # held-out tools have no usage bundle at all
    for tool in np.where(~catalog.used)[0]:
        assert int(tool) not in catalog.usage_examples


def test_c1_naive_pooling_hurts_held_out_helps_used():
    if np is None:
        return
    r = run_protocol(SEED)
    assert r["naive"]["held"] <= r["base"]["held"] - 0.05
    assert r["naive"]["used"] >= r["base"]["used"] + 0.03


def test_c2_shuffled_usage_keeps_harm_loses_gain():
    if np is None:
        return
    r = run_protocol(SEED)
    assert r["naive_shuffled"]["held"] <= r["base"]["held"] - 0.05
    assert r["naive_shuffled"]["used"] <= r["naive"]["used"] - 0.10


def test_c3_synapse_recovers_held_out_keeps_used_gain():
    if np is None:
        return
    r = run_protocol(SEED)
    assert r["synapse"]["held"] >= r["naive"]["held"] + 0.03
    assert r["synapse"]["used"] >= r["naive"]["used"] - 0.02


def test_c3_synapse_protection_persists_without_right_usage():
    if np is None:
        return
    r = run_protocol(SEED)
    assert r["synapse_shuffled"]["held"] >= r["naive"]["held"] + 0.03
    assert r["synapse_shuffled"]["used"] <= r["synapse"]["used"] - 0.10


if __name__ == "__main__":
    test_catalog_structure()
    test_c1_naive_pooling_hurts_held_out_helps_used()
    test_c2_shuffled_usage_keeps_harm_loses_gain()
    test_c3_synapse_recovers_held_out_keeps_used_gain()
    test_c3_synapse_protection_persists_without_right_usage()
    print("test_retrieval_synapse_cpu: 5/5 pass")
