"""Causal Memory Policy memory scoring (arXiv:2610.02070).

Unbiased memory-utility estimation via retrieval-level intervention with
known propensities (SNIPW under a balanced assignment design), fixing the
positivity violation of store-level intervention.

Requires numpy (optional dependency): ``pip install numpy``.
"""

from arxiv_lab.memory.cmp import (
    MemoryStore,
    naive_store_intervention,
    cmp_balanced_design,
    auc,
)

__all__ = ["MemoryStore", "naive_store_intervention", "cmp_balanced_design", "auc"]
