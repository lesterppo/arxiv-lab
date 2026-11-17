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
from arxiv_lab.memory.rawrag import (
    ConversationWorld,
    ProfileStore,
    compare,
    run_setting,
    window_chunks,
)
from arxiv_lab.memory.matrix_steps import (
    DeltaMatrix,
    EpisodicMemory,
    Episode,
    HashEmbedder,
    ProcedureRecord,
    Step,
    consolidate_episodes,
    fnv1a_64,
    procedure_sign_vector,
)

__all__ = ["MemoryStore", "naive_store_intervention", "cmp_balanced_design",
           "auc", "ConversationWorld", "ProfileStore", "compare",
           "run_setting", "window_chunks", "DeltaMatrix", "EpisodicMemory",
           "Episode", "HashEmbedder", "ProcedureRecord", "Step",
           "consolidate_episodes", "fnv1a_64", "procedure_sign_vector"]
