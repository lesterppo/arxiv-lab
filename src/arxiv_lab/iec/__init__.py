"""IEC: intent-execution correspondence across harness hops
(arXiv:2610.04375).

A tool call emitted by an LLM traverses quoting layers, re-serialization,
and proxies; any hop can silently change what executes. ``observe`` names
the first mutating hop without executing anything; ``armor``/``deliver``
(IntAct) deliver the call in a hop-unalterable form or refuse it.

Pure stdlib.
"""

from arxiv_lab.iec.iec import (
    Call,
    ToolContract,
    ContractViolation,
    Hop,
    IdentityHop,
    JsonReserializeHop,
    ShellQuoteHop,
    UrlDecodeHop,
    TruncateHop,
    EnvExpandHop,
    HopReceipt,
    observe,
    first_mutating_hop,
    armor,
    deliver,
    Refusal,
    ExecutedCall,
    NaiveTrajectoryJudge,
    ARG_CLASSES,
    make_workload,
    production_hops,
)

__all__ = [
    "Call",
    "ToolContract",
    "ContractViolation",
    "Hop",
    "IdentityHop",
    "JsonReserializeHop",
    "ShellQuoteHop",
    "UrlDecodeHop",
    "TruncateHop",
    "EnvExpandHop",
    "HopReceipt",
    "observe",
    "first_mutating_hop",
    "armor",
    "deliver",
    "Refusal",
    "ExecutedCall",
    "NaiveTrajectoryJudge",
    "ARG_CLASSES",
    "make_workload",
    "production_hops",
]
