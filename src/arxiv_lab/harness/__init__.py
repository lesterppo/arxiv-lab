"""Agent-harness mechanisms from Mingbird (arXiv:2610.02001).

The paper argues a substantial share of small-open-model agent failures is
attributable to the *harness*, not the model, and introduces Mingbird, a
local-first agent harness whose mechanisms compensate point-by-point for
small-model failure forms. Three representative mechanisms are implemented
here:

- ``prefill.PrefillBudget`` — byte-level net-zero prefill budget: tool
  prefill (system text + tool schemas) is compressed to a fixed byte budget
  instead of overflowing the context.
- ``finish_gate.FinishGate`` — a finish gate that re-reads the task and
  requires explicit evidence for every success criterion before accepting a
  model's "task complete" claim (prevents silent abandonment).
- ``loop.LoopDetector`` — signature-level loop detection: fingerprints every
  action as (tool, canonical-arg-signature) and halts on exact-repeat streaks
  or short repeated cycles (prevents tool-demonstration loops).
- ``schema_shield.SchemaShield`` — interface-drift repair layer: validates
  each emitted tool call, retries missing tool names against the candidate
  set, and applies one one-shot constrained argument repair (no retraining).

Pure stdlib. No numpy needed.
"""

from arxiv_lab.harness.prefill import PrefillBudget
from arxiv_lab.harness.finish_gate import FinishGate
from arxiv_lab.harness.loop import LoopDetector
from arxiv_lab.harness.schema_shield import (
    DriftSimulator,
    RepairResult,
    SchemaShield,
    ToolCall,
    ToolDef,
    validate,
)

__all__ = [
    "PrefillBudget",
    "FinishGate",
    "LoopDetector",
    "SchemaShield",
    "ToolCall",
    "ToolDef",
    "DriftSimulator",
    "RepairResult",
    "validate",
]
