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
- ``harnessmm`` (arXiv:2610.32813) — guarded decision reuse: a persistent
  Executable Decision Graph reuses certified decisions only on exact
  action + literal-argument + dependency + receipt match, falling back to
  the actor on novel decisions, unsupported evidence, and abstention;
  validation selects the guard config for deployment (RSI).

Pure stdlib. No numpy needed.
"""

from arxiv_lab.harness.prefill import PrefillBudget
from arxiv_lab.harness.finish_gate import FinishGate
from arxiv_lab.harness.loop import LoopDetector
from arxiv_lab.harness.harnessmm import (
    ACTOR_COST,
    REUSE_COST,
    Decision,
    DecisionGraph,
    Step,
    generate_tasks,
    run_condition,
    run_experiment,
)
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
    "Decision",
    "DecisionGraph",
    "Step",
    "generate_tasks",
    "run_condition",
    "run_experiment",
    "ACTOR_COST",
    "REUSE_COST",
    "SchemaShield",
    "ToolCall",
    "ToolDef",
    "DriftSimulator",
    "RepairResult",
    "validate",
]
