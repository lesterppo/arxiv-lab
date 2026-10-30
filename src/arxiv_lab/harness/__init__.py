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
- ``context_attrition`` (arXiv:2610.36114) — fixed-length slicing of
  tool-result JSON deletes end-positioned fields unevenly (every cut
  result becomes broken JSON; evidence survival splits by field
  position); the evidence-preserving field-aware truncation at the same
  byte budget restores validity, survival, and first-attempt accuracy.
- ``casel`` (arXiv:2610.36570) — consequence-aware selective execution:
  per-mode expected-cost allocation over direct/verify/sandbox/clarify/
  approve/abstain from calibrated failure probability, state-dependent
  consequence, reversibility, ambiguity, and evidence freshness; beats
  an uncertainty-ranked gate on harmful-trajectory rate and mean harm
  at matched cost in simulation.
- ``masked_erosion`` (arXiv:2610.38392) — acceptance gates erode
  redundant safeguards unseen: deleting one of four depth-2 safeguards
  changes no observable harm while the partner cover works, but larger
  evaluation batches accept more genuine deletions and lose depth,
  which converts to harm under weaker-model reuse; freezing safety
  text or auditing one-layer ablations preserves depth.

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
from arxiv_lab.harness.context_attrition import (
    BUDGET_CHARS,
    POLICIES,
    diagnose,
    field_aware_truncate,
    generate_tool_result,
    naive_slice,
    no_cut,
)
from arxiv_lab.harness.casel import (
    HARM_WEIGHT,
    run_casel_experiment,
    risk_ledger_check,
    threshold_validity_check,
)
from arxiv_lab.harness.masked_erosion import (
    analytic_acceptance,
    deletion_experiment,
    gate_decide,
    run_experiment as run_masked_erosion_experiment,
    run_improvement_loop,
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
    "BUDGET_CHARS",
    "POLICIES",
    "diagnose",
    "field_aware_truncate",
    "generate_tool_result",
    "naive_slice",
    "no_cut",
    "HARM_WEIGHT",
    "run_casel_experiment",
    "risk_ledger_check",
    "threshold_validity_check",
    "analytic_acceptance",
    "deletion_experiment",
    "gate_decide",
    "run_masked_erosion_experiment",
    "run_improvement_loop",
]
