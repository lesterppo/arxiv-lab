"""Training-track mechanisms (arXiv read-and-test).

- ``prospective_hindsight`` — Prospective Hindsight: self-calibrating RL via
  prediction-reality gaps (arXiv:2610.02740). Surprise-weighted advantages
  amplify the gradient on rollouts where the agent's prospective prediction
  most disagrees with the retrospective outcome, giving a descent pathway on
  miscalibration as a byproduct of optimization.
- ``arbor`` — ARBOR: conditional rank allocation; per-question selection of
  rank-one components from a shared low-rank basis via an additive gate
  (arXiv:2610.06765). Avoids the approximation floor of a fixed update of
  the same active rank under orthogonal subtasks.
- ``code_lora`` — CoDe-LoRA: dual-branch continual LoRA; consolidation
  branch accumulates shared updates via null-space projection with dynamic
  scaling, decoupling branch routes per-task experts by prototype cosine
  similarity with a confidence fallback (arXiv:2610.08312).
- ``rpo`` — Root-token Policy Optimization (arXiv:2610.17767): train only
  the root token (the think/don't-think decision) via group-relative
  rewards and group-wise advantages; group rollouts with forced arms
  learn difficulty-aware adaptive thinking at a fraction of full
  training compute.
- ``sst`` — Subspace Tuning (arXiv:2610.22293): restrict each weight update
  to the bottom singular subspace of the weight matrix, protecting the
  principal directions that carry prior-task knowledge.
- ``local_proxies`` — Local proxies for memory-efficient training
  (arXiv:2610.23145): per-block proxy networks distilled to predict the
  global output; block gradients propagate locally through the proxy, so
  full-stack activations need not be retained. Exact on deep linear nets;
  beats zeroth-order (MeZO-style) optimizers at comparable memory.
"""

from arxiv_lab.training.prospective_hindsight import (
    surprise_weights,
    ph_advantages,
    miscalibration_rate,
    expected_calibration_error,
)
from arxiv_lab.training.arbor import (
    make_orthogonal_tasks,
    fixed_update_floor,
    additive_gate,
    conditional_update,
    oracle_gate_weights,
    question_for_task,
)
from arxiv_lab.training.code_lora import (
    dynamic_scaling,
    top_column_basis,
    consolidate_null_project,
    update_overlap_rho,
    code_lora_step,
    build_prototype,
    route_prototype,
)
from arxiv_lab.training.rpo import (
    group_advantages,
    kl_penalty,
    root_token_losses,
    simulate_adaptive_thinking,
    THINK_COST,
)
from arxiv_lab.training.sst import (
    principal_directions,
    project_grad_to_bottom_subspace,
    sst_update,
)
from arxiv_lab.training.local_proxies import (
    linear_exactness,
    residual_error_bound,
    three_way_comparison,
    lstsq_proxy,
    ResidualProxy,
    TinyMLP,
)

__all__ = [
    "surprise_weights",
    "ph_advantages",
    "miscalibration_rate",
    "expected_calibration_error",
    "make_orthogonal_tasks",
    "fixed_update_floor",
    "additive_gate",
    "conditional_update",
    "oracle_gate_weights",
    "question_for_task",
    "dynamic_scaling",
    "top_column_basis",
    "consolidate_null_project",
    "update_overlap_rho",
    "code_lora_step",
    "build_prototype",
    "route_prototype",
    "group_advantages",
    "kl_penalty",
    "root_token_losses",
    "simulate_adaptive_thinking",
    "THINK_COST",
    "principal_directions",
    "project_grad_to_bottom_subspace",
    "sst_update",
    "linear_exactness",
    "residual_error_bound",
    "three_way_comparison",
    "lstsq_proxy",
    "ResidualProxy",
    "TinyMLP",
]
