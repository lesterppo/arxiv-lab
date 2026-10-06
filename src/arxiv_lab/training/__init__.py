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
]
