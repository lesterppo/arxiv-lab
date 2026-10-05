"""Training-track mechanisms (arXiv read-and-test).

- ``prospective_hindsight`` — Prospective Hindsight: self-calibrating RL via
  prediction-reality gaps (arXiv:2610.02740). Surprise-weighted advantages
  amplify the gradient on rollouts where the agent's prospective prediction
  most disagrees with the retrospective outcome, giving a descent pathway on
  miscalibration as a byproduct of optimization.
"""

from arxiv_lab.training.prospective_hindsight import (
    surprise_weights,
    ph_advantages,
    miscalibration_rate,
    expected_calibration_error,
)

__all__ = [
    "surprise_weights",
    "ph_advantages",
    "miscalibration_rate",
    "expected_calibration_error",
]
