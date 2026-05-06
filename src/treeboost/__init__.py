"""TreeBoost: from-scratch implementation of Friedman's gradient boosting machine.

This package implements the regression-focused TreeBoost algorithms from
Friedman (2001), "Greedy Function Approximation: A Gradient Boosting Machine."

The public surface intentionally exposes:

- ``RegressionTree``: a small CART-style binary regression tree used as the
  weak learner that fits least-squares pseudo-responses (Algorithm 1, line 4).
- ``TreeBoostRegressor``: the stagewise additive model that drives Algorithms
  2 (LS Boost), 3 (LAD TreeBoost), and 4 (M TreeBoost / Huber).
- The loss objects ``LeastSquaresLoss``, ``LeastAbsoluteDeviationLoss``, and
  ``HuberLoss`` which encapsulate the loss-specific parts of the algorithm:
  the initial constant ``F0``, the negative-gradient pseudo-responses, and
  the per-terminal-region update ``gamma_jm``.

The implementation prioritizes readability and faithfulness to the paper
rather than raw performance.
"""

from .losses import (
    HuberLoss,
    LeastAbsoluteDeviationLoss,
    LeastSquaresLoss,
    Loss,
)
from .model import TreeBoostRegressor
from .tree import RegressionTree

__all__ = [
    "RegressionTree",
    "TreeBoostRegressor",
    "Loss",
    "LeastSquaresLoss",
    "LeastAbsoluteDeviationLoss",
    "HuberLoss",
]
