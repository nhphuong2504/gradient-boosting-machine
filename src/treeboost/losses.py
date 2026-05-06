"""Loss functions for regression TreeBoost.

Each loss object encapsulates the three loss-specific pieces of the generic
gradient boosting algorithm (Algorithm 1 in Friedman 2001):

1. ``initial_prediction(y)`` returns the constant ``F_0`` that minimizes the
   empirical loss on its own. For LS this is the mean; for LAD and Huber it
   is the median.
2. ``negative_gradient(y, F)`` returns the pseudo-responses ``y_tilde_i =
   -[dL/dF(x_i)]_{F = F_{m-1}}`` that the weak learner is fit against in
   line 4 of the algorithm.
3. ``leaf_update(y, F_prev, sample_indices)`` returns the per-terminal-region
   update ``gamma_jm`` (equation (18) of the paper) that replaces the raw
   least-squares leaf mean. This is the special trick that makes TreeBoost
   robust for non-LS losses.

The ``loss_value(y, F)`` method is purely for monitoring training progress;
the algorithm itself does not require it.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


class Loss(ABC):
    """Abstract loss for regression TreeBoost."""

    name: str = "loss"

    @abstractmethod
    def initial_prediction(self, y: np.ndarray) -> float:
        """Return the constant ``F_0`` that minimizes empirical loss."""

    @abstractmethod
    def negative_gradient(self, y: np.ndarray, F: np.ndarray) -> np.ndarray:
        """Return pseudo-responses (negative gradient at current ``F``)."""

    @abstractmethod
    def leaf_update(
        self, y: np.ndarray, F: np.ndarray, indices: np.ndarray
    ) -> float:
        """Return ``gamma_jm`` for a single terminal region.

        ``indices`` is an integer array selecting the rows that fall into
        the terminal region. ``y`` and ``F`` are the full training arrays
        ``(y_i)`` and the current predictions ``(F_{m-1}(x_i))``.
        """

    @abstractmethod
    def loss_value(self, y: np.ndarray, F: np.ndarray) -> float:
        """Return the mean per-sample loss (used only for diagnostics)."""

    def update_state(self, y: np.ndarray, F: np.ndarray) -> None:
        """Optional per-iteration hook (e.g. Huber's adaptive ``delta``)."""
        # Default no-op; Huber overrides.
        return None


# --------------------------------------------------------------- Least Squares
@dataclass
class LeastSquaresLoss(Loss):
    """Squared-error loss; reproduces Algorithm 2 (LS Boost).

    With ``L(y, F) = (y - F)^2 / 2`` the pseudo-responses are simply the
    current residuals ``y_i - F_{m-1}(x_i)``, the optimal initial constant
    is the mean of ``y``, and the per-leaf update is the leaf mean of the
    residuals -- which is exactly what a least-squares regression tree
    already produces. So the boosting driver does not need to overwrite the
    raw tree leaf values when this loss is used.
    """

    name: str = "least_squares"

    def initial_prediction(self, y: np.ndarray) -> float:
        return float(np.mean(y))

    def negative_gradient(self, y: np.ndarray, F: np.ndarray) -> np.ndarray:
        return y - F

    def leaf_update(
        self, y: np.ndarray, F: np.ndarray, indices: np.ndarray
    ) -> float:
        # Mean of residuals in the leaf.
        residuals = y[indices] - F[indices]
        return float(np.mean(residuals)) if residuals.size else 0.0

    def loss_value(self, y: np.ndarray, F: np.ndarray) -> float:
        return float(0.5 * np.mean((y - F) ** 2))


# ------------------------------------------------- Least Absolute Deviation
@dataclass
class LeastAbsoluteDeviationLoss(Loss):
    """LAD loss; reproduces Algorithm 3 (LAD TreeBoost).

    With ``L(y, F) = |y - F|`` the negative gradient is ``sign(y - F)`` and
    the optimal terminal-region update is the median of the residuals in
    the leaf (equation (18) specialized to LAD). The initial constant is
    the median of ``y``.
    """

    name: str = "lad"

    def initial_prediction(self, y: np.ndarray) -> float:
        return float(np.median(y))

    def negative_gradient(self, y: np.ndarray, F: np.ndarray) -> np.ndarray:
        return np.sign(y - F)

    def leaf_update(
        self, y: np.ndarray, F: np.ndarray, indices: np.ndarray
    ) -> float:
        if indices.size == 0:
            return 0.0
        residuals = y[indices] - F[indices]
        return float(np.median(residuals))

    def loss_value(self, y: np.ndarray, F: np.ndarray) -> float:
        return float(np.mean(np.abs(y - F)))


# --------------------------------------------------------------------- Huber
@dataclass
class HuberLoss(Loss):
    """Huber loss; reproduces Algorithm 4 (M TreeBoost) with adaptive ``delta``.

    The loss is

        L(y, F) = (y - F)^2 / 2,                    if |y - F| <= delta
                  delta * |y - F| - delta^2 / 2,    otherwise.

    Following the paper, ``delta`` is set at the start of each iteration to
    the ``alpha``-quantile of the absolute residuals (with
    ``alpha = 1 - quantile``). The pseudo-response is the residual when
    ``|r| <= delta`` and ``delta * sign(r)`` otherwise. The terminal-region
    update uses Friedman's one-step approximation of the Huber location
    M-estimator initialized at the leaf median:

        gamma_j = r_tilde_j +
                  (1 / N_j) * sum_{x_i in R_j} sign(r_i - r_tilde_j)
                                   * min(delta, |r_i - r_tilde_j|)

    where ``r_tilde_j = median_{x_i in R_j}(r_i)``.
    """

    name: str = "huber"
    quantile: float = 0.9  # Outliers are residuals above this quantile.

    # Set per iteration in ``update_state``.
    delta_: float = 0.0

    def initial_prediction(self, y: np.ndarray) -> float:
        return float(np.median(y))

    def update_state(self, y: np.ndarray, F: np.ndarray) -> None:
        residuals = np.abs(y - F)
        if residuals.size == 0:
            self.delta_ = 0.0
            return
        if not (0.0 < self.quantile < 1.0):
            raise ValueError("HuberLoss.quantile must be strictly in (0, 1).")
        self.delta_ = float(np.quantile(residuals, self.quantile))

    def negative_gradient(self, y: np.ndarray, F: np.ndarray) -> np.ndarray:
        delta = self.delta_
        residuals = y - F
        out = np.where(
            np.abs(residuals) <= delta,
            residuals,
            delta * np.sign(residuals),
        )
        return out

    def leaf_update(
        self, y: np.ndarray, F: np.ndarray, indices: np.ndarray
    ) -> float:
        if indices.size == 0:
            return 0.0
        residuals = y[indices] - F[indices]
        median_r = float(np.median(residuals))
        delta = self.delta_
        diff = residuals - median_r
        clipped = np.minimum(delta, np.abs(diff))
        return median_r + float(np.mean(np.sign(diff) * clipped))

    def loss_value(self, y: np.ndarray, F: np.ndarray) -> float:
        delta = self.delta_
        if delta <= 0.0:
            # Before the first ``update_state`` call: fall back to LAD-like
            # mean-absolute behaviour purely for monitoring.
            return float(np.mean(np.abs(y - F)))
        residuals = y - F
        abs_r = np.abs(residuals)
        quad_part = 0.5 * residuals**2
        lin_part = delta * abs_r - 0.5 * delta**2
        return float(np.mean(np.where(abs_r <= delta, quad_part, lin_part)))


def get_loss(name: str, **kwargs) -> Loss:
    """Look up a loss object by short name.

    Recognized names: ``"ls"``, ``"lad"``, ``"huber"``.
    """
    name = name.lower()
    if name in {"ls", "least_squares", "squared_error"}:
        return LeastSquaresLoss()
    if name in {"lad", "absolute", "absolute_error"}:
        return LeastAbsoluteDeviationLoss()
    if name == "huber":
        return HuberLoss(**kwargs)
    raise ValueError(f"Unknown loss: {name!r}")
