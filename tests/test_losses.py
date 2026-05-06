"""Unit tests for the loss functions used by TreeBoost."""

from __future__ import annotations

import numpy as np
import pytest

from treeboost.losses import (
    HuberLoss,
    LeastAbsoluteDeviationLoss,
    LeastSquaresLoss,
    get_loss,
)


# ---------------------------------------------------------------- Least Squares
def test_ls_initial_prediction_is_mean():
    y = np.array([1.0, 2.0, 3.0, 4.0])
    assert LeastSquaresLoss().initial_prediction(y) == pytest.approx(2.5)


def test_ls_negative_gradient_equals_residual():
    y = np.array([1.0, 2.0, 3.0])
    F = np.array([0.5, 2.5, 2.0])
    np.testing.assert_allclose(
        LeastSquaresLoss().negative_gradient(y, F), y - F
    )


def test_ls_leaf_update_is_residual_mean():
    y = np.array([1.0, 2.0, 3.0, 5.0])
    F = np.array([0.0, 0.0, 0.0, 0.0])
    loss = LeastSquaresLoss()
    indices = np.array([0, 1, 3])
    expected = ((1.0 - 0.0) + (2.0 - 0.0) + (5.0 - 0.0)) / 3
    assert loss.leaf_update(y, F, indices) == pytest.approx(expected)


def test_ls_loss_value_matches_formula():
    y = np.array([1.0, 2.0])
    F = np.array([0.0, 0.0])
    # mean((y - F)^2 / 2) = mean([0.5, 2.0]) = 1.25
    assert LeastSquaresLoss().loss_value(y, F) == pytest.approx(1.25)


# --------------------------------------------------- Least Absolute Deviation
def test_lad_initial_prediction_is_median():
    y = np.array([1.0, 2.0, 3.0, 4.0, 100.0])
    assert LeastAbsoluteDeviationLoss().initial_prediction(y) == pytest.approx(3.0)


def test_lad_negative_gradient_is_sign_of_residual():
    y = np.array([1.0, 2.0, 3.0])
    F = np.array([0.5, 2.0, 5.0])
    expected = np.array([1.0, 0.0, -1.0])
    np.testing.assert_allclose(
        LeastAbsoluteDeviationLoss().negative_gradient(y, F), expected
    )


def test_lad_leaf_update_is_residual_median():
    y = np.array([1.0, 2.0, 3.0, 5.0, 100.0])
    F = np.zeros_like(y)
    indices = np.arange(5)
    # Median of residuals [1, 2, 3, 5, 100] = 3
    assert LeastAbsoluteDeviationLoss().leaf_update(y, F, indices) == pytest.approx(3.0)


def test_lad_loss_value_is_mean_absolute_error():
    y = np.array([1.0, 2.0, 3.0])
    F = np.array([0.0, 1.0, 5.0])
    # |y - F| = [1, 1, 2]; mean = 4/3.
    assert LeastAbsoluteDeviationLoss().loss_value(y, F) == pytest.approx(4.0 / 3.0)


# ---------------------------------------------------------------------- Huber
def test_huber_pseudo_response_is_clipped():
    """Inside ``|r| <= delta`` it equals ``r``; outside it equals ``delta * sign(r)``."""
    y = np.array([0.0, 0.0, 0.0, 0.0])
    F = np.array([0.5, -0.5, 5.0, -5.0])  # residuals = -0.5, 0.5, -5, 5
    huber = HuberLoss(quantile=0.75)
    huber.delta_ = 1.0  # set explicitly to make the math obvious
    pseudo = huber.negative_gradient(y, F)
    expected = np.array([-0.5, 0.5, -1.0, 1.0])
    np.testing.assert_allclose(pseudo, expected)


def test_huber_update_state_sets_delta_to_quantile():
    huber = HuberLoss(quantile=0.5)
    y = np.array([0.0, 0.0, 0.0, 0.0])
    F = np.array([1.0, 2.0, 3.0, 4.0])  # |r| = 4, 3, 2, 1
    huber.update_state(y, F)
    # 0.5-quantile of [1,2,3,4] is 2.5 (numpy linear interpolation)
    assert huber.delta_ == pytest.approx(2.5)


def test_huber_initial_prediction_is_median():
    y = np.array([1.0, 2.0, 100.0])
    assert HuberLoss().initial_prediction(y) == pytest.approx(2.0)


def test_huber_leaf_update_matches_paper_formula():
    """Verify equation from Algorithm 4 (M TreeBoost).

    With residuals r and median r~,
        gamma = r~ + (1/N) * sum sign(r - r~) * min(delta, |r - r~|).
    """
    y = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
    F = np.array([-3.0, -1.0, 0.0, 1.0, 3.0])  # residuals = 3, 1, 0, -1, -3
    indices = np.arange(5)
    huber = HuberLoss(quantile=0.8)
    # Pretend update_state has run: set delta to 1.5 manually so the math
    # is reproducible.
    huber.delta_ = 1.5

    residuals = np.array([3.0, 1.0, 0.0, -1.0, -3.0])
    median_r = float(np.median(residuals))  # 0.0
    diff = residuals - median_r
    expected = median_r + np.mean(
        np.sign(diff) * np.minimum(huber.delta_, np.abs(diff))
    )
    got = huber.leaf_update(y, F, indices)
    assert got == pytest.approx(expected)


def test_huber_reduces_to_squared_for_small_residuals():
    """Inside the quadratic zone Huber matches LS up to a constant of 0.5."""
    y = np.array([0.0, 0.0, 0.0])
    F = np.array([0.1, -0.2, 0.3])
    huber = HuberLoss()
    huber.delta_ = 10.0  # All residuals are inside the quadratic zone.
    expected = float(np.mean(0.5 * (y - F) ** 2))
    assert huber.loss_value(y, F) == pytest.approx(expected)


def test_get_loss_dispatches_correctly():
    assert isinstance(get_loss("ls"), LeastSquaresLoss)
    assert isinstance(get_loss("lad"), LeastAbsoluteDeviationLoss)
    huber = get_loss("huber", quantile=0.7)
    assert isinstance(huber, HuberLoss)
    assert huber.quantile == 0.7
    with pytest.raises(ValueError):
        get_loss("nonsense")
