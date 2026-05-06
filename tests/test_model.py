"""Integration tests for ``TreeBoostRegressor``."""

from __future__ import annotations

import numpy as np
import pytest

from treeboost import TreeBoostRegressor


def _make_clean_dataset(n: int = 400, seed: int = 0):
    rng = np.random.default_rng(seed)
    X = rng.uniform(-3.0, 3.0, size=(n, 3))
    y = (
        np.sin(X[:, 0])
        + 0.5 * X[:, 1] ** 2
        - 0.3 * X[:, 2]
        + 0.05 * rng.normal(size=n)
    )
    return X, y


def _make_contaminated_dataset(n: int = 400, frac_outliers: float = 0.1, seed: int = 1):
    rng = np.random.default_rng(seed)
    X = rng.uniform(-3.0, 3.0, size=(n, 3))
    y = np.sin(X[:, 0]) + 0.5 * X[:, 1] - 0.2 * X[:, 2]
    # Inject a small fraction of large symmetric outliers.
    n_out = int(frac_outliers * n)
    idx = rng.choice(n, size=n_out, replace=False)
    y[idx] += rng.choice([-1.0, 1.0], size=n_out) * (10 + 5 * rng.normal(size=n_out))
    return X, y, idx


def test_initial_prediction_for_zero_estimators_is_constant():
    X, y = _make_clean_dataset()
    model = TreeBoostRegressor(loss="ls", n_estimators=0).fit(X, y)
    preds = model.predict(X)
    np.testing.assert_allclose(preds, np.full_like(preds, y.mean()))


def test_lad_zero_estimators_returns_median():
    X, y = _make_clean_dataset()
    model = TreeBoostRegressor(loss="lad", n_estimators=0).fit(X, y)
    preds = model.predict(X)
    np.testing.assert_allclose(preds, np.full_like(preds, np.median(y)))


def test_training_loss_is_monotonically_non_increasing_for_ls():
    """LS Boost without shrinkage is greedy on training MSE, so it should not
    increase the training loss between iterations on simple data."""
    X, y = _make_clean_dataset(n=200, seed=42)
    model = TreeBoostRegressor(
        loss="ls", n_estimators=30, learning_rate=1.0, max_leaves=4
    ).fit(X, y)
    losses = np.asarray(model.train_loss_)
    # Allow a tiny floating-point slack.
    diffs = np.diff(losses)
    assert (diffs <= 1e-9).all(), f"Training loss increased: {diffs[diffs > 0]}"


def test_lad_training_loss_decreases_overall():
    """Loss is not strictly monotone for LAD due to discrete sign updates,
    but it should drop substantially over many iterations."""
    X, y = _make_clean_dataset(n=200, seed=7)
    model = TreeBoostRegressor(
        loss="lad", n_estimators=80, learning_rate=0.1, max_leaves=4
    ).fit(X, y)
    assert model.train_loss_[-1] < 0.5 * model.train_loss_[0]


def test_huber_training_loss_decreases_overall():
    X, y = _make_clean_dataset(n=200, seed=11)
    model = TreeBoostRegressor(
        loss="huber",
        n_estimators=80,
        learning_rate=0.1,
        max_leaves=4,
        huber_quantile=0.9,
    ).fit(X, y)
    assert model.train_loss_[-1] < 0.5 * model.train_loss_[0]


def test_staged_predict_matches_final_predict():
    X, y = _make_clean_dataset(n=150, seed=2)
    model = TreeBoostRegressor(
        loss="ls", n_estimators=20, learning_rate=0.3, max_leaves=4
    ).fit(X, y)
    last = None
    for stage in model.staged_predict(X):
        last = stage
    np.testing.assert_allclose(last, model.predict(X))


def test_staged_loss_records_initial_constant():
    X, y = _make_clean_dataset(n=120, seed=5)
    model = TreeBoostRegressor(
        loss="ls", n_estimators=10, learning_rate=0.2, max_leaves=4
    ).fit(X, y)
    losses = model.staged_loss(X, y)
    # First entry should be the loss of the constant prediction F0.
    F0 = np.full_like(y, y.mean())
    assert losses[0] == pytest.approx(0.5 * np.mean((y - F0) ** 2))
    assert len(losses) == 11  # 1 (F_0) + n_estimators


def test_robust_losses_resist_outliers_better_than_ls():
    """Compare clean-target MSE/MAE on contaminated training data.

    Robust variants should approximate the clean signal more closely than LS
    on the same noisy training data.
    """
    rng = np.random.default_rng(0)
    n = 600
    X_train = rng.uniform(-3.0, 3.0, size=(n, 3))
    f = np.sin(X_train[:, 0]) + 0.5 * X_train[:, 1] - 0.2 * X_train[:, 2]
    y_train = f + 0.1 * rng.normal(size=n)
    n_out = int(0.10 * n)
    idx = rng.choice(n, size=n_out, replace=False)
    y_train[idx] += rng.choice([-1.0, 1.0], size=n_out) * (15 + 5 * rng.normal(size=n_out))

    X_test = rng.uniform(-3.0, 3.0, size=(400, 3))
    f_test = np.sin(X_test[:, 0]) + 0.5 * X_test[:, 1] - 0.2 * X_test[:, 2]

    common = dict(n_estimators=120, learning_rate=0.1, max_leaves=6)
    ls = TreeBoostRegressor(loss="ls", **common).fit(X_train, y_train)
    lad = TreeBoostRegressor(loss="lad", **common).fit(X_train, y_train)
    huber = TreeBoostRegressor(loss="huber", huber_quantile=0.85, **common).fit(
        X_train, y_train
    )

    mse = lambda m: float(np.mean((m.predict(X_test) - f_test) ** 2))
    mae = lambda m: float(np.mean(np.abs(m.predict(X_test) - f_test)))

    # Robust losses should beat LS on the *clean* signal (MSE) by a clear
    # margin under heavy contamination.
    assert mse(lad) < mse(ls)
    assert mse(huber) < mse(ls)
    # Also strictly on MAE.
    assert mae(lad) < mae(ls)
    assert mae(huber) < mae(ls)
