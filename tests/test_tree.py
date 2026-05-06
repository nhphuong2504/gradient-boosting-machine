"""Unit tests for the custom CART regression tree."""

from __future__ import annotations

import numpy as np
import pytest

from treeboost.tree import RegressionTree


def test_tree_with_one_leaf_predicts_global_mean():
    """A tree with ``max_leaves=1`` cannot split, so predictions are y.mean()."""
    rng = np.random.default_rng(0)
    X = rng.normal(size=(20, 2))
    y = rng.normal(size=20)
    tree = RegressionTree(max_leaves=1).fit(X, y)
    preds = tree.predict(X)
    assert tree.n_leaves_ == 1
    np.testing.assert_allclose(preds, np.full(20, y.mean()))


def test_tree_finds_obvious_axis_aligned_split():
    """Two well-separated clusters along feature 0 should give a clean split."""
    X = np.array([[0.0], [0.5], [1.0], [10.0], [10.5], [11.0]])
    y = np.array([-1.0, -1.0, -1.0, 1.0, 1.0, 1.0])
    tree = RegressionTree(max_leaves=2).fit(X, y)
    assert tree.n_leaves_ == 2
    preds = tree.predict(X)
    # Each side should be predicted as the side's mean (here +-1 exactly).
    np.testing.assert_allclose(preds[:3], -1.0)
    np.testing.assert_allclose(preds[3:], 1.0)
    # The split threshold should be somewhere between 1.0 and 10.0.
    root = tree.root_
    assert root.feature == 0
    assert 1.0 < root.threshold < 10.0


def test_tree_apply_returns_unique_leaf_ids():
    """``apply`` must return integers in [0, n_leaves)."""
    rng = np.random.default_rng(1)
    X = rng.normal(size=(50, 3))
    y = X[:, 0] + 0.1 * rng.normal(size=50)
    tree = RegressionTree(max_leaves=4).fit(X, y)
    leaf_ids = tree.apply(X)
    assert leaf_ids.shape == (50,)
    assert leaf_ids.min() >= 0
    assert leaf_ids.max() < tree.n_leaves_
    # Every leaf should be reachable on training data.
    assert set(np.unique(leaf_ids)) == set(range(tree.n_leaves_))


def test_set_leaf_values_overrides_predictions():
    """``set_leaf_values`` must overwrite leaf means with custom gammas."""
    X = np.array([[0.0], [1.0], [10.0], [11.0]])
    y = np.array([0.0, 0.0, 1.0, 1.0])
    tree = RegressionTree(max_leaves=2).fit(X, y)
    tree.set_leaf_values(np.array([5.0, -5.0]))
    leaf_ids = tree.apply(X)
    preds = tree.predict(X)
    expected = np.array([5.0 if lid == 0 else -5.0 for lid in leaf_ids])
    np.testing.assert_allclose(preds, expected)


def test_set_leaf_values_validates_shape():
    X = np.array([[0.0], [1.0], [10.0], [11.0]])
    y = np.array([0.0, 0.0, 1.0, 1.0])
    tree = RegressionTree(max_leaves=2).fit(X, y)
    with pytest.raises(ValueError):
        tree.set_leaf_values(np.array([1.0, 2.0, 3.0]))


def test_tree_respects_max_leaves():
    rng = np.random.default_rng(2)
    X = rng.normal(size=(120, 4))
    # Make y depend on multiple features so deep splits are tempting.
    y = X[:, 0] + X[:, 1] ** 2 - X[:, 2]
    for J in [2, 4, 8]:
        tree = RegressionTree(max_leaves=J).fit(X, y)
        assert tree.n_leaves_ <= J


def test_tree_handles_constant_target():
    """No useful split exists; tree should remain a single leaf."""
    X = np.linspace(-1, 1, 30).reshape(-1, 1)
    y = np.full(30, 3.5)
    tree = RegressionTree(max_leaves=8).fit(X, y)
    assert tree.n_leaves_ == 1
    np.testing.assert_allclose(tree.predict(X), 3.5)


def test_tree_min_samples_leaf_is_respected():
    rng = np.random.default_rng(3)
    X = rng.normal(size=(40, 1))
    y = rng.normal(size=40)
    tree = RegressionTree(max_leaves=8, min_samples_leaf=10).fit(X, y)
    leaf_ids = tree.apply(X)
    counts = np.bincount(leaf_ids, minlength=tree.n_leaves_)
    assert (counts >= 10).all()
