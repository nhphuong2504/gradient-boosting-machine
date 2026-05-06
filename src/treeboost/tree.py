"""A small CART-style regression tree used as the weak learner.

Friedman's Gradient Boosting algorithm fits each weak learner ``h(x; a_m)``
to the *pseudo-responses* ``y_tilde_i = -[dL/dF(x_i)]_{F = F_{m-1}}`` by
least squares (Algorithm 1, line 4 of Friedman 2001). For TreeBoost the
weak learner is a J-terminal-node regression tree.

This module implements such a tree from scratch:

- Each split chooses a (feature, threshold) pair that maximizes the
  reduction in squared error, equivalent to maximizing
  ``n_L * mean_L^2 + n_R * mean_R^2 - n * mean^2``.
- Growth is controlled by ``max_leaves`` (the J in the paper) and standard
  guards (``min_samples_split``, ``min_samples_leaf``).
- The tree exposes ``apply(X)`` which returns the leaf index for each
  row. The boosting driver uses these leaf indices to compute
  loss-specific terminal-region updates ``gamma_jm`` and to overwrite the
  raw least-squares leaf values with them.

Only ``numpy`` is used; no scikit-learn or other tree libraries are imported.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np


@dataclass
class _Node:
    """Internal binary tree node.

    Either a split node (``feature`` and ``threshold`` set, ``left``/``right``
    set) or a leaf (``leaf_id`` set, ``value`` set to the least-squares mean
    of the targets in the leaf).
    """

    # Split fields (None on leaves).
    feature: Optional[int] = None
    threshold: Optional[float] = None
    left: Optional["_Node"] = None
    right: Optional["_Node"] = None

    # Leaf fields (None on internal nodes).
    leaf_id: Optional[int] = None
    value: Optional[float] = None

    # Bookkeeping for split-search candidates.
    n_samples: int = 0
    sum_y: float = 0.0
    sum_y_sq: float = 0.0

    @property
    def is_leaf(self) -> bool:
        return self.left is None and self.right is None


@dataclass
class _Candidate:
    """A frontier leaf with its best split, used for best-first growth."""

    node: _Node
    indices: np.ndarray
    gain: float
    feature: int
    threshold: float
    left_indices: np.ndarray
    right_indices: np.ndarray
    left_value: float
    right_value: float


@dataclass
class RegressionTree:
    """CART-style regression tree fit by minimizing squared error.

    Parameters
    ----------
    max_leaves:
        Maximum number of terminal nodes (the ``J`` in Friedman's TreeBoost).
        Trees are grown best-first so that ``max_leaves`` directly controls
        complexity in the same sense as the paper.
    max_depth:
        Optional hard cap on depth. ``None`` means unlimited; ``max_leaves``
        is usually the binding constraint.
    min_samples_split:
        A node is only considered for splitting if it has at least this many
        samples.
    min_samples_leaf:
        Both children of a candidate split must have at least this many
        samples for the split to be valid.

    Notes
    -----
    The tree fits *pseudo-responses* in the boosting context. Its raw leaf
    values are the least-squares means of those pseudo-responses; downstream
    code typically overwrites them with loss-specific ``gamma_jm`` values via
    ``set_leaf_values``.
    """

    max_leaves: int = 8
    max_depth: Optional[int] = None
    min_samples_split: int = 2
    min_samples_leaf: int = 1

    # Populated after ``fit``.
    root_: Optional[_Node] = field(default=None, init=False, repr=False)
    n_leaves_: int = field(default=0, init=False, repr=False)
    leaves_: List[_Node] = field(default_factory=list, init=False, repr=False)

    # ------------------------------------------------------------------ fit
    def fit(self, X: np.ndarray, y: np.ndarray) -> "RegressionTree":
        """Fit the tree to ``(X, y)`` minimizing squared error.

        Parameters
        ----------
        X: array of shape (n_samples, n_features).
        y: array of shape (n_samples,).
        """
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        if X.ndim != 2:
            raise ValueError("X must be 2-d (n_samples, n_features).")
        if y.shape[0] != X.shape[0]:
            raise ValueError("X and y must have the same number of rows.")
        if self.max_leaves < 1:
            raise ValueError("max_leaves must be >= 1.")

        n_samples = X.shape[0]
        indices = np.arange(n_samples)

        root = self._make_leaf(y, indices)
        self.root_ = root

        if self.max_leaves == 1 or n_samples < self.min_samples_split:
            self._finalize_leaves()
            return self

        # Best-first growth: keep a list of "open" leaves with their best
        # split, expand the one with the largest impurity reduction, repeat.
        frontier: List[_Candidate] = []
        first = self._best_split(root, indices, X, y, depth=0)
        if first is not None:
            frontier.append(first)

        # We currently have 1 leaf (root). Each successful split converts
        # one leaf into two, so each split increases the leaf count by 1.
        leaf_count = 1
        while frontier and leaf_count < self.max_leaves:
            # Pick the candidate with the largest gain.
            best_idx = max(range(len(frontier)), key=lambda i: frontier[i].gain)
            cand = frontier.pop(best_idx)
            if cand.gain <= 0.0:
                # No remaining beneficial split; stop early.
                break

            self._apply_split(cand, X, y)
            leaf_count += 1

            # Try to find new splits inside each new child. Only consider
            # depth caps via ``max_depth`` if provided.
            child_depth = self._depth_of(cand.node) + 1
            if self.max_depth is None or child_depth < self.max_depth:
                left_cand = self._best_split(
                    cand.node.left, cand.left_indices, X, y, depth=child_depth
                )
                if left_cand is not None:
                    frontier.append(left_cand)
                right_cand = self._best_split(
                    cand.node.right, cand.right_indices, X, y, depth=child_depth
                )
                if right_cand is not None:
                    frontier.append(right_cand)

        self._finalize_leaves()
        return self

    # ----------------------------------------------------------- prediction
    def apply(self, X: np.ndarray) -> np.ndarray:
        """Return the leaf id (0..n_leaves_-1) reached by each row of ``X``."""
        if self.root_ is None:
            raise RuntimeError("Tree must be fit before calling apply().")
        X = np.asarray(X, dtype=np.float64)
        out = np.empty(X.shape[0], dtype=np.int64)
        for i in range(X.shape[0]):
            out[i] = self._traverse_leaf(X[i]).leaf_id
        return out

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return the (current) leaf value for each row of ``X``."""
        if self.root_ is None:
            raise RuntimeError("Tree must be fit before calling predict().")
        X = np.asarray(X, dtype=np.float64)
        out = np.empty(X.shape[0], dtype=np.float64)
        for i in range(X.shape[0]):
            out[i] = self._traverse_leaf(X[i]).value
        return out

    def set_leaf_values(self, values: np.ndarray) -> None:
        """Overwrite leaf values with externally computed gammas.

        Used by the boosting driver to replace the least-squares leaf means
        (computed during the tree's own least-squares fit) with the
        loss-specific terminal-region updates ``gamma_jm``.
        """
        values = np.asarray(values, dtype=np.float64)
        if values.shape != (self.n_leaves_,):
            raise ValueError(
                f"Expected {self.n_leaves_} leaf values, got shape {values.shape}."
            )
        for leaf, v in zip(self.leaves_, values):
            leaf.value = float(v)

    # --------------------------------------------------------------- helpers
    def _make_leaf(self, y: np.ndarray, indices: np.ndarray) -> _Node:
        sub = y[indices]
        node = _Node(
            n_samples=int(sub.size),
            sum_y=float(sub.sum()),
            sum_y_sq=float(np.dot(sub, sub)),
            value=float(sub.mean()) if sub.size else 0.0,
        )
        return node

    def _traverse_leaf(self, x_row: np.ndarray) -> _Node:
        node = self.root_
        while not node.is_leaf:
            if x_row[node.feature] <= node.threshold:
                node = node.left
            else:
                node = node.right
        return node

    def _depth_of(self, target: _Node) -> int:
        # Cheap recursive depth lookup: trees here are tiny so this is fine.
        def _walk(node: _Node, d: int) -> Optional[int]:
            if node is target:
                return d
            if node.is_leaf:
                return None
            for child in (node.left, node.right):
                if child is None:
                    continue
                got = _walk(child, d + 1)
                if got is not None:
                    return got
            return None

        depth = _walk(self.root_, 0)
        return 0 if depth is None else depth

    def _finalize_leaves(self) -> None:
        """Assign sequential leaf ids and collect leaves in deterministic order."""
        leaves: List[_Node] = []

        def _walk(node: _Node) -> None:
            if node.is_leaf:
                node.leaf_id = len(leaves)
                leaves.append(node)
                return
            _walk(node.left)
            _walk(node.right)

        _walk(self.root_)
        self.leaves_ = leaves
        self.n_leaves_ = len(leaves)

    # ----------------------------------------------------------- split logic
    def _best_split(
        self,
        node: _Node,
        indices: np.ndarray,
        X: np.ndarray,
        y: np.ndarray,
        depth: int,
    ) -> Optional[_Candidate]:
        """Find the best (feature, threshold) split for ``node``."""
        if indices.size < self.min_samples_split:
            return None
        if self.max_depth is not None and depth >= self.max_depth:
            return None

        n = indices.size
        sub_y = y[indices]
        total_sum = sub_y.sum()
        # SSE of the parent: sum(y^2) - (sum(y))^2 / n. We don't need the
        # constant sum(y^2) part for the *gain*; we only need the right
        # change. We'll express gain as the increase in
        # n_L*mean_L^2 + n_R*mean_R^2 over n*mean^2, which equals the
        # decrease in SSE.
        parent_term = (total_sum * total_sum) / n

        best_gain = 0.0
        best_feature = -1
        best_threshold = 0.0
        best_left_idx: Optional[np.ndarray] = None
        best_right_idx: Optional[np.ndarray] = None
        best_left_mean = 0.0
        best_right_mean = 0.0

        n_features = X.shape[1]
        for j in range(n_features):
            col = X[indices, j]
            order = np.argsort(col, kind="mergesort")
            col_sorted = col[order]
            y_sorted = sub_y[order]

            # Cumulative sum of y in sorted order: prefix sum left of split,
            # suffix is total - prefix.
            cum_y = np.cumsum(y_sorted)

            # We can split between i and i+1 iff col_sorted[i] != col_sorted[i+1].
            # Iterate through valid split points using a vectorized mask.
            # Number of left samples = i + 1 for split between i and i+1.
            # Skip splits that would leave a child below min_samples_leaf.
            min_leaf = max(1, self.min_samples_leaf)
            if n - 2 * min_leaf < 0:
                continue

            # Indices i in [min_leaf - 1, n - min_leaf - 1] inclusive, with
            # the additional constraint that the next value differs.
            i_start = min_leaf - 1
            i_end = n - min_leaf - 1
            if i_start > i_end:
                continue

            # Vectorized candidate evaluation across allowed positions.
            i_range = np.arange(i_start, i_end + 1)
            # Only positions where the next value differs are real split points.
            diff_mask = col_sorted[i_range] != col_sorted[i_range + 1]
            if not diff_mask.any():
                continue
            valid_i = i_range[diff_mask]

            n_left = valid_i + 1
            n_right = n - n_left
            sum_left = cum_y[valid_i]
            sum_right = total_sum - sum_left

            # Reduction in SSE for each candidate split.
            gains = (sum_left * sum_left) / n_left + (
                sum_right * sum_right
            ) / n_right - parent_term

            local_best = int(np.argmax(gains))
            local_gain = float(gains[local_best])
            if local_gain > best_gain:
                split_pos = int(valid_i[local_best])
                threshold = 0.5 * (
                    col_sorted[split_pos] + col_sorted[split_pos + 1]
                )
                left_local = order[: split_pos + 1]
                right_local = order[split_pos + 1 :]
                best_gain = local_gain
                best_feature = j
                best_threshold = float(threshold)
                best_left_idx = indices[left_local]
                best_right_idx = indices[right_local]
                best_left_mean = float(sum_left[local_best] / n_left[local_best])
                best_right_mean = float(
                    sum_right[local_best] / n_right[local_best]
                )

        if best_feature < 0 or best_gain <= 0.0:
            return None

        return _Candidate(
            node=node,
            indices=indices,
            gain=best_gain,
            feature=best_feature,
            threshold=best_threshold,
            left_indices=best_left_idx,
            right_indices=best_right_idx,
            left_value=best_left_mean,
            right_value=best_right_mean,
        )

    def _apply_split(self, cand: _Candidate, X: np.ndarray, y: np.ndarray) -> None:
        """Convert a candidate into a real split on the tree."""
        node = cand.node
        node.feature = cand.feature
        node.threshold = cand.threshold

        left = self._make_leaf(y, cand.left_indices)
        right = self._make_leaf(y, cand.right_indices)
        node.left = left
        node.right = right
        # Internal nodes no longer carry a leaf value; clear it for clarity.
        node.value = None
