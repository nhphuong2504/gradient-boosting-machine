"""TreeBoostRegressor: stagewise additive model driver.

Implements Friedman's gradient boosting machine for regression. The same
class handles all three regression variants in the paper (LS, LAD, Huber)
because the only loss-specific code lives behind the ``Loss`` interface in
``losses.py``.

The training loop directly mirrors Algorithm 1 specialized to regression
trees (Algorithm 2/3/4 in the paper):

    F_0(x) = argmin_c sum_i L(y_i, c)              # see Loss.initial_prediction

    for m in 1..M:
        delta_m = chosen by loss (e.g. Huber quantile)   # Loss.update_state
        y_tilde_i = -dL/dF(y_i, F_{m-1}(x_i))            # Loss.negative_gradient
        Fit J-leaf regression tree to (X, y_tilde)       # RegressionTree.fit
        For each leaf R_jm:
            gamma_jm = argmin_gamma sum_{i in R_jm} L(y_i, F_{m-1}(x_i)+gamma)
                                                          # Loss.leaf_update
        F_m(x) = F_{m-1}(x) + nu * sum_j gamma_jm * 1{x in R_jm}

A ``learning_rate`` (``nu``) is included as a standard practical addition;
setting it to 1.0 recovers the paper's stagewise updates.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, List, Optional, Tuple, Union

import numpy as np

from .losses import Loss, get_loss
from .tree import RegressionTree


@dataclass
class TreeBoostRegressor:
    """Friedman-style regression TreeBoost.

    Parameters
    ----------
    loss:
        Either a string (``"ls"``, ``"lad"``, ``"huber"``) or a ``Loss``
        instance. Strings are resolved through :func:`losses.get_loss`.
    n_estimators:
        Number of boosting iterations ``M``.
    learning_rate:
        Shrinkage factor ``nu`` applied to each iteration's update. Default
        of 0.1 matches the paper's recommendation; 1.0 recovers the
        un-shrunk algorithm.
    max_leaves:
        Number of terminal nodes ``J`` per tree.
    max_depth:
        Optional hard depth cap for each tree.
    min_samples_split / min_samples_leaf:
        Standard tree growth guards.
    huber_quantile:
        Convenience parameter used only when ``loss="huber"`` and a
        non-default quantile is desired.
    track_loss:
        If True, the per-iteration training loss (and validation loss if
        ``X_val``/``y_val`` are provided to ``fit``) is recorded in
        ``train_loss_`` / ``val_loss_``.
    random_state:
        Reserved for future stochastic extensions; currently unused but
        accepted to keep the API stable.
    """

    loss: Union[str, Loss] = "ls"
    n_estimators: int = 100
    learning_rate: float = 0.1
    max_leaves: int = 8
    max_depth: Optional[int] = None
    min_samples_split: int = 2
    min_samples_leaf: int = 1
    huber_quantile: float = 0.9
    track_loss: bool = True
    random_state: Optional[int] = None

    # ----------------------------------------------------- learned attributes
    loss_: Optional[Loss] = field(default=None, init=False, repr=False)
    F0_: float = field(default=0.0, init=False, repr=False)
    trees_: List[RegressionTree] = field(default_factory=list, init=False, repr=False)
    train_loss_: List[float] = field(default_factory=list, init=False, repr=False)
    val_loss_: List[float] = field(default_factory=list, init=False, repr=False)
    n_features_: int = field(default=0, init=False, repr=False)

    # ------------------------------------------------------------------- fit
    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
    ) -> "TreeBoostRegressor":
        """Train the boosted ensemble on ``(X, y)``.

        Parameters
        ----------
        X: (n_samples, n_features) feature matrix.
        y: (n_samples,) target vector.
        X_val, y_val: optional validation set for loss tracking only; they
            do *not* affect training.
        """
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        if X.ndim != 2:
            raise ValueError("X must be 2-d.")
        if y.shape[0] != X.shape[0]:
            raise ValueError("X and y must have the same number of rows.")
        if self.n_estimators < 0:
            raise ValueError("n_estimators must be >= 0.")
        if not (0.0 < self.learning_rate <= 1.0):
            raise ValueError("learning_rate must be in (0, 1].")

        # Resolve loss object once.
        if isinstance(self.loss, str):
            kwargs = {}
            if self.loss.lower() == "huber":
                kwargs["quantile"] = self.huber_quantile
            self.loss_ = get_loss(self.loss, **kwargs)
        else:
            self.loss_ = self.loss

        # Step 1 of Algorithm 1: F_0 is the loss-minimizing constant.
        self.F0_ = self.loss_.initial_prediction(y)
        F = np.full(y.shape, self.F0_, dtype=np.float64)

        self.trees_ = []
        self.train_loss_ = []
        self.val_loss_ = []
        self.n_features_ = X.shape[1]

        # Pre-compute validation predictions if needed.
        F_val: Optional[np.ndarray] = None
        if X_val is not None and y_val is not None:
            X_val = np.asarray(X_val, dtype=np.float64)
            y_val = np.asarray(y_val, dtype=np.float64)
            F_val = np.full(y_val.shape, self.F0_, dtype=np.float64)

        if self.track_loss:
            self.train_loss_.append(self.loss_.loss_value(y, F))
            if F_val is not None:
                self.val_loss_.append(self.loss_.loss_value(y_val, F_val))

        # Steps 2-7 of Algorithm 1.
        for _ in range(self.n_estimators):
            # Loss-specific pre-iteration state (e.g. Huber's adaptive delta).
            self.loss_.update_state(y, F)

            # Pseudo-responses (negative gradient).
            y_tilde = self.loss_.negative_gradient(y, F)

            # Fit a J-leaf regression tree to the pseudo-responses.
            tree = RegressionTree(
                max_leaves=self.max_leaves,
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
            )
            tree.fit(X, y_tilde)

            # Compute terminal-region updates gamma_jm and overwrite leaf values.
            leaf_indices = tree.apply(X)
            gammas = self._compute_leaf_updates(tree, leaf_indices, y, F)
            tree.set_leaf_values(gammas)

            # Update F on training data using the (now gamma-valued) tree.
            increment = tree.predict(X)
            F += self.learning_rate * increment

            self.trees_.append(tree)

            if self.track_loss:
                self.train_loss_.append(self.loss_.loss_value(y, F))
                if F_val is not None:
                    F_val += self.learning_rate * tree.predict(X_val)
                    self.val_loss_.append(self.loss_.loss_value(y_val, F_val))

        return self

    # ----------------------------------------------------------- prediction
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return ``F_M(x)`` for each row of ``X``."""
        self._check_fitted()
        X = np.asarray(X, dtype=np.float64)
        F = np.full(X.shape[0], self.F0_, dtype=np.float64)
        for tree in self.trees_:
            F += self.learning_rate * tree.predict(X)
        return F

    def staged_predict(self, X: np.ndarray) -> Iterable[np.ndarray]:
        """Yield the prediction after each boosting iteration.

        Useful for plotting test-loss curves vs ``M``.
        """
        self._check_fitted()
        X = np.asarray(X, dtype=np.float64)
        F = np.full(X.shape[0], self.F0_, dtype=np.float64)
        # Yield F_0 first so the caller can track from m = 0.
        yield F.copy()
        for tree in self.trees_:
            F = F + self.learning_rate * tree.predict(X)
            yield F.copy()

    def staged_loss(
        self, X: np.ndarray, y: np.ndarray
    ) -> List[float]:
        """Return per-iteration loss on ``(X, y)`` using the trained loss."""
        self._check_fitted()
        y = np.asarray(y, dtype=np.float64)
        return [self.loss_.loss_value(y, F) for F in self.staged_predict(X)]

    # --------------------------------------------------------------- helpers
    def _compute_leaf_updates(
        self,
        tree: RegressionTree,
        leaf_indices: np.ndarray,
        y: np.ndarray,
        F: np.ndarray,
    ) -> np.ndarray:
        """Compute ``gamma_jm`` for every leaf of ``tree`` (equation (18))."""
        gammas = np.zeros(tree.n_leaves_, dtype=np.float64)
        for j in range(tree.n_leaves_):
            in_leaf = np.where(leaf_indices == j)[0]
            gammas[j] = self.loss_.leaf_update(y, F, in_leaf)
        return gammas

    def _check_fitted(self) -> None:
        if not self.trees_ and self.loss_ is None:
            raise RuntimeError(
                "TreeBoostRegressor has not been fit yet; call .fit(X, y) first."
            )
