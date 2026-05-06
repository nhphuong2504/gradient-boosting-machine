"""Regression TreeBoost experiments for the analysis writeup.

Runs three reproducible experiments and writes plots + a metrics table to
``experiments/results``:

1. Convergence on a clean nonlinear regression problem (Friedman #1-style).
   Shows training and held-out loss curves for LS, LAD, and Huber.

2. Robustness on the same signal contaminated with heavy-tailed outliers.
   Compares MSE / MAE on the *clean* test signal across the three losses.

3. Effect of the learning rate (shrinkage) on LS Boost on the clean
   problem, mirroring Section 5 of the paper.

The script keeps the Python API tight: each experiment is a small, named
function that returns a dict of results, and ``main`` orchestrates them.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

# Make ``src/`` importable when running this file directly.
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import matplotlib

matplotlib.use("Agg")  # non-interactive backend; works headless
import matplotlib.pyplot as plt  # noqa: E402

from treeboost import TreeBoostRegressor  # noqa: E402

RESULTS_DIR = HERE / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------- data helpers
def friedman1_signal(X: np.ndarray) -> np.ndarray:
    """A nonlinear regression target with mixed effects.

    f(x) = 10 sin(pi x0 x1) + 20 (x2 - 0.5)^2 + 10 x3 + 5 x4

    Inspired by Friedman's #1 benchmark, but features are drawn uniformly
    in [0, 1] which is enough to exercise non-linearity and interactions.
    """
    return (
        10.0 * np.sin(np.pi * X[:, 0] * X[:, 1])
        + 20.0 * (X[:, 2] - 0.5) ** 2
        + 10.0 * X[:, 3]
        + 5.0 * X[:, 4]
    )


def make_clean_split(
    n_train: int = 600, n_test: int = 600, n_features: int = 5, seed: int = 0
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    X_train = rng.uniform(0.0, 1.0, size=(n_train, n_features))
    X_test = rng.uniform(0.0, 1.0, size=(n_test, n_features))
    f_train = friedman1_signal(X_train)
    f_test = friedman1_signal(X_test)
    sigma = 1.0
    y_train = f_train + sigma * rng.normal(size=n_train)
    y_test = f_test + sigma * rng.normal(size=n_test)
    return X_train, y_train, X_test, y_test


def make_contaminated_split(
    n_train: int = 600,
    n_test: int = 600,
    n_features: int = 5,
    frac_outliers: float = 0.10,
    outlier_scale: float = 25.0,
    seed: int = 1,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return a contaminated training set and a *clean* test set.

    The clean test set lets us measure how much each loss is fooled by
    training-time outliers, which is the core motivation for LAD/Huber.
    """
    rng = np.random.default_rng(seed)
    X_train = rng.uniform(0.0, 1.0, size=(n_train, n_features))
    X_test = rng.uniform(0.0, 1.0, size=(n_test, n_features))
    f_train = friedman1_signal(X_train)
    f_test = friedman1_signal(X_test)
    y_train = f_train + 1.0 * rng.normal(size=n_train)

    n_out = int(frac_outliers * n_train)
    out_idx = rng.choice(n_train, size=n_out, replace=False)
    signs = rng.choice([-1.0, 1.0], size=n_out)
    y_train[out_idx] += signs * (outlier_scale + 0.5 * outlier_scale * rng.standard_t(df=3, size=n_out))

    return X_train, y_train, X_test, f_test, out_idx


# ---------------------------------------------------------------- experiments
def experiment_convergence(
    n_estimators: int = 200,
    learning_rate: float = 0.1,
    max_leaves: int = 8,
) -> Dict[str, np.ndarray]:
    """Compare LS / LAD / Huber convergence on the clean Friedman signal."""
    X_train, y_train, X_test, y_test = make_clean_split()
    out: Dict[str, np.ndarray] = {}

    common = dict(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_leaves=max_leaves,
        track_loss=True,
    )

    for name in ["ls", "lad", "huber"]:
        model = TreeBoostRegressor(loss=name, **common).fit(
            X_train, y_train, X_val=X_test, y_val=y_test
        )
        out[f"{name}_train_loss"] = np.asarray(model.train_loss_)
        out[f"{name}_val_loss"] = np.asarray(model.val_loss_)
        # Test-set MSE / MAE for a fair cross-loss comparison.
        preds = model.predict(X_test)
        out[f"{name}_test_mse"] = float(np.mean((preds - y_test) ** 2))
        out[f"{name}_test_mae"] = float(np.mean(np.abs(preds - y_test)))

    # Plot per-loss convergence on its own training-loss scale.
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6), sharex=True)
    for ax, name in zip(axes, ["ls", "lad", "huber"]):
        ax.plot(out[f"{name}_train_loss"], label="train")
        ax.plot(out[f"{name}_val_loss"], label="val")
        ax.set_title(f"{name.upper()} TreeBoost loss")
        ax.set_xlabel("boosting iteration m")
        ax.set_ylabel(f"{name} loss")
        ax.grid(True, alpha=0.3)
        ax.legend()
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "convergence_clean.png", dpi=150)
    plt.close(fig)

    return out


def experiment_robustness(
    n_estimators: int = 250,
    learning_rate: float = 0.1,
    max_leaves: int = 8,
    huber_quantile: float = 0.85,
) -> Dict[str, float]:
    """Compare losses under heavy-tailed training contamination."""
    X_train, y_train, X_test, f_test, out_idx = make_contaminated_split()
    metrics: Dict[str, float] = {}
    fitted = {}

    common = dict(
        n_estimators=n_estimators, learning_rate=learning_rate, max_leaves=max_leaves
    )
    losses = {
        "ls": TreeBoostRegressor(loss="ls", **common),
        "lad": TreeBoostRegressor(loss="lad", **common),
        "huber": TreeBoostRegressor(loss="huber", huber_quantile=huber_quantile, **common),
    }
    for name, model in losses.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        metrics[f"{name}_clean_mse"] = float(np.mean((preds - f_test) ** 2))
        metrics[f"{name}_clean_mae"] = float(np.mean(np.abs(preds - f_test)))
        fitted[name] = model

    # Visualize contamination and the prediction quality on a 1-D slice.
    # Plot true-vs-predicted on the clean test set for each loss.
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharex=True, sharey=True)
    for ax, name in zip(axes, ["ls", "lad", "huber"]):
        preds = fitted[name].predict(X_test[: len(f_test)])
        ax.scatter(f_test, preds, s=8, alpha=0.5)
        lo = float(min(f_test.min(), preds.min()))
        hi = float(max(f_test.max(), preds.max()))
        ax.plot([lo, hi], [lo, hi], "k--", lw=1)
        ax.set_title(
            f"{name.upper()} — clean-test MSE = {metrics[f'{name}_clean_mse']:.3f}"
        )
        ax.set_xlabel("true f(x)")
        ax.set_ylabel("predicted F(x)")
        ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "robustness_true_vs_pred.png", dpi=150)
    plt.close(fig)

    # Bar chart of the two metrics across losses for the writeup.
    fig, ax = plt.subplots(figsize=(7, 4))
    width = 0.35
    x = np.arange(3)
    mses = [metrics[f"{l}_clean_mse"] for l in ["ls", "lad", "huber"]]
    maes = [metrics[f"{l}_clean_mae"] for l in ["ls", "lad", "huber"]]
    ax.bar(x - width / 2, mses, width, label="MSE")
    ax.bar(x + width / 2, maes, width, label="MAE")
    ax.set_xticks(x)
    ax.set_xticklabels(["LS", "LAD", "Huber"])
    ax.set_ylabel("error vs clean target on test set")
    ax.set_title("Robustness under 10% heavy-tailed contamination")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "robustness_bar.png", dpi=150)
    plt.close(fig)

    metrics["n_outliers"] = int(out_idx.size)
    return metrics


def experiment_shrinkage(
    n_estimators: int = 400,
    max_leaves: int = 6,
    rates: Tuple[float, ...] = (1.0, 0.5, 0.1, 0.05),
) -> Dict[str, List[float]]:
    """Show how the learning rate trades off speed vs final quality (LS Boost)."""
    X_train, y_train, X_test, y_test = make_clean_split(seed=2)
    curves: Dict[str, List[float]] = {}

    fig, ax = plt.subplots(figsize=(7, 4.2))
    for nu in rates:
        model = TreeBoostRegressor(
            loss="ls",
            n_estimators=n_estimators,
            learning_rate=nu,
            max_leaves=max_leaves,
        ).fit(X_train, y_train, X_val=X_test, y_val=y_test)
        curves[f"val_loss_nu={nu}"] = list(model.val_loss_)
        ax.plot(model.val_loss_, label=f"nu = {nu}")

    ax.set_xlabel("boosting iteration m")
    ax.set_ylabel("validation MSE / 2")
    ax.set_title("LS Boost: effect of learning rate (shrinkage)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "shrinkage_curves.png", dpi=150)
    plt.close(fig)

    return curves


# ---------------------------------------------------------------------- main
def main() -> None:
    print("Running convergence experiment ...")
    conv = experiment_convergence()
    print(
        "  test-set metrics (clean): "
        + ", ".join(
            f"{l}: MSE={conv[f'{l}_test_mse']:.3f} MAE={conv[f'{l}_test_mae']:.3f}"
            for l in ["ls", "lad", "huber"]
        )
    )

    print("Running robustness experiment ...")
    rob = experiment_robustness()
    print(
        f"  ({rob['n_outliers']} outliers injected) "
        + ", ".join(
            f"{l}: clean-MSE={rob[f'{l}_clean_mse']:.3f} clean-MAE={rob[f'{l}_clean_mae']:.3f}"
            for l in ["ls", "lad", "huber"]
        )
    )

    print("Running shrinkage experiment ...")
    experiment_shrinkage()

    # Persist a tidy summary of headline numbers so the analysis writeup can
    # quote them directly.
    summary_path = RESULTS_DIR / "summary.txt"
    with summary_path.open("w", encoding="utf-8") as f:
        f.write("# TreeBoost experiment summary\n\n")
        f.write("## Clean Friedman-#1 test set (n_train=n_test=600, sigma=1.0)\n")
        for l in ["ls", "lad", "huber"]:
            f.write(
                f"  {l.upper():>5}: test MSE = {conv[f'{l}_test_mse']:.4f}, "
                f"test MAE = {conv[f'{l}_test_mae']:.4f}\n"
            )

        f.write("\n## Contaminated training set (10% heavy-tail outliers)\n")
        f.write("Metrics computed against the *clean* test signal f(x):\n")
        for l in ["ls", "lad", "huber"]:
            f.write(
                f"  {l.upper():>5}: clean-MSE = {rob[f'{l}_clean_mse']:.4f}, "
                f"clean-MAE = {rob[f'{l}_clean_mae']:.4f}\n"
            )

    print(f"Wrote summary to {summary_path}")
    print(f"Plots saved under {RESULTS_DIR}")


if __name__ == "__main__":
    main()
