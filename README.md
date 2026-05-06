# TreeBoost: Friedman's Gradient Boosting Machine, from scratch

A from-scratch Python implementation of the regression-focused parts of

> Friedman, J. H. (2001). *Greedy Function Approximation: A Gradient
> Boosting Machine.* Annals of Statistics, 29(5), 1189–1232.

The implementation covers Algorithms 2 (LS Boost), 3 (LAD TreeBoost), and 4
(M TreeBoost / Huber) and is built on a custom CART-style regression tree —
no scikit-learn, XGBoost, LightGBM, or CatBoost are used for the core
algorithm. Only `numpy` is used for low-level array math, plus `matplotlib`
for plots and `pytest` for tests.

## Project layout

```
paper/                        # the source paper
src/treeboost/
    __init__.py               # public API
    tree.py                   # custom CART-style regression tree
    losses.py                 # LS, LAD, Huber: F0, pseudo-responses, gamma_jm
    model.py                  # TreeBoostRegressor stagewise driver
tests/                        # pytest suite (30 tests)
experiments/
    run_regression_experiments.py
    results/                  # generated plots and summary.txt
analysis.md                   # writeup explaining what the results show
requirements.txt
conftest.py                   # makes ``src/`` importable for tests/scripts
```

## Quickstart

```bash
python -m venv venv
venv\Scripts\activate         # Windows
# source venv/bin/activate    # Unix
pip install -r requirements.txt

# Run the test suite
python -m pytest -q

# Run all three experiments and regenerate plots + summary.txt
python experiments/run_regression_experiments.py
```

`experiments/results/` will then contain:

- `convergence_clean.png` — train/val loss curves for LS, LAD, Huber
- `robustness_bar.png` — clean-test MSE/MAE under 10% contamination
- `robustness_true_vs_pred.png` — calibration plots per loss
- `shrinkage_curves.png` — effect of `learning_rate` for LS Boost
- `summary.txt` — headline metrics quoted in `analysis.md`

## API

```python
from treeboost import TreeBoostRegressor

model = TreeBoostRegressor(
    loss="huber",         # one of "ls", "lad", "huber"
    n_estimators=200,
    learning_rate=0.1,    # paper's nu
    max_leaves=8,         # paper's J
    huber_quantile=0.9,   # only used by Huber
)
model.fit(X_train, y_train, X_val=X_test, y_val=y_test)
preds = model.predict(X_test)
loss_curve = model.train_loss_   # list of length n_estimators + 1
```

`staged_predict(X)` and `staged_loss(X, y)` are also available for plotting
test loss vs `m`.

## Mapping to the paper

| Paper concept                            | Where it lives                                                                                |
|------------------------------------------|------------------------------------------------------------------------------------------------|
| `F_0(x) = argmin_c sum L(y, c)`          | `Loss.initial_prediction` (mean for LS, median for LAD/Huber)                                  |
| `y_tilde_i = -[dL/dF]_{F_{m-1}}`         | `Loss.negative_gradient`                                                                       |
| Tree fit by least squares to `y_tilde`   | `RegressionTree.fit` (best-first growth, SSE-reduction split search)                           |
| Terminal-region update `gamma_jm` (eq 18)| `Loss.leaf_update`, applied via `RegressionTree.set_leaf_values`                                |
| Adaptive Huber `delta_m` (Algorithm 4)   | `HuberLoss.update_state` (alpha-quantile of absolute residuals)                                |
| Shrinkage `nu` (Section 5)               | `TreeBoostRegressor.learning_rate`                                                             |

See `analysis.md` for what the experiments reveal about each piece.
