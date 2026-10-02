# xgb_trees

My default XGBoost implementation, for R&D and learning.

## Setup

```bash
make env                 # create the conda env from environment.yml
conda activate xgb_trees
pre-commit install       # run ruff on every commit
```

## Quick start

```python
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from xgb_trees import XGBDefaultClassifier

X, y = load_breast_cancer(return_X_y=True, as_frame=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, stratify=y)

clf = XGBDefaultClassifier().fit(X_train, y_train)  # all my defaults
print(roc_auc_score(y_test, clf.predict_proba(X_test)[:, 1]))
print(clf.gamma_, clf.n_trees_)  # HQ penalty, rounds trained
print(clf.params_)  # every parameter actually used
```

`XGBDefaultRegressor` works the same way (`fit` / `predict`, plus `phi_`).

### Defaults

| Parameter | Default | Override with |
|---|---|---|
| `base_score` | mean(y_train) | `base_score=0.3` |
| `n_estimators` | 2000, stop at the first tree with no split | `n_estimators=500`, `stop_on_empty_tree=False` |
| `gamma` | HQ: 3·ln(ln n)·phi | `criterion="aic"` (2·phi), `"bic"` (ln n·phi), or `gamma=1.0` |
| `phi` | 1 (classifier); residual variance of an under-fitted model (regressor) | `phi=2.0` |
| `learning_rate` | 0.1 | a number, or `"optuna"` to tune in [0.05, 0.3] |
| `max_depth` | 6 | a number, or `"optuna"` to tune in {3, 4, 5, 6} |
| `tree_method`, `max_bin` | `"hist"`, max(256, (2n)^(1/3)) | `tree_method=...`, `max_bin=...` |
| `enable_categorical` | True (use pandas `category` columns) | `enable_categorical=False` |
| anything else | XGBoost's default | `xgb_params={"subsample": 0.8, "n_jobs": 4}` |

### Tuning with Optuna

```python
from sklearn.datasets import load_diabetes

from xgb_trees import XGBDefaultRegressor

X, y = load_diabetes(return_X_y=True, as_frame=True)

reg = XGBDefaultRegressor(
    learning_rate="optuna",  # TPE search on an 80/20 holdout, then refit on all data
    max_depth="optuna",
    n_optuna_trials=30,
).fit(X, y)
print(reg.optuna_best_params_)  # {'learning_rate': 0.0536, 'max_depth': 6}
print(reg.phi_, reg.gamma_, reg.n_trees_)
```

Only the best values are kept (`optuna_best_params_`, also merged into `params_`), not the
Optuna study, so a saved model does not depend on the installed Optuna version.

Both classes are scikit-learn estimators, so `clone`, `Pipeline`, `cross_val_score`
and `GridSearchCV` work as usual.

## Daily commands

```bash
make format      # auto-fix style with ruff
make check       # ruff + mypy + pytest
```

## Layout

```
src/xgb_trees/
  estimators.py  XGBDefaultClassifier, XGBDefaultRegressor
  defaults.py    gamma, max_bin and phi helpers
  callbacks.py   StopOnEmptyTree
tests/           pytest tests
notebooks/       exploratory notebooks
```
