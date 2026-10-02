# tree_ensembles

Tree ensembles for my R&D and learning: my default XGBoost setup (`tree_ensembles.xgb`),
with BART to come.

## Install from GitHub

To use the package in another project or on another machine (Python >= 3.12).
pip also installs numpy, pandas, scikit-learn, xgboost and optuna.

This repo is private, so pip needs GitHub access. Over HTTPS, git uses your saved GitHub
login, or asks for your username and a
[personal access token](https://github.com/settings/tokens) (not your password):

```bash
pip install "git+https://github.com/yingboli/tree_ensembles.git"
```

Don't put the token in the URL; it would be saved in shell history or notebooks.
With an SSH key added to GitHub, this also works:

```bash
pip install "git+ssh://git@github.com/yingboli/tree_ensembles.git"
```

Pin a release tag (see [CHANGELOG.md](CHANGELOG.md)) or a commit, so later changes here
don't break that project:

```bash
pip install "git+https://github.com/yingboli/tree_ensembles.git@v0.2.0"
```

## Development setup

```bash
make env                 # create the conda env from environment.yml (installs this repo editable)
conda activate tree_ensembles
pre-commit install       # run ruff on every commit
```

## Quick start

```python
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from tree_ensembles.xgb import XGBDefaultClassifier

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
| `base_score` | weighted mean(y_train); 0 with an offset | `base_score=0.3` |
| `n_estimators` | 2000, stop at the first tree with no split (and a tiny leaf) | `n_estimators=500`, `stop_on_empty_tree=False` |
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

from tree_ensembles.xgb import XGBDefaultRegressor

X, y = load_diabetes(return_X_y=True, as_frame=True)

reg = XGBDefaultRegressor(
    learning_rate="optuna",  # TPE search on an 80/20 holdout, then refit on all data
    max_depth="optuna",
    n_optuna_trials=30,
).fit(X, y)
print(reg.optuna_best_params_)  # {'learning_rate': 0.0536, 'max_depth': 6}
print(reg.phi_, reg.gamma_, reg.n_trees_)

# Or score the trials on your own validation set (e.g. the most recent period,
# or held-out groups). The final model is then trained on X_train only.
X_train, X_val, y_train, y_val = X[:350], X[350:], y[:350], y[350:]
reg = XGBDefaultRegressor(learning_rate="optuna", max_depth="optuna")
reg.fit(X_train, y_train, X_valid=X_val, y_valid=y_val)
print(reg.optuna_best_params_)
```

Only the best values are kept (`optuna_best_params_`, also merged into `params_`), not the
Optuna study, so a saved model does not depend on the installed Optuna version.

### Sample weights and offsets

```python
import numpy as np
from sklearn.datasets import load_diabetes

from tree_ensembles.xgb import XGBDefaultRegressor

X, y = load_diabetes(return_X_y=True, as_frame=True)
weights = np.where(X["sex"] > 0, 2.0, 1.0)  # frequency weights
offset = np.full(len(y), 150.0)  # offset on the margin scale

reg = XGBDefaultRegressor().fit(X, y, sample_weight=weights, base_margin=offset)
pred = reg.predict(X, base_margin=offset)  # pass the offset again
```

- `sample_weight` are **frequency weights**: weight 2 means the row counts twice, so the
  information criterion uses n = sum of weights. Weights that sum to less than e raise an error.
- `base_margin` is an offset on the margin scale (log-odds for the classifier), passed to
  XGBoost as given; there is no separate intercept. `predict` without an offset uses 0.
- Boosting only stops at an empty tree once its leaf is tiny, so an overall shift not
  covered by the offset is still learned.
- For tuning with your own validation set, also pass `sample_weight_valid` /
  `base_margin_valid` (the latter is required when an offset is used).

Both classes are scikit-learn estimators, so `clone`, `Pipeline`, `cross_val_score`
and `GridSearchCV` work as usual.

## Daily commands

```bash
make format      # auto-fix style with ruff
make check       # ruff + mypy + pytest
```

## Layout

```
src/tree_ensembles/
  xgb/               XGBoost (from tree_ensembles.xgb import ...)
    estimators.py    XGBDefaultClassifier, XGBDefaultRegressor
    defaults.py      gamma, max_bin and phi helpers
    callbacks.py     StopOnEmptyTree
  bart/              BART (to come)
tests/
  xgb/               pytest tests for tree_ensembles.xgb
notebooks/           exploratory notebooks
CHANGELOG.md         what changed in each version
```
