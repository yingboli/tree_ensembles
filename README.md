# tree_ensembles

Tree ensembles for my R&D and learning: my default XGBoost setup (`tree_ensembles.xgb`)
and BART with posterior tools (`tree_ensembles.bart`, built on [bartz](https://github.com/bartz-org/bartz)).

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
pip install "git+https://github.com/yingboli/tree_ensembles.git@v0.3.0"
```

BART needs the optional `bart` extra (bartz, JAX and matplotlib):

```bash
pip install "tree_ensembles[bart] @ git+https://github.com/yingboli/tree_ensembles.git@v0.3.0"
```

## Development setup

```bash
make env                 # create the conda env (installs this repo editable) and its Jupyter kernel
conda activate tree_ensembles
pre-commit install       # run ruff on every commit
```

In Jupyter, pick the kernel **tree_ensembles**. After recreating or renaming the env,
`make kernel` registers it again.

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
| `learning_rate` | 0.1 | a number, or `"optuna"` to tune in [0.05, 0.3] (log scale) |
| `max_depth` | 6 | a number, or `"optuna"` to tune in {3, 4, 5, 6} |
| `tree_method`, `max_bin` | `"hist"`, max(256, (2n)^(1/3)) | `tree_method=...`, `max_bin=...` |
| `enable_categorical` | True (use pandas `category` columns) | `enable_categorical=False` |
| anything else | XGBoost's default | keyword arguments, e.g. `subsample=0.8, n_jobs=4` |

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
print(reg.optuna_best_params_)  # {'learning_rate': 0.151, 'max_depth': 3}
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

### Saving

```python
clf.dump("models/clf")  # model.ubj + estimator.pkl
clf = XGBDefaultClassifier().load("models/clf")  # like XGBoost's load_model
```

The trees are saved in XGBoost's own format (`save_model`), which later XGBoost versions can
read and which keeps the levels of `category` columns; the rest (settings, `params_`,
`classes_`, ...) is a small pickle.

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

## BART

```python
import jax

jax.config.update("jax_num_cpu_devices", 4)  # optional: run the 4 chains in parallel on CPU

from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split

from tree_ensembles.bart import BartRegressor

X, y = load_diabetes(return_X_y=True, as_frame=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

bart = BartRegressor(num_trees=200, n_save=1000, n_burn=1000, num_chains=4)
bart.fit(X_train, y_train)

bart.predict(X_test)  # posterior mean of f(x)
bart.predict_dist(X_test)  # mean, sd of f(x), posterior predictive sd
bart.predict_interval(X_test, prob=0.95, kind="predictive")  # 95% HPDI for a new y
bart.predict_samples(X_test)  # all posterior draws of f(x), shape (draws, rows)

print(bart.posterior_summary())  # sigma, tree size: mean, sd, 95% HPDI, R-hat, ESS
print(bart.diagnostics())  # convergence, including f(x) at 20 of the training rows

forest = bart.forest_summary()
print(forest.depth_distribution())  # share of trees by depth (0 = single leaf)
print(forest.variable_usage())  # splits per feature
print(bart.split_points().head())  # most used cut values
print(bart.format_tree(chain=0, draw=0, tree=0))  # one tree, readable
print(bart.variable_importance(X_test))  # split / row / gain shares with 95% HPDI
print(bart.interactions().head())  # feature pairs splitting as parent and child
```

### Trees as tables and plots (BART and XGBoost)

```python
from tree_ensembles.tree_plot import plot_tree, plot_trees
from tree_ensembles.xgb import XGBDefaultRegressor

table = bart.trees_to_dataframe(chains=0, draws=-1, X=X_test)  # one row per node
plot_tree(table[table["tree"] == 0])  # one tree: splits, leaf values, rows reaching each node
plot_trees(table[table["tree"] < 6])  # several trees, shared leaf color scale

xgb_table = XGBDefaultRegressor().fit(X_train, y_train).trees_to_dataframe(trees=[0, 1])
plot_trees(xgb_table, max_depth=2)  # same plots for XGBoost; deep trees cut at depth 2
```

Both `trees_to_dataframe` methods return the same columns (node, depth, is_leaf, feature,
cutpoint, condition, left, right, missing, leaf_value), so `tree_plot` draws either.
BART splits read `x <= cut`, XGBoost splits `x < cut`. Node sizes differ: BART's `num_rows`
counts the rows of the `X` you pass; XGBoost's `cover` is the sum of the hessians of the
training rows (the row count only for squared error without weights).

`BartClassifier` works the same way (probit BART), with `predict_proba` and intervals for p(x).
`bart.parameter_draws()` returns the raw (chain, draw) draws behind `posterior_summary()`.
A full walk-through on two TabReD datasets is in `notebooks/bart_demo.ipynb`.
Plots: `from tree_ensembles.bart.plots import plot_trace, plot_rank, plot_tree_sizes,
plot_variable_usage`.

- **Convergence:** `diagnostics()` warns when R-hat > 1.01 or ESS < 400 (Vehtari et al. 2021).
  Use several chains (default 4); in BART, sigma and the tree sizes often mix slowly, so check
  them before trusting intervals. Fix with a longer `n_burn` / `n_save` or more chains.
  It also reports `leaf_fill` (mean leaves per tree / most allowed by `maxdepth`, bartz's
  "leaves 2.6/32") and warns above 0.25: BART wants many small trees, so use more `num_trees`.
- **Variable importance:** `variable_importance(X)` gives, per feature and with a 95% HPDI
  from the posterior draws: `split_share` (share of splits), `inclusion_prob`, `row_share`
  (splits weighted by the rows they divide, like XGBoost's cover) and `gain_share` (share of the
  fitted function's variation from splits on the feature, computed from the trees' predictions,
  not from y). With the sparse (DART) prior, `BartRegressor(sparse=SparseConfig())`, it adds
  the posterior splitting probability `split_prob`.
- **Categorical features:** BART has no categorical splits: bartz splits every column at numeric
  cutpoints, so integer codes would be split into ranges of codes, whose order usually means
  nothing (binary 0/1 features are fine). By default (`order_categories=True`), each level of a pandas
  `category` column is replaced by its partially pooled mean of y (a normal
  hierarchical model with empirical Bayes variances, `PooledMeanEncoder`), which orders similar
  levels next to each other. With `order_categories=False` they raise an error; encode them
  yourself instead (e.g. one-hot). Only `category` columns count as categorical: text
  (`object` / `string`) columns raise an error asking for `.astype("category")` or
  `pd.to_numeric`. XGBoost handles pandas
  `category` columns natively (`enable_categorical=True`). Ordering by
  residuals instead of y (once, or at every split, which needs bartz changes) is discussed in
  `bart/categories.py`.
- **Missing values:** handled for you. bartz bins NaN poorly (every NaN counts as a distinct
  value when choosing cutpoints, crowding out real cutpoints and creating splits that do
  nothing; reported upstream). Until that is fixed, the BART estimators median-impute NaN
  (training medians, `impute_strategy="median"` or `"mean"`) and add a `<name>_missing` column for
  each feature missing in more than 50% of the training rows (`missing_indicator_threshold`;
  0 adds one for every feature with NaN). `model_feature_names_` lists the columns bartz sees.
  `impute_strategy=None` passes NaN to bartz unchanged. The step is also available on its own as
  `MissingValueImputer`.
- **Tree prior and thinning:** `power`, `base` (a node at depth d splits with probability
  `base / (1 + d) ** power`) and `n_skip` (keep every n-th draw) are named arguments.
- **Other bartz options** (`sparse`, `k`, `sigma_df`, ...) are keyword arguments, e.g.
  `BartRegressor(k=3.0)`. Sample weights and offsets are not supported yet.
- **Saving:** `bart.dump("folder")` / `BartRegressor().load("folder")` (like XGBoost's `load_model`) use bartz's own format,
  which depends on the bartz/JAX versions: good for caching, not for archiving. With many trees
  the file is large (about 4 GB for 10,000 trees x 1,000 draws).

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
  bart/              BART (from tree_ensembles.bart import ...)
    estimators.py    BartRegressor, BartClassifier
    intervals.py     hpdi, quantile_interval
    diagnostics.py   split_rhat, ess_bulk, ess_tail, summarize_draws
    missing.py       MissingValueImputer (NaN workaround for bartz)
    categories.py    PooledMeanEncoder (order categories by pooled mean y)
    trees.py         forest summaries (the only code reading bartz internals)
    importance.py    variable_importance, interactions
    plots.py         trace, rank, tree-size and variable-usage plots
  tree_plot.py       plot_tree, plot_trees (BART and XGBoost trees)
tests/
  xgb/               pytest tests for tree_ensembles.xgb
  bart/              pytest tests for tree_ensembles.bart
notebooks/           bart_demo.ipynb (BART walk-through) and exploratory notebooks
CHANGELOG.md         what changed in each version
```
