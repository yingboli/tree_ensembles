# Changelog

All notable changes to this project. Format based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versions follow
[Semantic Versioning](https://semver.org/).

## [Unreleased]

- `load` fills the estimator it is called on, like XGBoost's `load_model`:
  `reg = BartRegressor(); reg.load(folder)` or `reg = BartRegressor().load(folder)`. Breaking:
  `BartRegressor.load(folder)` becomes `BartRegressor().load(folder)`. Loading the other class or
  an empty folder raises a clear error; using an unfitted model raises `NotFittedError`.

### Fixed

- Models saved or pickled with 0.3.0 without extra parameters (stored as `bartz_params=None` /
  `xgb_params=None`) failed in `repr`, `get_params`, `clone` and refitting after loading; `None`
  is now read as "no extra parameters".

### Added

- `trees_to_dataframe` for BART (`BartRegressor` / `BartClassifier`) and XGBoost
  (`XGBDefault*`): every node of the selected trees as a table, like XGBoost's own, with the
  same columns for both (BART: optional `num_rows` from any `X`; XGBoost keeps its `cover`).
- BART `variable_importance(X)`: split share, inclusion probability, row-weighted split share,
  fit-variance gain share and (with the sparse prior) splitting probability, each with a
  posterior mean and HPDI; `interactions()`: feature pairs splitting as parent and child.
- BART: pandas `category` columns are ordered by default (`order_categories=True`): each level
  is replaced by its partially pooled (empirical Bayes) mean of y (`PooledMeanEncoder`), since
  BART has no categorical features; `order_categories=False` makes them raise a `TypeError`.
  Text (`object` / `string`) columns raise a `TypeError` that says to convert them with
  `.astype("category")` or `pd.to_numeric`, instead of a bare float-conversion error (or, for
  numbers stored as text, being read silently as numbers). The demo notebook no longer uses the categorical `x_cat`.
- `tree_ensembles.tree_plot`: `plot_tree` and `plot_trees` draw BART or XGBoost trees from
  that table (splits, leaf values, node sizes; `max_depth` to cut deep trees).
- `CLAUDE.md` with project conventions.
- XGBoost `dump(folder)` / `load(folder)` for `XGBDefaultClassifier` / `XGBDefaultRegressor`,
  like BART's: the trees in XGBoost's own format (`model.ubj`, readable by later XGBoost
  versions, keeps `category` levels) and the rest of the estimator in `estimator.pkl`.

### Changed

- BART `diagnostics()` reports the average acceptance rate as a single row (mean only) and no
  longer checks it for convergence, so it no longer adds to the convergence warning.
- BART `diagnostics()` summarizes f(x) at the probe rows in two rows (median and worst R-hat /
  ESS) and reports the share of passing points; `per_point=True` shows every point.
- BART `diagnostics()` reports `leaf_fill`, the mean leaves per tree out of the most that
  `maxdepth` allows (bartz's "leaves 2.6/32" progress number, as a share), and warns when it is
  above `leaf_fill_max=0.25`: large trees are a sign that more trees (`num_trees`) are needed.
- Extra XGBoost / bartz parameters are plain keyword arguments (`**xgb_params`,
  `**bartz_params`) instead of a dict, e.g. `XGBDefaultRegressor(subsample=0.8)` or
  `BartRegressor(k=3.0)`; they work with `get_params`, `set_params`, `clone` and
  `GridSearchCV`. Replace `xgb_params={"subsample": 0.8}` with `subsample=0.8`.
- Optuna samples `learning_rate` on a log scale (`log=True`) within [0.05, 0.3].
- Module-level settings became arguments: `fit(..., n_probe=20)` for the rows checked by
  `diagnostics()`, `batch_size=2000` in `predict_dist` / `predict_interval`, and
  `rhat_max=1.01`, `ess_min=400` in `summarize_draws`, `posterior_summary` and `diagnostics`.

## [0.3.0] - 2026-10-05

### Added

- `tree_ensembles.bart`: `BartRegressor` and `BartClassifier` (probit) wrapping bartz, with
  posterior draws, predictive sd, HPDI intervals, posterior summaries, convergence diagnostics
  (rank-normalized split R-hat, bulk/tail ESS), forest summaries (depth, leaves, variable usage,
  split points, readable trees), plots, and dump/load. Install with the `bart` extra.
- BART estimators impute NaN before fitting (median by default) and add missing-indicator
  columns for features missing in more than 50% of rows (`MissingValueImputer`), a workaround
  for how bartz bins NaN.
- `notebooks/bart_demo.ipynb`: BART walk-through on cooking-time (regression) and
  homesite-insurance (classification), compared with the XGBoost defaults.

## [0.2.0] - 2026-10-02

### Changed

- Renamed the project from `xgb_trees` to `tree_ensembles`, to make room for BART.
- XGBoost code moved into the `tree_ensembles.xgb` subpackage. Update imports:
  `from xgb_trees import X` → `from tree_ensembles.xgb import X`.

## [0.1.0] - 2026-10-02

### Added

- Project skeleton: ruff, mypy, pytest, pre-commit, Makefile, conda `environment.yml`.
- `XGBDefaultClassifier` and `XGBDefaultRegressor`: scikit-learn estimators with my
  XGBoost defaults (base_score = mean(y), gamma from AIC/HQ/BIC, hist with adaptive
  max_bin, categorical support, stop at the first empty tree).
- Optional Optuna TPE tuning of `learning_rate` and `max_depth`, on a random holdout or
  on a validation set passed to `fit(..., X_valid=, y_valid=)`.
- Sample weights (frequency weights) and offsets (`base_margin`).

### Changed

- Only Optuna's best parameters are stored (`optuna_best_params_`), not the Study
  object, so saved models don't depend on the Optuna version.
