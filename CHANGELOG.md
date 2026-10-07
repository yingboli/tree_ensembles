# Changelog

All notable changes to this project. Format based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versions follow
[Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added

- `trees_to_dataframe` for BART (`BartRegressor` / `BartClassifier`) and XGBoost
  (`XGBDefault*`): every node of the selected trees as a table, like XGBoost's own, with the
  same columns for both (BART: optional `num_rows` from any `X`; XGBoost keeps its `cover`).
- `tree_ensembles.tree_plot`: `plot_tree` and `plot_trees` draw BART or XGBoost trees from
  that table (splits, leaf values, node sizes; `max_depth` to cut deep trees).
- `CLAUDE.md` with project conventions.

### Changed

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
