# Changelog

All notable changes to this project. Format based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versions follow
[Semantic Versioning](https://semver.org/).

## [Unreleased]

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
