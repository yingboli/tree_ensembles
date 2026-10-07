# tree_ensembles

Personal R&D / learning package: my default XGBoost setup (`tree_ensembles.xgb`) and BART via
bartz with posterior tools (`tree_ensembles.bart`). See README.md and CHANGELOG.md.

## Commands

- Env: `conda activate tree_ensembles` (`make env` creates it and the Jupyter kernel).
- Checks: `make check` (ruff, mypy, pytest). Format: `make format`.
- Before every commit: `pre-commit run --all-files` must pass. Commit/push only when asked.

## Conventions

- Readable over clever: small functions, plain numpy/pandas, comments only where non-obvious.
- numpydoc docstrings (Parameters / Returns / Notes) on every public function and class.
- No module-level settings read by functions: make them arguments with defaults.
- Estimators are scikit-learn compatible; extra library params go through `**xgb_params` /
  `**bartz_params` (`KwargsEstimator` in `_base.py`).
- Every change gets tests; user-visible changes get a CHANGELOG entry under "Unreleased".
- Verify claims about XGBoost/bartz behavior with a small experiment before relying on them.

## BART specifics

- bartz is pinned to 0.12.x; `bart/trees.py` is the only module reading bartz internals.
- bartz bins NaN poorly (issue drafted in `~/Dropbox/Yingbo/Projects/2026 bart/bartz_issue/`);
  estimators impute by default (`bart/missing.py`).
- Depth follows XGBoost: levels of splits (0 = single leaf); bartz `maxdepth` = XGBoost
  `max_depth` + 1.
- JAX runs asynchronously: time fits by waiting for a prediction.

## Working with me

- Don't post to external repos/services (issues, PRs) without my review of the final text.
- Data lives in `data/` (git-ignored); never commit data, model dumps or notebook checkpoints.
