# xgb_trees

My default XGBoost implementation, for R&D and learning.

## Setup

```bash
make env                 # create the conda env from environment.yml
conda activate xgb_trees
pre-commit install       # run ruff on every commit
```

## Daily commands

```bash
make format      # auto-fix style with ruff
make check       # ruff + mypy + pytest
```

## Layout

```
src/xgb_trees/   package code (import xgb_trees)
tests/           pytest tests
notebooks/       exploratory notebooks
```
