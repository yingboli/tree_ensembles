"""Small, fast BART fits shared by the bart tests (JAX compiles once per session)."""

from typing import Any

import numpy as np
import pandas as pd
import pytest

from tree_ensembles.bart import BartClassifier, BartRegressor

SMALL: dict[str, Any] = {
    "num_trees": 50,
    "n_save": 300,
    "n_burn": 300,
    "num_chains": 2,
    "show_progress": False,
}


@pytest.fixture(scope="session")
def regression_data() -> tuple[pd.DataFrame, np.ndarray]:
    """y = sin(2a) + 0.5 b + N(0, 0.3^2); c is pure noise."""
    rng = np.random.default_rng(0)
    n = 1000
    X = pd.DataFrame(rng.normal(size=(n, 3)), columns=["a", "b", "c"])
    y = np.sin(2 * X["a"].to_numpy()) + 0.5 * X["b"].to_numpy() + 0.3 * rng.normal(size=n)
    return X, y


@pytest.fixture(scope="session")
def fitted_regressor(regression_data: tuple) -> BartRegressor:
    X, y = regression_data
    return BartRegressor(**SMALL).fit(X[:800], y[:800])


@pytest.fixture(scope="session")
def fitted_classifier(regression_data: tuple) -> BartClassifier:
    X, y = regression_data
    labels = np.where(y > 0, "yes", "no")
    return BartClassifier(**SMALL).fit(X[:800], labels[:800])
