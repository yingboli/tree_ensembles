"""Small functions that compute my default XGBoost parameter values from the data."""

import math

import numpy as np
import pandas as pd
import xgboost as xgb


def ic_gamma(criterion: str, n: int, phi: float) -> float:
    """Split penalty gamma from an information criterion.

    XGBoost's split gain is the drop in deviance times phi (SSE for squared error,
    2 * neg. log-likelihood for logistic), so a split adding one leaf must "pay"
    the criterion's per-parameter penalty on that scale.
    """
    penalties = {
        "aic": 2.0,
        "hq": 3.0 * math.log(math.log(n)),
        "bic": math.log(n),
    }
    if criterion not in penalties:
        raise ValueError(f"criterion must be one of {list(penalties)}, got {criterion!r}")
    return penalties[criterion] * phi


def default_max_bin(n: int) -> int:
    """max(256, (2n)^(1/3)); only exceeds 256 for n above ~8.4 million."""
    return max(256, math.ceil((2 * n) ** (1 / 3)))


def estimate_phi(X: pd.DataFrame | np.ndarray, y: np.ndarray, random_state: int = 0) -> float:
    """Rough residual variance from a deliberately under-fitted XGBoost model."""
    model = xgb.XGBRegressor(
        n_estimators=100,
        max_depth=2,
        learning_rate=0.1,
        tree_method="hist",
        enable_categorical=True,
        base_score=float(np.mean(y)),
        random_state=random_state,
    )
    model.fit(X, y)
    residuals = y - model.predict(X)
    return float(np.mean(residuals**2))
