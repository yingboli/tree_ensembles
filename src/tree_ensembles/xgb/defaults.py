"""Small functions that compute my default XGBoost parameter values from the data."""

import math

import numpy as np
import pandas as pd
import xgboost as xgb


def ic_gamma(criterion: str, n: float, phi: float) -> float:
    """Split penalty gamma from an information criterion.

    XGBoost's split gain is the drop in (weighted) deviance times phi (SSE for squared
    error, 2 * neg. log-likelihood for logistic), so a split adding one leaf must "pay"
    the criterion's per-parameter penalty on that scale. With frequency weights,
    n is the sum of the weights.

    Parameters
    ----------
    criterion : {"aic", "hq", "bic"}
        Penalty per split: 2 (AIC), 3 * ln(ln n) (Hannan-Quinn) or ln n (BIC).
    n : float
        Number of observations (sum of the weights with frequency weights).
    phi : float
        Dispersion: 1 for logistic loss, the noise variance for squared error.

    Returns
    -------
    float
        The split penalty, to pass to XGBoost as `gamma`.

    Raises
    ------
    ValueError
        For an unknown criterion, or n <= e for "hq" / "bic" (the penalty would be
        negative or meaningless), which usually means weights that are not frequencies.
    """
    if criterion not in ("aic", "hq", "bic"):
        raise ValueError(f"criterion must be 'aic', 'hq' or 'bic', got {criterion!r}")
    if criterion == "aic":
        return 2.0 * phi
    if n <= math.e:
        raise ValueError(
            f"n = {n:.3g} is too small for {criterion!r}; with sample weights, "
            "n is the sum of the weights (frequency weights)"
        )
    if criterion == "hq":
        return 3.0 * math.log(math.log(n)) * phi
    return math.log(n) * phi


def default_max_bin(n: int) -> int:
    """Number of histogram bins per feature: max(256, ceil((2 n) ** (1/3))).

    The cube-root rule only exceeds XGBoost's default of 256 for n above ~8.4 million rows.

    Parameters
    ----------
    n : int
        Number of training rows.

    Returns
    -------
    int
    """
    return max(256, math.ceil((2 * n) ** (1 / 3)))


def estimate_phi(
    X: pd.DataFrame | np.ndarray,
    y: np.ndarray,
    sample_weight: np.ndarray | None = None,
    base_margin: np.ndarray | None = None,
    random_state: int = 0,
) -> float:
    """Rough (weighted) residual variance from a deliberately under-fitted XGBoost model.

    Fits 100 depth-2 trees with learning rate 0.1 and returns the weighted mean squared
    training residual. Being under-fitted, it errs on the high side, which makes the
    split penalty a little more cautious.

    Parameters
    ----------
    X : DataFrame or array of shape (n, p)
        Training features.
    y : ndarray of shape (n,)
        Training target.
    sample_weight : ndarray of shape (n,), optional
        Frequency weights.
    base_margin : ndarray of shape (n,), optional
        Offset on the margin scale.
    random_state : int, default 0
        XGBoost seed.

    Returns
    -------
    float
        Estimated noise variance phi.
    """
    model = xgb.XGBRegressor(
        n_estimators=100,
        max_depth=2,
        learning_rate=0.1,
        tree_method="hist",
        enable_categorical=True,
        base_score=float(np.average(y, weights=sample_weight)),
        random_state=random_state,
    )
    model.fit(X, y, sample_weight=sample_weight, base_margin=base_margin)
    residuals = y - model.predict(X, base_margin=base_margin)
    return float(np.average(residuals**2, weights=sample_weight))
