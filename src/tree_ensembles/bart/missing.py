"""Missing-value preprocessing for BART, a temporary workaround for bartz's NaN handling.

bartz accepts NaN but bins it poorly: every NaN counts as a distinct value when choosing
cutpoints, which crowds out real cutpoints and creates splits that do nothing (see trees.py
and https://github.com/bartz-org/bartz/issues). Until that is fixed upstream, the BART
estimators impute NaN before fitting and can add missing-indicator columns.
"""

from typing import Literal

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class MissingValueImputer(TransformerMixin, BaseEstimator):
    """Impute NaN per feature and add indicator columns for often-missing features.

    Fill values and the choice of indicator columns are learned on the training rows and
    reused for new rows (also for features that had no NaN in training).

    Parameters
    ----------
    strategy : {"median", "mean"}, default "median"
        Value used to fill NaN in each feature, computed on the observed training values.
        A feature with no observed value is filled with 0.
    indicator_threshold : float, default 0.5
        A feature gets a 0/1 `<name>_missing` column when its missing rate in the training
        rows is above this. 0 adds an indicator for every feature with any NaN; 1 adds none.

    Attributes
    ----------
    fill_values_ : ndarray of shape (p,)
        Value that replaces NaN in each feature.
    missing_rate_ : ndarray of shape (p,)
        Share of missing training rows per feature.
    indicator_features_ : ndarray of int
        Indices of the features that get a missing indicator, in column order.

    Examples
    --------
    >>> imputer = MissingValueImputer(indicator_threshold=0.3).fit(X_train)
    >>> imputer.transform(X_test)              # filled features, then indicator columns
    >>> imputer.get_feature_names_out(names)   # names, then "<name>_missing"
    """

    def __init__(
        self,
        strategy: Literal["median", "mean"] = "median",
        indicator_threshold: float = 0.5,
    ) -> None:
        self.strategy = strategy
        self.indicator_threshold = indicator_threshold

    def fit(self, X: np.ndarray, y: object = None) -> "MissingValueImputer":
        """Learn the fill values and which features get an indicator.

        Parameters
        ----------
        X : array of shape (n, p)
            Training features, may contain NaN.
        y : ignored

        Returns
        -------
        self
        """
        if self.strategy not in ("median", "mean"):
            raise ValueError(f"strategy must be 'median' or 'mean', got {self.strategy!r}")
        if not 0 <= self.indicator_threshold <= 1:
            raise ValueError(
                f"indicator_threshold must be in [0, 1], got {self.indicator_threshold}"
            )
        X = np.asarray(X, dtype=np.float64)
        missing = np.isnan(X)  # (n, p), True where a value is missing
        self.missing_rate_ = missing.mean(axis=0)

        # One fill value per feature (column), from that column's observed values.
        # This only computes the values; transform() then replaces the NaN cells with them.
        has_observed = ~missing.all(axis=0)  # (p,), features with at least one value
        fill_values = np.zeros(X.shape[1])  # 0 for a feature with no observed value
        columns = X[:, has_observed]
        if self.strategy == "median":
            fill_values[has_observed] = np.nanmedian(columns, axis=0)  # one median per column
        else:
            fill_values[has_observed] = np.nanmean(columns, axis=0)  # one mean per column
        self.fill_values_ = fill_values
        self.indicator_features_ = np.flatnonzero(self.missing_rate_ > self.indicator_threshold)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Fill NaN and append the indicator columns.

        Parameters
        ----------
        X : array of shape (m, p)
            Features with the same columns as in fit, may contain NaN.

        Returns
        -------
        ndarray of shape (m, p + number of indicator features), float32
            The filled features, then one 0/1 column per indicator feature.
        """
        X = np.asarray(X, dtype=np.float32)
        missing = np.isnan(X)
        filled = np.where(missing, self.fill_values_.astype(np.float32), X)
        indicators = missing[:, self.indicator_features_].astype(np.float32)
        return np.hstack([filled, indicators])

    def get_feature_names_out(self, input_features: list[str]) -> list[str]:
        """Names of the transformed columns: the inputs, then `<name>_missing`."""
        names = list(input_features)
        return names + [f"{names[j]}_missing" for j in self.indicator_features_]
