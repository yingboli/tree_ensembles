"""Order categorical levels by a partially pooled (empirical Bayes) mean of y.

bartz has no categorical features: every column is split at numeric cutpoints, so a
categorical column given as integer codes is split into contiguous ranges of codes, whose
order may mean nothing. Replacing each level by its partially pooled mean of y puts similar
levels next to each other, so a single split can separate low-y levels from high-y ones.

Model, per categorical column (one-way random effects):

    y_i = mu + alpha_j + e_i,   alpha_j ~ N(0, tau^2),   e_i ~ N(0, sigma^2),

with j the level of row i. The posterior mean of level j's mean is

    theta_j = mu + B_j * (ybar_j - mu),   B_j = tau^2 / (tau^2 + sigma^2 / n_j),

so a level with many rows keeps about its own mean, and a rare level is pulled toward mu.
mu, sigma^2 and tau^2 are estimated from the data (empirical Bayes, method of moments of the
unbalanced one-way ANOVA):

    sigma^2 = within-level sum of squares / (N - J)
    tau^2   = max(0, (MSB - sigma^2) / n0),  MSB = sum_j n_j (ybar_j - ybar)^2 / (J - 1),
              n0 = (N - sum_j n_j^2 / N) / (J - 1)
    mu      = mean of y.

Missing values form a level of their own; levels not seen in training get mu.

Alternatives (not implemented):

- Order by residuals at every split. Within the MCMC, order the levels at each node by the
  mean partial residual (y minus the other trees) of the rows reaching it, as LightGBM does
  with gradients; for squared loss the best two-way grouping of levels is contiguous in
  that order (Fisher 1958). This needs changes to bartz itself: it stores a split as a
  cutpoint index over a fixed order of each feature and routes rows by bin >= split, while
  such a split is a "level in subset" rule that changes per node and draw. It needs a new
  tree representation, grow/prune moves, likelihood, prediction and a careful
  Metropolis-Hastings ratio. Published BART work on categorical splits (e.g. flexBART)
  assigns random subsets of levels to each branch.
- Order by residuals once (two-stage). Fit BART without the categorical columns, order the
  levels by the partially pooled mean of y - f_hat(x) (the same model as above with the
  residuals as target), then refit. No bartz change. Unlike ordering by y, it uses only
  what a category adds beyond correlated numeric features. But it is still one global
  order (no interactions), uses y twice and doubles the fitting time.
"""

from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

_MISSING = "<missing>"  # the level that missing values form


def categorical_columns(X: Any) -> list[str]:
    """Names of the pandas `category` columns of X (none for an array)."""
    if not isinstance(X, pd.DataFrame):
        return []
    return [str(c) for c in X.columns if isinstance(X[c].dtype, pd.CategoricalDtype)]


def text_columns(X: Any) -> list[str]:
    """Names of the `object` and `string` columns of X (none for an array).

    BART does not guess whether such a column holds categories or numbers stored as text;
    it asks for `category` (or numeric) columns instead.
    """
    if not isinstance(X, pd.DataFrame):
        return []
    return [
        str(c) for c in X.columns if X[c].dtype == object or isinstance(X[c].dtype, pd.StringDtype)
    ]


def _levels(column: pd.Series) -> pd.Series:
    """The column's values as objects, with missing values replaced by the _MISSING level."""
    values = column.astype(object)
    return values.where(column.notna(), _MISSING)


def pooled_level_means(levels: pd.Series, y: np.ndarray) -> pd.DataFrame:
    """Empirical Bayes partially pooled mean of y for every level (see the module notes).

    Parameters
    ----------
    levels : Series of shape (n,)
        The level of every row.
    y : ndarray of shape (n,)
        The target (0/1 for a binary target).

    Returns
    -------
    DataFrame indexed by level with columns n, mean_y, weight (B_j: 1 = own mean, 0 = fully
    pooled) and pooled_mean (theta_j), plus the estimates as attributes: attrs["mu"],
    attrs["sigma2"] and attrs["tau2"].
    """
    y = np.asarray(y, dtype=float)
    stats = pd.DataFrame({"level": levels.to_numpy(), "y": y}).groupby("level", sort=False)["y"]
    n_j, mean_j = stats.count().to_numpy(dtype=float), stats.mean().to_numpy(dtype=float)
    n, n_levels = len(y), len(n_j)
    mu = float(y.mean())

    within = float(((y - stats.transform("mean").to_numpy(dtype=float)) ** 2).sum())
    sigma2 = within / (n - n_levels) if n > n_levels else float(y.var())
    if n_levels > 1:
        msb = float((n_j * (mean_j - mu) ** 2).sum()) / (n_levels - 1)
        n0 = (n - float((n_j**2).sum()) / n) / (n_levels - 1)
        tau2 = max(0.0, (msb - sigma2) / n0)
    else:
        tau2 = 0.0
    with np.errstate(invalid="ignore", divide="ignore"):
        weight = np.where(tau2 > 0, tau2 / (tau2 + sigma2 / n_j), 0.0)

    table = pd.DataFrame(
        {
            "n": n_j.astype(int),
            "mean_y": mean_j,
            "weight": weight,
            "pooled_mean": mu + weight * (mean_j - mu),
        },
        index=pd.Index(stats.count().index, name="level"),
    )
    table.attrs.update(mu=mu, sigma2=sigma2, tau2=tau2)
    return table.sort_values("pooled_mean")


class PooledMeanEncoder(TransformerMixin, BaseEstimator):
    """Replace each categorical level by its partially pooled mean of y (empirical Bayes).

    Trees only use the order of a column's values, so this orders the levels by their
    pooled mean of y; the values themselves are in units of y. See the module notes for the
    model.

    Parameters
    ----------
    columns : list of str, optional
        Columns to encode. Default: the pandas `category` columns of X in fit.

    Attributes
    ----------
    columns_ : list of str
        The encoded columns.
    levels_ : dict of str -> DataFrame
        Per column, the table from pooled_level_means (n, mean_y, weight, pooled_mean per
        level, and the mu / sigma2 / tau2 estimates in its attrs).

    Notes
    -----
    The encoding uses y of the training rows, so it slightly favors the levels as seen in
    training; partial pooling limits this for rare levels.
    """

    def __init__(self, columns: list[str] | None = None) -> None:
        self.columns = columns

    def fit(self, X: pd.DataFrame, y: Any) -> "PooledMeanEncoder":
        """Estimate the pooled level means of every encoded column.

        Parameters
        ----------
        X : DataFrame
            Features; only the encoded columns are used.
        y : array-like of shape (n,)
            Target (0/1 for a binary target).

        Returns
        -------
        self
        """
        self.columns_ = list(self.columns) if self.columns is not None else categorical_columns(X)
        y_arr = np.asarray(y, dtype=float)
        self.levels_ = {c: pooled_level_means(_levels(X[c]), y_arr) for c in self.columns_}
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """X with every encoded column replaced by the pooled mean of its level (float).

        Missing values use the missing level's mean if training had one, else mu; levels
        not seen in training get mu.
        """
        X = X.copy()
        for c in self.columns_:
            table = self.levels_[c]
            mapping = table["pooled_mean"].to_dict()
            X[c] = _levels(X[c]).map(mapping).fillna(table.attrs["mu"]).astype(float)
        return X
