"""Credible intervals from posterior draws."""

import math

import numpy as np


def hpdi(draws: np.ndarray, prob: float = 0.95, axis: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Highest posterior density interval: the narrowest interval holding `prob` of the draws.

    Sort the draws and slide a window of k = ceil(prob * n) consecutive draws; the
    narrowest window is the HPDI. Computed independently for every position along
    the other axes (e.g. one interval per test point). Assumes a unimodal posterior.

    Parameters
    ----------
    draws : ndarray
        Posterior draws, with draws along `axis`.
    prob : float, default 0.95
        Probability inside the interval, in (0, 1).
    axis : int, default 0
        Axis of the draws.

    Returns
    -------
    (lower, upper) : tuple of ndarrays
        Shaped like `draws` without `axis`.
    """
    if not 0 < prob < 1:
        raise ValueError(f"prob must be in (0, 1), got {prob}")
    x = np.sort(np.moveaxis(np.asarray(draws, dtype=float), axis, 0), axis=0)
    n = x.shape[0]
    k = math.ceil(prob * n)  # number of draws inside the interval
    widths = x[k - 1 :] - x[: n - k + 1]  # window i covers x[i] ... x[i + k - 1]
    start = np.argmin(widths, axis=0)[None]
    lower = np.take_along_axis(x, start, axis=0)[0]
    upper = np.take_along_axis(x, start + k - 1, axis=0)[0]
    return lower, upper


def quantile_interval(
    draws: np.ndarray, prob: float = 0.95, axis: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    """Equal-tailed interval: the (1 - prob)/2 and (1 + prob)/2 quantiles of the draws.

    Same probability in each tail; wider than the HPDI for a skewed posterior.

    Parameters
    ----------
    draws : ndarray
        Posterior draws, with draws along `axis`.
    prob : float, default 0.95
        Probability inside the interval, in (0, 1).
    axis : int, default 0
        Axis of the draws.

    Returns
    -------
    (lower, upper) : tuple of ndarrays
        Shaped like `draws` without `axis`.
    """
    if not 0 < prob < 1:
        raise ValueError(f"prob must be in (0, 1), got {prob}")
    lower, upper = np.quantile(draws, [(1 - prob) / 2, (1 + prob) / 2], axis=axis)
    return lower, upper
