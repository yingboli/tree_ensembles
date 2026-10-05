"""MCMC convergence diagnostics: R-hat and effective sample size.

Follows Vehtari, Gelman, Simpson, Carpenter and Buerkner (2021), "Rank-normalization,
folding, and localization: an improved R-hat for assessing convergence of MCMC",
the same definitions used by Stan and ArviZ.

All functions take draws of one scalar quantity shaped (chain, draw); a 1-D array is
treated as a single chain.
"""

import numpy as np
import pandas as pd
from scipy.stats import norm, rankdata

from tree_ensembles.bart.intervals import hpdi

# Usual thresholds (Vehtari et al. 2021): R-hat below 1.01 and ESS above 400.
RHAT_MAX = 1.01
ESS_MIN = 400


def _as_chains(draws: np.ndarray) -> np.ndarray:
    """Draws as a float (chain, draw) array; a 1-D array becomes a single chain."""
    x = np.asarray(draws, dtype=float)
    return x[None, :] if x.ndim == 1 else x


def _split_chains(x: np.ndarray) -> np.ndarray:
    """Split every chain in two halves (drops the middle draw if odd), doubling the chains."""
    half = x.shape[1] // 2
    return np.concatenate([x[:, :half], x[:, -half:]])


def _rank_normalize(x: np.ndarray) -> np.ndarray:
    """Replace draws by normal scores of their ranks, pooled over all chains."""
    ranks = rankdata(x, method="average").reshape(x.shape)
    return norm.ppf((ranks - 3 / 8) / (x.size + 1 / 4))


def _rhat(x: np.ndarray) -> float:
    """Classic potential scale reduction factor on (chain, draw)."""
    n = x.shape[1]
    within = np.mean(np.var(x, axis=1, ddof=1))
    between = n * np.var(np.mean(x, axis=1), ddof=1)
    var_plus = (n - 1) / n * within + between / n
    return float(np.sqrt(var_plus / within))


def split_rhat(draws: np.ndarray) -> float:
    """Rank-normalized split R-hat: max of the bulk and the folded (tail) version.

    Compares the variance between chains with the variance within chains, after splitting
    every chain in two halves (to also catch a trend within a chain) and replacing draws by
    normal scores of their ranks (robust to heavy tails). The folded version, on
    |draw - median|, checks the spread. Values near 1 mean the chains agree; above 1.01
    is a warning sign.

    Parameters
    ----------
    draws : ndarray of shape (chain, draw), or (draw,) for one chain

    Returns
    -------
    float
        NaN if all draws are equal.
    """
    x = _split_chains(_as_chains(draws))
    if np.ptp(x) == 0:
        return float("nan")
    bulk = _rhat(_rank_normalize(x))
    folded = _rhat(_rank_normalize(np.abs(x - np.median(x))))
    return max(bulk, folded)


def _autocovariance(x: np.ndarray) -> np.ndarray:
    """Autocovariance of every chain at lags 0 ... n - 1, via FFT."""
    n = x.shape[1]
    centered = x - x.mean(axis=1, keepdims=True)
    f = np.fft.rfft(centered, n=2 * n)
    return np.fft.irfft(f * np.conj(f))[:, :n] / n


def _ess(x: np.ndarray) -> float:
    """Effective sample size of (chain, draw), with Geyer's initial monotone sequence.

    ESS = total draws / tau, where tau = 1 + 2 * (sum of autocorrelations) is the integrated
    autocorrelation time. Autocorrelations are summed in pairs while the pair sums stay
    positive and non-increasing, which cuts off the noisy tail of the estimate.
    """
    m, n = x.shape
    if np.ptp(x) == 0 or n < 4:
        return float("nan")
    acov = _autocovariance(x)
    mean_var = np.mean(acov[:, 0]) * n / (n - 1)
    var_plus = mean_var * (n - 1) / n
    if m > 1:
        var_plus += np.var(np.mean(x, axis=1), ddof=1)
    # autocorrelation combining all chains
    rho = 1 - (mean_var - np.mean(acov, axis=0)) / var_plus
    rho[0] = 1.0

    # Sum autocorrelations in pairs (rho[2t] + rho[2t+1]) while the pair sums stay
    # positive, forcing them to be non-increasing (Geyer's initial monotone sequence).
    total = 0.0
    previous_pair = np.inf
    for t in range(0, n - 1, 2):
        pair = rho[t] + rho[t + 1]
        if pair <= 0:
            break
        pair = min(pair, previous_pair)
        total += pair
        previous_pair = pair
    tau = max(2 * total - 1, 1 / np.log10(m * n))  # integrated autocorrelation time
    return float(m * n / tau)


def ess_bulk(draws: np.ndarray) -> float:
    """Bulk effective sample size: ESS of the rank-normalized split chains.

    How many independent draws the chains are worth for estimating the center of the
    posterior (mean, median). Aim for more than 400.

    Parameters
    ----------
    draws : ndarray of shape (chain, draw), or (draw,) for one chain

    Returns
    -------
    float
        NaN if all draws are equal.
    """
    x = _split_chains(_as_chains(draws))
    if np.ptp(x) == 0:
        return float("nan")
    return _ess(_rank_normalize(x))


def ess_tail(draws: np.ndarray) -> float:
    """Tail effective sample size: the smaller ESS of the 5% and 95% quantile indicators.

    How many independent draws the chains are worth for the tails, i.e. for intervals.
    Aim for more than 400.

    Parameters
    ----------
    draws : ndarray of shape (chain, draw), or (draw,) for one chain

    Returns
    -------
    float
    """
    x = _split_chains(_as_chains(draws))
    q05, q95 = np.quantile(x, [0.05, 0.95])
    return min(_ess((x <= q05).astype(float)), _ess((x <= q95).astype(float)))


def summarize_draws(draws: dict[str, np.ndarray], prob: float = 0.95) -> pd.DataFrame:
    """One row per quantity: mean, sd, HPDI, R-hat, bulk/tail ESS and an `ok` flag.

    Parameters
    ----------
    draws : dict of str -> ndarray of shape (chain, draw)
        Draws of each scalar quantity.
    prob : float, default 0.95
        Probability inside the HPDI.

    Returns
    -------
    DataFrame indexed by name, with columns mean, sd, hpdi_<prob>_low, hpdi_<prob>_high,
    rhat, ess_bulk, ess_tail and ok (R-hat < RHAT_MAX and both ESS > ESS_MIN).
    """
    rows = []
    for name, d in draws.items():
        x = _as_chains(d)
        lower, upper = hpdi(x.ravel(), prob)
        rhat, bulk, tail = split_rhat(x), ess_bulk(x), ess_tail(x)
        rows.append(
            {
                "name": name,
                "mean": x.mean(),
                "sd": x.std(ddof=1),
                f"hpdi_{prob:g}_low": float(lower),
                f"hpdi_{prob:g}_high": float(upper),
                "rhat": rhat,
                "ess_bulk": bulk,
                "ess_tail": tail,
                "ok": bool(rhat < RHAT_MAX and min(bulk, tail) > ESS_MIN),
            }
        )
    return pd.DataFrame(rows).set_index("name")
