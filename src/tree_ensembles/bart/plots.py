"""Small matplotlib helpers for MCMC draws and fitted forests. Each returns the Figure."""

from typing import TYPE_CHECKING

import numpy as np
from scipy.stats import rankdata

from tree_ensembles.bart.trees import ForestSummary

if TYPE_CHECKING:
    from matplotlib.figure import Figure


def _subplots(ncols: int, width: float = 5.0, height: float = 3.5) -> tuple["Figure", np.ndarray]:
    """A figure with `ncols` panels in one row, each `width` x `height` inches."""
    import matplotlib.pyplot as plt  # imported here so the package works without matplotlib

    fig, axes = plt.subplots(1, ncols, figsize=(width * ncols, height), squeeze=False)
    return fig, axes[0]


def plot_trace(draws: np.ndarray, name: str = "") -> "Figure":
    """Trace (left) and density (right) of one quantity, one line per chain.

    Well-mixed chains look like overlapping noise with matching densities; a chain
    sitting at a different level, or a trend, signals poor convergence.

    Parameters
    ----------
    draws : ndarray of shape (chain, draw)
        E.g. model.parameter_draws()["sigma"].
    name : str, optional
        Quantity name for the titles.

    Returns
    -------
    matplotlib Figure
        In Jupyter, end the line with ";" (or assign the result, e.g. fig = ...) so the
        figure is shown once: otherwise it is drawn both by matplotlib and as the cell's
        returned value. Keep the Figure to save it, e.g. fig.savefig("trees.pdf").
    """
    x = np.atleast_2d(draws)
    fig, (trace, density) = _subplots(2)
    for c, chain in enumerate(x):
        trace.plot(chain, lw=0.7, alpha=0.8, label=f"chain {c}")
        density.hist(chain, bins=40, density=True, histtype="step", label=f"chain {c}")
    trace.set(title=f"Trace {name}", xlabel="draw")
    density.set(title=f"Posterior {name}", xlabel=name)
    density.legend()
    fig.tight_layout()
    return fig


def plot_rank(draws: np.ndarray, name: str = "", bins: int = 20) -> "Figure":
    """Rank plot: histogram of each chain's ranks among all draws (Vehtari et al. 2021).

    All draws are ranked together; if the chains mix well, every chain's histogram of
    ranks is roughly flat. A chain with mostly low or high ranks is stuck elsewhere.

    Parameters
    ----------
    draws : ndarray of shape (chain, draw)
    name : str, optional
        Quantity name for the title.
    bins : int, default 20
        Histogram bins.

    Returns
    -------
    matplotlib Figure
        In Jupyter, end the line with ";" (or assign the result, e.g. fig = ...) so the
        figure is shown once: otherwise it is drawn both by matplotlib and as the cell's
        returned value. Keep the Figure to save it, e.g. fig.savefig("trees.pdf").
    """
    x = np.atleast_2d(draws)
    ranks = rankdata(x).reshape(x.shape)
    fig, axes = _subplots(len(x), width=2.5)
    for c, ax in enumerate(axes):
        ax.hist(ranks[c], bins=bins, range=(1, x.size))
        ax.set(title=f"chain {c}", xticks=[], yticks=[])
    fig.suptitle(f"Rank plot {name}")
    fig.tight_layout()
    return fig


def plot_tree_sizes(summary: ForestSummary) -> "Figure":
    """Share of trees by depth (left) and by number of leaves (right), over all draws.

    Parameters
    ----------
    summary : ForestSummary
        From model.forest_summary().

    Returns
    -------
    matplotlib Figure
        In Jupyter, end the line with ";" (or assign the result, e.g. fig = ...) so the
        figure is shown once: otherwise it is drawn both by matplotlib and as the cell's
        returned value. Keep the Figure to save it, e.g. fig.savefig("trees.pdf").

    Notes
    -----
    The depth reported by tree_ensembles.bart follows the XGBoost meaning:
    forest_summary().depth, depth_distribution(), mean_tree_depth and plot_tree_sizes all
    report levels of splits (0 for a tree without a split). So a BART depth of 3 means the
    same as an XGBoost depth of 3, and only the bartz `maxdepth` setting is shifted by one
    (bartz maxdepth = XGBoost max_depth + 1).
    """
    fig, (depth, leaves) = _subplots(2)
    d, n = summary.depth_distribution(), summary.leaves_distribution()
    depth.bar(d.index, d.to_numpy())
    depth.set(
        title="Tree depth", xlabel="depth (0 = single leaf, 1 = one split)", ylabel="share of trees"
    )
    leaves.bar(n.index, n.to_numpy())
    leaves.set(title="Leaves per tree", xlabel="number of leaves")
    fig.tight_layout()
    return fig


def plot_variable_usage(summary: ForestSummary, top: int = 20) -> "Figure":
    """Mean number of splits per draw for the `top` most used features.

    Parameters
    ----------
    summary : ForestSummary
        From model.forest_summary().
    top : int, default 20
        Number of features to show, most used at the top.

    Returns
    -------
    matplotlib Figure
        In Jupyter, end the line with ";" (or assign the result, e.g. fig = ...) so the
        figure is shown once: otherwise it is drawn both by matplotlib and as the cell's
        returned value. Keep the Figure to save it, e.g. fig.savefig("trees.pdf").
    """
    usage = summary.variable_usage().head(top).iloc[::-1]  # most used at the top
    fig, (ax,) = _subplots(1, width=6, height=0.3 * len(usage) + 1)
    ax.barh(usage.index, usage["mean_splits_per_draw"])
    ax.set(title="Variable usage", xlabel="mean splits per posterior draw")
    fig.tight_layout()
    return fig
