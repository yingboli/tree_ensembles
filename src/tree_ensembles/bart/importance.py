"""Variable importance and interactions for BART, with posterior uncertainty.

Every importance is computed per posterior draw, so each comes with a posterior mean and an
HPDI. The measures (one row per feature in `variable_importance`):

- split share: share of all decision nodes in a draw that split on the feature (the
  "variable inclusion proportion" of Chipman, George and McCulloch 2010);
- inclusion probability: share of draws in which the feature is used at least once;
- row-weighted split share (needs X): like the split share, but each split counts the rows
  of X reaching it, so splits near the root weigh more (like XGBoost's "cover" importance);
- fit-variance gain share (needs X): share of the variation of the fitted function that
  comes from splits on the feature (see `split_gains`);
- splitting probability: the posterior of the sparse (DART) prior's probabilities
  (Linero 2018), only when fit with sparse=SparseConfig(...).

`interactions` counts how often two features split as parent and child (as bartMachine's
interaction_investigator does).
"""

import numpy as np
import pandas as pd
from bartz import Bart

from tree_ensembles.bart.intervals import hpdi
from tree_ensembles.bart.trees import (
    TreeArrays,
    cut_table,
    goes_right,
    internal_nodes,
    split_probabilities,
    tree_arrays,
)


def split_gains(
    var: np.ndarray,
    split: np.ndarray,
    leaf: np.ndarray,
    cuts: np.ndarray,
    nan_bin: np.ndarray,
    X: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Rows reaching, and fit-variance gain of, every decision node of one draw's trees.

    For a decision node whose children get n_L and n_R rows of X, with m_L and m_R the
    average prediction *of that tree* over the rows in each child,

        gain = n_L * n_R / (n_L + n_R) * (m_L - m_R) ** 2,

    the between-group sum of squares of the tree's predictions. Over all decision nodes
    of a tree the gains add up to n * Var(tree predictions on X), so a feature's share of
    the gain is the share of the fitted function's variation coming from splits on it.

    Parameters
    ----------
    var, split : ndarray of shape (tree, half heap size)
        Feature and split index of every node (see trees.TreeArrays), for one draw.
    leaf : ndarray of shape (tree, heap size)
        Leaf values in units of the latent prediction (already times leaf_unit).
    cuts : ndarray of shape (feature, max cuts)
        Cut values, from trees.cut_table.
    nan_bin : ndarray of shape (feature,)
        Bin a NaN lands in per feature (trees.TreeArrays.nan_bin).
    X : ndarray of shape (n, feature)
        Rows in the columns bartz was fit on.

    Returns
    -------
    internal : ndarray of bool, shape (tree, half heap size)
        Which nodes are decision nodes.
    rows : ndarray, shape (tree, half heap size)
        Rows of X reaching each node.
    gain : ndarray, shape (tree, half heap size)
        Fit-variance gain of each decision node (0 elsewhere).
    """
    n_trees, half = split.shape
    n = len(X)
    internal = internal_nodes(split)

    # route every row down every tree at once (one level of splits per pass)
    node = np.ones((n_trees, n), dtype=int)
    row_index = np.arange(n)[None, :]
    for _ in range(int(np.log2(half))):
        slot = np.minimum(node, half - 1)
        s = np.take_along_axis(split, slot, axis=1).astype(int)
        active = (node < half) & (s > 0)
        j = np.take_along_axis(var, slot, axis=1).astype(int)
        cut = cuts[j, np.maximum(s - 1, 0)]
        right = goes_right(X[row_index, j], s, cut, nan_bin[j])
        node = np.where(active, 2 * node + right, node)
    prediction = np.take_along_axis(leaf, node, axis=1)  # each row's leaf value per tree

    # count rows and sum predictions at every node, by walking each row's leaf up to the root
    size = 2 * half
    offset = (np.arange(n_trees) * size)[:, None]
    counts = np.zeros(n_trees * size)
    sums = np.zeros(n_trees * size)
    ancestor = node.copy()
    for _ in range(int(np.log2(size))):
        on = ancestor > 0
        index = (offset + ancestor)[on]
        counts += np.bincount(index, minlength=n_trees * size)
        sums += np.bincount(index, weights=prediction[on], minlength=n_trees * size)
        ancestor = ancestor // 2
    counts, sums = counts.reshape(n_trees, size), sums.reshape(n_trees, size)

    nodes = np.arange(half)
    n_left, n_right = counts[:, 2 * nodes], counts[:, 2 * nodes + 1]
    with np.errstate(invalid="ignore", divide="ignore"):
        m_left = sums[:, 2 * nodes] / n_left
        m_right = sums[:, 2 * nodes + 1] / n_right
        gain = n_left * n_right / (n_left + n_right) * (m_left - m_right) ** 2
    gain = np.where(internal & (n_left > 0) & (n_right > 0), gain, 0.0)
    return internal, counts[:, :half], gain


def _picked_draws(n_chains: int, n_draws: int, max_draws: int) -> list[tuple[int, int]]:
    """Up to `max_draws` (chain, draw) pairs, evenly spaced over all chains and draws."""
    total = n_chains * n_draws
    flat = np.unique(np.linspace(0, total - 1, min(max_draws, total)).round().astype(int))
    return [(int(i // n_draws), int(i % n_draws)) for i in flat]


def _shares(per_feature: np.ndarray) -> np.ndarray:
    """Each row (one draw) divided by its sum; rows summing to 0 give all zeros."""
    total = per_feature.sum(axis=1, keepdims=True)
    return np.divide(
        per_feature, total, out=np.zeros_like(per_feature, dtype=float), where=total > 0
    )


def _summarize(name: str, draws: np.ndarray, prob: float) -> dict[str, np.ndarray]:
    """Posterior mean and HPDI of each feature's value, from draws shaped (draw, feature)."""
    low, high = hpdi(draws, prob, axis=0)
    return {name: draws.mean(axis=0), f"{name}_low": low, f"{name}_high": high}


def variable_importance(
    bart: Bart,
    feature_names: list[str],
    X: np.ndarray | None = None,
    prob: float = 0.95,
    max_draws: int = 200,
) -> pd.DataFrame:
    """Several variable importances per feature, each with a posterior mean and HPDI.

    Parameters
    ----------
    bart : bartz.Bart
        A fitted model.
    feature_names : list of str
        One name per feature, in the column order bartz was fit on.
    X : ndarray of shape (n, p), optional
        Rows in the columns bartz was fit on (e.g. training or test rows). Needed for the
        row-weighted split share and the fit-variance gain share.
    prob : float, default 0.95
        Probability inside the HPDI columns.
    max_draws : int, default 200
        The X-based measures route every row through every tree of a draw, so they use at
        most this many draws, evenly spaced over the chains. The split-based measures use
        all draws.

    Returns
    -------
    DataFrame indexed by feature, sorted by the most informative measure available
    (gain share, else split share), with columns (mean, then `_low` / `_high` HPDI):
        mean_splits_per_draw   decision nodes on the feature, per draw (posterior mean)
        split_share            share of a draw's decision nodes that split on the feature
        inclusion_prob         share of draws in which the feature is used at least once
                               (close to 1 for most features with many trees, unless the
                               sparse prior is used)
        row_share              (with X) share of the rows reaching decision nodes, i.e.
                               splits weighted by how many rows they divide
        gain_share             (with X) share of the fit-variance gain (see split_gains)
        split_prob             (sparse prior only) posterior splitting probability

    Notes
    -----
    The gain is computed from the trees' own predictions, not from y: it measures how much
    each feature drives the fitted function (on the latent probit scale for the
    classifier), not how much it improves the likelihood.
    """
    arrays = tree_arrays(bart)
    n_chains, n_draws = arrays.split.shape[:2]
    p = len(feature_names)
    internal = internal_nodes(arrays.split)

    # split counts per draw, from all draws
    chain_draw = np.broadcast_to(
        np.arange(n_chains * n_draws).reshape(n_chains, n_draws, 1, 1), internal.shape
    )
    key = chain_draw[internal] * p + arrays.var[internal].astype(np.int64)
    counts = np.bincount(key, minlength=n_chains * n_draws * p).reshape(-1, p).astype(float)
    columns: dict[str, np.ndarray] = {"mean_splits_per_draw": counts.mean(axis=0)}
    columns |= _summarize("split_share", _shares(counts), prob)
    columns["inclusion_prob"] = (counts > 0).mean(axis=0)

    if X is not None:
        X = np.asarray(X, dtype=np.float32)
        cuts = cut_table(arrays)
        rows_per_feature, gain_per_feature = [], []
        for c, d in _picked_draws(n_chains, n_draws, max_draws):
            var = arrays.var[c, d]
            leaf = arrays.leaf[c, d] * arrays.leaf_unit
            is_split, rows, gain = split_gains(
                var, arrays.split[c, d], leaf, cuts, arrays.nan_bin, X
            )
            features = var[is_split].astype(int)
            rows_per_feature.append(np.bincount(features, weights=rows[is_split], minlength=p))
            gain_per_feature.append(np.bincount(features, weights=gain[is_split], minlength=p))
        columns |= _summarize("row_share", _shares(np.array(rows_per_feature)), prob)
        columns |= _summarize("gain_share", _shares(np.array(gain_per_feature)), prob)

    split_prob = split_probabilities(bart)
    if split_prob is not None:
        columns |= _summarize("split_prob", split_prob.reshape(-1, p), prob)

    table = pd.DataFrame(columns, index=pd.Index(feature_names, name="feature"))
    order = "gain_share" if "gain_share" in table else "split_share"
    return table.sort_values(order, ascending=False)


def interactions(bart: Bart, feature_names: list[str]) -> pd.DataFrame:
    """How often two different features split as parent and child, over all draws.

    A split on feature b directly under a split on feature a means the tree uses b only
    for the rows on one side of a's cut: an interaction between a and b. Pairs are
    unordered; repeated splits on the same feature are left out.

    Parameters
    ----------
    bart : bartz.Bart
        A fitted model.
    feature_names : list of str
        One name per feature, in the column order bartz was fit on.

    Returns
    -------
    DataFrame with columns feature_a, feature_b, mean_per_draw (parent-child pairs per
    draw, summed over trees) and share (of all parent-child pairs of different features),
    most frequent first.
    """
    arrays: TreeArrays = tree_arrays(bart)
    n_chains, n_draws, _, half = arrays.split.shape
    internal = internal_nodes(arrays.split)
    child = np.arange(2, half)  # nodes with a parent (slot 0 is unused, 1 is the root)
    is_pair = internal[..., child]  # the child splits; its parent is a split by definition
    a = arrays.var[..., child // 2][is_pair].astype(np.int64)
    b = arrays.var[..., child][is_pair].astype(np.int64)
    different = a != b
    first, second = np.minimum(a, b)[different], np.maximum(a, b)[different]
    p = len(feature_names)
    counts = np.bincount(first * p + second, minlength=p * p)

    pairs = np.flatnonzero(counts)
    names = np.asarray(feature_names, dtype=object)
    table = pd.DataFrame(
        {
            "feature_a": names[pairs // p],
            "feature_b": names[pairs % p],
            "mean_per_draw": counts[pairs] / (n_chains * n_draws),
            "share": counts[pairs] / max(int(counts.sum()), 1),
        }
    )
    return table.sort_values("mean_per_draw", ascending=False, ignore_index=True)
