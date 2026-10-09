"""Insights into the fitted BART trees: depth, leaves, splits, variable usage.

This is the only module that reads bartz internals (`Bart._main_trace`, `Bart._binner`),
which is why bartz is pinned to a minor version. How bartz stores a tree:

- Each tree is a heap of nodes numbered from 1 (the root); node i has children 2i and 2i + 1.
- `var_tree[i]` is the feature used at node i, `split_tree[i]` its split index;
  `split_tree[i] == 0` means node i is a leaf.
- A point goes right when its binned value is >= split, i.e. in original units when
  x > cutpoints[feature][split - 1].
- Missing values (NaN): bartz's quantile binner treats every NaN as a distinct value, so
  a feature with NaN also gets cutpoints at NaN. A NaN x always lands in a bin above every
  real cutpoint, so at a real cutpoint missing rows go right (like +inf). At a NaN
  cutpoint, rows with values always go left, and missing rows go:
    * left too, for a feature with many NaN (NaN fills the cutpoint slots and missing
      rows are binned below them): a useless split, both groups in one child;
    * right, for a feature with few NaN (the slots end in padding and missing rows are
      binned above everything): an "is missing" split.
  `nan_bin` (the bin a NaN lands in, per feature) decides which case applies.
- A draw's latent prediction is offset + leaf_unit * (sum over trees of the leaf values).
"""

from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np
import pandas as pd
from bartz import Bart


@dataclass
class TreeArrays:
    """The trees of every posterior draw, as numpy arrays shaped (chain, draw, tree, node)."""

    var: np.ndarray  # feature index at each decision node
    split: np.ndarray  # split index (0 = leaf); length = half the heap size
    leaf: np.ndarray  # leaf values in units of `leaf_unit`; length = full heap size
    leaf_unit: float
    offset: float
    cutpoints: list[np.ndarray]  # per feature, the cut values in original units
    nan_bin: np.ndarray  # per feature, the bin a NaN value lands in (see the module notes)


def tree_arrays(bart: Bart) -> TreeArrays:
    """Copy the trees of every posterior draw out of a fitted bartz model.

    Parameters
    ----------
    bart : bartz.Bart
        A fitted model (e.g. BartRegressor().fit(...).bart_).

    Returns
    -------
    TreeArrays
        Heap arrays shaped (chain, draw, tree, node), with a chain axis of size 1 if the
        model was run with num_chains=None, plus the leaf unit, offset and cutpoints.
    """
    trace = bart._main_trace
    var, split, leaf = (np.asarray(a) for a in (trace.var_tree, trace.split_tree, trace.leaf_tree))
    if not trace.has_chains:  # num_chains=None: add a chain axis of size 1
        var, split, leaf = var[None], split[None], leaf[None]
    splits = np.asarray(bart._binner._splits)
    max_split = np.asarray(bart._binner.max_split)
    nan_column = jnp.full((len(max_split), 1), jnp.nan, dtype=jnp.float32)
    nan_bin = np.asarray(bart._binner.bin(nan_column))[:, 0].astype(int)
    return TreeArrays(
        var=var,
        split=split,
        leaf=leaf,
        leaf_unit=float(trace.leaf_unit),
        offset=float(trace.offset),
        cutpoints=[splits[j, : max_split[j]] for j in range(len(max_split))],
        nan_bin=nan_bin,
    )


def internal_nodes(split: np.ndarray) -> np.ndarray:
    """Mask of the decision nodes actually in use (reachable from the root and split != 0).

    Unreachable heap slots can hold stale values, so a node counts only if its parent
    is itself an internal node.

    Parameters
    ----------
    split : ndarray of shape (..., half_heap_size)
        Split indices of one or many trees (0 = leaf); slot 0 is unused.

    Returns
    -------
    ndarray of bool, same shape as `split`
    """
    is_split = split != 0
    internal = np.zeros_like(is_split)
    internal[..., 1] = is_split[..., 1]  # the root (slot 0 is unused)
    for i in range(2, split.shape[-1]):
        internal[..., i] = is_split[..., i] & internal[..., i // 2]
    return internal


def node_depths(size: int) -> np.ndarray:
    """Depth of every heap slot: node 1 is depth 0, nodes 2-3 depth 1, 4-7 depth 2, ...

    Parameters
    ----------
    size : int
        Number of heap slots (slot 0 is unused and gets depth 0).

    Returns
    -------
    ndarray of int, shape (size,)
    """
    index = np.arange(size)
    return np.where(index > 0, np.floor(np.log2(np.maximum(index, 1))), 0).astype(int)


@dataclass
class ForestSummary:
    """Per-tree structure of every posterior draw, plus variable usage.

    `depth`, `n_leaves` and `n_splits` are shaped (chain, draw, tree). A single-leaf tree
    (no split) has depth 0, a tree with one split has depth 1 and 2 leaves; every tree has
    n_leaves = n_splits + 1.

    Notes
    -----
    The depth reported by tree_ensembles.bart follows the XGBoost meaning:
    forest_summary().depth, depth_distribution(), mean_tree_depth and plot_tree_sizes all
    report levels of splits (0 for a tree without a split). So a BART depth of 3 means the
    same as an XGBoost depth of 3, and only the bartz `maxdepth` setting is shifted by one
    (bartz maxdepth = XGBoost max_depth + 1).
    """

    depth: np.ndarray
    n_leaves: np.ndarray
    n_splits: np.ndarray
    split_counts: np.ndarray  # (chain, draw, feature): number of decision nodes using it
    feature_names: list[str]

    def tree_sizes(self) -> pd.DataFrame:
        """Average tree size in every draw.

        Returns
        -------
        DataFrame indexed by (chain, draw) with columns mean_depth, mean_leaves and
        share_single_leaf (trees without any split). Useful as a convergence trace.

        Notes
        -----
        The depth reported by tree_ensembles.bart follows the XGBoost meaning:
        forest_summary().depth, depth_distribution(), mean_tree_depth and plot_tree_sizes all
        report levels of splits (0 for a tree without a split). So a BART depth of 3 means the
        same as an XGBoost depth of 3, and only the bartz `maxdepth` setting is shifted by one
        (bartz maxdepth = XGBoost max_depth + 1).
        """
        chains, draws = self.depth.shape[:2]
        index = pd.MultiIndex.from_product([range(chains), range(draws)], names=["chain", "draw"])
        return pd.DataFrame(
            {
                "mean_depth": self.depth.mean(axis=-1).ravel(),
                "mean_leaves": self.n_leaves.mean(axis=-1).ravel(),
                "share_single_leaf": (self.n_splits == 0).mean(axis=-1).ravel(),
            },
            index=index,
        )

    def depth_distribution(self) -> pd.Series:
        """Share of trees at each depth (0 = single leaf), pooled over all chains and draws.

        Returns
        -------
        Series indexed by depth, summing to 1.

        Notes
        -----
        The depth reported by tree_ensembles.bart follows the XGBoost meaning:
        forest_summary().depth, depth_distribution(), mean_tree_depth and plot_tree_sizes all
        report levels of splits (0 for a tree without a split). So a BART depth of 3 means the
        same as an XGBoost depth of 3, and only the bartz `maxdepth` setting is shifted by one
        (bartz maxdepth = XGBoost max_depth + 1).
        """
        return pd.Series(self.depth.ravel()).value_counts(normalize=True).sort_index()

    def leaves_distribution(self) -> pd.Series:
        """Share of trees with each number of leaves, pooled over all chains and draws.

        Returns
        -------
        Series indexed by number of leaves, summing to 1.
        """
        return pd.Series(self.n_leaves.ravel()).value_counts(normalize=True).sort_index()

    def variable_usage(self) -> pd.DataFrame:
        """How much each feature is used to split, most used first.

        Returns
        -------
        DataFrame indexed by feature with columns mean_splits_per_draw (decision nodes on
        that feature, summed over the trees of a draw, averaged over draws) and
        share_of_splits (summing to 1). A rough variable importance.
        """
        per_draw = self.split_counts.reshape(-1, len(self.feature_names)).mean(axis=0)
        usage = pd.DataFrame(
            {"mean_splits_per_draw": per_draw, "share_of_splits": per_draw / per_draw.sum()},
            index=pd.Index(self.feature_names, name="feature"),
        )
        return usage.sort_values("mean_splits_per_draw", ascending=False)


def forest_summary(bart: Bart, feature_names: list[str]) -> ForestSummary:
    """Depth, leaves and splits of every tree in every draw, and split counts per feature.

    Parameters
    ----------
    bart : bartz.Bart
        A fitted model.
    feature_names : list of str
        One name per feature, in training column order.

    Returns
    -------
    ForestSummary

    Notes
    -----
    The depth reported by tree_ensembles.bart follows the XGBoost meaning:
    forest_summary().depth, depth_distribution(), mean_tree_depth and plot_tree_sizes all
    report levels of splits (0 for a tree without a split). So a BART depth of 3 means the
    same as an XGBoost depth of 3, and only the bartz `maxdepth` setting is shifted by one
    (bartz maxdepth = XGBoost max_depth + 1).
    """
    trees = tree_arrays(bart)
    internal = internal_nodes(trees.split)
    n_splits = internal.sum(axis=-1)

    # depth of a tree = depth of its deepest decision node + 1 (0 for a single-leaf tree)
    depths = node_depths(trees.split.shape[-1])
    depth = np.where(n_splits > 0, np.max(np.where(internal, depths + 1, 0), axis=-1), 0)

    # count decision nodes per (draw, feature) in one pass with bincount
    p = len(feature_names)
    chains, draws = internal.shape[:2]
    draw_id = np.broadcast_to(
        np.arange(chains * draws).reshape(chains, draws, 1, 1), internal.shape
    )
    key = draw_id[internal] * p + trees.var[internal].astype(np.int64)
    split_counts = np.bincount(key, minlength=chains * draws * p).reshape(chains, draws, p)
    return ForestSummary(
        depth=depth,
        n_leaves=n_splits + 1,
        n_splits=n_splits,
        split_counts=split_counts,
        feature_names=feature_names,
    )


def split_points(bart: Bart, feature_names: list[str]) -> pd.DataFrame:
    """How often each cut value is used, pooled over all trees and draws, most used first.

    Parameters
    ----------
    bart : bartz.Bart
        A fitted model.
    feature_names : list of str
        One name per feature, in training column order.

    Returns
    -------
    DataFrame with columns feature, cutpoint (a point goes right if x > cutpoint) and
    count. A NaN cutpoint marks either a useless split or an "is missing" split, see the
    module notes.
    """
    trees = tree_arrays(bart)
    internal = internal_nodes(trees.split)
    var = trees.var[internal].astype(np.int64)
    split = trees.split[internal].astype(np.int64)
    n_cuts = int(split.max()) + 1 if split.size else 1
    counts = np.bincount(var * n_cuts + split, minlength=len(feature_names) * n_cuts)
    rows = [
        (feature_names[j], float(trees.cutpoints[j][s - 1]), int(c))
        for (j, s), c in zip(np.ndindex(len(feature_names), n_cuts), counts, strict=True)
        if c > 0
    ]
    table = pd.DataFrame(rows, columns=["feature", "cutpoint", "count"])
    return table.sort_values("count", ascending=False, ignore_index=True)


def format_tree(bart: Bart, feature_names: list[str], chain: int, draw: int, tree: int) -> str:
    """One tree in readable form, with feature names, cut values and leaf contributions.

    Parameters
    ----------
    bart : bartz.Bart
        A fitted model.
    feature_names : list of str
        One name per feature, in training column order.
    chain, draw, tree : int
        Which tree to print.

    Returns
    -------
    str
        Nested "if feature <= cut:" / "else:" lines ending in "leaf <value>", where the
        value is the tree's contribution to the latent prediction.
    """
    trees = tree_arrays(bart)
    var = trees.var[chain, draw, tree]
    split = trees.split[chain, draw, tree]
    leaf = trees.leaf[chain, draw, tree] * trees.leaf_unit
    lines: list[str] = []

    def visit(node: int, indent: str) -> None:
        """Append the lines of the subtree under `node`, depth first."""
        is_leaf = node >= len(split) or split[node] == 0
        if is_leaf:
            lines.append(f"{indent}leaf {leaf[node]:+.4g}")
            return
        j, s = var[node], split[node]
        name = feature_names[j]
        cut = trees.cutpoints[j][s - 1]
        missing_right = trees.nan_bin[j] >= s
        if np.isnan(cut) and not missing_right:
            lines.append(f"{indent}split on {name} at a NaN cutpoint: every row goes left")
            visit(2 * node, indent + "    ")
            return
        if np.isnan(cut):
            lines.append(f"{indent}if {name} is not missing:")
        else:
            lines.append(f"{indent}if {name} <= {cut:.4g}{'' if missing_right else ' or missing'}:")
        visit(2 * node, indent + "    ")
        if np.isnan(cut):
            lines.append(f"{indent}else:  # {name} is missing")
        else:
            lines.append(
                f"{indent}else:  # {name} > {cut:.4g}{' or missing' if missing_right else ''}"
            )
        visit(2 * node + 1, indent + "    ")

    visit(1, "")
    return "\n".join(lines)


def goes_right(
    x: np.ndarray, split: np.ndarray, cut: np.ndarray, nan_bin: np.ndarray
) -> np.ndarray:
    """bartz's split rule in original units, including its handling of NaN (see top).

    A row with a value goes right if x > cut (never at a NaN cutpoint). A missing row goes
    right if the bin NaN lands in is >= the split index.

    Parameters
    ----------
    x : ndarray
        Feature values.
    split : ndarray
        Split indices (1-based position of the cutpoint), broadcastable with `x`.
    cut : ndarray
        Cut values cutpoints[feature][split - 1] (NaN for a split at a NaN cutpoint).
    nan_bin : ndarray
        The bin a NaN value of that feature lands in.

    Returns
    -------
    ndarray of bool
        True where the row goes to the right child.
    """
    missing = np.isnan(x)
    with np.errstate(invalid="ignore"):
        observed_right = ~np.isnan(cut) & (x > cut)
    return np.where(missing, nan_bin >= split, observed_right)


def predict_one_draw(bart: Bart, X: np.ndarray, chain: int, draw: int) -> np.ndarray:
    """Latent prediction of one posterior draw, by walking the trees in original units.

    Slow and meant for checking/teaching: it should equal bartz's own latent prediction
    (Bart.predict(..., kind="latent_samples")) for that draw.

    Parameters
    ----------
    bart : bartz.Bart
        A fitted model.
    X : ndarray of shape (m, p)
        Rows to predict, in original units.
    chain, draw : int
        Which posterior draw.

    Returns
    -------
    ndarray of shape (m,)
        offset + leaf_unit * (sum of the leaf values reached in every tree).
    """
    trees = tree_arrays(bart)
    X = np.asarray(X, dtype=np.float32)
    total = np.zeros(len(X))
    for t in range(trees.split.shape[2]):
        var, split = trees.var[chain, draw, t], trees.split[chain, draw, t]
        node = np.ones(len(X), dtype=int)
        for _ in range(int(np.log2(len(split)))):  # at most the tree depth
            inside = node < len(split)
            s = np.where(inside, split[np.minimum(node, len(split) - 1)], 0)
            active = s > 0
            j = var[np.minimum(node, len(split) - 1)]
            cut = np.array(
                [
                    trees.cutpoints[jj][ss - 1] if a else 0.0
                    for jj, ss, a in zip(j, s, active, strict=True)
                ]
            )
            x = X[np.arange(len(X)), j]
            right = goes_right(x, s, cut, trees.nan_bin[j])
            node = np.where(active, 2 * node + right, node)
        total += trees.leaf[chain, draw, t][node]
    return trees.offset + trees.leaf_unit * total


def _selection(index: int | list[int] | None, size: int) -> np.ndarray:
    """Positions picked by an int, a list of ints (negative ints count from the end) or None."""
    if index is None:
        return np.arange(size)
    return np.arange(size)[np.atleast_1d(np.asarray(index, dtype=int))]


def trees_to_dataframe(
    bart: Bart,
    feature_names: list[str],
    chains: int | list[int] | None = None,
    draws: int | list[int] | None = None,
    trees: int | list[int] | None = None,
    X: np.ndarray | None = None,
) -> pd.DataFrame:
    """All nodes of the selected trees as a table, like XGBoost's Booster.trees_to_dataframe.

    Parameters
    ----------
    bart : bartz.Bart
        A fitted model.
    feature_names : list of str
        One name per feature, in the column order bartz was fit on.
    chains, draws, trees : int, list of int or None, default None (all)
        Which trees to include; negative ints count from the end (e.g. draws=-1 is the last
        draw). The full table has one row per node of every tree in every draw, which can
        be millions of rows, so select a few draws or trees when exploring.
    X : ndarray of shape (n, p), optional
        Rows in the columns bartz was fit on. If given, a `num_rows` column counts how many of
        them reach each node.

    Returns
    -------
    DataFrame with one row per node, sorted by chain, draw, tree and node, with columns:
        chain, draw, tree  which tree
        node               heap index: 1 is the root, node i has children 2i and 2i + 1
        depth              0 for the root (depths count levels of splits, like XGBoost)
        is_leaf            True for leaves
        feature            split feature (None for leaves)
        cutpoint           a row goes to `left` if x <= cutpoint, else to `right`; NaN marks
                           a split at a NaN cutpoint (see the module notes)
        condition          the split as text, true for rows going left, e.g. "age <= 41.5"
        left, right        heap index of the children (<NA> for leaves)
        missing            child that a missing (NaN) value goes to: "left" or "right"
        leaf_value         the leaf's contribution to the latent prediction (NaN for splits);
                           a draw's prediction is offset + the sum over trees of its leaves
        num_rows           number of rows of X reaching the node: arriving at a split, or
                           ending in a leaf (only when X is given; unlike XGBoost's cover,
                           a plain row count)
    """
    arrays = tree_arrays(bart)
    n_chains, n_draws, n_trees, half = arrays.split.shape
    c_idx, d_idx, t_idx = (
        _selection(chains, n_chains),
        _selection(draws, n_draws),
        _selection(trees, n_trees),
    )
    pick = np.ix_(c_idx, d_idx, t_idx)
    split, var, leaf = arrays.split[pick], arrays.var[pick], arrays.leaf[pick]

    # which heap slots are real nodes: the root, and every child of a decision node
    internal = np.concatenate([internal_nodes(split), np.zeros_like(split, dtype=bool)], axis=-1)
    in_tree = np.zeros_like(internal)
    in_tree[..., 1] = True
    in_tree[..., 2:] = internal[..., np.arange(2, 2 * half) // 2]
    c, d, t, node = np.nonzero(in_tree)
    is_leaf = ~internal[c, d, t, node]

    # split details (only meaningful for decision nodes, which all sit below `half`)
    slot = np.minimum(node, half - 1)
    feat = var[c, d, t, slot].astype(int)
    s = split[c, d, t, slot].astype(int)
    cuts = cut_table(arrays)
    cutpoint = np.where(is_leaf, np.nan, cuts[feat, np.maximum(s - 1, 0)])
    missing_right = arrays.nan_bin[feat] >= s
    names = np.asarray(feature_names, dtype=object)[feat]
    condition = [
        None
        if leaf_
        else f"{name} <= {cut:.4g}"
        if not np.isnan(cut)
        else f"{name} is not missing"
        if right
        else f"{name}: NaN cutpoint (all rows left)"
        for leaf_, name, cut, right in zip(is_leaf, names, cutpoint, missing_right, strict=True)
    ]

    table = pd.DataFrame(
        {
            "chain": c_idx[c],
            "draw": d_idx[d],
            "tree": t_idx[t],
            "node": node,
            "depth": node_depths(2 * half)[node],
            "is_leaf": is_leaf,
            "feature": [
                None if leaf_ else name for leaf_, name in zip(is_leaf, names, strict=True)
            ],
            "cutpoint": cutpoint,
            "condition": condition,
            "left": pd.array(np.where(is_leaf, -1, 2 * node), dtype="Int64"),
            "right": pd.array(np.where(is_leaf, -1, 2 * node + 1), dtype="Int64"),
            "missing": [
                None if leaf_ else ("right" if right else "left")
                for leaf_, right in zip(is_leaf, missing_right, strict=True)
            ],
            "leaf_value": np.where(is_leaf, leaf[c, d, t, node] * arrays.leaf_unit, np.nan),
        }
    )
    table.loc[table["is_leaf"], ["left", "right"]] = pd.NA
    if X is not None:
        table["num_rows"] = _num_rows(arrays, np.asarray(X, dtype=np.float32), table)
    return table


def _num_rows(arrays: TreeArrays, X: np.ndarray, table: pd.DataFrame) -> np.ndarray:
    """Number of rows of X reaching each node listed in `table` (one tree at a time)."""
    half = arrays.split.shape[-1]
    counts = np.zeros(len(table), dtype=int)
    ids = table[["chain", "draw", "tree"]].to_numpy()
    combos, which = np.unique(ids, axis=0, return_inverse=True)
    for k, (c, d, t) in enumerate(combos):
        rows = np.flatnonzero(which.ravel() == k)  # the table rows of this tree
        var, split = arrays.var[c, d, t], arrays.split[c, d, t]
        per_node = np.zeros(2 * half, dtype=int)
        node = np.ones(len(X), dtype=int)
        per_node[1] = len(X)
        for _ in range(int(np.log2(half))):  # one level of splits per pass
            inside = node < half
            s = np.where(inside, split[np.minimum(node, half - 1)], 0)
            active = s > 0
            j = var[np.minimum(node, half - 1)].astype(int)
            cut = np.array(
                [
                    arrays.cutpoints[jj][ss - 1] if a else 0.0
                    for jj, ss, a in zip(j, s, active, strict=True)
                ]
            )
            right = goes_right(X[np.arange(len(X)), j], s, cut, arrays.nan_bin[j])
            node = np.where(active, 2 * node + right, node)
            np.add.at(per_node, node[active], 1)
        counts[rows] = per_node[table["node"].to_numpy()[rows]]
    return counts


def cut_table(arrays: TreeArrays) -> np.ndarray:
    """Cut values as a (feature, split index - 1) array, NaN-padded to the longest feature."""
    longest = max(max(len(cuts) for cuts in arrays.cutpoints), 1)
    table = np.full((len(arrays.cutpoints), longest), np.nan)
    for j, cuts in enumerate(arrays.cutpoints):
        table[j, : len(cuts)] = cuts
    return table


def split_probabilities(bart: Bart) -> np.ndarray | None:
    """Posterior draws of each feature's splitting probability under the sparse (DART) prior.

    Returns
    -------
    ndarray of shape (chain, draw, feature), or None when the sparse prior is off.
    """
    trace = bart._main_trace
    if trace.varprob is None:
        return None
    varprob = np.asarray(trace.varprob)
    return varprob if trace.has_chains else varprob[None]
