"""Draw decision trees from a tree table, for BART and XGBoost alike (matplotlib).

The table is what `trees_to_dataframe` returns for either model: one row per node with
columns node, depth, is_leaf, condition, left, right, missing, leaf_value and optionally
num_rows (BART, with X) or cover (XGBoost), plus chain / draw / tree to identify each tree.
"""

from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure


def _tree_keys(table: pd.DataFrame) -> list[str]:
    """Columns that identify a tree: chain, draw, tree for BART; tree for XGBoost."""
    return [k for k in ("chain", "draw", "tree") if k in table.columns]


def _records(tree: pd.DataFrame) -> list[dict[str, Any]]:
    """The rows of a tree table as plain dicts, one per node."""
    return tree.to_dict("records")  # type: ignore[return-value]


def _number(value: float) -> str:
    """Whole numbers with thousands separators (20,000); others with 4 significant digits."""
    value = float(value)
    return f"{value:,.0f}" if value.is_integer() else f"{value:.4g}"


def _cut(tree: pd.DataFrame, max_depth: int | None) -> pd.DataFrame:
    """The nodes down to `max_depth`, with the splits at that depth marked `cut_off`."""
    if max_depth is None:
        return tree.assign(cut_off=False)
    shown = tree[tree["depth"] <= max_depth]
    return shown.assign(cut_off=~shown["is_leaf"] & (shown["depth"] == max_depth))


def _ends(row: dict[str, Any]) -> bool:
    """True for a node drawn without children: a leaf, or a split cut off by max_depth."""
    return bool(row["is_leaf"] or row["cut_off"])


def _layout(tree: pd.DataFrame) -> dict[int, tuple[float, int]]:
    """x position and depth of every node of one (cut) tree.

    Nodes drawn without children get x = 0, 1, 2, ... from left to right; a split sits
    above the middle of its two children. Works from the left/right links, so any binary
    tree can be drawn.
    """
    rows = {int(r["node"]): r for r in _records(tree)}
    children = {int(r[s]) for r in rows.values() if not _ends(r) for s in ("left", "right")}
    root = next(n for n in rows if n not in children)
    positions: dict[int, tuple[float, int]] = {}
    next_slot = 0

    def place(node: int, depth: int) -> None:
        nonlocal next_slot
        row = rows[node]
        if _ends(row):
            positions[node] = (float(next_slot), depth)
            next_slot += 1
            return
        left, right = int(row["left"]), int(row["right"])
        place(left, depth + 1)
        place(right, depth + 1)
        positions[node] = ((positions[left][0] + positions[right][0]) / 2, depth)

    place(root, 0)
    return positions


def _figure_size(tree: pd.DataFrame) -> tuple[float, float]:
    """A width that fits the bottom boxes side by side, and a height that fits the depth."""
    ends = int((tree["is_leaf"] | tree["cut_off"]).sum())
    return max(4.0, 1.7 * ends), 1.3 * (int(tree["depth"].max()) + 1) + 0.6


def plot_tree(
    tree: pd.DataFrame,
    ax: "Axes | None" = None,
    show_missing: bool = True,
    value_range: float | None = None,
    max_depth: int | None = None,
) -> "Figure":
    """Draw one tree: split conditions in grey boxes, leaf values in colored boxes.

    Parameters
    ----------
    tree : DataFrame
        The rows of a single tree from trees_to_dataframe (BartRegressor /
        BartClassifier.trees_to_dataframe, or XGBDefault*.trees_to_dataframe).
    ax : matplotlib Axes, optional
        Where to draw; a new figure is created if None.
    show_missing : bool, default True
        Add ", missing" to the edge that missing (NaN) values follow.
    value_range : float, optional
        Leaf colors run from -value_range (blue) to +value_range (red). Default: the
        largest absolute leaf value in `tree`. plot_trees passes one shared range.
    max_depth : int, optional
        Draw only the nodes down to this depth (0 = the root only); a split whose children
        are not drawn shows "...". Useful for deep XGBoost trees, too wide to read in full.

    Returns
    -------
    matplotlib Figure

    Notes
    -----
    The left child ("yes") is where the condition is true. BART splits read
    "x <= cut", XGBoost splits "x < cut" (each library's own rule). If the table has a
    num_rows column (BART, with X given: rows reaching the node) or a cover column (XGBoost:
    sum of the hessians of the training rows), each box also shows it as "rows" or "cover".
    """
    import matplotlib.pyplot as plt  # imported here so the package works without matplotlib
    from matplotlib.colors import Normalize

    keys = _tree_keys(tree)
    if keys and len(tree.groupby(keys)) != 1:
        raise ValueError("plot_tree draws one tree; select it first, or use plot_trees")
    leaves = tree.loc[tree["is_leaf"], "leaf_value"].to_numpy(dtype=float)
    vmax = value_range or float(np.max(np.abs(leaves))) or 1.0
    color = plt.get_cmap("RdBu_r")
    norm = Normalize(-vmax, vmax)
    shown = _cut(tree, max_depth)
    if ax is None:
        _, ax = plt.subplots(figsize=_figure_size(shown))
    positions = _layout(shown)
    # BART: rows of X reaching the node; XGBoost: cover (sum of hessians)
    size_column, size_label = next(
        (
            (c, label)
            for c, label in (("num_rows", "rows"), ("cover", "cover"))
            if c in tree.columns
        ),
        (None, ""),
    )

    for row in _records(shown):
        x, depth = positions[int(row["node"])]
        size = f"\n{size_label} {_number(row[size_column])}" if size_column else ""
        if row["is_leaf"]:
            value = float(row["leaf_value"])
            box: dict[str, Any] = {"boxstyle": "round", "fc": color(norm(value))}
            text = f"{value:+.3g}{size}"
            ink = "white" if abs(float(norm(value)) - 0.5) > 0.3 else "black"  # readable on dark
            ax.text(x, -depth, text, ha="center", va="center", fontsize=8, bbox=box, color=ink)
            continue
        box = {"boxstyle": "round", "fc": "0.92"}
        text = f"{row['condition']}{size}" + ("\n..." if row["cut_off"] else "")
        ax.text(x, -depth, text, ha="center", va="center", fontsize=8, bbox=box)
        if row["cut_off"]:
            continue
        for side, label in (("left", "yes"), ("right", "no")):
            cx, cdepth = positions[int(row[side])]
            ax.plot([x, cx], [-depth, -cdepth], color="0.6", lw=1, zorder=0)
            edge = f"{label}, missing" if show_missing and row["missing"] == side else label
            mid = ((x + cx) / 2, -(depth + cdepth) / 2)
            ax.text(*mid, edge, ha="center", va="center", fontsize=7, color="0.35")

    xs = [p[0] for p in positions.values()]
    ax.set(xlim=(min(xs) - 0.7, max(xs) + 0.7), ylim=(-int(shown["depth"].max()) - 0.6, 0.6))
    ax.set_axis_off()
    if keys:
        ax.set_title(", ".join(f"{k} {tree[k].iloc[0]}" for k in keys), fontsize=9)
    return ax.figure  # type: ignore[return-value]


def plot_trees(
    trees: pd.DataFrame,
    ncols: int = 3,
    max_trees: int = 12,
    show_missing: bool = True,
    max_depth: int | None = None,
) -> "Figure":
    """Draw several trees side by side, with one shared color scale for the leaf values.

    Parameters
    ----------
    trees : DataFrame
        Rows of several trees from trees_to_dataframe (BART or XGBoost).
    ncols : int, default 3
        Trees per row of the figure.
    max_trees : int, default 12
        Only the first `max_trees` trees are drawn (in the table's order).
    show_missing : bool, default True
        Add ", missing" to the edge that missing values follow.
    max_depth : int, optional
        Draw each tree only down to this depth (see plot_tree).

    Returns
    -------
    matplotlib Figure
    """
    import matplotlib.pyplot as plt

    groups = [g for _, g in trees.groupby(_tree_keys(trees), sort=False)][:max_trees]
    shown = pd.concat(groups)
    vmax = float(np.nanmax(np.abs(shown["leaf_value"].to_numpy(dtype=float)))) or 1.0
    nrows = int(np.ceil(len(groups) / ncols))
    sizes = [_figure_size(_cut(g, max_depth)) for g in groups]  # each panel fits the largest
    width, height = max(w for w, _ in sizes), max(h for _, h in sizes)
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * width, nrows * height), squeeze=False)
    for ax, tree in zip(axes.ravel(), groups, strict=False):
        plot_tree(tree, ax=ax, show_missing=show_missing, value_range=vmax, max_depth=max_depth)
    for ax in axes.ravel()[len(groups) :]:
        ax.set_axis_off()
    fig.tight_layout()
    return fig
