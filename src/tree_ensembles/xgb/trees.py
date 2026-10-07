"""XGBoost trees as a table in the same format as tree_ensembles.bart's, for tree_plot."""

from typing import Any

import numpy as np
import pandas as pd
import xgboost as xgb


def _node_number(node_id: Any) -> int | None:
    """XGBoost node id "3-12" (tree 3, node 12) -> 12; None for a leaf's missing child."""
    if node_id is None or node_id is pd.NA or (isinstance(node_id, float) and np.isnan(node_id)):
        return None
    return int(str(node_id).split("-")[1])


def _condition(row: dict[str, Any]) -> str | None:
    """The split as text, true for rows going left (XGBoost's "Yes" child)."""
    if row["Feature"] == "Leaf":
        return None
    if isinstance(row.get("Category"), list):  # categorical split: listed codes go left
        return f"{row['Feature']} in {row['Category']}"
    return f"{row['Feature']} < {row['Split']:.4g}"


def trees_to_dataframe(
    model: xgb.XGBModel | xgb.Booster, trees: int | list[int] | None = None
) -> pd.DataFrame:
    """All nodes of an XGBoost model's trees, in the format used by tree_plot.

    Converts Booster.trees_to_dataframe() to the columns of
    tree_ensembles.bart's trees_to_dataframe, so the same plots work for both.

    Parameters
    ----------
    model : xgboost.XGBModel or xgboost.Booster
        A fitted model (e.g. XGBDefaultRegressor().fit(...).model_).
    trees : int, list of int or None, default None (all)
        Which trees (boosting rounds for a single-output model) to include.

    Returns
    -------
    DataFrame with one row per node and columns:
        tree               tree index
        node               node number within the tree (0 is the root)
        depth              0 for the root (levels of splits)
        is_leaf            True for leaves
        feature            split feature (None for leaves)
        cutpoint           a row goes to `left` if x < cutpoint (note: strict, unlike
                           BART's <=); NaN for categorical splits
        condition          the split as text, true for rows going left, e.g. "age < 41.5" or
                           "color in [0, 2]" (category codes)
        left, right        node number of the children (<NA> for leaves)
        missing            child that a missing (NaN) value goes to: "left" or "right"
        leaf_value         the leaf's contribution to the margin (NaN for splits); the
                           prediction is base_score (or the offset) + the sum over trees
        gain               loss reduction of the split (NaN for leaves)
        cover              XGBoost's cover: sum of the hessians of the training rows reaching
                           the node (the row count for squared error, unweighted)
    """
    booster = model.get_booster() if isinstance(model, xgb.XGBModel) else model
    raw = booster.trees_to_dataframe()
    if trees is not None:
        raw = raw[raw["Tree"].isin(np.atleast_1d(trees))]
    records: list[dict[str, Any]] = raw.to_dict("records")  # type: ignore[assignment]

    is_leaf = np.array([r["Feature"] == "Leaf" for r in records], dtype=bool)
    gain = raw["Gain"].to_numpy(dtype=float)
    table = pd.DataFrame(
        {
            "tree": raw["Tree"].to_numpy(),
            "node": raw["Node"].to_numpy(),
            "depth": 0,
            "is_leaf": is_leaf,
            "feature": [None if r["Feature"] == "Leaf" else r["Feature"] for r in records],
            "cutpoint": pd.to_numeric(raw["Split"], errors="coerce").to_numpy(dtype=float),
            "condition": [_condition(r) for r in records],
            "left": pd.array([_node_number(r["Yes"]) for r in records], dtype="Int64"),
            "right": pd.array([_node_number(r["No"]) for r in records], dtype="Int64"),
            "missing": [
                None
                if r["Feature"] == "Leaf"
                else ("left" if r["Missing"] == r["Yes"] else "right")
                for r in records
            ],
            "leaf_value": np.where(is_leaf, gain, np.nan),
            "gain": np.where(is_leaf, np.nan, gain),
            "cover": raw["Cover"].to_numpy(dtype=float),
        }
    )
    table["depth"] = _depths(table)
    return table


def _depths(table: pd.DataFrame) -> np.ndarray:
    """Depth of every node, following the left/right links down from each root (node 0)."""
    depth = np.zeros(len(table), dtype=int)
    records: list[dict[str, Any]] = table.to_dict("records")  # type: ignore[assignment]
    for tree_id in table["tree"].unique():
        # row position of every node of this tree
        position = {int(r["node"]): i for i, r in enumerate(records) if r["tree"] == tree_id}
        stack = [(position[0], 0)]
        while stack:
            i, d = stack.pop()
            depth[i] = d
            row = records[i]
            if not row["is_leaf"]:
                stack += [(position[int(row["left"])], d + 1), (position[int(row["right"])], d + 1)]
    return depth
