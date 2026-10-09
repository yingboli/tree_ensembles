import warnings

import numpy as np
import pandas as pd
import pytest
from bartz import SparseConfig

from tree_ensembles.bart import BartRegressor
from tree_ensembles.bart.importance import split_gains


def test_split_gains_worked_example() -> None:
    """Root splits x0 (300 rows left: leaf +0.2); right child splits x1 (500: -0.1, 200: +0.4)."""
    #               slot: 0  1  2  3
    var = np.array([[0, 0, 0, 1]])
    split = np.array([[0, 1, 0, 1]])  # node 1 and node 3 split; node 2 is a leaf
    leaf = np.zeros((1, 8))
    leaf[0, [2, 6, 7]] = [0.2, -0.1, 0.4]
    cuts = np.array([[0.0], [1.0]])  # x0 <= 0 goes left; x1 <= 1 goes left
    X = np.array([[-1.0, 0.0]] * 300 + [[1.0, 0.0]] * 500 + [[1.0, 2.0]] * 200)
    internal, rows, gain = split_gains(var, split, leaf, cuts, np.array([1, 1]), X)

    assert internal[0].tolist() == [False, True, False, True]
    assert rows[0, 1] == 1000 and rows[0, 3] == 700
    assert gain[0, 1] == pytest.approx(300 * 700 / 1000 * (0.2 - 30 / 700) ** 2)  # 5.19
    assert gain[0, 3] == pytest.approx(500 * 200 / 700 * 0.5**2)  # 35.71
    predictions = np.array([0.2] * 300 + [-0.1] * 500 + [0.4] * 200)
    assert gain.sum() == pytest.approx(((predictions - predictions.mean()) ** 2).sum())


def test_variable_importance(fitted_regressor: BartRegressor, regression_data: tuple) -> None:
    X, _ = regression_data
    split_only = fitted_regressor.variable_importance()
    assert {"split_share", "inclusion_prob"} <= set(split_only.columns)
    assert "gain_share" not in split_only.columns

    table = fitted_regressor.variable_importance(X[:300], max_draws=50)
    for name in ("split_share", "row_share", "gain_share"):
        assert table[name].sum() == pytest.approx(1.0)
        assert (table[f"{name}_low"] <= table[name]).all() and (
            table[name] <= table[f"{name}_high"]
        ).all()
    assert table.index[-1] == "c"  # y does not depend on c
    assert table["gain_share"].to_dict()["a"] > 0.5  # sin(2a) drives most of f
    assert table["inclusion_prob"].between(0, 1).all()


def test_interactions_match_the_tree_table(fitted_regressor: BartRegressor) -> None:
    pairs = fitted_regressor.interactions()
    assert (pairs["feature_a"] != pairs["feature_b"]).all()
    assert pairs["share"].sum() == pytest.approx(1.0)

    # count parent-child pairs of different features directly from the node table
    nodes = fitted_regressor.trees_to_dataframe()
    splits = nodes[~nodes["is_leaf"]].set_index(["chain", "draw", "tree", "node"])["feature"]
    children = splits[splits.index.get_level_values("node") >= 2]
    parents = splits.reindex(
        pd.MultiIndex.from_arrays(
            [children.index.get_level_values(level) for level in ("chain", "draw", "tree")]
            + [children.index.get_level_values("node") // 2]
        )
    ).to_numpy()
    expected = sum(a != b for a, b in zip(parents, children.to_numpy(), strict=True))
    assert pairs["mean_per_draw"].sum() * 2 * 300 == pytest.approx(expected)


def test_sparse_prior_adds_split_probabilities(regression_data: tuple) -> None:
    X, y = regression_data
    reg = BartRegressor(
        num_trees=20, n_save=50, n_burn=50, num_chains=1, show_progress=False, sparse=SparseConfig()
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        reg.fit(X[:300], y[:300])
    table = reg.variable_importance()
    assert table["split_prob"].sum() == pytest.approx(1.0, abs=1e-4)
    assert table["split_prob"].idxmin() == "c"  # the sparse prior pushes the noise feature down
    assert table["split_prob"].to_dict()["c"] < 0.1
