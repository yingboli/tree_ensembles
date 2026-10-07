import numpy as np
import pandas as pd
import pytest

from tree_ensembles.bart import BartRegressor
from tree_ensembles.bart.trees import internal_nodes, node_depths, predict_one_draw, tree_arrays


def test_internal_nodes_ignore_unreachable_slots() -> None:
    #           slot: 0  1  2  3  4  5  6  7
    split = np.array([0, 5, 0, 3, 9, 0, 0, 0])  # node 4 is under leaf 2: unreachable
    assert internal_nodes(split).nonzero()[0].tolist() == [1, 3]


def test_node_depths() -> None:
    assert node_depths(8).tolist() == [0, 0, 1, 1, 2, 2, 2, 2]


def test_tree_walk_matches_bartz(fitted_regressor: BartRegressor, regression_data: tuple) -> None:
    X, _ = regression_data
    X_test = X[:40].to_numpy(dtype=np.float32)
    latent = fitted_regressor._draws(X[:40], "latent_samples")
    for chain, draw in [(0, 0), (1, 123), (1, 299)]:
        ours = predict_one_draw(fitted_regressor.bart_, X_test, chain, draw)
        np.testing.assert_allclose(ours, latent[chain, draw], atol=1e-5)


def test_tree_walk_with_nan_matches_bartz() -> None:
    rng = np.random.default_rng(1)
    X = rng.normal(size=(300, 2)).astype(np.float32)
    X[::5, 1] = np.nan
    y = X[:, 0] + np.nan_to_num(X[:, 1])
    reg = BartRegressor(
        num_trees=20,
        n_save=20,
        n_burn=50,
        num_chains=1,
        show_progress=False,
        impute_strategy=None,  # pass NaN to bartz to test its own handling
    )
    with pytest.warns(UserWarning, match="NaN"):
        reg.fit(X, y)
    latent = reg._draws(X[:50], "latent_samples")
    for draw in range(20):
        np.testing.assert_allclose(
            predict_one_draw(reg.bart_, X[:50], 0, draw), latent[0, draw], atol=1e-5
        )


def test_forest_summary(fitted_regressor: BartRegressor) -> None:
    fs = fitted_regressor.forest_summary()
    assert fs.depth.shape == (2, 300, 50)
    assert (fs.n_leaves == fs.n_splits + 1).all()
    assert ((fs.n_splits == 0) == (fs.depth == 0)).all()
    vc = np.asarray(fitted_regressor.bart_.varcount).reshape(2, 300, 3)
    np.testing.assert_array_equal(fs.split_counts, vc)  # same counts as bartz

    usage = fs.variable_usage()
    assert usage.index[-1] == "c"  # the noise feature is used least
    assert usage["share_of_splits"].sum() == 1.0 or np.isclose(usage["share_of_splits"].sum(), 1)
    assert np.isclose(fs.depth_distribution().sum(), 1)
    assert isinstance(fs.tree_sizes(), pd.DataFrame) and len(fs.tree_sizes()) == 600
    assert list(fs.tree_sizes().columns) == ["mean_depth", "mean_leaves", "share_single_leaf"]
    assert ((fs.depth == 1) == (fs.n_leaves == 2)).all()  # depth 1 = one split, 2 leaves


def test_split_points_and_format_tree(fitted_regressor: BartRegressor) -> None:
    points = fitted_regressor.split_points()
    assert set(points["feature"]) <= {"a", "b", "c"}
    assert points["count"].sum() == fitted_regressor.forest_summary().n_splits.sum()
    text = fitted_regressor.format_tree(chain=0, draw=0, tree=0)
    assert "leaf" in text


def test_tree_walk_matches_bartz_for_both_kinds_of_nan_features() -> None:
    """Many NaN: missing rows sit below the NaN cutpoints. Few NaN: they sit above all."""
    rng = np.random.default_rng(2)
    n = 400
    X = np.column_stack(
        [
            rng.normal(size=n),
            rng.integers(0, 30, size=n).astype(np.float32),  # 30 distinct values ...
            rng.normal(size=n),
        ]
    ).astype(np.float32)
    X[:4, 1] = np.nan  # ... and only 4 missing: "is missing" splits are possible
    X[rng.random(n) < 0.4, 2] = np.nan  # 40% missing: NaN fills the cutpoint slots
    y = X[:, 0] + 3.0 * np.isnan(X[:, 1]) + np.nan_to_num(X[:, 2])
    reg = BartRegressor(
        num_trees=30,
        n_save=30,
        n_burn=100,
        num_chains=1,
        show_progress=False,
        impute_strategy=None,  # pass NaN to bartz to test its own handling
    )
    with pytest.warns(UserWarning, match="NaN"):
        reg.fit(X, y)
    # without imputation, NaN reach the trees and the missing direction is reported
    assert set(reg.trees_to_dataframe(draws=0).dropna(subset=["condition"])["missing"]) <= {
        "left",
        "right",
    }

    trees = tree_arrays(reg.bart_)
    real_cuts = [int(np.sum(~np.isnan(c))) for c in trees.cutpoints]
    assert trees.nan_bin[1] > real_cuts[1] + 1  # few NaN: binned above the NaN cutpoints
    assert trees.nan_bin[2] == real_cuts[2]  # many NaN: right above the real cutpoints

    latent = reg._draws(X, "latent_samples")
    for draw in range(30):
        np.testing.assert_allclose(
            predict_one_draw(reg.bart_, X, 0, draw), latent[0, draw], atol=1e-5
        )


def _walk(table: pd.DataFrame, x: np.ndarray, feature_index: dict[str, int]) -> float:
    """Sum of the leaf values one row reaches, walking every tree of the table."""
    total = 0.0
    for _, tree in table.groupby(["chain", "draw", "tree"]):
        nodes = {int(r["node"]): r for r in tree.to_dict("records")}
        node = min(nodes)  # the root
        while not nodes[node]["is_leaf"]:
            row = nodes[node]
            value = x[feature_index[row["feature"]]]
            node = int(row["left"] if value <= row["cutpoint"] else row["right"])
        total += float(nodes[node]["leaf_value"])
    return total


def test_trees_to_dataframe_reproduces_predictions(
    fitted_regressor: BartRegressor, regression_data: tuple
) -> None:
    X, _ = regression_data
    table = fitted_regressor.trees_to_dataframe(chains=1, draws=-1)  # all trees of one draw
    assert set(table["chain"]) == {1} and set(table["draw"]) == {299}
    assert table.groupby("tree").size().size == 50
    latent = fitted_regressor._draws(X[:5], "latent_samples")[1, 299]
    offset = float(fitted_regressor.bart_._main_trace.offset)
    index = {name: j for j, name in enumerate(fitted_regressor.model_feature_names_)}
    for i in range(5):
        walked = offset + _walk(table, X.to_numpy()[i], index)
        assert walked == pytest.approx(latent[i], abs=1e-4)


def test_trees_to_dataframe_columns_and_num_rows(
    fitted_regressor: BartRegressor, regression_data: tuple
) -> None:
    X, _ = regression_data
    table = fitted_regressor.trees_to_dataframe(chains=0, draws=[0, -1], trees=[0, 1], X=X[:100])
    assert len(table.groupby(["chain", "draw", "tree"])) == 4
    splits, leaves = table[~table["is_leaf"]], table[table["is_leaf"]]
    assert splits["condition"].str.contains(" <= ").all()
    assert leaves["left"].isna().all() and leaves["leaf_value"].notna().all()
    for _, tree in table.groupby(["draw", "tree"]):
        rows = tree.set_index("node")["num_rows"]
        assert rows.iloc[0] == 100  # every row reaches the root
        assert tree.loc[tree["is_leaf"], "num_rows"].sum() == 100  # and exactly one leaf
        for row in tree[~tree["is_leaf"]].itertuples():
            assert rows[row.left] + rows[row.right] == rows[row.node]
