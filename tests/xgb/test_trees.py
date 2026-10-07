import numpy as np
import pandas as pd
import pytest
import xgboost as xgb

from tree_ensembles.xgb import XGBDefaultRegressor


@pytest.fixture(scope="module")
def fitted() -> tuple[XGBDefaultRegressor, pd.DataFrame]:
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(500, 3)), columns=["a", "b", "c"])
    X.loc[::9, "b"] = np.nan  # so some splits learn a missing direction
    y = X["a"] + np.where(X["b"].isna(), 2.0, X["b"]) + 0.1 * rng.normal(size=500)
    return XGBDefaultRegressor(n_estimators=30, max_depth=3).fit(X, y), X


def test_walking_the_table_reproduces_xgboost(fitted: tuple) -> None:
    reg, X = fitted
    table = reg.trees_to_dataframe()
    assert table["depth"].max() <= 3
    assert table.loc[~table["is_leaf"], "condition"].str.contains(" < ").all()
    margin = reg.model_.get_booster().predict(xgb.DMatrix(X), output_margin=True)
    for i in [0, 9, 18, 1, 2]:  # rows 0, 9, 18 have a missing b
        row_x = X.iloc[i].to_dict()
        total = float(reg.params_["base_score"])
        for _, tree in table.groupby("tree"):
            nodes = {int(r["node"]): r for r in tree.to_dict("records")}
            node = 0
            while not nodes[node]["is_leaf"]:
                split = nodes[node]
                value = row_x[split["feature"]]
                if np.isnan(value):
                    side = split["missing"]
                else:
                    side = "left" if value < split["cutpoint"] else "right"
                node = int(split[side])
            total += float(nodes[node]["leaf_value"])
        assert total == pytest.approx(margin[i], abs=1e-4)


def test_select_trees_and_categorical_condition() -> None:
    rng = np.random.default_rng(1)
    X = pd.DataFrame({"c": pd.Categorical(rng.choice(list("xyz"), 300))})
    y = (X["c"] == "x") * 2.0 + 0.1 * rng.normal(size=300)
    reg = XGBDefaultRegressor(n_estimators=5).fit(X, y)
    table = reg.trees_to_dataframe(trees=[0, 1])
    assert set(table["tree"]) == {0, 1}
    assert str(table["condition"].iloc[0]).startswith("c in [")
