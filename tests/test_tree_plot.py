import matplotlib
import numpy as np
import pandas as pd
import pytest

matplotlib.use("Agg")  # no display needed

from tree_ensembles.tree_plot import plot_tree, plot_trees  # noqa: E402
from tree_ensembles.xgb import XGBDefaultRegressor  # noqa: E402


@pytest.fixture(scope="module")
def xgb_table() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(300, 3)), columns=["a", "b", "c"])
    y = X["a"] * X["b"] + rng.normal(size=300)
    return XGBDefaultRegressor(n_estimators=6, max_depth=3).fit(X, y).trees_to_dataframe()


def test_plot_one_xgboost_tree(xgb_table: pd.DataFrame) -> None:
    fig = plot_tree(xgb_table[xgb_table["tree"] == 0])
    texts = [t.get_text() for t in fig.axes[0].texts]
    assert any(" < " in t for t in texts) and any(t.startswith(("yes", "no")) for t in texts)
    assert any("\ncover " in t for t in texts)  # XGBoost's node size is its cover
    # every XGBoost split shows which branch missing values follow
    n_splits = int((~xgb_table[xgb_table["tree"] == 0]["is_leaf"]).sum())
    assert sum(t in ("yes, missing", "no, missing") for t in texts) == n_splits


def test_plot_tree_needs_one_tree(xgb_table: pd.DataFrame) -> None:
    with pytest.raises(ValueError, match="one tree"):
        plot_tree(xgb_table)


def test_plot_several_trees(xgb_table: pd.DataFrame) -> None:
    fig = plot_trees(xgb_table, ncols=2, max_trees=4)
    drawn = [ax for ax in fig.axes if ax.texts]
    assert len(drawn) == 4


def test_max_depth_cuts_the_drawing(xgb_table: pd.DataFrame) -> None:
    tree = xgb_table[xgb_table["tree"] == 0]
    full = plot_tree(tree).axes[0].texts
    cut = plot_tree(tree, max_depth=1).axes[0].texts
    assert len(cut) < len(full)
    assert sum(t.get_text().endswith("...") for t in cut) >= 1


def test_node_sizes_are_easy_to_read() -> None:
    from tree_ensembles.tree_plot import _number

    assert _number(20000.0) == "20,000"
    assert _number(142.25) == "142.2"
