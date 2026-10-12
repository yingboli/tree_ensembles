import matplotlib
import numpy as np

matplotlib.use("Agg")  # no display needed

from tree_ensembles.bart import BartRegressor  # noqa: E402
from tree_ensembles.bart.plots import (  # noqa: E402
    plot_feature_importance,
    plot_rank,
    plot_trace,
    plot_tree_sizes,
)


def test_draw_plots() -> None:
    draws = np.random.default_rng(0).normal(size=(4, 200))
    assert len(plot_trace(draws, "sigma").axes) == 2
    assert len(plot_rank(draws).axes) == 4


def test_forest_plots(fitted_regressor: BartRegressor) -> None:
    summary = fitted_regressor.forest_summary()
    assert len(plot_tree_sizes(summary).axes) == 2
    with_prior = plot_tree_sizes(summary, show_prior=True)
    assert all(len(ax.containers) == 2 for ax in with_prior.axes)  # posterior and prior bars
    plain = plot_feature_importance(summary, top=2)
    assert len(plain.axes) == 1 and not plain.axes[0].collections  # no error bars
    with_hpdi = plot_feature_importance(summary, top=2, prob=0.9)
    assert len(with_hpdi.axes[0].collections) == 1  # the HPDI error bars
    assert with_hpdi.axes[0].get_title() == "Feature importance"


def test_plot_bart_trees(fitted_regressor: BartRegressor) -> None:
    from tree_ensembles.tree_plot import plot_tree, plot_trees

    X = np.random.default_rng(0).normal(size=(30, 3))
    table = fitted_regressor.trees_to_dataframe(chains=0, draws=-1, trees=[0, 1, 2], X=X)
    assert "num_rows" in table.columns and "cover" not in table.columns
    fig = plot_tree(table[table["tree"] == 0])
    assert fig.axes[0].get_title() == "chain 0, draw 299, tree 0"
    assert any("\nrows 30" in t.get_text() for t in fig.axes[0].texts)  # the root
    assert table["missing"].isna().all()  # NaN are imputed, so no missing direction
    assert not any("missing" in t.get_text() for t in fig.axes[0].texts)  # plain yes / no
    assert len([ax for ax in plot_trees(table).axes if ax.texts]) == 3
