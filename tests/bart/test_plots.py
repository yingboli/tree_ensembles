import matplotlib
import numpy as np

matplotlib.use("Agg")  # no display needed

from tree_ensembles.bart import BartRegressor  # noqa: E402
from tree_ensembles.bart.plots import (  # noqa: E402
    plot_rank,
    plot_trace,
    plot_tree_sizes,
    plot_variable_usage,
)


def test_draw_plots() -> None:
    draws = np.random.default_rng(0).normal(size=(4, 200))
    assert len(plot_trace(draws, "sigma").axes) == 2
    assert len(plot_rank(draws).axes) == 4


def test_forest_plots(fitted_regressor: BartRegressor) -> None:
    summary = fitted_regressor.forest_summary()
    assert len(plot_tree_sizes(summary).axes) == 2
    assert len(plot_variable_usage(summary, top=2).axes) == 1
