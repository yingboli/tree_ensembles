import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone
from sklearn.metrics import roc_auc_score

from tree_ensembles.bart import BartClassifier, BartRegressor


def test_regressor_predictions(fitted_regressor: BartRegressor, regression_data: tuple) -> None:
    X, y = regression_data
    pred = fitted_regressor.predict(X[800:])
    assert pred.shape == (200,)
    assert np.sqrt(np.mean((pred - y[800:]) ** 2)) < 0.45


def test_sigma_recovered(fitted_regressor: BartRegressor) -> None:
    summary = fitted_regressor.posterior_summary()
    sigma = summary.loc["sigma"]
    assert sigma["hpdi_0.95_low"] < 0.3 * 1.15 and sigma["hpdi_0.95_high"] > 0.3 * 0.85
    assert {"mean_tree_leaves", "mean_tree_depth"} <= set(summary.index)


def test_default_summary_uses_95_percent_hpdi(fitted_regressor: BartRegressor) -> None:
    assert "hpdi_0.95_low" in fitted_regressor.posterior_summary().columns


def test_parameter_draws(fitted_regressor: BartRegressor) -> None:
    draws = fitted_regressor.parameter_draws()
    assert set(draws) == {"sigma", "mean_tree_leaves", "mean_tree_depth"}
    assert all(d.shape == (2, 300) for d in draws.values())


def test_predictions_in_row_batches(
    fitted_regressor: BartRegressor, regression_data: tuple
) -> None:
    X, _ = regression_data
    whole = fitted_regressor.predict_dist(X[:50])
    pd.testing.assert_frame_equal(fitted_regressor.predict_dist(X[:50], batch_size=7), whole)
    np.testing.assert_allclose(fitted_regressor.predict(X[:50]), whole["mean"], rtol=1e-5)
    pd.testing.assert_frame_equal(
        fitted_regressor.predict_interval(X[:50], batch_size=7),
        fitted_regressor.predict_interval(X[:50]),
    )
    with pytest.raises(ValueError, match="batch_size"):
        fitted_regressor.predict_dist(X[:5], batch_size=0)


def test_n_probe_and_thresholds(regression_data: tuple) -> None:
    X, y = regression_data
    params: dict[str, Any] = {
        "num_trees": 10,
        "n_save": 20,
        "n_burn": 20,
        "num_chains": 2,
        "show_progress": False,
    }
    reg = BartRegressor(**params).fit(X[:200], y[:200], n_probe=5)
    assert reg.X_probe_.shape == (5, 3)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        table = reg.diagnostics(rhat_max=np.inf, ess_min=0)  # thresholds anyone passes
    # 3 parameters, accept_rate, leaf_fill, 2 f(x) rows
    assert table["ok"].all() and len(table) == 5 + 2
    assert table.attrs["f_ok_share"] == 1.0
    assert not reg.posterior_summary(ess_min=1e9)["ok"].any()


def test_predict_dist_and_intervals(
    fitted_regressor: BartRegressor, regression_data: tuple
) -> None:
    X, y = regression_data
    X_test, y_test = X[800:], y[800:]
    dist = fitted_regressor.predict_dist(X_test)
    assert list(dist.index) == list(X_test.index)
    assert (dist["predictive_sd"] > dist["sd"]).all()  # adds the noise variance

    pred = fitted_regressor.predict_interval(X_test, kind="predictive")  # default 95%
    coverage = np.mean((y_test >= pred["lower"]) & (y_test <= pred["upper"]))
    assert 0.88 < coverage < 0.99  # about 95% on 200 test rows

    mean = fitted_regressor.predict_interval(X_test, kind="mean")
    quant = fitted_regressor.predict_interval(X_test, method="quantile")
    assert ((mean["upper"] - mean["lower"]) < (pred["upper"] - pred["lower"])).all()
    assert np.median((mean["upper"] - mean["lower"]) / (quant["upper"] - quant["lower"])) <= 1


def test_draw_shapes_follow_chain_order(
    fitted_regressor: BartRegressor, regression_data: tuple
) -> None:
    X, _ = regression_data
    draws = fitted_regressor.predict_samples(X[:5])
    assert draws.shape == (2 * 300, 5)
    by_chain = fitted_regressor._draws(X[:5], "mean_samples")
    np.testing.assert_array_equal(by_chain[1, 0], draws[300])  # chain 1 starts at row 300


def test_diagnostics_table(fitted_regressor: BartRegressor) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # short test chains do not pass the thresholds
        table = fitted_regressor.diagnostics(per_point=True)
        summary = fitted_regressor.diagnostics()
    assert {"sigma", "accept_rate", "f(x_probe[0])", "f(x_probe[19])"} <= set(table.index)
    points = table.loc[[f"f(x_probe[{i}])" for i in range(20)]]
    worst = summary.loc["f(x_probe): worst"]
    assert worst["rhat"] == pytest.approx(points["rhat"].max())
    assert worst["ess_bulk"] == pytest.approx(points["ess_bulk"].min())
    assert summary.loc["f(x_probe): median", "rhat"] == pytest.approx(points["rhat"].median())
    assert summary.attrs["f_ok_share"] == pytest.approx(points["ok"].astype(bool).mean())
    assert not any(name.startswith("f(x_probe[") for name in summary.index)
    checked = table.drop(index=["accept_rate", "leaf_fill"])
    assert np.isfinite(checked[["rhat", "ess_bulk", "ess_tail"]].to_numpy(dtype=float)).all()
    # the acceptance rate and leaf fill are only reported as averages, not checked
    reported = table.loc[["accept_rate", "leaf_fill"]]
    assert ((reported["mean"] > 0) & (reported["mean"] < 1)).all()
    assert reported["rhat"].isna().all() and reported["ok"].isna().all()
    # leaf_fill is bartz's "leaves 2.6/32": mean leaves per tree over 2 ** (maxdepth - 1)
    assert table.attrs["max_leaves"] == 2 ** (fitted_regressor.maxdepth - 1)
    mean_leaves = table.loc["mean_tree_leaves", "mean"]
    assert table.loc["leaf_fill", "mean"] == pytest.approx(mean_leaves / table.attrs["max_leaves"])


def test_diagnostics_warns_for_large_trees(fitted_regressor: BartRegressor) -> None:
    passing = {"rhat_max": np.inf, "ess_min": 0}  # no convergence warning
    with pytest.warns(UserWarning, match="Large trees"):
        fitted_regressor.diagnostics(leaf_fill_max=0.0, **passing)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        fitted_regressor.diagnostics(leaf_fill_max=1.0, **passing)
    assert not any("Large trees" in str(w.message) for w in caught)


def test_diagnostics_warns_for_short_chains(fitted_regressor: BartRegressor) -> None:
    with pytest.warns(UserWarning, match="convergence"):
        fitted_regressor.diagnostics()


def test_classifier(fitted_classifier: BartClassifier, regression_data: tuple) -> None:
    X, y = regression_data
    truth = (y[800:] > 0).astype(int)
    proba = fitted_classifier.predict_proba(X[800:])
    assert proba.shape == (200, 2)
    np.testing.assert_allclose(proba.sum(axis=1), 1, rtol=1e-6)
    assert roc_auc_score(truth, proba[:, 1]) > 0.85
    assert abs(proba[:, 1].mean() - truth.mean()) < 0.1
    assert set(fitted_classifier.predict(X[800:])) <= {"yes", "no"}

    interval = fitted_classifier.predict_interval(X[800:])
    assert ((interval["lower"] >= 0) & (interval["upper"] <= 1)).all()
    with pytest.raises(ValueError, match="kind='mean'"):
        fitted_classifier.predict_interval(X[800:], kind="predictive")
    assert "sigma" not in fitted_classifier.posterior_summary().index


def test_predictive_samples_of_classifier_are_binary(
    fitted_classifier: BartClassifier, regression_data: tuple
) -> None:
    X, _ = regression_data
    assert set(np.unique(fitted_classifier.predict_samples(X[:10], kind="predictive"))) <= {0, 1}


def test_multiclass_raises() -> None:
    with pytest.raises(ValueError, match="binary"):
        BartClassifier().fit(np.zeros((6, 1)), [0, 1, 2, 0, 1, 2])


def test_reserved_bartz_param_raises(regression_data: tuple) -> None:
    X, y = regression_data
    with pytest.raises(ValueError, match="random_state"):
        BartRegressor(seed=1).fit(X, y)


def test_extra_bartz_params_reach_bartz(regression_data: tuple) -> None:
    X, y = regression_data
    with pytest.raises(TypeError, match="not_a_bartz_argument"):  # forwarded to bartz.Bart
        BartRegressor(not_a_bartz_argument=1, show_progress=False).fit(X[:50], y[:50])


def test_nan_without_imputation_warns() -> None:
    X = np.random.default_rng(0).normal(size=(50, 2))
    X[0, 0] = np.nan
    reg = BartRegressor(
        impute_strategy=None, num_trees=5, n_save=5, n_burn=5, num_chains=1, show_progress=False
    )
    with pytest.warns(UserWarning, match="impute_strategy=None"):
        reg.fit(X, X[:, 1])
    assert reg.imputer_ is None and reg.model_feature_names_ == reg.feature_names_in_


@pytest.fixture(scope="module")
def nan_data() -> tuple[pd.DataFrame, np.ndarray]:
    """a: no NaN; b: 20% NaN; c: 70% NaN and its missingness shifts y by 2."""
    rng = np.random.default_rng(5)
    n = 600
    X = pd.DataFrame(rng.normal(size=(n, 3)), columns=["a", "b", "c"])
    X.loc[rng.random(n) < 0.2, "b"] = np.nan
    c_missing = rng.random(n) < 0.7
    X.loc[c_missing, "c"] = np.nan
    y = X["a"].to_numpy() + 2.0 * c_missing + 0.3 * rng.normal(size=n)
    return X, y


def test_imputation_is_the_default(nan_data: tuple) -> None:
    X, y = nan_data
    params: dict[str, Any] = {
        "num_trees": 30,
        "n_save": 50,
        "n_burn": 50,
        "num_chains": 2,
        "show_progress": False,
    }
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)  # no NaN warning when imputing
        warnings.filterwarnings("ignore", message=".*shard.*")
        reg = BartRegressor(**params).fit(X, y)

    # only c is missing more than 50% of the time, so only c gets an indicator
    assert reg.model_feature_names_ == ["a", "b", "c", "c_missing"]
    assert reg.feature_names_in_ == ["a", "b", "c"]
    assert np.isnan(np.asarray(reg.X_probe_, dtype=float)).any()  # probe rows kept as given

    pred = reg.predict(X)  # NaN in new rows are imputed the same way
    assert np.isfinite(pred).all()
    assert np.corrcoef(pred, y)[0, 1] > 0.9  # the trees use c_missing
    assert "c_missing" in reg.forest_summary().variable_usage().index
    assert reg.split_points()["cutpoint"].notna().all()  # no NaN cutpoints any more
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # tiny test chains do not converge
        assert "f(x_probe): worst" in reg.diagnostics().index

    zero = BartRegressor(missing_indicator_threshold=0.0, **params).fit(X, y)
    assert zero.model_feature_names_ == ["a", "b", "c", "b_missing", "c_missing"]


def test_wrong_number_of_features(fitted_regressor: BartRegressor) -> None:
    with pytest.raises(ValueError, match="shape"):
        fitted_regressor.predict(np.zeros((3, 5)))


def test_clone_and_set_params_keep_extra_params() -> None:
    model = BartRegressor(num_trees=30, k=3.0)
    copy = clone(model)
    assert copy.get_params()["num_trees"] == 30 and copy.get_params()["k"] == 3.0
    copy.set_params(k=1.5, sigma_df=10.0, n_save=50)
    assert copy.bartz_params == {"k": 1.5, "sigma_df": 10.0} and copy.n_save == 50


def test_dump_and_load(
    fitted_regressor: BartRegressor, regression_data: tuple, tmp_path: Path
) -> None:
    X, _ = regression_data
    fitted_regressor.dump(tmp_path / "model")
    loaded = BartRegressor().load(tmp_path / "model")
    assert isinstance(loaded, BartRegressor)
    np.testing.assert_allclose(loaded.predict(X[:10]), fitted_regressor.predict(X[:10]))
    pd.testing.assert_frame_equal(loaded.posterior_summary(), fitted_regressor.posterior_summary())


def test_tree_prior_and_thinning_params(regression_data: tuple) -> None:
    X, y = regression_data
    params: dict[str, Any] = {
        "num_trees": 10,
        "n_save": 20,
        "n_burn": 20,
        "num_chains": 1,
        "show_progress": False,
    }
    reg = BartRegressor(n_skip=3, power=1.0, base=0.5, **params).fit(X[:200], y[:200])
    assert reg.get_params()["n_skip"] == 3
    assert reg.bart_.n_save == 20  # n_skip thins: 20 saved draws from 60 iterations
    with pytest.raises(ValueError, match="set by this class"):
        BartRegressor(outcome_type="binary").fit(X, y)


def test_dataframe_columns_must_match(
    fitted_regressor: BartRegressor, regression_data: tuple
) -> None:
    X, _ = regression_data
    with pytest.raises(ValueError, match="missing: \\['c'\\], unexpected: \\['d'\\]"):
        fitted_regressor.predict(X[:5].rename(columns={"c": "d"}))
    with pytest.raises(ValueError, match="different order"):
        fitted_regressor.predict(X[:5][["b", "a", "c"]])
    fitted_regressor.predict(X[:5].to_numpy())  # a plain array is accepted


def test_array_fit_accepts_any_dataframe_names() -> None:
    rng = np.random.default_rng(0)
    X = rng.normal(size=(100, 2))
    params: dict[str, Any] = {
        "num_trees": 10,
        "n_save": 20,
        "n_burn": 20,
        "num_chains": 1,
        "show_progress": False,
    }
    reg = BartRegressor(**params).fit(X, X[:, 0])
    reg.predict(pd.DataFrame(X, columns=["u", "v"]))  # fitted without names: nothing to check


def test_model_saved_by_version_0_3_still_works(
    fitted_regressor: BartRegressor, regression_data: tuple, tmp_path: Path
) -> None:
    """Version 0.3.0 stored bartz_params=None when no extra parameters were given."""
    X, y = regression_data
    fitted_regressor.dump(tmp_path / "model")
    loaded = BartRegressor().load(tmp_path / "model")
    loaded.bartz_params = None  # type: ignore[assignment]  # what a 0.3.0 dump holds
    assert "num_trees=50" in repr(loaded)
    assert "k" not in loaded.get_params()
    assert clone(loaded).get_params()["num_trees"] == 50
    loaded.set_params(n_save=20, n_burn=20).fit(X[:100], y[:100])  # refitting works too


def test_load_fills_the_estimator_in_place(
    fitted_regressor: BartRegressor,
    fitted_classifier: BartClassifier,
    regression_data: tuple,
    tmp_path: Path,
) -> None:
    X, _ = regression_data
    fitted_regressor.dump(tmp_path / "reg")
    reg = BartRegressor(num_trees=999)  # settings are replaced by the dumped ones
    returned = reg.load(tmp_path / "reg")
    assert returned is reg and reg.num_trees == 50
    np.testing.assert_allclose(reg.predict(X[:5]), fitted_regressor.predict(X[:5]))

    fitted_classifier.dump(tmp_path / "clf")
    with pytest.raises(TypeError, match="BartClassifier"):
        BartRegressor().load(tmp_path / "clf")
    with pytest.raises(FileNotFoundError, match="No model dumped"):
        BartRegressor().load(tmp_path / "empty")


def test_unfitted_model_says_not_fitted(regression_data: tuple, tmp_path: Path) -> None:
    from sklearn.exceptions import NotFittedError

    X, _ = regression_data
    reg = BartRegressor()
    for call in (
        lambda: reg.predict(X[:3]),
        lambda: BartClassifier().predict_proba(X[:3]),
        lambda: reg.dump(tmp_path / "m"),
        reg.posterior_summary,
        reg.forest_summary,
        lambda: reg.trees_to_dataframe(draws=0),
    ):
        with pytest.raises(NotFittedError):
            call()
