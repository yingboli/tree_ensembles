import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from tree_ensembles.bart import BartRegressor, PooledMeanEncoder
from tree_ensembles.bart.categories import pooled_level_means


def test_pooled_means_match_the_formulas() -> None:
    rng = np.random.default_rng(0)
    sizes = {"a": 200, "b": 50, "c": 10, "d": 3}
    effects = {"a": -1.0, "b": 0.0, "c": 1.0, "d": 3.0}
    levels = pd.Series([lv for lv, n in sizes.items() for _ in range(n)])
    y = levels.map(effects).to_numpy() + rng.normal(size=len(levels))
    table = pooled_level_means(levels, y)

    # the same estimates, computed directly
    n_j = np.array(list(sizes.values()), dtype=float)
    means = np.array([y[levels == lv].mean() for lv in sizes])
    n, k, mu = len(y), len(sizes), y.mean()
    sigma2 = sum(((y[levels == lv] - y[levels == lv].mean()) ** 2).sum() for lv in sizes) / (n - k)
    msb = (n_j * (means - mu) ** 2).sum() / (k - 1)
    n0 = (n - (n_j**2).sum() / n) / (k - 1)
    tau2 = max(0.0, (msb - sigma2) / n0)
    weight = tau2 / (tau2 + sigma2 / n_j)
    expected = mu + weight * (means - mu)

    assert table.attrs["tau2"] == pytest.approx(tau2) and table.attrs["sigma2"] == pytest.approx(
        sigma2
    )
    for lv, value in zip(sizes, expected, strict=True):
        assert table["pooled_mean"].to_dict()[lv] == pytest.approx(value)
    assert (
        table["weight"].to_dict()["a"] > table["weight"].to_dict()["d"]
    )  # rare level: more pooling


def test_no_level_differences_pool_everything() -> None:
    rng = np.random.default_rng(1)
    levels = pd.Series(np.repeat(["a", "b", "c"], 100))
    y = rng.normal(size=300)  # same distribution in every level
    table = pooled_level_means(levels, y)
    if table.attrs["tau2"] == 0:
        assert np.allclose(table["pooled_mean"], y.mean())
    assert table["weight"].max() < 0.5


def test_encoder_missing_and_unseen_levels() -> None:
    X = pd.DataFrame({"c": pd.Categorical(["x", "x", "y", "y", None, None]), "z": range(6)})
    y = np.array([0.0, 0.0, 10.0, 10.0, 5.0, 5.0])
    encoder = PooledMeanEncoder().fit(X, y)
    assert encoder.columns_ == ["c"]
    out = encoder.transform(X)
    assert out["c"].dtype == float and list(out["z"]) == list(range(6))
    assert (
        out["c"].iloc[0] < out["c"].iloc[4] < out["c"].iloc[2]
    )  # ordered by mean y, missing between

    new = pd.DataFrame({"c": pd.Categorical(["w"]), "z": [0]})  # level never seen in fit
    assert encoder.transform(new)["c"].iloc[0] == pytest.approx(y.mean())


@pytest.fixture(scope="module")
def category_data() -> tuple[pd.DataFrame, np.ndarray]:
    rng = np.random.default_rng(2)
    n = 400
    color = rng.choice(["red", "green", "blue", "gray"], size=n)
    X = pd.DataFrame({"a": rng.normal(size=n), "color": pd.Categorical(color)})
    effect = pd.Series(color).map({"red": 2.0, "green": -2.0, "blue": 1.5, "gray": -1.5}).to_numpy()
    return X, X["a"].to_numpy() + effect + 0.3 * rng.normal(size=n)


SMALL: dict[str, Any] = {
    "num_trees": 20,
    "n_save": 50,
    "n_burn": 50,
    "num_chains": 1,
    "show_progress": False,
}


def test_bart_rejects_categories_without_ordering(category_data: tuple) -> None:
    X, y = category_data
    with pytest.raises(TypeError, match="order_categories=True"):
        BartRegressor(order_categories=False, **SMALL).fit(X, y)  # text categories
    codes = X.assign(color=X["color"].cat.codes.astype("category"))
    with pytest.raises(TypeError, match="categorical columns"):
        BartRegressor(order_categories=False, **SMALL).fit(codes, y)  # numeric categories too

    reg = BartRegressor(**SMALL).fit(
        X.assign(color=X["color"].cat.codes), y
    )  # plain numbers: no encoder
    with pytest.raises(TypeError, match="categorical columns"):
        reg.predict(codes[:5])  # category columns at predict time as well


@pytest.mark.parametrize("dtype", [object, "string"])
def test_bart_rejects_text_columns(category_data: tuple, dtype: Any) -> None:
    X, y = category_data
    text = X.assign(color=X["color"].astype(dtype))
    with pytest.raises(TypeError, match=r"astype\('category'\)"):
        BartRegressor(**SMALL).fit(text, y)
    numbers_as_text = X.assign(color=X["color"].cat.codes.astype(str).astype(dtype))
    with pytest.raises(TypeError, match="pd.to_numeric"):
        BartRegressor(**SMALL).fit(numbers_as_text, y)  # not silently read as numbers
    reg = BartRegressor(**SMALL).fit(X, y)
    with pytest.raises(TypeError, match="text"):
        reg.predict(text[:5])  # at predict time as well


def test_bart_orders_categories(category_data: tuple, tmp_path: Path) -> None:
    X, y = category_data
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        warnings.filterwarnings("ignore", message=".*shard.*")
        reg = BartRegressor(**SMALL).fit(X, y)  # ordering is the default
    levels = reg.category_encoder_.levels_["color"]  # type: ignore[union-attr]
    assert list(levels.index) == ["green", "gray", "blue", "red"]  # ordered by pooled mean y
    assert np.corrcoef(reg.predict(X), y)[0, 1] > 0.9
    assert isinstance(reg.X_probe_, pd.DataFrame)  # probe rows kept as given
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # short chains do not converge
        assert (
            "f(x_probe): worst" in reg.diagnostics().index
        )  # probe rows are encoded like any new rows

    new = pd.DataFrame({"a": [0.0], "color": pd.Categorical(["purple"])})  # unseen level
    assert np.isfinite(reg.predict(new)).all()
    with pytest.raises(ValueError, match="DataFrame"):
        reg.predict(X.to_numpy())

    reg.dump(tmp_path / "m")
    loaded = BartRegressor().load(tmp_path / "m")
    np.testing.assert_allclose(loaded.predict(X[:5]), reg.predict(X[:5]))
