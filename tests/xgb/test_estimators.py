import math
from typing import Any

import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import r2_score, roc_auc_score
from sklearn.model_selection import cross_val_score, train_test_split

from tree_ensembles.xgb import XGBDefaultClassifier, XGBDefaultRegressor


@pytest.fixture
def binary_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    X, y = load_breast_cancer(return_X_y=True, as_frame=True)
    return train_test_split(X, y, test_size=0.25, random_state=0, stratify=y)


@pytest.fixture
def regression_data() -> tuple[pd.DataFrame, np.ndarray]:
    """y = signal + N(0, 1) noise, so the true residual variance is 1."""
    rng = np.random.default_rng(0)
    n = 2000
    X = pd.DataFrame(rng.normal(size=(n, 4)), columns=["a", "b", "c", "d"])
    y = 2 * X["a"] + np.sin(3 * X["b"]) + (X["c"] > 0) + rng.normal(size=n)
    return X, y.to_numpy()


def test_classifier_defaults(binary_data: tuple) -> None:
    X_train, X_test, y_train, y_test = binary_data
    clf = XGBDefaultClassifier().fit(X_train, y_train)

    n = len(y_train)
    assert clf.params_["n_estimators"] == 2000
    assert clf.params_["learning_rate"] == 0.1
    assert clf.params_["max_depth"] == 6
    assert clf.params_["tree_method"] == "hist"
    assert clf.params_["max_bin"] == 256
    assert clf.params_["base_score"] == pytest.approx(y_train.mean())
    assert clf.phi_ == 1.0
    assert clf.gamma_ == pytest.approx(3 * math.log(math.log(n)))
    assert clf.n_trees_ < 2000
    assert roc_auc_score(y_test, clf.predict_proba(X_test)[:, 1]) > 0.95


def test_regressor_defaults(regression_data: tuple) -> None:
    X, y = regression_data
    reg = XGBDefaultRegressor().fit(X, y)

    assert 0.8 < reg.phi_ < 1.3
    assert reg.n_trees_ < 2000
    assert r2_score(y, reg.predict(X)) > 0.7


def test_user_overrides(regression_data: tuple) -> None:
    X, y = regression_data
    reg = XGBDefaultRegressor(gamma=0.0, max_depth=3, xgb_params={"subsample": 0.8})
    reg.set_params(n_estimators=20).fit(X, y)
    assert reg.params_["gamma"] == 0.0
    assert reg.params_["max_depth"] == 3
    assert reg.params_["subsample"] == 0.8
    assert reg.n_trees_ == 20

    bic = XGBDefaultRegressor(criterion="bic", phi=2.0).fit(X, y)
    assert bic.gamma_ == pytest.approx(math.log(len(y)) * 2.0)


def test_named_param_in_xgb_params_raises(regression_data: tuple) -> None:
    X, y = regression_data
    with pytest.raises(ValueError):
        XGBDefaultRegressor(xgb_params={"gamma": 1.0}).fit(X, y)


def test_optuna_tuning(binary_data: tuple) -> None:
    X_train, _, y_train, _ = binary_data
    clf = XGBDefaultClassifier(learning_rate="optuna", max_depth="optuna", n_optuna_trials=3)
    clf.fit(X_train, y_train)

    best = clf.optuna_best_params_
    assert best is not None and set(best) == {"learning_rate", "max_depth"}
    assert 0.05 <= best["learning_rate"] <= 0.3
    assert best["max_depth"] in {3, 4, 5, 6}
    assert clf.params_["learning_rate"] == best["learning_rate"]
    assert clf.params_["max_depth"] == best["max_depth"]
    # The fitted object holds no Optuna objects, so loading it never needs Optuna.
    assert not any(type(v).__module__.startswith("optuna") for v in vars(clf).values())


def test_no_tuning_leaves_optuna_best_params_empty(regression_data: tuple) -> None:
    X, y = regression_data
    assert XGBDefaultRegressor(n_estimators=5).fit(X, y).optuna_best_params_ is None


def test_tuning_on_user_validation_data(
    regression_data: tuple, monkeypatch: pytest.MonkeyPatch
) -> None:
    X, y = regression_data
    X_train, X_val, y_train, y_val = X[:1500], X[1500:], y[:1500], y[1500:]

    # Record which data each Optuna run trains and scores on.
    seen: dict[str, int] = {}
    original_tune = XGBDefaultRegressor._tune

    def spy_tune(self: Any, fit_data: Any, valid: Any, params: Any) -> Any:
        seen["n_train"], seen["n_valid"] = len(fit_data.y), len(valid.y)
        return original_tune(self, fit_data, valid, params)

    monkeypatch.setattr(XGBDefaultRegressor, "_tune", spy_tune)
    reg = XGBDefaultRegressor(learning_rate="optuna", max_depth="optuna", n_optuna_trials=3)
    reg.fit(X_train, y_train, X_valid=X_val, y_valid=y_val)

    assert seen == {"n_train": 1500, "n_valid": 500}
    # The final model is trained on X_train only: same as fitting with the tuned values.
    assert reg.params_["base_score"] == pytest.approx(y_train.mean())
    best = reg.optuna_best_params_
    assert best is not None
    same = XGBDefaultRegressor(**best).fit(X_train, y_train)
    np.testing.assert_allclose(reg.predict(X_val), same.predict(X_val))


def test_validation_data_checks(binary_data: tuple) -> None:
    X_train, X_test, y_train, y_test = binary_data
    tuned = XGBDefaultClassifier(max_depth="optuna", n_optuna_trials=2)

    with pytest.raises(ValueError, match="both"):
        tuned.fit(X_train, y_train, X_valid=X_test)
    with pytest.raises(ValueError, match="not in y"):
        tuned.fit(X_train, y_train, X_valid=X_test, y_valid=y_test + 5)
    with pytest.warns(UserWarning, match="ignored"):
        XGBDefaultClassifier(n_estimators=5).fit(X_train, y_train, X_valid=X_test, y_valid=y_test)


def test_string_labels_in_validation_data() -> None:
    rng = np.random.default_rng(0)
    X = pd.DataFrame({"x": rng.normal(size=400)})
    y = np.where(X["x"] + rng.normal(size=400) > 0, "yes", "no")
    clf = XGBDefaultClassifier(max_depth="optuna", n_optuna_trials=2)
    clf.fit(X[:300], y[:300], X_valid=X[300:], y_valid=y[300:])
    assert clf.optuna_best_params_ is not None


def test_categorical_feature_and_string_labels() -> None:
    rng = np.random.default_rng(0)
    n = 500
    X = pd.DataFrame({"x": rng.normal(size=n), "c": pd.Categorical(rng.choice(list("abc"), n))})
    y = np.where((X["c"] == "a") | (X["x"] > 1), "yes", "no")
    clf = XGBDefaultClassifier().fit(X, y)
    assert set(clf.predict(X)) <= {"yes", "no"}


def test_multiclass_raises() -> None:
    X = np.random.default_rng(0).normal(size=(30, 2))
    with pytest.raises(ValueError):
        XGBDefaultClassifier().fit(X, np.arange(30) % 3)


def test_sklearn_compatible(binary_data: tuple) -> None:
    X_train, _, y_train, _ = binary_data
    clf = clone(XGBDefaultClassifier(criterion="aic", xgb_params={"subsample": 0.9}))
    assert clf.get_params()["xgb_params"] == {"subsample": 0.9}
    scores = cross_val_score(clf, X_train, y_train, cv=3)
    assert scores.mean() > 0.9


# --- sample weights -------------------------------------------------------------


def test_unit_weights_match_no_weights(regression_data: tuple) -> None:
    X, y = regression_data
    plain = XGBDefaultRegressor().fit(X, y)
    weighted = XGBDefaultRegressor().fit(X, y, sample_weight=np.ones(len(y)))
    assert weighted.gamma_ == pytest.approx(plain.gamma_)
    np.testing.assert_allclose(weighted.predict(X), plain.predict(X), rtol=1e-6)


def test_weight_two_matches_duplicated_rows(regression_data: tuple) -> None:
    """Frequency weights: weight 2 means the row appears twice, so n = sum of weights.

    Floating-point details differ between the two, so the fitted trees are close but
    not identical; phi is fixed so gamma can be compared exactly.
    """
    X, y = regression_data
    doubled = XGBDefaultRegressor(phi=1.0).fit(pd.concat([X, X]), np.concatenate([y, y]))
    weighted = XGBDefaultRegressor(phi=1.0).fit(X, y, sample_weight=np.full(len(y), 2.0))

    assert weighted.gamma_ == pytest.approx(3 * math.log(math.log(2 * len(y))))
    assert weighted.gamma_ == pytest.approx(doubled.gamma_)
    assert weighted.params_["base_score"] == pytest.approx(doubled.params_["base_score"])
    assert np.corrcoef(weighted.predict(X), doubled.predict(X))[0, 1] > 0.99


def test_too_small_weight_sum_raises(regression_data: tuple) -> None:
    X, y = regression_data
    w = np.full(len(y), 1 / len(y))  # weights normalized to sum 1: not frequency weights
    with pytest.raises(ValueError, match="sum of the weights"):
        XGBDefaultRegressor().fit(X, y, sample_weight=w)


def test_cross_val_score_with_weights(binary_data: tuple) -> None:
    X_train, _, y_train, _ = binary_data
    w = np.where(y_train == 1, 1.0, 2.0)
    scores = cross_val_score(
        XGBDefaultClassifier(), X_train, y_train, cv=3, params={"sample_weight": w}
    )
    assert scores.mean() > 0.9


# --- offset (base_margin) -------------------------------------------------------


@pytest.fixture
def offset_data() -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """y = offset + 3 + signal + noise: the +3 shift is not in the offset."""
    rng = np.random.default_rng(1)
    n = 3000
    X = pd.DataFrame(rng.normal(size=(n, 3)), columns=["a", "b", "c"])
    offset = rng.normal(size=n)
    y = offset + 3 + 2 * X["a"].to_numpy() + rng.normal(size=n)
    return X, y, offset


def test_regressor_with_offset(offset_data: tuple) -> None:
    X, y, offset = offset_data
    reg = XGBDefaultRegressor().fit(X, y, base_margin=offset)
    pred = reg.predict(X, base_margin=offset)

    assert reg.params_["base_score"] == 0.0  # so no offset in predict means offset 0
    assert np.mean(pred) == pytest.approx(np.mean(y), abs=0.02)  # the +3 shift is learned
    assert r2_score(y, pred) > 0.75
    np.testing.assert_allclose(reg.predict(X), reg.predict(X, base_margin=np.zeros(len(y))))


def test_offset_shift_learned_even_when_first_tree_is_empty(offset_data: tuple) -> None:
    X, y, offset = offset_data
    y_no_signal = y - 2 * X["a"].to_numpy()  # only offset + 3 + noise: no split is worth it
    reg = XGBDefaultRegressor().fit(X, y_no_signal, base_margin=offset)
    pred = reg.predict(X, base_margin=offset)

    assert reg.n_trees_ > 1
    # The learned shift is the sample mean of y - offset (about 3, plus noise).
    assert np.mean(pred - offset) == pytest.approx(np.mean(y_no_signal - offset), abs=0.01)


def test_classifier_with_offset() -> None:
    rng = np.random.default_rng(2)
    n = 4000
    X = pd.DataFrame(rng.normal(size=(n, 2)), columns=["a", "b"])
    offset = rng.normal(size=n)
    margin = offset - 1.5 + X["a"].to_numpy()
    y = (rng.random(n) < 1 / (1 + np.exp(-margin))).astype(int)

    clf = XGBDefaultClassifier().fit(X, y, base_margin=offset)
    proba = clf.predict_proba(X, base_margin=offset)[:, 1]

    assert clf.params_["base_score"] == 0.0
    assert np.mean(proba) == pytest.approx(np.mean(y), abs=0.01)
    assert roc_auc_score(y, proba) > 0.7


def test_base_score_with_offset_warns(offset_data: tuple) -> None:
    X, y, offset = offset_data
    with pytest.warns(UserWarning, match="base_score is ignored"):
        XGBDefaultRegressor(base_score=3.0, n_estimators=5).fit(X, y, base_margin=offset)


def test_offset_needs_validation_offset(offset_data: tuple) -> None:
    X, y, offset = offset_data
    tuned = XGBDefaultRegressor(max_depth="optuna", n_optuna_trials=2)
    with pytest.raises(ValueError, match="base_margin_valid"):
        tuned.fit(X[:2000], y[:2000], base_margin=offset[:2000], X_valid=X[2000:], y_valid=y[2000:])


def test_tuning_with_weights_and_offset(offset_data: tuple) -> None:
    X, y, offset = offset_data
    w = np.random.default_rng(3).integers(1, 4, size=len(y)).astype(float)
    reg = XGBDefaultRegressor(learning_rate="optuna", n_optuna_trials=2)

    reg.fit(X, y, sample_weight=w, base_margin=offset)  # internal holdout
    assert reg.optuna_best_params_ is not None

    reg.fit(
        X[:2000],
        y[:2000],
        sample_weight=w[:2000],
        base_margin=offset[:2000],
        X_valid=X[2000:],
        y_valid=y[2000:],
        sample_weight_valid=w[2000:],
        base_margin_valid=offset[2000:],
    )
    assert reg.optuna_best_params_ is not None
