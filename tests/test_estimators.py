import math
from typing import Any

import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import r2_score, roc_auc_score
from sklearn.model_selection import cross_val_score, train_test_split

from xgb_trees import XGBDefaultClassifier, XGBDefaultRegressor


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

    def spy_tune(self: Any, X_tr: Any, y_tr: Any, X_va: Any, y_va: Any, params: Any) -> Any:
        seen["n_train"], seen["n_valid"] = len(y_tr), len(y_va)
        return original_tune(self, X_tr, y_tr, X_va, y_va, params)

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
        XGBDefaultClassifier(n_estimators=5).fit(X_train, y_train, X_test, y_test)


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
