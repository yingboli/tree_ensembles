"""XGBoost classifier/regressor with my default settings.

Defaults (each one can be overridden by passing your own value):
    base_score   = (weighted) mean(y_train); not used with an offset (base_margin)
    n_estimators = 2000, stopped once a tree has no split and a tiny leaf
    gamma        = information-criterion penalty * phi   (AIC, HQ or BIC; default HQ)
    phi          = 1 for binary:logistic, residual variance from an under-fitted model otherwise
    eta          = 0.1, or "optuna" to tune it in [0.05, 0.3]
    max_depth    = 6, or "optuna" to tune it in {3, 4, 5, 6}
                   (tuning scores on fit(..., X_valid=, y_valid=) if given,
                   otherwise on a random holdout of size valid_size)
    tree_method  = "hist", max_bin = max(256, (2n)^(1/3)), enable_categorical = True
Any other XGBoost parameter goes in `xgb_params`.

fit() also takes sample_weight (frequency weights: n = sum of weights) and
base_margin (an offset on the margin scale, passed to XGBoost as given).
"""

import warnings
from typing import Any, Literal, NamedTuple

import numpy as np
import optuna
import pandas as pd
import xgboost as xgb
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.metrics import log_loss, root_mean_squared_error
from sklearn.model_selection import train_test_split

from xgb_trees.callbacks import StopOnEmptyTree
from xgb_trees.defaults import default_max_bin, estimate_phi, ic_gamma

ArrayLike = pd.DataFrame | np.ndarray


class _Data(NamedTuple):
    """The rows for one fit or one validation score."""

    X: ArrayLike
    y: np.ndarray
    sample_weight: np.ndarray | None = None
    base_margin: np.ndarray | None = None

    def take(self, idx: np.ndarray) -> "_Data":
        """Select rows by position."""
        X = self.X.iloc[idx] if isinstance(self.X, pd.DataFrame) else self.X[idx]
        w, m = self.sample_weight, self.base_margin
        return _Data(X, self.y[idx], None if w is None else w[idx], None if m is None else m[idx])


def _as_float_array(a: Any) -> np.ndarray | None:
    return None if a is None else np.asarray(a, dtype=float)


# Parameters this class sets itself; they must not also be passed in `xgb_params`.
_MANAGED_PARAMS = {
    "objective",
    "base_score",
    "n_estimators",
    "gamma",
    "learning_rate",
    "max_depth",
    "max_bin",
    "tree_method",
    "enable_categorical",
    "random_state",
    "callbacks",
}


class _XGBDefaultBase(BaseEstimator):
    _objective: str
    _stratify: bool

    def __init__(
        self,
        criterion: Literal["aic", "hq", "bic"] = "hq",
        phi: float | None = None,
        gamma: float | None = None,
        base_score: float | None = None,
        n_estimators: int = 2000,
        learning_rate: float | Literal["optuna"] = 0.1,
        max_depth: int | Literal["optuna"] = 6,
        max_bin: int | None = None,
        tree_method: str = "hist",
        enable_categorical: bool = True,
        stop_on_empty_tree: bool = True,
        n_optuna_trials: int = 30,
        valid_size: float = 0.2,
        random_state: int = 0,
        xgb_params: dict[str, Any] | None = None,
    ) -> None:
        self.criterion = criterion
        self.phi = phi
        self.gamma = gamma
        self.base_score = base_score
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.max_bin = max_bin
        self.tree_method = tree_method
        self.enable_categorical = enable_categorical
        self.stop_on_empty_tree = stop_on_empty_tree
        self.n_optuna_trials = n_optuna_trials
        self.valid_size = valid_size
        self.random_state = random_state
        self.xgb_params = xgb_params

    # --- hooks implemented by the subclasses ---------------------------------

    def _prepare_y(self, y: Any) -> np.ndarray:
        raise NotImplementedError

    def _prepare_y_valid(self, y: Any) -> np.ndarray:
        raise NotImplementedError

    def _default_phi(self, data: _Data) -> float:
        raise NotImplementedError

    def _new_model(self, params: dict[str, Any]) -> xgb.XGBModel:
        raise NotImplementedError

    def _valid_loss(self, model: xgb.XGBModel, data: _Data) -> float:
        raise NotImplementedError

    def _fit_model(self, params: dict[str, Any], data: _Data) -> xgb.XGBModel:
        model = self._new_model(params)
        return model.fit(
            data.X, data.y, sample_weight=data.sample_weight, base_margin=data.base_margin
        )

    # --- fitting -------------------------------------------------------------

    def fit(
        self,
        X: ArrayLike,
        y: Any,
        sample_weight: Any = None,
        base_margin: Any = None,
        *,
        X_valid: ArrayLike | None = None,
        y_valid: Any = None,
        sample_weight_valid: Any = None,
        base_margin_valid: Any = None,
    ) -> "_XGBDefaultBase":
        """Fit on (X, y).

        sample_weight: frequency weights (n in the information criterion = sum of weights).
        base_margin: offset on the margin scale; pass it again to predict().
        The *_valid arguments are used only to score Optuna trials.
        """
        train = _Data(
            X, self._prepare_y(y), _as_float_array(sample_weight), _as_float_array(base_margin)
        )
        valid = None
        if X_valid is not None or y_valid is not None:
            if X_valid is None or y_valid is None:
                raise ValueError("Pass both X_valid and y_valid, or neither")
            if (base_margin is None) != (base_margin_valid is None):
                raise ValueError(
                    "With X_valid, pass base_margin_valid exactly when base_margin is given"
                )
            valid = _Data(
                X_valid,
                self._prepare_y_valid(y_valid),
                _as_float_array(sample_weight_valid),
                _as_float_array(base_margin_valid),
            )

        extra = dict(self.xgb_params or {})
        repeated = _MANAGED_PARAMS & extra.keys()
        if repeated:
            raise ValueError(f"Set {sorted(repeated)} with the named arguments, not xgb_params")

        w = train.sample_weight
        n_info = float(np.sum(w)) if w is not None else len(train.y)
        self.phi_ = self.phi if self.phi is not None else self._default_phi(train)
        self.gamma_ = (
            self.gamma if self.gamma is not None else ic_gamma(self.criterion, n_info, self.phi_)
        )
        # The empty-tree stop waits until an overall shift has been learned (see callbacks.py).
        stop = StopOnEmptyTree(leaf_tol=1e-3 * np.sqrt(self.phi_))
        params: dict[str, Any] = {
            "objective": self._objective,
            "n_estimators": self.n_estimators,
            "gamma": self.gamma_,
            "learning_rate": self.learning_rate,
            "max_depth": self.max_depth,
            "max_bin": self.max_bin if self.max_bin is not None else default_max_bin(len(train.y)),
            "tree_method": self.tree_method,
            "enable_categorical": self.enable_categorical,
            "random_state": self.random_state,
            "callbacks": [stop] if self.stop_on_empty_tree else None,
            **extra,
        }
        if train.base_margin is None:
            params["base_score"] = (
                self.base_score
                if self.base_score is not None
                else float(np.average(train.y, weights=w))
            )
        else:
            # With an offset, XGBoost ignores base_score during training; set it to 0 so
            # that predicting without an offset means offset = 0.
            params["base_score"] = 0.0
            if self.base_score is not None:
                warnings.warn(
                    "base_score is ignored by XGBoost when base_margin is given", stacklevel=2
                )

        # Keep only plain best values, not the Study object, so a saved model
        # does not depend on the installed Optuna version.
        self.optuna_best_params_: dict[str, Any] | None = None
        if "optuna" in (self.learning_rate, self.max_depth):
            if valid is None:
                idx_tr, idx_va = train_test_split(
                    np.arange(len(train.y)),
                    test_size=self.valid_size,
                    random_state=self.random_state,
                    stratify=train.y if self._stratify else None,
                )
                fit_data, valid = train.take(idx_tr), train.take(idx_va)
            else:
                fit_data = train
            self.optuna_best_params_ = self._tune(fit_data, valid, params)
            params.update(self.optuna_best_params_)
        elif valid is not None:
            warnings.warn(
                "X_valid/y_valid are only used for Optuna tuning and nothing is set to "
                '"optuna"; they are ignored.',
                stacklevel=2,
            )

        # The final model is always trained on (X, y) only.
        self.params_ = params
        self.model_ = self._fit_model(params, train)
        self.n_trees_ = self.model_.get_booster().num_boosted_rounds()
        return self

    def _tune(self, fit_data: _Data, valid: _Data, params: dict[str, Any]) -> dict[str, Any]:
        """Tune learning_rate and/or max_depth with Optuna TPE: fit on fit_data, score on valid."""

        def objective(trial: optuna.Trial) -> float:
            trial_params = dict(params)
            if self.learning_rate == "optuna":
                trial_params["learning_rate"] = trial.suggest_float("learning_rate", 0.05, 0.3)
            if self.max_depth == "optuna":
                trial_params["max_depth"] = trial.suggest_int("max_depth", 3, 6)
            return self._valid_loss(self._fit_model(trial_params, fit_data), valid)

        optuna.logging.set_verbosity(optuna.logging.WARNING)
        sampler = optuna.samplers.TPESampler(seed=self.random_state)
        study = optuna.create_study(direction="minimize", sampler=sampler)
        study.optimize(objective, n_trials=self.n_optuna_trials)
        return dict(study.best_params)

    def predict(self, X: ArrayLike, base_margin: Any = None) -> np.ndarray:
        return self.model_.predict(X, base_margin=_as_float_array(base_margin))


class XGBDefaultClassifier(ClassifierMixin, _XGBDefaultBase):
    """Binary classifier (binary:logistic) with my default settings; phi = 1."""

    _objective = "binary:logistic"
    _stratify = True

    def _prepare_y(self, y: Any) -> np.ndarray:
        self.classes_, y_encoded = np.unique(np.asarray(y), return_inverse=True)
        if len(self.classes_) != 2:
            raise ValueError(f"Only binary targets are supported, got {len(self.classes_)} classes")
        return y_encoded

    def _prepare_y_valid(self, y: Any) -> np.ndarray:
        """Encode validation labels with the classes found in the training labels."""
        y = np.asarray(y)
        unseen = set(np.unique(y)) - set(self.classes_)
        if unseen:
            raise ValueError(f"y_valid has labels not in y: {sorted(unseen)}")
        return np.searchsorted(self.classes_, y)

    def _default_phi(self, data: _Data) -> float:
        return 1.0

    def _new_model(self, params: dict[str, Any]) -> xgb.XGBModel:
        return xgb.XGBClassifier(**params)

    def _valid_loss(self, model: xgb.XGBModel, data: _Data) -> float:
        proba = model.predict_proba(data.X, base_margin=data.base_margin)[:, 1]
        return float(log_loss(data.y, proba, sample_weight=data.sample_weight, labels=[0, 1]))

    def predict(self, X: ArrayLike, base_margin: Any = None) -> np.ndarray:
        return self.classes_[self.model_.predict(X, base_margin=_as_float_array(base_margin))]

    def predict_proba(self, X: ArrayLike, base_margin: Any = None) -> np.ndarray:
        return self.model_.predict_proba(X, base_margin=_as_float_array(base_margin))


class XGBDefaultRegressor(RegressorMixin, _XGBDefaultBase):
    """Regressor (reg:squarederror) with my default settings; phi from an under-fitted model."""

    _objective = "reg:squarederror"
    _stratify = False

    def _prepare_y(self, y: Any) -> np.ndarray:
        return np.asarray(y, dtype=float)

    def _prepare_y_valid(self, y: Any) -> np.ndarray:
        return np.asarray(y, dtype=float)

    def _default_phi(self, data: _Data) -> float:
        return estimate_phi(*data, random_state=self.random_state)

    def _new_model(self, params: dict[str, Any]) -> xgb.XGBModel:
        return xgb.XGBRegressor(**params)

    def _valid_loss(self, model: xgb.XGBModel, data: _Data) -> float:
        pred = model.predict(data.X, base_margin=data.base_margin)
        return float(root_mean_squared_error(data.y, pred, sample_weight=data.sample_weight))
