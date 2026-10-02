"""XGBoost classifier/regressor with my default settings.

Defaults (each one can be overridden by passing your own value):
    base_score   = mean(y_train)
    n_estimators = 2000, stopped early once a tree has no split
    gamma        = information-criterion penalty * phi   (AIC, HQ or BIC; default HQ)
    phi          = 1 for binary:logistic, residual variance from an under-fitted model otherwise
    eta          = 0.1, or "optuna" to tune it in [0.05, 0.3]
    max_depth    = 6, or "optuna" to tune it in {3, 4, 5, 6}
    tree_method  = "hist", max_bin = max(256, (2n)^(1/3)), enable_categorical = True
Any other XGBoost parameter goes in `xgb_params`.
"""

from typing import Any, Literal

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

    def _default_phi(self, X: ArrayLike, y: np.ndarray) -> float:
        raise NotImplementedError

    def _new_model(self, params: dict[str, Any]) -> xgb.XGBModel:
        raise NotImplementedError

    def _valid_loss(self, model: xgb.XGBModel, X: ArrayLike, y: np.ndarray) -> float:
        raise NotImplementedError

    # --- fitting -------------------------------------------------------------

    def fit(self, X: ArrayLike, y: Any) -> "_XGBDefaultBase":
        y = self._prepare_y(y)
        n = len(y)

        extra = dict(self.xgb_params or {})
        repeated = _MANAGED_PARAMS & extra.keys()
        if repeated:
            raise ValueError(f"Set {sorted(repeated)} with the named arguments, not xgb_params")

        self.phi_ = self.phi if self.phi is not None else self._default_phi(X, y)
        self.gamma_ = (
            self.gamma if self.gamma is not None else ic_gamma(self.criterion, n, self.phi_)
        )
        params: dict[str, Any] = {
            "objective": self._objective,
            "base_score": self.base_score if self.base_score is not None else float(np.mean(y)),
            "n_estimators": self.n_estimators,
            "gamma": self.gamma_,
            "learning_rate": self.learning_rate,
            "max_depth": self.max_depth,
            "max_bin": self.max_bin if self.max_bin is not None else default_max_bin(n),
            "tree_method": self.tree_method,
            "enable_categorical": self.enable_categorical,
            "random_state": self.random_state,
            "callbacks": [StopOnEmptyTree()] if self.stop_on_empty_tree else None,
            **extra,
        }

        self.study_: optuna.Study | None = None
        if "optuna" in (self.learning_rate, self.max_depth):
            params.update(self._tune(X, y, params))

        self.params_ = params
        self.model_ = self._new_model(params).fit(X, y)
        self.n_trees_ = self.model_.get_booster().num_boosted_rounds()
        return self

    def _tune(self, X: ArrayLike, y: np.ndarray, params: dict[str, Any]) -> dict[str, Any]:
        """Tune learning_rate and/or max_depth with Optuna TPE on one holdout split."""
        X_tr, X_va, y_tr, y_va = train_test_split(
            X,
            y,
            test_size=self.valid_size,
            random_state=self.random_state,
            stratify=y if self._stratify else None,
        )

        def objective(trial: optuna.Trial) -> float:
            trial_params = dict(params)
            if self.learning_rate == "optuna":
                trial_params["learning_rate"] = trial.suggest_float("learning_rate", 0.05, 0.3)
            if self.max_depth == "optuna":
                trial_params["max_depth"] = trial.suggest_int("max_depth", 3, 6)
            model = self._new_model(trial_params).fit(X_tr, y_tr)
            return self._valid_loss(model, X_va, y_va)

        optuna.logging.set_verbosity(optuna.logging.WARNING)
        sampler = optuna.samplers.TPESampler(seed=self.random_state)
        self.study_ = optuna.create_study(direction="minimize", sampler=sampler)
        self.study_.optimize(objective, n_trials=self.n_optuna_trials)
        return self.study_.best_params

    def predict(self, X: ArrayLike) -> np.ndarray:
        return self.model_.predict(X)


class XGBDefaultClassifier(ClassifierMixin, _XGBDefaultBase):
    """Binary classifier (binary:logistic) with my default settings; phi = 1."""

    _objective = "binary:logistic"
    _stratify = True

    def _prepare_y(self, y: Any) -> np.ndarray:
        self.classes_, y_encoded = np.unique(np.asarray(y), return_inverse=True)
        if len(self.classes_) != 2:
            raise ValueError(f"Only binary targets are supported, got {len(self.classes_)} classes")
        return y_encoded

    def _default_phi(self, X: ArrayLike, y: np.ndarray) -> float:
        return 1.0

    def _new_model(self, params: dict[str, Any]) -> xgb.XGBModel:
        return xgb.XGBClassifier(**params)

    def _valid_loss(self, model: xgb.XGBModel, X: ArrayLike, y: np.ndarray) -> float:
        return float(log_loss(y, model.predict_proba(X)[:, 1]))

    def predict(self, X: ArrayLike) -> np.ndarray:
        return self.classes_[self.model_.predict(X)]

    def predict_proba(self, X: ArrayLike) -> np.ndarray:
        return self.model_.predict_proba(X)


class XGBDefaultRegressor(RegressorMixin, _XGBDefaultBase):
    """Regressor (reg:squarederror) with my default settings; phi from an under-fitted model."""

    _objective = "reg:squarederror"
    _stratify = False

    def _prepare_y(self, y: Any) -> np.ndarray:
        return np.asarray(y, dtype=float)

    def _default_phi(self, X: ArrayLike, y: np.ndarray) -> float:
        return estimate_phi(X, y, random_state=self.random_state)

    def _new_model(self, params: dict[str, Any]) -> xgb.XGBModel:
        return xgb.XGBRegressor(**params)

    def _valid_loss(self, model: xgb.XGBModel, X: ArrayLike, y: np.ndarray) -> float:
        return float(root_mean_squared_error(y, model.predict(X)))
