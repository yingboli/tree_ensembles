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

from tree_ensembles.xgb.callbacks import StopOnEmptyTree
from tree_ensembles.xgb.defaults import default_max_bin, estimate_phi, ic_gamma

ArrayLike = pd.DataFrame | np.ndarray


class _Data(NamedTuple):
    """The rows for one fit or one validation score, kept together so they are split alike.

    sample_weight and base_margin are None when not used.
    """

    X: ArrayLike
    y: np.ndarray
    sample_weight: np.ndarray | None = None
    base_margin: np.ndarray | None = None

    def take(self, idx: np.ndarray) -> "_Data":
        """Return the rows at positions `idx` (of X, y, weights and offset together)."""
        X = self.X.iloc[idx] if isinstance(self.X, pd.DataFrame) else self.X[idx]
        w, m = self.sample_weight, self.base_margin
        return _Data(X, self.y[idx], None if w is None else w[idx], None if m is None else m[idx])


def _as_float_array(a: Any) -> np.ndarray | None:
    """`a` as a float numpy array, or None if `a` is None."""
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
    """Shared fitting logic of XGBDefaultClassifier and XGBDefaultRegressor (use those classes).

    Parameters
    ----------
    criterion : {"aic", "hq", "bic"}, default "hq"
        Information criterion that sets the split penalty: gamma = penalty * phi, with
        penalty 2 (AIC), 3 * ln(ln n) (Hannan-Quinn) or ln n (BIC). A split must reduce
        the deviance by more than this. With sample weights, n is the sum of the weights.
    phi : float, optional
        Dispersion used in the penalty. Default: 1 for the classifier; for the regressor,
        the residual variance of a deliberately under-fitted XGBoost model
        (see defaults.estimate_phi).
    gamma : float, optional
        Set XGBoost's split penalty directly instead of deriving it from `criterion`.
    base_score : float, optional
        Starting prediction (a probability for the classifier). Default: the (weighted)
        mean of y. Ignored by XGBoost when an offset (base_margin) is passed to fit.
    n_estimators : int, default 2000
        Maximum number of boosting rounds; usually far fewer are trained because of
        `stop_on_empty_tree`.
    learning_rate : float or "optuna", default 0.1
        Shrinkage (eta). "optuna" tunes it with Optuna TPE in [0.05, 0.3].
    max_depth : int or "optuna", default 6
        Maximum tree depth. "optuna" tunes it in {3, 4, 5, 6}.
    max_bin : int, optional
        Histogram bins per feature. Default: max(256, ceil((2 n) ** (1/3))), n = rows.
    tree_method : str, default "hist"
        XGBoost tree construction method.
    enable_categorical : bool, default True
        Use pandas "category" columns natively.
    stop_on_empty_tree : bool, default True
        Stop boosting at the first round whose trees have no split and a leaf value of at
        most 1e-3 * sqrt(phi), i.e. when no split is worth its penalty any more
        (see callbacks.StopOnEmptyTree).
    n_optuna_trials : int, default 30
        Number of Optuna trials when `learning_rate` or `max_depth` is "optuna".
    valid_size : float, default 0.2
        Share of rows held out to score Optuna trials when fit gets no X_valid. Only used
        for tuning: the final model is always fit on all of (X, y).
    random_state : int, default 0
        Seed for XGBoost, the tuning holdout split and the Optuna sampler.
    xgb_params : dict, optional
        Any other XGBoost parameter, e.g. {"subsample": 0.8, "n_jobs": 4}. Must not repeat
        one of the named parameters above (that raises a ValueError in fit).

    Attributes
    ----------
    model_ : xgboost.XGBClassifier or xgboost.XGBRegressor
        The fitted XGBoost model.
    params_ : dict
        Every parameter passed to XGBoost for the final fit (after tuning).
    phi_ : float
        The dispersion used in the penalty.
    gamma_ : float
        The split penalty used.
    n_trees_ : int
        Number of boosting rounds actually trained.
    optuna_best_params_ : dict or None
        Best tuned values as plain numbers (not the Optuna Study, so a saved model does
        not depend on the Optuna version), or None when nothing was tuned.
    classes_ : ndarray of shape (2,)
        Classifier only: the two class labels; predict_proba columns follow this order.
    """

    _objective: str  # XGBoost objective, set by each subclass
    _stratify: bool  # stratify the tuning holdout split by y (classifier only)

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
        """Check and encode the training target (e.g. labels to 0/1)."""
        raise NotImplementedError

    def _prepare_y_valid(self, y: Any) -> np.ndarray:
        """Encode a validation target the same way as the training target."""
        raise NotImplementedError

    def _default_phi(self, data: _Data) -> float:
        """Dispersion phi when the user does not give one."""
        raise NotImplementedError

    def _new_model(self, params: dict[str, Any]) -> xgb.XGBModel:
        """An unfitted XGBoost model of the right kind."""
        raise NotImplementedError

    def _valid_loss(self, model: xgb.XGBModel, data: _Data) -> float:
        """Loss of a fitted model on validation data (lower is better), for tuning."""
        raise NotImplementedError

    def _fit_model(self, params: dict[str, Any], data: _Data) -> xgb.XGBModel:
        """Fit a new XGBoost model with `params` on `data` (weights and offset included)."""
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
        """Fit the model on all of (X, y), tuning first if anything is set to "optuna".

        Parameters
        ----------
        X : DataFrame or array of shape (n, p)
            Training features. pandas "category" columns are used natively.
        y : array-like of shape (n,)
            Training target; two classes for the classifier.
        sample_weight : array-like of shape (n,), optional
            Frequency weights: weight 2 counts a row twice, and the information criterion
            uses n = sum of the weights.
        base_margin : array-like of shape (n,), optional
            Offset on the margin scale (log-odds for the classifier), passed to XGBoost as
            given; there is no separate intercept. Pass it again to predict().
        X_valid, y_valid : optional
            Validation data used only to score Optuna trials (instead of a random holdout
            of `valid_size`). The final model is still fit on (X, y) only. Ignored, with a
            warning, when nothing is tuned.
        sample_weight_valid, base_margin_valid : array-like, optional
            Weights and offset of the validation rows. base_margin_valid is required
            exactly when base_margin is given.

        Returns
        -------
        self
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
        """Tune learning_rate and/or max_depth with Optuna TPE.

        Every trial fits on `fit_data` with the other parameters fixed (`params`) and is
        scored by `_valid_loss` on `valid`. Returns the best values as a plain dict.
        """

        def objective(trial: optuna.Trial) -> float:
            """Validation loss of one Optuna trial (lower is better)."""
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
        """Predicted values for X.

        Parameters
        ----------
        X : DataFrame or array of shape (m, p)
            Same columns as in fit.
        base_margin : array-like of shape (m,), optional
            Offset for these rows, if the model was fit with one; None means offset 0.

        Returns
        -------
        ndarray of shape (m,)
        """
        return self.model_.predict(X, base_margin=_as_float_array(base_margin))


def _shared_doc(cls: type) -> str:
    """The class docstring without its first (summary) line, to reuse in subclasses."""
    return (cls.__doc__ or "").split("\n", 1)[1]


class XGBDefaultClassifier(ClassifierMixin, _XGBDefaultBase):
    __doc__ = """Binary XGBoost classifier (binary:logistic) with my default settings.

    The split penalty uses phi = 1 (deviance scale of the logistic loss). Labels can be any
    two values; they are encoded to 0/1 and decoded back in predict.
    """ + _shared_doc(_XGBDefaultBase)  # its Parameters and Attributes

    _objective = "binary:logistic"
    _stratify = True

    def _prepare_y(self, y: Any) -> np.ndarray:
        """Encode the two labels to 0/1 (in sorted order) and remember them in classes_."""
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
        """phi = 1 for the logistic loss."""
        return 1.0

    def _new_model(self, params: dict[str, Any]) -> xgb.XGBModel:
        """An unfitted xgboost.XGBClassifier."""
        return xgb.XGBClassifier(**params)

    def _valid_loss(self, model: xgb.XGBModel, data: _Data) -> float:
        """Weighted log loss on the validation rows."""
        proba = model.predict_proba(data.X, base_margin=data.base_margin)[:, 1]
        return float(log_loss(data.y, proba, sample_weight=data.sample_weight, labels=[0, 1]))

    def predict(self, X: ArrayLike, base_margin: Any = None) -> np.ndarray:
        """Predicted class labels (probability threshold 0.5).

        Parameters
        ----------
        X : DataFrame or array of shape (m, p)
            Same columns as in fit.
        base_margin : array-like of shape (m,), optional
            Log-odds offset for these rows, if the model was fit with one; None means 0.

        Returns
        -------
        ndarray of shape (m,) with values from `classes_`.
        """
        return self.classes_[self.model_.predict(X, base_margin=_as_float_array(base_margin))]

    def predict_proba(self, X: ArrayLike, base_margin: Any = None) -> np.ndarray:
        """Predicted class probabilities.

        Parameters
        ----------
        X : DataFrame or array of shape (m, p)
            Same columns as in fit.
        base_margin : array-like of shape (m,), optional
            Log-odds offset for these rows, if the model was fit with one; None means 0.

        Returns
        -------
        ndarray of shape (m, 2): columns P(classes_[0]) and P(classes_[1]).
        """
        return self.model_.predict_proba(X, base_margin=_as_float_array(base_margin))


class XGBDefaultRegressor(RegressorMixin, _XGBDefaultBase):
    __doc__ = """XGBoost regressor (reg:squarederror) with my default settings.

    By default phi, the noise variance in the split penalty, is estimated as the residual
    variance of a deliberately under-fitted XGBoost model.
    """ + _shared_doc(_XGBDefaultBase)  # its Parameters and Attributes

    _objective = "reg:squarederror"
    _stratify = False

    def _prepare_y(self, y: Any) -> np.ndarray:
        """y as float."""
        return np.asarray(y, dtype=float)

    def _prepare_y_valid(self, y: Any) -> np.ndarray:
        """y_valid as float."""
        return np.asarray(y, dtype=float)

    def _default_phi(self, data: _Data) -> float:
        """Noise variance from an under-fitted XGBoost model (defaults.estimate_phi)."""
        return estimate_phi(*data, random_state=self.random_state)

    def _new_model(self, params: dict[str, Any]) -> xgb.XGBModel:
        """An unfitted xgboost.XGBRegressor."""
        return xgb.XGBRegressor(**params)

    def _valid_loss(self, model: xgb.XGBModel, data: _Data) -> float:
        """Weighted RMSE on the validation rows."""
        pred = model.predict(data.X, base_margin=data.base_margin)
        return float(root_mean_squared_error(data.y, pred, sample_weight=data.sample_weight))
