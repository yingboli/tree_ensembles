"""Shared base for estimators that pass extra keyword arguments to an underlying library."""

from typing import Any

from sklearn.base import BaseEstimator


class KwargsEstimator(BaseEstimator):
    """A scikit-learn estimator whose __init__ also takes **kwargs for another library.

    scikit-learn only sees the named arguments of __init__, so on its own a **kwargs
    argument would be dropped by get_params, clone, set_params and GridSearchCV. This base
    class adds the extra arguments to get_params and accepts them in set_params, like
    XGBoost's own scikit-learn wrapper. Subclasses store them in a dict attribute whose
    name is given by `_kwargs_attr`.
    """

    _kwargs_attr: str  # name of the dict attribute holding the extra keyword arguments

    def _extra_kwargs(self) -> dict[str, Any]:
        """The extra keyword arguments as a dict.

        Models saved with version 0.3.0 or earlier stored None when there were none
        (they took a dict argument then), so None is read as {}.
        """
        return getattr(self, self._kwargs_attr, None) or {}

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        """Named parameters plus the extra keyword arguments."""
        params = super().get_params(deep=deep)
        params.update(self._extra_kwargs())
        return params

    def set_params(self, **params: Any) -> "KwargsEstimator":
        """Set named parameters as usual; any other name becomes an extra keyword argument."""
        named = set(self._get_param_names())
        extra = {k: v for k, v in params.items() if k not in named}
        setattr(self, self._kwargs_attr, {**self._extra_kwargs(), **extra})
        super().set_params(**{k: v for k, v in params.items() if k in named})
        return self
