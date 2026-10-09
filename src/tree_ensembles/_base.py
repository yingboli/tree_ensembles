"""Shared base for the estimators: extra keyword arguments, and dumping to a folder."""

import pickle
from pathlib import Path
from typing import Any

from sklearn.base import BaseEstimator

STATE_FILE = "estimator.pkl"  # the estimator's own state, next to the library's model file


def dump_state(estimator: BaseEstimator, folder: Path, model_attr: str) -> None:
    """Pickle the estimator's class and attributes, except the library model, to a folder.

    The library model (`model_attr`) is written separately in the library's own format.

    Parameters
    ----------
    estimator : BaseEstimator
        A fitted estimator.
    folder : Path
        Folder to write; created if needed.
    model_attr : str
        Name of the attribute holding the library model, left out of the pickle.
    """
    folder.mkdir(parents=True, exist_ok=True)
    state = {k: v for k, v in estimator.__dict__.items() if k != model_attr}
    with (folder / STATE_FILE).open("wb") as f:
        pickle.dump((type(estimator), state), f)


def load_state(estimator: BaseEstimator, folder: Path, model_file: str) -> None:
    """Replace the estimator's attributes by those written with dump_state.

    Parameters
    ----------
    estimator : BaseEstimator
        The estimator to fill; all its own attributes are dropped first.
    folder : Path
        Folder written by dump_state.
    model_file : str
        Name of the library's model file, which must be in the folder too.

    Raises
    ------
    FileNotFoundError
        If the folder has no dump in it.
    TypeError
        If the folder holds another class (e.g. a classifier loaded into a regressor).
    """
    missing = [name for name in (STATE_FILE, model_file) if not (folder / name).exists()]
    if missing:
        raise FileNotFoundError(f"No model dumped in {folder}: {missing} not found")
    with (folder / STATE_FILE).open("rb") as f:
        cls, state = pickle.load(f)
    if cls is not type(estimator):
        raise TypeError(f"{folder} holds a {cls.__name__}; load it with {cls.__name__}().load(...)")
    estimator.__dict__.clear()  # drop anything from an earlier fit
    estimator.__dict__.update(state)


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
