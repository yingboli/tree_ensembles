"""My default XGBoost setup: XGBDefaultClassifier, XGBDefaultRegressor and StopOnEmptyTree."""

from tree_ensembles.xgb.callbacks import StopOnEmptyTree
from tree_ensembles.xgb.estimators import XGBDefaultClassifier, XGBDefaultRegressor

__all__ = ["StopOnEmptyTree", "XGBDefaultClassifier", "XGBDefaultRegressor"]
