"""A small, readable wrapper around XGBoost's scikit-learn API."""

from typing import Any

import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score, roc_auc_score

DEFAULT_PARAMS: dict[str, Any] = {
    "n_estimators": 200,
    "max_depth": 4,
    "learning_rate": 0.1,
    "random_state": 0,
}


def train_classifier(X: pd.DataFrame, y: pd.Series, **params: Any) -> xgb.XGBClassifier:
    """Fit a binary XGBoost classifier. Keyword args override DEFAULT_PARAMS."""
    model = xgb.XGBClassifier(**{**DEFAULT_PARAMS, **params})
    model.fit(X, y)
    return model


def evaluate(model: xgb.XGBClassifier, X: pd.DataFrame, y: pd.Series) -> dict[str, float]:
    """Return accuracy and ROC AUC on (X, y)."""
    proba = model.predict_proba(X)[:, 1]
    return {
        "accuracy": float(accuracy_score(y, proba > 0.5)),
        "roc_auc": float(roc_auc_score(y, proba)),
    }
