import numpy as np
import xgboost as xgb

from xgb_trees import StopOnEmptyTree


def _fit(gamma: float) -> int:
    rng = np.random.default_rng(0)
    X = rng.normal(size=(500, 3))
    y = X[:, 0] + rng.normal(size=500)
    model = xgb.XGBRegressor(n_estimators=50, gamma=gamma, callbacks=[StopOnEmptyTree()])
    model.fit(X, y)
    return model.get_booster().num_boosted_rounds()


def test_stops_after_first_empty_tree() -> None:
    assert _fit(gamma=1e9) == 1


def test_runs_all_rounds_without_penalty() -> None:
    assert _fit(gamma=0.0) == 50
