"""XGBoost training callbacks."""

import xgboost as xgb


class StopOnEmptyTree(xgb.callback.TrainingCallback):
    """Stop boosting once a round produces only single-leaf trees (no splits) with tiny leaves.

    With gamma set by an information criterion, an empty tree means no split is
    worth its penalty any more. But an empty tree with a large leaf value means the
    model is still learning an overall shift (e.g. an offset without an intercept),
    so boosting continues until that leaf is at most `leaf_tol` (margin scale).
    The last empty tree is kept; it only shifts predictions by at most `leaf_tol`.

    Parameters
    ----------
    leaf_tol : float, default 1e-3
        Largest absolute leaf value (margin scale, after the learning rate) for which an
        empty tree stops training. The XGBoost estimators use 1e-3 * sqrt(phi).

    Examples
    --------
    >>> model = xgboost.XGBRegressor(n_estimators=2000, gamma=5.0,
    ...                              callbacks=[StopOnEmptyTree()])
    """

    def __init__(self, leaf_tol: float = 1e-3) -> None:
        super().__init__()
        self.leaf_tol = leaf_tol

    def after_iteration(
        self, model: xgb.Booster, epoch: int, evals_log: xgb.callback.TrainingCallback.EvalsLog
    ) -> bool:
        """Called by XGBoost after each round; returning True stops training."""
        for tree in model[epoch : epoch + 1].get_dump():
            if not tree.startswith("0:leaf="):  # the tree has a split
                return False
            if abs(float(tree.split("=")[1])) > self.leaf_tol:  # still learning a shift
                return False
        return True
