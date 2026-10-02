"""XGBoost training callbacks."""

import xgboost as xgb


class StopOnEmptyTree(xgb.callback.TrainingCallback):
    """Stop boosting as soon as a round produces only single-leaf trees (no splits).

    With gamma set by an information criterion, an empty tree means no split is
    worth its penalty any more. The empty tree itself is kept; it only shifts the
    intercept slightly.
    """

    def after_iteration(
        self, model: xgb.Booster, epoch: int, evals_log: xgb.callback.TrainingCallback.EvalsLog
    ) -> bool:
        trees = model[epoch : epoch + 1].get_dump()
        return all(tree.startswith("0:leaf") for tree in trees)
