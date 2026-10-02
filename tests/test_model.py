from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

from xgb_trees import evaluate, train_classifier


def test_train_and_evaluate() -> None:
    X, y = load_breast_cancer(return_X_y=True, as_frame=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

    model = train_classifier(X_train, y_train, n_estimators=50)
    metrics = evaluate(model, X_test, y_test)

    assert metrics["roc_auc"] > 0.9
