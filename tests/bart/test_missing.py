import numpy as np
import pytest

from tree_ensembles.bart import MissingValueImputer

NAN = np.nan


def test_median_fill_and_indicators() -> None:
    X = np.array(
        [
            [1.0, NAN, NAN],
            [2.0, 10.0, NAN],
            [3.0, 20.0, NAN],
            [4.0, 30.0, 5.0],
        ]
    )  # missing rates 0, 0.25, 0.75
    imputer = MissingValueImputer().fit(X)
    np.testing.assert_allclose(imputer.missing_rate_, [0, 0.25, 0.75])
    np.testing.assert_allclose(imputer.fill_values_, [2.5, 20.0, 5.0])
    assert imputer.indicator_features_.tolist() == [2]  # only the 75% one is above 0.5

    out = imputer.transform(X)
    np.testing.assert_allclose(out[:, :3], [[1, 20, 5], [2, 10, 5], [3, 20, 5], [4, 30, 5]])
    np.testing.assert_array_equal(out[:, 3], [1, 1, 1, 0])
    assert imputer.get_feature_names_out(["a", "b", "c"]) == ["a", "b", "c", "c_missing"]


def test_mean_and_threshold() -> None:
    X = np.array([[1.0, NAN], [2.0, 4.0], [6.0, 8.0]])
    imputer = MissingValueImputer(strategy="mean", indicator_threshold=0.0).fit(X)
    np.testing.assert_allclose(imputer.fill_values_, [3.0, 6.0])
    assert imputer.indicator_features_.tolist() == [1]  # any NaN counts with threshold 0
    assert MissingValueImputer(indicator_threshold=1.0).fit(X).indicator_features_.size == 0


def test_new_rows_use_training_values() -> None:
    imputer = MissingValueImputer().fit(np.array([[1.0, 5.0], [3.0, 7.0]]))  # no NaN at all
    out = imputer.transform(np.array([[NAN, NAN]]))
    np.testing.assert_allclose(out, [[2.0, 6.0]])  # filled with training medians, no indicators


def test_all_missing_feature_is_filled_with_zero() -> None:
    imputer = MissingValueImputer().fit(np.array([[1.0, NAN], [2.0, NAN]]))
    np.testing.assert_allclose(imputer.transform(np.array([[NAN, NAN]])), [[1.5, 0.0, 1.0]])


@pytest.mark.parametrize(
    "kwargs", [{"strategy": "mode"}, {"indicator_threshold": -0.1}, {"indicator_threshold": 2}]
)
def test_invalid_settings(kwargs: dict) -> None:
    with pytest.raises(ValueError):
        MissingValueImputer(**kwargs).fit(np.zeros((3, 2)))
