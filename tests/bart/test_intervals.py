import numpy as np
import pytest

from tree_ensembles.bart.intervals import hpdi, quantile_interval


def test_hpdi_normal() -> None:
    draws = np.random.default_rng(0).normal(size=200_000)
    lower, upper = hpdi(draws, prob=0.95)
    assert lower == pytest.approx(-1.96, abs=0.02)
    assert upper == pytest.approx(1.96, abs=0.02)


def test_hpdi_shorter_than_quantile_interval_for_skewed() -> None:
    draws = np.random.default_rng(0).gamma(shape=1.5, size=100_000)
    h_low, h_high = hpdi(draws, prob=0.95)
    q_low, q_high = quantile_interval(draws, prob=0.95)
    assert h_high - h_low < q_high - q_low
    assert np.mean((draws >= h_low) & (draws <= h_high)) == pytest.approx(0.95, abs=1e-3)


def test_hpdi_per_column() -> None:
    rng = np.random.default_rng(0)
    draws = np.column_stack([rng.normal(0, 1, 10_000), rng.normal(5, 2, 10_000)])
    lower, upper = hpdi(draws, prob=0.95, axis=0)
    assert lower.shape == (2,)
    np.testing.assert_allclose(upper - lower, [2 * 1.96, 4 * 1.96], rtol=0.05)


def test_prob_out_of_range() -> None:
    with pytest.raises(ValueError):
        hpdi(np.arange(10.0), prob=1.0)
