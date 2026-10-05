import numpy as np
import pytest

from tree_ensembles.bart.diagnostics import ess_bulk, ess_tail, split_rhat, summarize_draws


def _ar1(rho: float, chains: int, n: int, seed: int = 0) -> np.ndarray:
    """AR(1) chains with stationary N(0, 1) marginals."""
    rng = np.random.default_rng(seed)
    x = np.empty((chains, n))
    x[:, 0] = rng.normal(size=chains)
    noise = rng.normal(size=(chains, n)) * np.sqrt(1 - rho**2)
    for t in range(1, n):
        x[:, t] = rho * x[:, t - 1] + noise[:, t]
    return x


def test_iid_chains() -> None:
    x = np.random.default_rng(0).normal(size=(4, 1000))
    assert split_rhat(x) == pytest.approx(1.0, abs=0.01)
    assert ess_bulk(x) == pytest.approx(4000, rel=0.1)
    assert ess_tail(x) == pytest.approx(4000, rel=0.2)


def test_chains_with_different_means_have_large_rhat() -> None:
    x = np.random.default_rng(0).normal(size=(4, 1000)) + np.array([[0], [0], [0], [1.0]])
    assert split_rhat(x) > 1.1


def test_trending_single_chain_detected_by_split() -> None:
    x = np.random.default_rng(0).normal(size=1000) + np.linspace(0, 3, 1000)
    assert split_rhat(x) > 1.1


def test_ar1_ess_matches_theory() -> None:
    rho, chains, n = 0.9, 4, 5000
    expected = chains * n * (1 - rho) / (1 + rho)  # about 1053
    assert ess_bulk(_ar1(rho, chains, n)) == pytest.approx(expected, rel=0.15)


def test_constant_draws_give_nan() -> None:
    assert np.isnan(split_rhat(np.ones((2, 100))))
    assert np.isnan(ess_bulk(np.ones((2, 100))))


def test_summarize_draws() -> None:
    rng = np.random.default_rng(0)
    table = summarize_draws({"good": rng.normal(size=(4, 500)), "stuck": _ar1(0.999, 4, 500)})
    assert list(table.index) == ["good", "stuck"]
    assert table.loc["good", "ok"]
    assert not table.loc["stuck", "ok"]
    assert {"mean", "sd", "hpdi_0.95_low", "hpdi_0.95_high", "rhat", "ess_bulk"} <= set(
        table.columns
    )
