import math

import numpy as np
import pytest

from xgb_trees.defaults import default_max_bin, estimate_phi, ic_gamma


def test_ic_gamma() -> None:
    n, phi = 1000, 2.5
    assert ic_gamma("aic", n, phi) == pytest.approx(2 * phi)
    assert ic_gamma("hq", n, phi) == pytest.approx(3 * math.log(math.log(n)) * phi)
    assert ic_gamma("bic", n, phi) == pytest.approx(math.log(n) * phi)


def test_ic_gamma_unknown_criterion() -> None:
    with pytest.raises(ValueError):
        ic_gamma("cv", 1000, 1.0)


def test_ic_gamma_too_small_n() -> None:
    assert ic_gamma("aic", 1.0, 1.0) == 2.0  # AIC does not depend on n
    for criterion in ("hq", "bic"):
        with pytest.raises(ValueError, match="too small"):
            ic_gamma(criterion, 2.0, 1.0)


def test_estimate_phi_weighted() -> None:
    """Noise variance 1 on rows with weight 1, 9 on rows with weight 0 (ignored)."""
    rng = np.random.default_rng(0)
    n = 4000
    X = rng.normal(size=(n, 2))
    w = (np.arange(n) % 2).astype(float)
    y = X[:, 0] + rng.normal(size=n) * np.where(w == 1, 1.0, 3.0)
    assert 0.8 < estimate_phi(X, y, sample_weight=w) < 1.3


def test_default_max_bin() -> None:
    assert default_max_bin(1000) == 256
    assert default_max_bin(10_000_000) > 256
