import math

import pytest

from xgb_trees.defaults import default_max_bin, ic_gamma


def test_ic_gamma() -> None:
    n, phi = 1000, 2.5
    assert ic_gamma("aic", n, phi) == pytest.approx(2 * phi)
    assert ic_gamma("hq", n, phi) == pytest.approx(3 * math.log(math.log(n)) * phi)
    assert ic_gamma("bic", n, phi) == pytest.approx(math.log(n) * phi)


def test_ic_gamma_unknown_criterion() -> None:
    with pytest.raises(ValueError):
        ic_gamma("cv", 1000, 1.0)


def test_default_max_bin() -> None:
    assert default_max_bin(1000) == 256
    assert default_max_bin(10_000_000) > 256
