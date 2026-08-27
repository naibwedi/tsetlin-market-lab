import numpy as np

from src.common.odds import implied_prob, overround, remove_vig_proportional


def test_implied_prob_roundtrip():
    assert np.isclose(implied_prob(2.0), 0.5)
    assert np.isnan(implied_prob(0.9))


def test_remove_vig_sums_to_one():
    ip = np.array([[0.5, 0.30, 0.28], [0.45, 0.30, 0.40]])  # overround > 0
    fp = remove_vig_proportional(ip)
    assert np.allclose(fp.sum(axis=-1), 1.0)
    assert (overround(ip) > 0).all()
