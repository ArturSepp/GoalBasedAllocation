"""Cross-check the Laplace pricer against the Fourier and Monte Carlo references.

The reference pricers live in ``examples/regime_switch_smile.py`` (see conftest
for the path setup).
"""
import numpy as np
import pytest

from goal_based_allocation import price_vanilla, Regime

import regime_switch_smile as rss   # from examples/, via conftest path insert

SPOT, TTM = 100.0, 10.0


@pytest.mark.parametrize('tenor', [0.5, 1.0, 5.0, 10.0])
@pytest.mark.parametrize('strike', [80.0, 120.0, 200.0])
def test_laplace_matches_fourier(rn_params, tenor, strike):
    lap = price_vanilla(rn_params, SPOT, strike, tenor, Regime.GROWTH)
    fou = rss.price_vanilla_fourier(rn_params, SPOT, strike, tenor, Regime.GROWTH)
    assert abs(lap - fou) < 1e-4


@pytest.mark.slow
@pytest.mark.parametrize('regime', [Regime.GROWTH, Regime.STRESS])
def test_laplace_matches_monte_carlo(rn_params, regime):
    strikes = np.array([80.0, 100.0, 120.0, 150.0])
    lap = price_vanilla(rn_params, SPOT, strikes, TTM, regime, 'call')
    mc, se = rss.price_vanilla_mc(rn_params, SPOT, strikes, TTM, regime, 'call',
                                  n_paths=100_000, seed=7)
    z = (lap - mc) / se
    assert np.max(np.abs(z)) < 4.0
