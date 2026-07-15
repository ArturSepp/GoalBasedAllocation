"""Tests for the Laplace-transform vanilla option pricer (core package)."""
import numpy as np
import pytest
from scipy.stats import norm

from goal_based_allocation import (
    price_vanilla, implied_vol, RiskNeutralParams, OptionType, Regime,
)

SPOT, TTM = 100.0, 10.0
STRIKES = np.array([80.0, 100.0, 120.0, 150.0, 200.0])


@pytest.mark.parametrize('regime', [Regime.GROWTH, Regime.STRESS])
def test_put_call_parity(rn_params, regime):
    call = price_vanilla(rn_params, SPOT, STRIKES, TTM, regime, OptionType.CALL)
    put = price_vanilla(rn_params, SPOT, STRIKES, TTM, regime, OptionType.PUT)
    target = SPOT - STRIKES * np.exp(-rn_params.rate * TTM)
    assert np.max(np.abs(call - put - target)) < 1e-5


def test_black_scholes_limit():
    """Vanishing jumps with equal regime vols collapse to Black-Scholes."""
    flat = RiskNeutralParams(sigma_0=0.20, sigma_1=0.20,
                             lambda_01=0.01, lambda_10=0.01,
                             eta_0=1e-3, eta_1=1e-3, rate=0.03)
    t = 1.0
    px = price_vanilla(flat, SPOT, STRIKES, t, Regime.GROWTH, OptionType.CALL)
    sd = 0.20 * np.sqrt(t)
    fwd = SPOT * np.exp(0.03 * t)
    d1 = (np.log(fwd / STRIKES) + 0.5 * sd ** 2) / sd
    bs = np.exp(-0.03 * t) * (fwd * norm.cdf(d1) - STRIKES * norm.cdf(d1 - sd))
    assert np.max(np.abs(px - bs)) < 1e-3


@pytest.mark.parametrize('regime', [Regime.GROWTH, Regime.STRESS])
def test_calls_decrease_in_strike(rn_params, regime):
    call = price_vanilla(rn_params, SPOT, STRIKES, TTM, regime, OptionType.CALL)
    assert np.all(np.diff(call) < 0.0)


@pytest.mark.parametrize('opt', [OptionType.CALL, OptionType.PUT])
def test_prices_nonnegative(rn_params, opt):
    px = price_vanilla(rn_params, SPOT, STRIKES, TTM, Regime.GROWTH, opt)
    assert np.all(px >= 0.0)


def test_scalar_input_returns_float_matching_array(rn_params):
    arr = price_vanilla(rn_params, SPOT, STRIKES, TTM, Regime.GROWTH, OptionType.CALL)
    scalar = price_vanilla(rn_params, SPOT, float(STRIKES[1]), TTM,
                           Regime.GROWTH, OptionType.CALL)
    assert isinstance(scalar, float)
    assert abs(scalar - arr[1]) < 1e-10


def test_implied_vol_is_finite_and_reasonable(rn_params):
    iv = implied_vol(rn_params, SPOT, STRIKES, 5.0, Regime.GROWTH)
    assert np.all(np.isfinite(iv))
    assert np.all((iv > 0.0) & (iv < 1.0))


def test_from_rates_roundtrip():
    p = RiskNeutralParams.from_rates(sigma_0=0.18, sigma_1=0.28,
                                     lambda_01=0.1, lambda_10=1.0,
                                     eta_01=3.0, eta_10=8.0, rate=0.03)
    assert abs(p.eta_0 - 1.0 / 3.0) < 1e-12
    assert abs(p.eta_1 - 1.0 / 8.0) < 1e-12


def test_martingale_condition_enforced():
    with pytest.raises(ValueError):
        RiskNeutralParams(sigma_0=0.2, sigma_1=0.2, lambda_01=0.1, lambda_10=1.0,
                          eta_0=0.3, eta_1=1.0, rate=0.02)   # eta_1 >= 1


def test_positive_params_enforced():
    with pytest.raises(ValueError):
        RiskNeutralParams(sigma_0=-0.2, sigma_1=0.2, lambda_01=0.1, lambda_10=1.0,
                          eta_0=0.3, eta_1=0.1, rate=0.02)


def test_invalid_pricing_inputs_raise(rn_params):
    with pytest.raises(ValueError):
        price_vanilla(rn_params, SPOT, -100.0, TTM)      # negative strike
    with pytest.raises(ValueError):
        price_vanilla(rn_params, -1.0, STRIKES, TTM)     # negative spot
    with pytest.raises(ValueError):
        price_vanilla(rn_params, SPOT, STRIKES, 0.0)     # non-positive ttm
