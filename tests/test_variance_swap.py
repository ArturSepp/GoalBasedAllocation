"""Tests for the variance-swap module (closed form vs Monte Carlo, limits, identities).

Written as unittest.TestCase so the file is collected identically by pytest and
by unittest (PyCharm's default runner). No pytest-only features are used.
"""
import unittest
import numpy as np

from goal_based_allocation import (
    RiskNeutralParams, Regime,
    variance_swap_strike, decompose_variance, occupation_times,
    jump_skew_gap, variance_risk_premium, VarianceConvention,
)
from goal_based_allocation.variance_swap import (
    variance_swap_strike_mc,
    implied_crash_size_from_var_swap,
    skew_overidentification_test,
)

REGIMES = (Regime.GROWTH, Regime.STRESS)


def make_params() -> RiskNeutralParams:
    """Base risk-neutral parameters (Sturm's case, rate convention)."""
    return RiskNeutralParams.from_rates(sigma_0=0.18, sigma_1=0.28,
                                        lambda_01=0.10, lambda_10=1.0,
                                        eta_01=3.0, eta_10=8.0, rate=0.03)


def make_params_p() -> RiskNeutralParams:
    """Physical parameters: mean crash size 1/4."""
    return RiskNeutralParams.from_rates(sigma_0=0.18, sigma_1=0.28,
                                        lambda_01=0.10, lambda_10=1.0,
                                        eta_01=4.0, eta_10=8.0, rate=0.03)


class TestVarianceSwap(unittest.TestCase):
    """Runs under both pytest and unittest/PyCharm."""

    # ==============================================================================
    # occupation times
    # ==============================================================================

    def test_occupation_times_sum_to_ttm(self):
        params = make_params()
        for regime in REGIMES:
            for ttm in (0.5, 1.0, 5.0, 50.0):
                t0, t1 = occupation_times(params, ttm, regime)
                assert abs(t0 + t1 - ttm) < 1e-10
                assert t0 >= 0.0 and t1 >= 0.0


    def test_occupation_times_converge_to_stationary(self):
        params = make_params()
        ttm = 1e4
        ls = params.lambda_01 + params.lambda_10
        for regime in REGIMES:
            t0, _ = occupation_times(params, ttm, regime)
            assert abs(t0 / ttm - params.lambda_10 / ls) < 1e-3


    # ==============================================================================
    # strike identities and limits
    # ==============================================================================

    def test_no_jump_limit_recovers_sigma_squared(self):
        flat = RiskNeutralParams(sigma_0=0.2, sigma_1=0.2,
                                 lambda_01=0.05, lambda_10=0.05,
                                 eta_0=1e-4, eta_1=1e-4, rate=0.03)
        kv = variance_swap_strike(flat, 1.0, Regime.GROWTH,
                                  VarianceConvention.QUADRATIC_VARIATION)
        kl = variance_swap_strike(flat, 1.0, Regime.GROWTH,
                                  VarianceConvention.LOG_CONTRACT)
        assert abs(kv - 0.04) < 1e-6
        assert abs(kl - 0.04) < 1e-6


    def test_decomposition_sums_to_total(self):
        d = decompose_variance(make_params(), 1.0, Regime.GROWTH)
        assert abs(d.diffusion + d.jump_crash + d.jump_recovery - d.total) < 1e-12
        assert 0.0 < d.jump_fraction < 1.0


    def test_skew_gap_negative_when_crash_dominated(self):
        p = RiskNeutralParams.from_rates(sigma_0=0.18, sigma_1=0.28,
                                         lambda_01=0.30, lambda_10=1.0,
                                         eta_01=2.5, eta_10=8.0, rate=0.03)
        assert jump_skew_gap(p, 1.0, Regime.GROWTH) < 0.0


    # ==============================================================================
    # closed form vs Monte Carlo
    # ==============================================================================

    def test_qv_strike_matches_monte_carlo(self):
        params = make_params()
        for regime in REGIMES:
            for ttm in (0.5, 2.0, 10.0):
                cf = variance_swap_strike(params, ttm, regime,
                                          VarianceConvention.QUADRATIC_VARIATION)
                mc, se = variance_swap_strike_mc(params, ttm, regime,
                                                 VarianceConvention.QUADRATIC_VARIATION,
                                                 n_paths=300_000, seed=3)
                assert abs(cf - mc) < 4.0 * se, f"QV mismatch at {regime}, T={ttm}"


    def test_log_contract_strike_matches_monte_carlo(self):
        params = make_params()
        for regime in REGIMES:
            for ttm in (0.5, 2.0, 10.0):
                cf = variance_swap_strike(params, ttm, regime,
                                          VarianceConvention.LOG_CONTRACT)
                mc, se = variance_swap_strike_mc(params, ttm, regime,
                                                 VarianceConvention.LOG_CONTRACT,
                                                 n_paths=300_000, seed=5)
                assert abs(cf - mc) < 4.0 * se, f"log mismatch at {regime}, T={ttm}"


    # ==============================================================================
    # variance risk premium
    # ==============================================================================

    def test_jump_size_premium_shows_up_in_jump_term(self):
        """Fatter risk-neutral crashes (smaller eta_01 rate) raise the jump variance."""
        base = make_params_p()
        fatter = RiskNeutralParams.from_rates(sigma_0=0.18, sigma_1=0.28,
                                              lambda_01=0.10, lambda_10=1.0,
                                              eta_01=2.5, eta_10=8.0, rate=0.03)
        vrp = variance_risk_premium(base, fatter, 1.0, Regime.GROWTH)
        assert vrp.total > 0.0
        assert vrp.jump > 0.0
        assert abs(vrp.diffusion) < 1e-12   # sigmas unchanged


    # ==============================================================================
    # input validation
    # ==============================================================================

    def test_invalid_inputs_raise(self):
        params = make_params()
        with self.assertRaises(ValueError):
            occupation_times(params, -1.0, Regime.GROWTH)
        with self.assertRaises(ValueError):
            occupation_times(params, 1.0, 2)


    # ==============================================================================
    # size-premium calibration (no-intensity-premium normalisation)
    # ==============================================================================

    def test_inversion_roundtrip(self):
        """Calibrating to a strike generated by known eta_0^Q must recover it exactly."""
        params_p = make_params_p()
        ttm = 1.0
        eta_true = 0.42
        params_q = RiskNeutralParams(sigma_0=params_p.sigma_0, sigma_1=params_p.sigma_1,
                                     lambda_01=params_p.lambda_01,
                                     lambda_10=params_p.lambda_10,
                                     eta_0=eta_true, eta_1=params_p.eta_1,
                                     rate=params_p.rate)
        k_mkt = variance_swap_strike(params_q, ttm, Regime.GROWTH)
        cal = implied_crash_size_from_var_swap(params_p, k_mkt, ttm, Regime.GROWTH)
        assert abs(cal.eta_0_q - eta_true) < 1e-12
        assert abs(variance_swap_strike(cal.params_q, ttm, Regime.GROWTH) - k_mkt) < 1e-14


    def test_inversion_roundtrip_both_regimes(self):
        params_p = make_params_p()
        ttm = 2.0
        eta_true = 0.35
        params_q = RiskNeutralParams(sigma_0=params_p.sigma_0, sigma_1=params_p.sigma_1,
                                     lambda_01=params_p.lambda_01,
                                     lambda_10=params_p.lambda_10,
                                     eta_0=eta_true, eta_1=params_p.eta_1,
                                     rate=params_p.rate)
        for regime in REGIMES:
            k_mkt = variance_swap_strike(params_q, ttm, regime)
            cal = implied_crash_size_from_var_swap(params_p, k_mkt, ttm, regime)
            assert abs(cal.eta_0_q - eta_true) < 1e-10, f"roundtrip failed in {regime}"


    def test_inversion_monotone_in_market_quote(self):
        """Richer variance quote -> larger implied crash size."""
        params_p = make_params_p()
        etas = [implied_crash_size_from_var_swap(params_p, v ** 2, 1.0).eta_0_q
                for v in (0.24, 0.26, 0.28)]
        assert etas[0] < etas[1] < etas[2]


    def test_below_floor_raises(self):
        with self.assertRaisesRegex(ValueError, "jump-free floor"):
            implied_crash_size_from_var_swap(make_params_p(), 0.03 ** 2, 1.0,
                                             Regime.GROWTH)


    def test_negative_premium_guard(self):
        params_p = make_params_p()
        k_p = variance_swap_strike(params_p, 1.0, Regime.GROWTH)
        with self.assertRaisesRegex(ValueError, "negative crash-size premium"):
            implied_crash_size_from_var_swap(params_p, k_p * 0.98, 1.0, Regime.GROWTH)
        # opting out returns the sub-physical value instead of raising
        cal = implied_crash_size_from_var_swap(params_p, k_p * 0.98, 1.0, Regime.GROWTH,
                                               enforce_crash_premium_sign=False)
        assert cal.eta_0_q < params_p.eta_0


    def test_calibration_preserves_normalisation(self):
        """Only eta_0 may differ from the physical set (lambda_Q = lambda_P etc.)."""
        params_p = make_params_p()
        cal = implied_crash_size_from_var_swap(params_p, 0.26 ** 2, 1.0, Regime.GROWTH)
        q = cal.params_q
        assert q.lambda_01 == params_p.lambda_01
        assert q.lambda_10 == params_p.lambda_10
        assert q.eta_1 == params_p.eta_1
        assert q.sigma_0 == params_p.sigma_0 and q.sigma_1 == params_p.sigma_1
        assert q.eta_0 > params_p.eta_0
        assert abs(cal.size_premium_ratio - q.eta_0 / params_p.eta_0) < 1e-14


    def test_skew_prediction_steeper_than_physical(self):
        """The calibrated size premium must steepen the put skew, most at low strikes."""
        params_p = make_params_p()
        cal = implied_crash_size_from_var_swap(params_p, 0.26 ** 2, 1.0, Regime.GROWTH)
        strikes = np.array([60.0, 80.0, 100.0])
        ivs = skew_overidentification_test(params_p, cal, 100.0, strikes, 1.0,
                                           Regime.GROWTH)
        premium_skew = ivs[:, 0] - ivs[:, 1]
        assert np.all(premium_skew > 0.0)
        assert premium_skew[0] > premium_skew[1] > premium_skew[2]


if __name__ == '__main__':
    unittest.main()