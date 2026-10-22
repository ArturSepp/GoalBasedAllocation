"""Moments of the MV-optimal terminal wealth from the Riccati solution (manuscript Theorem 3.8(c)-(d)).

Checks the coefficient identities b = -ell exp(-r_c tau) a and gamma = -(ell^2/4)(1 - exp(-2 r_c tau) a),
the regime-independent target, the dynamic information ratio Theta = 2 r_c T - ln a(T), the expected
terminal wealth and the efficient frontier, against the single-regime Zhou-Li closed form, the numbers
of the handbook chapter and a Monte Carlo simulation of the policy.
"""
import numpy as np
import pytest

from goal_based_allocation import (
    AssetSpecification, RegimeSwitchParams, create_paper_assets, create_paper_mandates,
    find_ell, find_ell_expected_wealth,
)
from goal_based_allocation.mandate_utils import mandate_effective_asset
from goal_based_allocation.riccati_solver import solve_riccati, simulate_mv_optimal

HORIZON, RATE = 10.0, 0.02


def _balanced_mandate():
    return mandate_effective_asset(create_paper_mandates(create_paper_assets())['balanced'])


def test_coefficient_identities_and_common_target_with_regime_switching_and_jumps():
    ell = 400.0
    ric = solve_riccati(_balanced_mandate(), HORIZON, ell, r=RATE, c=0.0)
    discount = np.exp(-ric.r_c * ric.tau_grid)
    np.testing.assert_allclose(ric.b, -ell * discount * ric.a, rtol=1e-10)
    np.testing.assert_allclose(ric.gamma, -0.25 * ell**2 * (1.0 - discount**2 * ric.a), rtol=1e-9, atol=1e-8)
    for tau in (0.0, 2.5, HORIZON):
        pi_star = ric.derived_at_tau(tau)['Pi_star']
        np.testing.assert_allclose(pi_star, 0.5 * ell * np.exp(-ric.r_c * tau), rtol=1e-10)
    np.testing.assert_allclose(ric.pi_star_0(0), 0.5 * ell * np.exp(-ric.r_c * HORIZON), rtol=1e-10)
    np.testing.assert_allclose(ric.pi_star_0(1), ric.pi_star_0(0), rtol=1e-10)


def test_single_regime_without_jumps_recovers_zhou_li():
    mu, sigma = 0.06, 0.15
    params = RegimeSwitchParams(sigma0=sigma, sigma1=sigma, lambda01=0.0, lambda10=0.0)
    single = AssetSpecification('single', params, mu_growth=mu, mu_stress=mu, pi0=100.0, pi_floor=1.0)
    ell = 400.0
    ric = solve_riccati(single, HORIZON, ell, r=RATE, c=0.0)
    sharpe = (mu - ric.r_h) / sigma
    np.testing.assert_allclose(ric.a[0, -1], np.exp((2.0 * ric.r_c - sharpe**2) * HORIZON), rtol=1e-9)
    np.testing.assert_allclose(ric.information_ratio(0), sharpe**2 * HORIZON, rtol=1e-8)
    # Zhou and Li (2000): E = d - (d - x) exp(-theta^2 T), Var = (E - x)^2 / (exp(theta^2 T) - 1)
    d, x = 0.5 * ell, single.pi0 * np.exp(ric.r_c * HORIZON)
    expected = d - (d - x) * np.exp(-sharpe**2 * HORIZON)
    np.testing.assert_allclose(ric.expected_wealth(0), expected, rtol=1e-8)
    np.testing.assert_allclose(ric.wealth_variance(0), (expected - x)**2 / np.expm1(sharpe**2 * HORIZON), rtol=1e-7)
    # the information ratio with -2 r_c T would be negative here, i.e. a negative variance
    assert -np.log(ric.a[0, -1]) - 2.0 * ric.r_c * HORIZON < 0.0


def test_expected_wealth_is_not_the_present_value_of_the_target():
    equity = create_paper_assets()['equity']
    ell, ric = find_ell(equity, HORIZON, target_return=0.04, r=RATE, c=0.0)
    # handbook chapter, mv_optimal_policy.md: ell = 364.42, Pi*(0) = 149.18, Theta = (0.1753, 0.1625),
    # E[Pi_T] = 131.80 and std = 22.07 from growth
    np.testing.assert_allclose(ell, 364.42, atol=5e-3)
    np.testing.assert_allclose(ric.pi_star_0(0), 149.18, atol=5e-3)
    np.testing.assert_allclose([ric.information_ratio(0), ric.information_ratio(1)], [0.1753, 0.1625], atol=5e-5)
    np.testing.assert_allclose(ric.expected_wealth(0), 131.80, atol=5e-3)
    np.testing.assert_allclose(np.sqrt(ric.wealth_variance(0)), 22.07, atol=5e-3)
    assert ric.expected_wealth(0) < ric.pi_star_0(0)
    # the value function and the envelope theorem give the same expected wealth
    envelope = np.exp(-ric.r_c * HORIZON) * ric.a[0, -1] * equity.pi0 - 2.0 * ric.gamma[0, -1] / ell
    np.testing.assert_allclose(ric.expected_wealth(0), envelope, rtol=1e-9)


def test_find_ell_expected_wealth_hits_the_requirement():
    equity = create_paper_assets()['equity']
    required = 100.0 * np.exp(0.4)
    ell, ric = find_ell_expected_wealth(equity, HORIZON, required, r=RATE, c=0.0, regime=0)
    np.testing.assert_allclose(ric.expected_wealth(0), required, rtol=1e-12)
    np.testing.assert_allclose([ell, ric.pi_star_0(0), ric.omega_star(0.0, 100.0, 0)],
                               [580.60, 237.68, 1.0413], atol=5e-3)
    # the regime-1 requirement needs a different multiplier because Theta differs by regime
    ell1, ric1 = find_ell_expected_wealth(equity, HORIZON, required, r=RATE, c=0.0, regime=1)
    np.testing.assert_allclose(ric1.expected_wealth(1), required, rtol=1e-12)
    assert ell1 != ell
    with pytest.raises(ValueError):
        find_ell_expected_wealth(equity, HORIZON, 100.0 * np.exp(RATE * HORIZON), r=RATE, c=0.0)


@pytest.mark.slow
def test_expected_wealth_and_variance_against_monte_carlo():
    """The simulator of the signed MV-optimal policy, floor far below start wealth, reproduces E[Pi_T] from regime 1."""
    base = _balanced_mandate()
    asset = AssetSpecification(name=base.name, params=base.params, mu_growth=base.mu_growth,
                               mu_stress=base.mu_stress, pi0=base.pi0, pi_floor=1e-6)
    ric = solve_riccati(asset, HORIZON, 400.0, r=RATE, c=0.0)
    sim = simulate_mv_optimal(ric, n_paths=100_000, steps_per_year=260, seed=1, policy='signed')
    terminal = sim['Pi_T']
    standard_error = terminal.std() / np.sqrt(terminal.size)
    assert abs(terminal.mean() - ric.expected_wealth(0)) < 4.0 * standard_error
    assert abs(terminal.var() / ric.wealth_variance(0) - 1.0) < 0.06
    # the present value of the target is far from the simulated mean
    assert abs(terminal.mean() - ric.pi_star_0(0)) > 50.0 * standard_error


def test_recovery_gap_jump_mean_matching():
    """The recovery jump of the log-gap is mean-matched conditional on wealth staying below the target."""
    from goal_based_allocation.riccati_solver import _eta_eff_recovery_quadrature, recovery_exit_probability
    eta1, w = create_paper_assets()['equity'].params.eta1, 0.2129   # equity class, stress coefficient of the worked example
    np.testing.assert_allclose(eta1, 0.130, atol=5e-4)
    ceiling = np.log1p(1.0 / w)
    p_exit = recovery_exit_probability(eta1, w)
    np.testing.assert_allclose(p_exit, np.exp(-ceiling / eta1), rtol=1e-12)
    assert 1.5e-6 < p_exit < 1.7e-6
    # sampled conditional mean of -ln(1 - w (e^J - 1)) over J < J*
    rng = np.random.default_rng(5)
    jumps = rng.exponential(eta1, 4_000_000)
    kept = jumps < ceiling
    sampled = np.mean(-np.log1p(-w * np.expm1(jumps[kept])))
    eta_eff = _eta_eff_recovery_quadrature(eta1, w)
    np.testing.assert_allclose(eta_eff, sampled, rtol=2e-3)
    np.testing.assert_allclose(eta_eff, 0.0333, atol=5e-5)
    # small jumps: the effective mean tends to w * eta1
    np.testing.assert_allclose(_eta_eff_recovery_quadrature(1e-3, w), w * 1e-3, rtol=2e-3)
    assert _eta_eff_recovery_quadrature(0.0, w) == 0.0 and recovery_exit_probability(eta1, 0.0) == 0.0
