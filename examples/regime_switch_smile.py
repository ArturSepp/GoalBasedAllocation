#!/usr/bin/env python3
"""Regime-switching implied-volatility smile, with reference pricers and checks.

Generates the regime-switching implied-volatility smile from the Laplace vanilla
pricer in ``goal_based_allocation`` (``price_vanilla`` / ``implied_vol``), and
bundles two reference pricers used only for validation:

  * ``price_vanilla_fourier`` - a Carr-Madan Fourier pricer with a moment-condition
    guard, kept as a cautionary cross-check (it can return a finite but wrong price
    when the damping violates E[S^(1+alpha)] < inf);
  * ``price_vanilla_mc`` - a Monte Carlo pricer with exact regime holding times.

Run with a ``UnitTests`` case; the default draws the smile and saves
``regime_switch_smile.png`` next to this script.

    python examples/regime_switch_smile.py
"""
from __future__ import annotations

# packages
import numpy as np
from enum import Enum
from typing import Tuple, Union
from pathlib import Path
# project
from goal_based_allocation import (
    RiskNeutralParams, OptionType, Regime, price_vanilla, implied_vol,
    create_paper_assets,
)


# ==============================================================================
# REFERENCE IMPLEMENTATIONS FOR VALIDATION
# ==============================================================================

def price_vanilla_fourier(params: RiskNeutralParams,
                          spot: float,
                          strike: float,
                          ttm: float,
                          regime: Union[int, Regime] = Regime.GROWTH,
                          damping: float = 0.75,
                          check_moment: bool = True,
                          upper: float = 200.0,
                          ) -> float:
    """Carr-Madan Fourier reference pricer (calls only), with a moment-condition guard.

    The characteristic function is φ_l(u) = e^{iu ln S₀}·[e^{T·M(u)}·1]_l with

        M(u) = [[iu b₀ − ½u²σ₀² − λ₀₁,      λ₀₁/(1 + iu η₀)         ],
                [λ₁₀/(1 − iu η₁),           iu b₁ − ½u²σ₁² − λ₁₀     ]]

    (the jump entries are the exponential mgfs in the paper's mean convention).

    Carr-Madan requires E[S_T^{1+α}] < ∞, i.e. α < 1/η₁ − 1. Violating it returns a
    FINITE but badly wrong price, since the matrix exponential is finite regardless.
    `check_moment=True` raises instead. Kept as a cautionary reference — prefer
    :func:`price_vanilla`.
    """
    from scipy.linalg import expm
    from scipy.integrate import quad

    regime = int(regime)
    alpha_max = params.max_carr_madan_damping()
    if check_moment and damping >= alpha_max:
        raise ValueError(
            f"Carr-Madan damping alpha={damping!r} violates the moment condition "
            f"E[S^(1+alpha)] < inf, which requires alpha < 1/eta_1 - 1 = {alpha_max:.4f}. "
            f"The price would be finite but wrong. Use price_vanilla (Laplace) instead.")

    nu0, nu1 = params.nu
    b0, b1 = nu0, nu1   # log-drifts already net of ½σ²

    def phi(u: complex) -> complex:
        mat = np.zeros((2, 2), dtype=complex)
        mat[0, 0] = 1j * u * b0 - 0.5 * u ** 2 * params.sigma_0 ** 2 - params.lambda_01
        mat[0, 1] = params.lambda_01 / (1.0 + 1j * u * params.eta_0)
        mat[1, 0] = params.lambda_10 / (1.0 - 1j * u * params.eta_1)
        mat[1, 1] = 1j * u * b1 - 0.5 * u ** 2 * params.sigma_1 ** 2 - params.lambda_10
        return np.exp(1j * u * np.log(spot)) * (expm(ttm * mat) @ np.ones(2))[regime]

    k_log = np.log(strike)

    def integrand(u: float) -> float:
        num = np.exp(-params.rate * ttm) * phi(u - (damping + 1.0) * 1j)
        den = damping ** 2 + damping - u ** 2 + 1j * (2.0 * damping + 1.0) * u
        return float(np.real(np.exp(-1j * u * k_log) * num / den))

    val, _ = quad(integrand, 0.0, upper, limit=400)
    return float(np.exp(-damping * k_log) / np.pi * val)


def price_vanilla_mc(params: RiskNeutralParams,
                     spot: float,
                     strikes: np.ndarray,
                     ttm: float,
                     regime: Union[int, Regime] = Regime.GROWTH,
                     option_type: Union[str, OptionType] = OptionType.CALL,
                     n_paths: int = 200_000,
                     seed: int = 7,
                     ) -> Tuple[np.ndarray, np.ndarray]:
    """Monte Carlo reference: exact regime holding times, jump at each transition.

    Returns (prices, standard_errors).
    """
    regime = int(regime)
    opt = OptionType(option_type)
    rng = np.random.default_rng(seed)
    nu0, nu1 = params.nu
    nu = np.array([nu0, nu1])
    sig = np.array([params.sigma_0, params.sigma_1])
    lam = np.array([params.lambda_01, params.lambda_10])
    eta = np.array([params.eta_0, params.eta_1])   # means

    x = np.zeros(n_paths)
    chi = np.full(n_paths, regime, dtype=int)
    t = np.zeros(n_paths)
    alive = np.ones(n_paths, dtype=bool)

    while alive.any():
        idx = np.where(alive)[0]
        cur = chi[idx]
        dt_jump = rng.exponential(1.0 / lam[cur])
        dt_left = ttm - t[idx]
        dt = np.minimum(dt_jump, dt_left)
        x[idx] += nu[cur] * dt + sig[cur] * np.sqrt(dt) * rng.standard_normal(len(idx))
        t[idx] += dt
        jumped = dt_jump < dt_left
        ji = idx[jumped]
        if len(ji) > 0:
            cj = chi[ji]
            size = rng.exponential(eta[cj])          # mean convention
            x[ji] += np.where(cj == 0, -size, size)  # crash down, recovery up
            chi[ji] = 1 - cj
        alive[idx[~jumped]] = False

    s_t = spot * np.exp(x)
    disc = np.exp(-params.rate * ttm)
    prices, errors = np.zeros(len(strikes)), np.zeros(len(strikes))
    for i, k in enumerate(strikes):
        pay = np.maximum(s_t - k, 0.0) if opt == OptionType.CALL else np.maximum(k - s_t, 0.0)
        prices[i] = disc * pay.mean()
        errors[i] = disc * pay.std() / np.sqrt(n_paths)
    return prices, errors


# ==============================================================================
# DEMO, CHECKS AND TESTS
# ==============================================================================

class UnitTests(Enum):
    DEMO_PRICES = 1
    DEMO_SMILE = 2
    CHECK_PUT_CALL_PARITY = 3
    CHECK_BLACK_SCHOLES_LIMIT = 4
    TEST_MONTE_CARLO = 5
    TEST_FOURIER_CROSSCHECK = 6
    TEST_FOURIER_FAILURE = 7


def paper_params(asset: str = 'equity', rate: float = 0.02) -> RiskNeutralParams:
    """Risk-neutral parameters from the paper's regime process (Table 1).

    Builds the option-pricing parameters from a paper asset's structural
    parameters (sigma, lambda, eta -- these are measure-independent), pinning
    the drift by the risk-neutral martingale condition. The default underlying
    is equity; 'bonds' or 'private_equity' also work. The paper's risk-free
    rate is 2%.
    """
    p = create_paper_assets()[asset].params
    return RiskNeutralParams(sigma_0=p.sigma0,
                             sigma_1=p.sigma1,
                             lambda_01=p.lambda01,
                             lambda_10=p.lambda10,
                             eta_0=p.eta0,
                             eta_1=p.eta1,
                             rate=rate)


def run_local_test(unit_test: UnitTests) -> None:

    params = paper_params()
    spot, ttm = 100.0, 10.0
    strikes = np.array([80.0, 100.0, 120.0, 150.0, 200.0])

    if unit_test == UnitTests.DEMO_PRICES:
        print(f"eta_0={params.eta_0:.4f}  eta_1={params.eta_1:.4f}  "
              f"finite jump variance: {params.has_finite_jump_variance}")
        print(f"max Carr-Madan damping: {params.max_carr_madan_damping():.2f}\n")
        for regime in (Regime.GROWTH, Regime.STRESS):
            for opt in (OptionType.CALL, OptionType.PUT):
                px = price_vanilla(params=params,
                                   spot=spot,
                                   strikes=strikes,
                                   ttm=ttm,
                                   regime=regime,
                                   option_type=opt)
                body = "  ".join(f"K={k:.0f}: {v:8.4f}" for k, v in zip(strikes, px))
                print(f"{regime.name:>6} {opt.value:>4}: {body}")

    elif unit_test == UnitTests.DEMO_SMILE:
        import matplotlib.pyplot as plt
        ref_vol = 0.20   # representative vol for sizing the strike grid
        fig, axs = plt.subplots(1, 2, figsize=(12, 4.5), tight_layout=True)
        for tenor, label, ax in zip([1.0 / 12.0, 1.0], ['1 month', '1 year'], axs):
            fwd = spot * np.exp(params.rate * tenor)
            if tenor < 0.5:                                # 1 month: fixed strike grid
                ks = np.linspace(60.0, 120.0, 40)
            else:                                          # longer tenor: +/- 3 sigma in log-moneyness
                width = 3.0 * ref_vol * np.sqrt(tenor)
                ks = fwd * np.exp(np.linspace(-width, width, 40))
            for regime in (Regime.GROWTH, Regime.STRESS):
                vols = implied_vol(params=params, spot=spot, strikes=ks,
                                   ttm=tenor, regime=regime)
                ax.plot(ks, 100.0 * vols, label=regime.name.capitalize(), lw=2)
            ax.axvline(fwd, color='grey', ls='--', lw=1, label='forward')
            ax.set_title(f"Implied volatility, T = {label}")
            ax.set_xlabel('strike')
            ax.set_ylabel('implied vol (%)')
            ax.legend()
            ax.grid(alpha=0.3)
        out = Path(__file__).with_name('regime_switch_smile.png')
        plt.savefig(out, dpi=120)
        print(f"saved {out}")

    elif unit_test == UnitTests.CHECK_PUT_CALL_PARITY:
        for regime in (Regime.GROWTH, Regime.STRESS):
            call = price_vanilla(params, spot, strikes, ttm, regime, OptionType.CALL)
            put = price_vanilla(params, spot, strikes, ttm, regime, OptionType.PUT)
            target = spot - strikes * np.exp(-params.rate * ttm)
            err = np.max(np.abs(call - put - target))
            print(f"{regime.name:>6}: max |C - P - (S0 - K e^-rT)| = {err:.3e}")
            assert err < 1e-5, f"put-call parity violated in regime {regime.name}"
        print("PASS")

    elif unit_test == UnitTests.CHECK_BLACK_SCHOLES_LIMIT:
        from scipy.stats import norm
        flat = RiskNeutralParams(sigma_0=0.20, sigma_1=0.20,
                                 lambda_01=0.01, lambda_10=0.01,
                                 eta_0=1e-3, eta_1=1e-3,   # vanishing jumps
                                 rate=0.03)
        tenor = 1.0
        px = price_vanilla(flat, spot, strikes, tenor, Regime.GROWTH, OptionType.CALL)
        sd = 0.20 * np.sqrt(tenor)
        fwd = spot * np.exp(0.03 * tenor)
        d1 = (np.log(fwd / strikes) + 0.5 * sd ** 2) / sd
        bs = np.exp(-0.03 * tenor) * (fwd * norm.cdf(d1) - strikes * norm.cdf(d1 - sd))
        err = np.max(np.abs(px - bs))
        print(f"max |Laplace - Black-Scholes| = {err:.3e}")
        assert err < 1e-3, "does not collapse to Black-Scholes"
        print("PASS")

    elif unit_test == UnitTests.TEST_MONTE_CARLO:
        for regime in (Regime.GROWTH, Regime.STRESS):
            lap = price_vanilla(params, spot, strikes, ttm, regime, OptionType.CALL)
            mc, se = price_vanilla_mc(params, spot, strikes, ttm, regime,
                                      OptionType.CALL, n_paths=400_000)
            print(f"\n--- {regime.name} ---")
            print(f"{'K':>6} {'Laplace':>10} {'MC':>10} {'MC se':>8} {'z':>7}")
            for i, k in enumerate(strikes):
                z = (lap[i] - mc[i]) / se[i]
                print(f"{k:6.0f} {lap[i]:10.4f} {mc[i]:10.4f} {se[i]:8.4f} {z:7.2f}")
                assert abs(z) < 4.0, f"Laplace disagrees with MC at K={k}"
        print("\nPASS")

    elif unit_test == UnitTests.TEST_FOURIER_CROSSCHECK:
        print(f"{'T':>6} {'K':>6} {'Laplace':>11} {'Fourier':>11} {'diff':>11}")
        for tenor in (0.5, 1.0, 5.0, 10.0, 20.0):
            for k in (80.0, 120.0, 200.0):
                lap = price_vanilla(params, spot, k, tenor, Regime.GROWTH)
                fou = price_vanilla_fourier(params, spot, k, tenor, Regime.GROWTH)
                print(f"{tenor:6.1f} {k:6.0f} {lap:11.4f} {fou:11.4f} {lap - fou:11.2e}")
                assert abs(lap - fou) < 1e-4, "Laplace and Fourier disagree"
        print("PASS")

    elif unit_test == UnitTests.TEST_FOURIER_FAILURE:
        # eta_10 = 2.5 (rate) -> eta_1 = 0.4 -> Carr-Madan needs alpha < 1.5
        bad = RiskNeutralParams.from_rates(sigma_0=0.18, sigma_1=0.28,
                                           lambda_01=0.10, lambda_10=1.0,
                                           eta_01=3.0, eta_10=2.5,
                                           rate=0.03)
        k = np.array([120.0])
        mc, se = price_vanilla_mc(bad, spot, k, ttm, Regime.GROWTH, n_paths=400_000)
        lap = price_vanilla(bad, spot, k, ttm, Regime.GROWTH)
        print(f"alpha_max = {bad.max_carr_madan_damping():.2f}, "
              f"finite jump variance: {bad.has_finite_jump_variance}\n")
        print(f"  Monte Carlo        : {mc[0]:9.4f}  (se {se[0]:.4f})")
        print(f"  Laplace            : {lap[0]:9.4f}   z = {(lap[0] - mc[0]) / se[0]:+.2f}")
        for alpha in (1.0, 1.5, 2.0):
            unguarded = price_vanilla_fourier(bad, spot, 120.0, ttm, Regime.GROWTH,
                                              damping=alpha, check_moment=False)
            flag = "" if alpha < bad.max_carr_madan_damping() else "  <-- MOMENT COND. VIOLATED"
            print(f"  Fourier (alpha={alpha:.1f}) : {unguarded:9.4f}   "
                  f"z = {(unguarded - mc[0]) / se[0]:+8.1f}{flag}")
        try:
            price_vanilla_fourier(bad, spot, 120.0, ttm, Regime.GROWTH, damping=2.0)
        except ValueError as err:
            print(f"\n  guard raises as expected: {err}")
        print("\nPASS")



if __name__ == '__main__':
    run_local_test(unit_test=UnitTests.DEMO_SMILE)
