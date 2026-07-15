"""
Vanilla Option Pricing under Regime-Switching Jump-Diffusions via Laplace Transform.

Companion code to: "Dynamic Mean-Variance Portfolio Allocation under Regime-Switching
Jump-Diffusions with Absorbing Barriers and Distribution Matching" (Sepp, 2026), applying the Laplace-transform
methodology of Sepp (2004), "Analytical Pricing of Double-Barrier Options under a
Double-Exponential Jump Diffusion Process".

MODEL (risk-neutral measure Q)
------------------------------
Regime χ_t ∈ {0 (growth), 1 (stress)}, generator Λ = [[-λ₀₁, λ₀₁], [λ₁₀, -λ₁₀]].
Between transitions the log-price X_t = ln S_t follows

    dX_t = ν^[i] dt + σ^[i] dW_t,      ν^[i] = μ_Q^[i] − ½(σ^[i])²

and at transitions the price jumps (paper notation, η = jump MEAN):

    0 → 1 (crash)   : X → X + J^[1],  J^[1] ~ −Exp(1/η₀),  E[e^{ΦJ}] = 1/(1 + η₀Φ)
    1 → 0 (recovery): X → X + J^[2],  J^[2] ~ +Exp(1/η₁),  E[e^{ΦJ}] = 1/(1 − η₁Φ)

The Q-drifts are pinned by the martingale condition E[dS/S] = r dt, giving

    μ_Q^[0] = r + λ₀₁ η₀/(1 + η₀),      μ_Q^[1] = r − λ₁₀ η₁/(1 − η₁),

i.e. exactly `RegimeSwitchParams.compute_drifts(r, r)`. Requires η₁ < 1 for
E[e^{J^[2]}] < ∞; the paper's η₁ < ½ additionally gives finite jump variance.

METHOD
------
Laplace-transform the forward (Fokker-Planck) system in time, f̂_i(q, y) =
∫₀^∞ e^{−qτ} f_i(τ, y) dτ. Exponential jumps have rational Laplace symbols, so the
ansatz f̂_i ~ a_i e^{ψy} reduces the coupled OIDE to the characteristic equation

    G⁰(ψ) G¹(ψ) − λ₀₁λ₁₀ = 0,
    G⁰(ψ) = (−½σ₀²ψ² + ν₀ψ + λ₀₁ + q)(1 − η₀ψ),
    G¹(ψ) = (−½σ₁²ψ² + ν₁ψ + λ₁₀ + q)(1 + η₁ψ),

the same degree-6 polynomial as `regime_switch_paper._solve_characteristic`, with
3 roots in each half-plane. Six coefficients are fixed by: continuity of both
components, the δ-source in the starting regime, and two spurious-pole conditions
removing the e^{y/η₀} and e^{−y/η₁} terms introduced when clearing the jump symbols.

Since f̂ is a piecewise sum of exponentials, the payoff integral

    Ĉ(p) = ∫ (S₀e^y − K)⁺ f̂(p + r, y) dy

is ANALYTIC. Discounting is the shift q = p + r, which also makes the convergence
requirement Re(ψ_neg) < −1 automatic on the Abate-Whitt contour Re(p) > 0.
One numerical inversion in time then returns the price.

WHY LAPLACE RATHER THAN FOURIER
-------------------------------
Prices agree with a Carr-Madan Fourier pricer to ~1e-7 (see the regime_switch_smile
example, TEST_FOURIER_CROSSCHECK),
but:
  1. No damping parameter α, hence no silent moment-condition failure. Carr-Madan needs
     E[S_T^{1+α}] < ∞, i.e. η₁ < 1/(1+α) in mean convention. Violating it returns a
     FINITE but badly wrong price (matrix exponentials never complain) — see the
     regime_switch_smile example, TEST_FOURIER_FAILURE, where the price collapses
     from 38.04 to 8.05.
  2. The same characteristic roots extend to barrier and first-passage payoffs, which
     the Fourier transform cannot handle. Vanillas are the zero-barrier special case of
     the double-barrier machinery, so the option layer and the wealth-floor layer of the
     portfolio problem share one engine.

LIMITATION
----------
The method requires a RATIONAL jump transform: exponential, Erlang (integer shape), or
hyperexponential. A Gamma density with non-integer shape is not rational and destroys the
polynomial structure. Use hyperexponential mixtures instead — dense in the completely
monotone class, and every formula here survives.
"""
from __future__ import annotations

# packages
import numpy as np
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Tuple, Union

# project
from .laplace_inversion import laplace_invert_abate_whitt
from .regime_switch_paper import RegimeSwitchParams


# ==============================================================================
# DATA CLASSES
# ==============================================================================

class OptionType(str, Enum):
    CALL = 'call'
    PUT = 'put'


class Regime(int, Enum):
    GROWTH = 0
    STRESS = 1


@dataclass(frozen=True)
class RiskNeutralParams:
    """Risk-neutral parameters for option pricing.

    Uses the paper's MEAN convention: η₀, η₁ are the expected absolute log-jump sizes.
    Use :meth:`from_rates` to build from the RATE convention (η_rate = 1/η_mean).

    Parameters
    ----------
    sigma_0, sigma_1 : float
        Diffusion volatilities, growth and stress regimes.
    lambda_01, lambda_10 : float
        Risk-neutral transition intensities (crash 0→1, recovery 1→0).
    eta_0 : float
        Mean crash log-jump size (>0). Any η₀ > 0 is admissible.
    eta_1 : float
        Mean recovery log-jump size. Requires η₁ < 1 for E[e^{J}] < ∞ (martingale);
        η₁ < ½ additionally gives finite jump variance (paper condition).
    rate : float
        Continuously-compounded risk-free rate r.
    """
    sigma_0: float
    sigma_1: float
    lambda_01: float
    lambda_10: float
    eta_0: float
    eta_1: float
    rate: float = 0.0

    def __post_init__(self) -> None:
        for name in ('sigma_0', 'sigma_1', 'lambda_01', 'lambda_10', 'eta_0', 'eta_1'):
            if getattr(self, name) <= 0.0:
                raise ValueError(f"{name} must be positive, got {getattr(self, name)!r}")
        if self.eta_1 >= 1.0:
            raise ValueError(
                f"eta_1 must be < 1 so that E[exp(J)] < inf and S is a Q-martingale, "
                f"got {self.eta_1!r}")

    @classmethod
    def from_rates(cls,
                   sigma_0: float,
                   sigma_1: float,
                   lambda_01: float,
                   lambda_10: float,
                   eta_01: float,          # RATE of the crash jump: J ~ Exp(eta_01)
                   eta_10: float,          # RATE of the recovery jump: J ~ Exp(eta_10)
                   rate: float = 0.0,
                   ) -> 'RiskNeutralParams':
        """Build from the RATE convention, η_mean = 1 / η_rate.

        Requires eta_10 > 1 (equivalently η₁ < 1) for the martingale condition.
        """
        if eta_01 <= 0.0 or eta_10 <= 0.0:
            raise ValueError(f"jump rates must be positive, got {eta_01!r}, {eta_10!r}")
        return cls(sigma_0=sigma_0,
                   sigma_1=sigma_1,
                   lambda_01=lambda_01,
                   lambda_10=lambda_10,
                   eta_0=1.0 / eta_01,
                   eta_1=1.0 / eta_10,
                   rate=rate)

    @property
    def has_finite_jump_variance(self) -> bool:
        """Var[e^{J^[2]}] < ∞, i.e. η₁ < ½ (paper condition)."""
        return self.eta_1 < 0.5

    @property
    def nu(self) -> Tuple[float, float]:
        """Risk-neutral log-drifts (ν₀, ν₁), from the martingale condition."""
        return self.to_regime_switch_params().compute_drifts(self.rate, self.rate)

    def to_regime_switch_params(self) -> RegimeSwitchParams:
        """Interop with the density/allocation machinery in `regime_switch_paper`."""
        return RegimeSwitchParams(sigma0=self.sigma_0,
                                  sigma1=self.sigma_1,
                                  lambda01=self.lambda_01,
                                  lambda10=self.lambda_10,
                                  eta0=self.eta_0,
                                  eta1=self.eta_1)

    def max_carr_madan_damping(self) -> float:
        """Largest admissible Carr-Madan damping α: requires η₁ < 1/(1+α).

        Provided as a guard for anyone using a Fourier pricer on this model —
        exceeding it returns a finite but wrong price. The Laplace pricer below
        has no damping parameter and is unaffected.
        """
        return 1.0 / self.eta_1 - 1.0


# ==============================================================================
# LAPLACE-DOMAIN SOLUTION
# ==============================================================================

def _solve_characteristic_rn(q: complex,
                             params: RiskNeutralParams,
                             regime: int,
                             ) -> Tuple[np.ndarray, np.ndarray, int]:
    """Solve the degree-6 characteristic system at Laplace argument q.

    Mirrors `regime_switch_paper._solve_characteristic` but (i) uses risk-neutral
    drifts and (ii) allows either starting regime.

    Returns
    -------
    psi : (6,) complex ndarray
        Roots, sorted ascending by real part.
    c_tot : (6,) complex ndarray
        Coefficients of the regime-summed transformed density,
        f̂(q, y) = Σ_k c_tot[k] e^{ψ_k y}, summing over the negative block for
        y > 0 and the positive block for y < 0.
    n_neg : int
        Number of roots in the left half-plane (must be 3).

    Raises
    ------
    ValueError
        If the root split is not 3/3, which signals inadmissible parameters.
    """
    s0h = 0.5 * params.sigma_0 ** 2
    s1h = 0.5 * params.sigma_1 ** 2
    nu0, nu1 = params.nu
    eta0, eta1 = params.eta_0, params.eta_1
    lam01, lam10 = params.lambda_01, params.lambda_10

    # Q^[i](ψ) = −½σ_i²ψ² + ν_iψ + (λ_ij + q),  descending powers
    q0 = np.array([-s0h, nu0, lam01 + q], dtype=complex)
    q1 = np.array([-s1h, nu1, lam10 + q], dtype=complex)

    # G⁰ = Q⁰·(1 − η₀ψ),  G¹ = Q¹·(1 + η₁ψ)
    g0 = np.polymul(q0, np.array([-eta0, 1.0], dtype=complex))
    g1 = np.polymul(q1, np.array([eta1, 1.0], dtype=complex))

    char = np.polymul(g0, g1)
    char[-1] -= lam01 * lam10

    psi = np.roots(char)
    psi = psi[np.argsort(psi.real)]

    n_neg = int(np.sum(psi.real < 0.0))
    if n_neg != 3:
        raise ValueError(f"expected 3 roots in the left half-plane, got {n_neg} "
                         f"at q={q!r}; check parameters")

    # Coupling ratio β_k = a₁/a₀ from the regime-1 equation
    q0_vals = -s0h * psi ** 2 + nu0 * psi + (lam01 + q)
    beta = q0_vals * (1.0 + eta1 * psi) / lam10

    # 6×6 matching system; sign = +1 on the negative (y>0) block, −1 on the positive
    sign = np.where(np.arange(6) < n_neg, 1.0, -1.0).astype(complex)

    mat = np.zeros((6, 6), dtype=complex)
    mat[0, :] = sign / (1.0 - eta0 * psi)          # kill spurious e^{+y/η₀}
    mat[1, :] = sign                                # continuity of f̂₀
    mat[2, :] = sign * psi                          # δ-source, regime 0
    mat[3, :] = sign * beta                         # continuity of f̂₁
    mat[4, :] = sign * beta * psi                   # δ-source, regime 1
    mat[5, :] = sign * beta / (1.0 + eta1 * psi)    # kill spurious e^{−y/η₁}

    rhs = np.zeros(6, dtype=complex)
    if regime == 0:
        rhs[2] = -1.0 / s0h
    else:
        rhs[4] = -1.0 / s1h

    a_coef = np.linalg.solve(mat, rhs)
    c_tot = a_coef * (1.0 + beta)   # payoff is independent of the terminal regime
    return psi, c_tot, n_neg


def _payoff_transform(q: complex,
                      params: RiskNeutralParams,
                      regime: int,
                      spot: float,
                      strikes: np.ndarray,
                      option_type: OptionType,
                      ) -> np.ndarray:
    """Analytic payoff integral Û(q) = ∫ payoff(S₀e^y) f̂(q, y) dy.

    Closed form because f̂ is a piecewise sum of exponentials. The ∫ e^{(1+ψ)y} dy
    term over y > 0 converges iff Re(ψ) < −1 on the negative block, which holds
    exactly when Re(q) > r.
    """
    psi, c_tot, n_neg = _solve_characteristic_rn(q=q, params=params, regime=regime)
    ps_n, cs_n = psi[:n_neg][None, :], c_tot[:n_neg][None, :]      # (1, 3)
    ps_p, cs_p = psi[n_neg:][None, :], c_tot[n_neg:][None, :]
    kk = np.log(strikes / spot)[:, None]                            # (n_k, 1)
    kv = strikes[:, None]

    if option_type == OptionType.CALL:
        lo = np.maximum(kk, 0.0)                                    # tail on y > max(k,0)
        val = (cs_n * (-spot * np.exp((1.0 + ps_n) * lo) / (1.0 + ps_n)
                       + kv * np.exp(ps_n * lo) / ps_n)).sum(axis=1)
        itm = (kk < 0.0)[:, 0]
        if np.any(itm):                                            # strip [k, 0]
            strip = (cs_p * (spot * (1.0 - np.exp((1.0 + ps_p) * kk)) / (1.0 + ps_p)
                             - kv * (1.0 - np.exp(ps_p * kk)) / ps_p)).sum(axis=1)
            val = val + np.where(itm, strip, 0.0)
    else:
        hi = np.minimum(kk, 0.0)                                    # tail on y < min(k,0)
        val = (cs_p * (kv * np.exp(ps_p * hi) / ps_p
                       - spot * np.exp((1.0 + ps_p) * hi) / (1.0 + ps_p))).sum(axis=1)
        itm = (kk > 0.0)[:, 0]
        if np.any(itm):                                            # strip [0, k]
            strip = (cs_n * (kv * (np.exp(ps_n * kk) - 1.0) / ps_n
                             - spot * (np.exp((1.0 + ps_n) * kk) - 1.0) / (1.0 + ps_n))
                     ).sum(axis=1)
            val = val + np.where(itm, strip, 0.0)

    return val


# ==============================================================================
# PUBLIC INTERFACE
# ==============================================================================

def price_vanilla(params: RiskNeutralParams,
                  spot: float,
                  strikes: Union[float, np.ndarray],
                  ttm: float,
                  regime: Union[int, Regime] = Regime.GROWTH,
                  option_type: Union[str, OptionType] = OptionType.CALL,
                  n_terms: int = 25,        # Abate-Whitt main sum terms
                  n_euler: int = 12,        # Euler acceleration terms
                  ) -> np.ndarray:
    """Price European vanillas by Laplace inversion. Strikes are priced jointly.

    Parameters
    ----------
    params : RiskNeutralParams
    spot : float
        Current asset price S₀.
    strikes : float or ndarray
        Strike(s) K.
    ttm : float
        Time to maturity in years.
    regime : int or Regime
        Current regime χ₀ (0 = growth, 1 = stress).
    option_type : str or OptionType
        'call' or 'put'.

    Returns
    -------
    ndarray of prices aligned with `strikes` (scalar if scalar input).

    Raises
    ------
    ValueError
        On non-positive spot/strike/ttm or an invalid regime.
    """
    is_scalar = np.ndim(strikes) == 0
    strikes_arr = np.atleast_1d(np.asarray(strikes, dtype=float))
    if np.any(strikes_arr <= 0.0):
        raise ValueError(f"strikes must be positive, got {strikes!r}")
    if spot <= 0.0:
        raise ValueError(f"spot must be positive, got {spot!r}")
    if ttm <= 0.0:
        raise ValueError(f"ttm must be positive, got {ttm!r}")
    regime = int(regime)
    if regime not in (0, 1):
        raise ValueError(f"regime must be 0 or 1, got {regime!r}")
    option_type = OptionType(option_type)

    def transform(p_nodes: np.ndarray) -> np.ndarray:
        out = np.zeros((len(p_nodes), len(strikes_arr)), dtype=complex)
        for i, p in enumerate(p_nodes):
            out[i, :] = _payoff_transform(q=p + params.rate,   # discounting = shift
                                          params=params,
                                          regime=regime,
                                          spot=spot,
                                          strikes=strikes_arr,
                                          option_type=option_type)
        return out

    prices = laplace_invert_abate_whitt(transform, ttm, N=n_terms, M=n_euler)
    prices = np.maximum(prices, 0.0)
    return float(prices[0]) if is_scalar else prices


def implied_vol(params: RiskNeutralParams,
                spot: float,
                strikes: Union[float, np.ndarray],
                ttm: float,
                regime: Union[int, Regime] = Regime.GROWTH,
                option_type: Union[str, OptionType] = OptionType.CALL,
                ) -> np.ndarray:
    """Black-Scholes implied vols of the model prices (for smile diagnostics)."""
    from scipy.optimize import brentq
    from scipy.stats import norm

    strikes_arr = np.atleast_1d(np.asarray(strikes, dtype=float))
    prices = np.atleast_1d(price_vanilla(params=params,
                                         spot=spot,
                                         strikes=strikes_arr,
                                         ttm=ttm,
                                         regime=regime,
                                         option_type=option_type))
    opt = OptionType(option_type)
    disc = np.exp(-params.rate * ttm)
    fwd = spot / disc

    def bs_price(vol: float, strike: float) -> float:
        sd = vol * np.sqrt(ttm)
        d1 = (np.log(fwd / strike) + 0.5 * sd ** 2) / sd
        d2 = d1 - sd
        if opt == OptionType.CALL:
            return disc * (fwd * norm.cdf(d1) - strike * norm.cdf(d2))
        return disc * (strike * norm.cdf(-d2) - fwd * norm.cdf(-d1))

    out = np.full(len(strikes_arr), np.nan)
    for i, (px, k) in enumerate(zip(prices, strikes_arr)):
        try:
            out[i] = brentq(lambda v: bs_price(v, k) - px, 1e-4, 5.0, xtol=1e-10)
        except ValueError:
            pass
    return out
