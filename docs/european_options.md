---
myst:
  html_meta:
    description: >-
      European option pricing under the two-regime jump-diffusion in goal-based-allocation: the
      risk-neutral drifts, the forward density in the Laplace domain, the closed-form payoff
      integral, one inversion for all strikes, implied volatility smiles from either regime, and
      an independent check against a Fourier pricer and the Black-Scholes limit.
---

# European options under regime switching

*Author: [Artur Sepp](https://github.com/ArturSepp)*

Implemented in [GoalBasedAllocation](https://github.com/ArturSepp/GoalBasedAllocation).
Software citation: [CITATION.cff](https://github.com/ArturSepp/GoalBasedAllocation/blob/main/CITATION.cff).

A European call or put on an asset that follows the regime-switching jump-diffusion under a
risk-neutral measure has a price whose Laplace transform in maturity is a closed form: the forward
density of the log-price is a sum of exponentials in the Laplace domain, and integrating the payoff
against exponentials is elementary. The package prices calls and puts this way, for all strikes at
once and from either starting regime, with one [numerical inversion](laplace_inversion.md); the same
characteristic roots solve the [barrier problems](laplace_barrier_framework.md) of the wealth floor.
This chapter derives the price, states its conditions, and checks it against an independent Fourier
pricer.

## Overview

`price_vanilla(params, spot, strikes, ttm, regime, option_type)` returns call or put prices for a
vector of strikes, and `implied_vol` their Black–Scholes implied volatilities. The parameters are
`RiskNeutralParams`, a validated container in the mean convention for jumps, with
`RiskNeutralParams.from_rates` for the rate convention. The use cases are the implied-volatility smile
of the model, its dependence on the current regime, and calibration of the jump parameters to option
quotes, as in the KOSPI study of the repository.

Option pricing is a secondary workflow of the package: it reuses the model of the allocation chapters
under a risk-neutral measure, but the package does not link the two measures. The
[variance-swap chapter](variance_swaps.md) compares physical and risk-neutral parameters through the
variance risk premium.

## Inputs, notation, and assumptions

| Convention | This article |
|---|---|
| Time and rates | Maturity `ttm` in years; continuously compounded riskless rate `rate` |
| Regimes | The starting regime is an input, `Regime.GROWTH = 0` or `Regime.STRESS = 1`; the payoff does not depend on the terminal regime |
| Jump sizes | Mean convention in `RiskNeutralParams`; `from_rates` takes the rates $1/\eta^{[1]}$ and $1/\eta^{[2]}$; $\eta^{[2]} \lt 1$ is required |
| Wealth coordinate | Log-return $y = \ln(S_T/S_0)$; strikes in price units, with $k = \ln(K/S_0)$ |
| Measure | Risk-neutral: the total expected return equals the riskless rate in both regimes |
| Numerical method | Degree-six roots and a $6 \times 6$ solve at each Laplace argument, closed-form payoff integral, Abate–Whitt with $N = 25$ and $M = 12$; checked against a Fourier pricer, put–call parity and the Black–Scholes limit |
| Package default | `price_vanilla(params, spot, strikes, ttm, regime=Regime.GROWTH, option_type=OptionType.CALL, n_terms=25, n_euler=12)`; `RiskNeutralParams(rate=0.0)` |

| Symbol | Meaning | Units and convention |
|---|---|---|
| $S_0$, $K$ | Spot and strike | Price units |
| $y$, $k$ | Log-return to maturity and log-moneyness $\ln(K/S_0)$ | Log units |
| $f^{[s]}(T, y)$ | Risk-neutral density of $y$ from regime $s$, summed over terminal regimes | Per unit of $y$ |
| $C$, $P$ | Call and put prices | Price units |
| $c_k$ | Coefficients of the transformed density at root $\psi_k$ | |
| $\varphi(v)$ | Characteristic function $\mathbb{E}[e^{ivy}]$ | Complex |
| $\alpha_{\mathrm{CM}}$ | Damping parameter of the Carr–Madan Fourier method | Dimensionless |

The regime parameters $\sigma^{[i]}$, $\lambda^{[12]}$, $\lambda^{[21]}$, $\eta^{[i]}$ and the riskless rate
$r$ follow the [conventions page](conventions.md#reserved-notation); here they are risk-neutral values.

## Methodology

### Risk-neutral drifts

**Proposition (martingale condition).** The discounted price $e^{-rt}S_t$ is a martingale under the
regime-switching jump-diffusion if and only if the total expected return of each regime equals $r$,
that is

$$
\nu^{[1]} = r + \lambda^{[12]}\frac{\eta^{[1]}}{1 + \eta^{[1]}} - \frac{(\sigma^{[1]})^2}{2}, \qquad
\nu^{[2]} = r - \lambda^{[21]}\frac{\eta^{[2]}}{1 - \eta^{[2]}} - \frac{(\sigma^{[2]})^2}{2} .
$$

**Proof.** In regime $i$ the instantaneous expected return, including the expected relative jump, is
$\bar\mu^{[i]}$ by the [model chapter](regime_switching_model.md#total-return-diffusion-drift-and-log-drift);
the discounted price has zero drift in every state exactly when $\bar\mu^{[i]} = r$ in both regimes.
The log drift then follows from the definitions of $\mu^{[i]}$ and $\nu^{[i]}$. $\square$

These are `RegimeSwitchParams.compute_drifts(r, r)`. The recovery compensator needs $\eta^{[2]} \lt 1$,
which `RiskNeutralParams` enforces.

### The price in the Laplace domain

The call price is $C = e^{-rT}\int(S_0e^{y} - K)^{+}f^{[s]}(T, y)\thinspace dy$. Its Laplace transform in maturity
is

$$
\hat C(p) = \int_{-\infty}^{\infty}(S_0e^{y} - K)^{+}\hat f^{[s]}(p + r, y)\thinspace dy,
$$

because $\int_0^{\infty}e^{-pT}e^{-rT}f(T, y)\thinspace dT = \hat f(p + r, y)$: discounting is a shift of the
Laplace variable.

**Proposition (closed-form payoff integral).** With $\hat f^{[s]}(q, y) = \sum_{k \le 3}c_ke^{\psi_ky}$ for
$y \gt 0$ and $\sum_{k \ge 4}c_ke^{\psi_ky}$ for $y \lt 0$, and $k \ge 0$,

$$
\hat C(p) = \sum_{k=1}^{3}c_k\left(-\frac{S_0e^{(1 + \psi_k)k}}{1 + \psi_k} + \frac{Ke^{\psi_kk}}{\psi_k}\right),
$$

with an additional strip term from $[k, 0]$ when $k \lt 0$; the put is the mirror image. The integral
converges because the decaying roots satisfy $\mathrm{Re}(\psi_k) \lt -1$ when $\mathrm{Re}(q) \gt r$,
which holds on the inversion contour $q = p + r$ with $\mathrm{Re}(p) \gt 0$.

**Proof.** On $y \gt \max(k, 0)$, $\int(S_0e^{y} - K)e^{\psi y}dy$ is a sum of two exponential integrals,
finite when $\mathrm{Re}(1 + \psi) \lt 0$. The coefficients $c_k$ are those of the
[unbounded solution](laplace_barrier_framework.md#unbounded-and-bounded-solutions), summed over the two
terminal regimes because the payoff ignores the regime. The moment of $e^{y}$ under $\hat f(q, \cdot)$ is
finite exactly when $q$ exceeds the growth rate $r$ of the forward. $\square$

The coefficients depend on the strike only through the integration limits, so all strikes share the
roots and the linear solve at each of the 38 Laplace arguments: one inversion prices the whole strike
vector. The starting regime enters only the right-hand side of the linear system, through the source
term of the forward equation.

**Identity (put–call parity).** $C - P = S_0 - Ke^{-rT}$ for every maturity and starting regime.

**Proof.** $(S_T - K)^{+} - (K - S_T)^{+} = S_T - K$, and $\mathbb{E}[e^{-rT}S_T] = S_0$ by the
martingale condition. $\square$

### Laplace against Fourier

A Fourier pricer needs the characteristic function, which is a matrix exponential:
$\varphi^{[s]}(v) = [e^{T\mathbf{M}(v)}(1, 1)^{\top}]_s$ with

$$
\mathbf{M}(v) = \begin{pmatrix} iv\nu^{[1]} - \frac{1}{2}v^2(\sigma^{[1]})^2 - \lambda^{[12]} & \lambda^{[12]}/(1 + iv\eta^{[1]}) \\ \lambda^{[21]}/(1 - iv\eta^{[2]}) & iv\nu^{[2]} - \frac{1}{2}v^2(\sigma^{[2]})^2 - \lambda^{[21]} \end{pmatrix} .
$$

The worked example prices with the formula of Lewis (2001), which evaluates $\varphi$ at $v - i/2$.

> **Pitfall.** The Carr–Madan method (Carr and Madan, 1999) needs $\mathbb{E}[S_T^{1 + \alpha_{\mathrm{CM}}}] \lt \infty$,
> which for exponential recoveries is $\eta^{[2]} \lt 1/(1 + \alpha_{\mathrm{CM}})$. A damping parameter
> beyond that bound returns a finite but wrong price, because the matrix exponential stays finite.
> `RiskNeutralParams.max_carr_madan_damping` returns the bound $1/\eta^{[2]} - 1$. The Laplace pricer has
> no damping parameter.

## Worked example

The risk-neutral parameters reuse the equity regime process of Table 1, with volatilities of 15% and
22.5%, intensities 0.1 and 1, jump means $1/3$ and $0.1304$, and $r = 2$%. The log drifts are 3.375%
in growth and $-15.53$% in stress, and the Carr–Madan bound is $\alpha_{\mathrm{CM}} \lt 6.67$. The first
block prices calls and puts at strikes from 60 to 200 for maturities of one and ten years from both
regimes, and checks that:

- a call with a strike near zero is worth the spot, the martingale condition;
- put–call parity holds within $1.1 \times 10^{-6}$;
- an independent Lewis–Fourier pricer with the matrix-exponential characteristic function, validated
  against the Black–Scholes formula to $10^{-12}$, agrees within $10^{-6}$;
- with equal regimes, rare transitions and jump means of $10^{-3}$ the price is within $10^{-3}$ of
  Black–Scholes at every strike, the tolerance of the example script's limit test.

```python
import numpy as np
from scipy.integrate import quad
from scipy.linalg import expm
from scipy.stats import norm
import goal_based_allocation as gba

equity = gba.create_paper_assets()['equity'].params
params = gba.RiskNeutralParams(sigma_0=equity.sigma0, sigma_1=equity.sigma1,
                               lambda_01=equity.lambda01, lambda_10=equity.lambda10,
                               eta_0=equity.eta0, eta_1=equity.eta1, rate=0.02)
np.testing.assert_allclose(params.nu, [0.03375, -0.1553125], rtol=1e-12)
np.testing.assert_allclose(params.nu, params.to_regime_switch_params().compute_drifts(0.02, 0.02), rtol=1e-12)
np.testing.assert_allclose(params.max_carr_madan_damping(), 1.0 / params.eta_1 - 1.0)
assert params.has_finite_jump_variance
same = gba.RiskNeutralParams.from_rates(sigma_0=0.15, sigma_1=0.225, lambda_01=0.1, lambda_10=1.0,
                                        eta_01=1.0 / equity.eta0, eta_10=1.0 / equity.eta1, rate=0.02)
np.testing.assert_allclose([same.eta_0, same.eta_1], [params.eta_0, params.eta_1], rtol=1e-12)


def lewis_call(prm, spot, strike, ttm, regime):
    """Call price by the Lewis (2001) Fourier integral with the regime-switching characteristic function."""
    nu0, nu1 = prm.nu

    def characteristic(v):
        generator = np.array([
            [1j * v * nu0 - 0.5 * v**2 * prm.sigma_0**2 - prm.lambda_01, prm.lambda_01 / (1.0 + 1j * v * prm.eta_0)],
            [prm.lambda_10 / (1.0 - 1j * v * prm.eta_1), 1j * v * nu1 - 0.5 * v**2 * prm.sigma_1**2 - prm.lambda_10],
        ])
        return (expm(ttm * generator) @ np.ones(2))[regime]

    forward = spot * np.exp(prm.rate * ttm)
    moneyness = np.log(strike / forward)

    def integrand(u):
        v = u - 0.5j
        return np.real(np.exp(-1j * u * moneyness - 1j * v * prm.rate * ttm) * characteristic(v)) / (u**2 + 0.25)

    return spot - np.sqrt(forward * strike) * np.exp(-prm.rate * ttm) / np.pi * quad(integrand, 0.0, np.inf, limit=500)[0]


def black_scholes_call(spot, strike, ttm, rate, vol):
    d1 = (np.log(spot / strike) + (rate + 0.5 * vol**2) * ttm) / (vol * np.sqrt(ttm))
    return spot * norm.cdf(d1) - strike * np.exp(-rate * ttm) * norm.cdf(d1 - vol * np.sqrt(ttm))


# the Fourier reference reproduces Black-Scholes when both regimes are equal and jumps vanish
flat = gba.RiskNeutralParams(sigma_0=0.2, sigma_1=0.2, lambda_01=0.5, lambda_10=0.5, eta_0=1e-9, eta_1=1e-9, rate=0.03)
for strike in (80.0, 100.0, 130.0):
    assert abs(lewis_call(flat, 100.0, strike, 2.0, 0) - black_scholes_call(100.0, strike, 2.0, 0.03, 0.2)) < 1e-12

strikes = np.array([60.0, 80.0, 100.0, 120.0, 150.0, 200.0])
for ttm in (1.0, 10.0):
    for regime in (gba.Regime.GROWTH, gba.Regime.STRESS):
        calls = gba.price_vanilla(params, 100.0, strikes, ttm, regime, gba.OptionType.CALL)
        puts = gba.price_vanilla(params, 100.0, strikes, ttm, regime, 'put')
        assert np.abs(calls - puts - (100.0 - strikes * np.exp(-0.02 * ttm))).max() < 1.1e-6
        reference = np.array([lewis_call(params, 100.0, k, ttm, int(regime)) for k in strikes[::2]])
        assert np.abs(calls[::2] - reference).max() < 1e-6
assert abs(gba.price_vanilla(params, 100.0, 1e-6, 10.0) - 100.0) < 1e-5

# Black-Scholes limit as in the example script: equal regimes, rare transitions, small jumps
nearly_flat = gba.RiskNeutralParams(sigma_0=0.2, sigma_1=0.2, lambda_01=0.01, lambda_10=0.01,
                                    eta_0=1e-3, eta_1=1e-3, rate=0.03)
limit = gba.price_vanilla(nearly_flat, 100.0, strikes, 1.0)
assert np.abs(limit - black_scholes_call(100.0, strikes, 1.0, 0.03, 0.2)).max() < 1e-3
```

The second block shows the smile. From growth, one-year implied volatilities fall from 30.2% at a
strike of 60 to 16.5% at 150 and rise again to 20.2% at 200: the crash makes low strikes expensive, and
the chance of spending part of the year in the more volatile stress regime fattens both tails. At the money the implied
volatility is 17.7%, above the diffusion volatility of 15%, because jumps add variance. From stress the
smile is 27.9%, 26.1% at the money and 29.9%: higher, since the volatility is 22.5%, and flatter, since
the next transition is the upward recovery. At ten years the two smiles converge, 20.2% and 21.3% at
the money, as the regime at inception matters less.

```python
smile = {(ttm, int(regime)): gba.implied_vol(params, 100.0, strikes, ttm, regime)
         for ttm in (1.0, 10.0) for regime in (gba.Regime.GROWTH, gba.Regime.STRESS)}
np.testing.assert_allclose(smile[(1.0, 0)], [0.3024, 0.2186, 0.1768, 0.1655, 0.1650, 0.2015], atol=5e-5)
np.testing.assert_allclose(smile[(1.0, 1)][[0, 2, 5]], [0.2794, 0.2610, 0.2992], atol=5e-5)
np.testing.assert_allclose([smile[(10.0, 0)][2], smile[(10.0, 1)][2]], [0.2018, 0.2133], atol=5e-5)
assert smile[(1.0, 0)][2] > params.sigma_0 and smile[(1.0, 1)][2] > params.sigma_1
# the regime at inception matters less at ten years than at one
assert abs(smile[(10.0, 1)][2] - smile[(10.0, 0)][2]) < abs(smile[(1.0, 1)][2] - smile[(1.0, 0)][2])
```

## Implementation in goal_based_allocation

| Quantity | Formula | Entry point |
|---|---|---|
| Risk-neutral parameters | Volatilities, intensities, jump means, rate | `RiskNeutralParams(sigma_0, sigma_1, lambda_01, lambda_10, eta_0, eta_1, rate=0.0)` |
| Rate convention | $\eta = 1/\text{rate}$ | `RiskNeutralParams.from_rates(sigma_0, sigma_1, lambda_01, lambda_10, eta_01, eta_10, rate=0.0)` |
| Risk-neutral log drifts | Martingale condition | `RiskNeutralParams.nu` |
| Calls and puts for a strike vector | One Laplace inversion | `price_vanilla(params, spot, strikes, ttm, regime=Regime.GROWTH, option_type=OptionType.CALL)` |
| Implied volatility | Black–Scholes inversion by Brent's method on $[10^{-4}, 5]$ | `implied_vol(params, spot, strikes, ttm, regime=Regime.GROWTH, option_type=OptionType.CALL)` |
| Carr–Madan damping bound | $1/\eta^{[2]} - 1$ | `RiskNeutralParams.max_carr_madan_damping()` |

The pricer lives in
[`vanilla_option_pricer.py`](https://github.com/ArturSepp/GoalBasedAllocation/blob/main/src/goal_based_allocation/vanilla_option_pricer.py),
and the Fourier and Monte Carlo reference pricers with the smile plot in
[`examples/regime_switch_smile.py`](https://github.com/ArturSepp/GoalBasedAllocation/blob/main/examples/regime_switch_smile.py).
API pages: {doc}`price_vanilla <api/generated/goal_based_allocation.price_vanilla>`,
{doc}`implied_vol <api/generated/goal_based_allocation.implied_vol>` and
{doc}`RiskNeutralParams <api/generated/goal_based_allocation.RiskNeutralParams>`.

Contract details:

- `RiskNeutralParams` is frozen and raises `ValueError` for non-positive volatilities, intensities or
  jump means and for $\eta^{[2]} \ge 1$. `has_finite_jump_variance` reports $\eta^{[2]} \lt 1/2$.
- `price_vanilla` raises `ValueError` for non-positive spot, strikes or maturity and for a regime other
  than 0 or 1, and returns a float for a scalar strike and an array otherwise. It clips negative
  inversion noise at zero.
- The root solver raises `ValueError` unless three roots lie in the left half-plane at every Laplace
  argument, which signals inadmissible parameters.
- `implied_vol` returns `NaN` where Brent's method finds no root in $[10^{-4}, 5]$, for example for a
  price below intrinsic value.
- `n_terms` and `n_euler` are the Abate–Whitt $N$ and $M$.

> **Pitfall.** Jump means close to zero make the characteristic polynomial ill-conditioned, because two
> of its roots move to $\pm 1/\eta$. In our checks with equal regimes, jump means of $10^{-3}$ and
> $10^{-4}$ priced within $3 \times 10^{-4}$ of Black–Scholes, while means of $10^{-5}$ and $10^{-6}$
> returned `NaN` or a wrong price without a warning. Approach the jump-free limit through small
> intensities instead, and keep jump means above about $10^{-3}$.

## Interpretation and limitations

- The method needs rational jump transforms: exponential jumps, or hyperexponential and Erlang
  mixtures. A gamma law with non-integer shape or a normal jump law destroys the polynomial structure.
- Only European calls and puts are implemented. Barrier and first-passage payoffs follow from the same
  roots, as the wealth-floor chapters show, but have no pricing function.
- The model has two regimes with jumps only at transitions; its smile is shaped by four jump and
  intensity parameters and two volatilities. A fit that needs intra-regime jumps or stochastic
  volatility belongs to [StochVolModels](https://github.com/ArturSepp/StochVolModels); conventional
  Black–Scholes and Bachelier pricing to
  [VanillaOptionPricers](https://github.com/ArturSepp/VanillaOptionPricers).
- Risk-neutral parameters are inputs. Their relation to the physical parameters of the allocation
  chapters, the jump-size and intensity premia, is the subject of the
  [variance-swap chapter](variance_swaps.md).

## See also

- [Variance swaps and the crash-size premium](variance_swaps.md)
- [Survival, densities and overshoot in the Laplace domain](laplace_barrier_framework.md)
- [Numerical Laplace inversion](laplace_inversion.md)
- [The regime-switching jump-diffusion](regime_switching_model.md)
- [Papers and research projects](papers.md)
- [Bibliography](bibliography.md)

## References

1. Sepp, A. (2004). Analytical Pricing of Double-Barrier Options under a Double-Exponential Jump Diffusion Process: Applications of Laplace Transform. *International Journal of Theoretical and Applied Finance*, 7(2), 151–175. [DOI: 10.1142/S0219024904002402](https://doi.org/10.1142/S0219024904002402). The Laplace-transform method for options under exponential jumps.
2. Sepp, A. (2026). Dynamic Mean-Variance Portfolio Allocation under Regime-Switching Jump-Diffusions with Absorbing Barriers and Distribution Matching. Working paper. [DOI: 10.2139/ssrn.6534579](https://doi.org/10.2139/ssrn.6534579). The regime-switching model and its characteristic polynomial.
3. Lewis, A. L. (2001). A Simple Option Formula for General Jump-Diffusion and Other Exponential Lévy Processes. Working paper. [DOI: 10.2139/ssrn.282110](https://doi.org/10.2139/ssrn.282110). The Fourier formula of the reference pricer.
4. Carr, P., and Madan, D. B. (1999). Option Valuation Using the Fast Fourier Transform. *Journal of Computational Finance*, 2(4), 61–73. [DOI: 10.21314/JCF.1999.043](https://doi.org/10.21314/JCF.1999.043). The damped Fourier method and its moment condition.
5. Black, F., and Scholes, M. (1973). The Pricing of Options and Corporate Liabilities. *Journal of Political Economy*, 81(3), 637–654. [DOI: 10.1086/260062](https://doi.org/10.1086/260062). The limit without jumps and implied volatility.
6. Kou, S. G. (2002). A Jump-Diffusion Model for Option Pricing. *Management Science*, 48(8), 1086–1101. [DOI: 10.1287/mnsc.48.8.1086.166](https://doi.org/10.1287/mnsc.48.8.1086.166). Option pricing with exponential jumps.
7. Sepp, A. goal-based-allocation: Semi-analytical dynamic mean-variance allocation and terminal-wealth risk under regime-switching jump-diffusions. [Software citation metadata](https://github.com/ArturSepp/GoalBasedAllocation/blob/main/CITATION.cff).
