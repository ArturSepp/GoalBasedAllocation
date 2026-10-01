---
myst:
  html_meta:
    description: >-
      Variance swaps under the two-regime jump-diffusion in goal-based-allocation: expected
      occupation times, the closed-form fair strike and its diffusion and jump decomposition, the
      log-contract strike and the jump skew gap, the variance risk premium, and the crash size
      implied by one variance-swap quote, checked by exact simulation and static replication.
---

# Variance swaps and the crash-size premium

*Author: [Artur Sepp](https://github.com/ArturSepp)*

Implemented in [GoalBasedAllocation](https://github.com/ArturSepp/GoalBasedAllocation).
Software citation: [CITATION.cff](https://github.com/ArturSepp/GoalBasedAllocation/blob/main/CITATION.cff).

A variance swap pays the realised variance of the log-price over its life against a fixed strike.
Under the regime-switching jump-diffusion the fair strike is a closed form: the expected time spent in
each regime weights the regime's diffusion variance and the second moment of the jump that leaves it.
The strike is therefore affine in the compound jump moments $\lambda\eta^2$, which makes one quote a
linear equation for the risk-neutral crash size. This chapter derives the strike, its log-contract
counterpart, the variance risk premium and the implied crash size, and checks each against an
independent computation, including static replication with the package's own option prices.

## Overview

The variance-swap module complements the [option pricer](european_options.md). Where an option price
needs a Laplace inversion, the variance-swap strike needs only the two expected occupation times of the
regime chain, which are elementary. Its decomposition into diffusion, crash and recovery parts shows
how much of the fair variance the jumps carry, and comparing strikes under physical and risk-neutral
parameters splits the variance risk premium by source.

The module also inverts the strike: holding the intensities and the recovery size at their physical
values, one variance-swap quote identifies the risk-neutral mean crash size in closed form, and the
put skew implied by that single parameter is a prediction that can be checked against option quotes.

## Inputs, notation, and assumptions

| Convention | This article |
|---|---|
| Time and rates | Maturity in years; strikes are annualised variances, $\mathrm{as\_vol}$ returns their square root |
| Regimes | The starting regime is an input; occupation times condition on it |
| Jump sizes | Mean convention; $\mathbb{E}[(J^{[i]})^2] = 2(\eta^{[i]})^2$ |
| Wealth coordinate | Log-price $X = \ln S$; realised variance is the quadratic variation of $X$ divided by the maturity |
| Measure | Risk-neutral for strikes; the same container holds physical parameters for the premium, whose drift does not enter the quadratic variation |
| Numerical method | Closed forms; checked against a matrix-exponential integral, exact simulation of quadratic variation and static replication with `price_vanilla` |
| Package default | `variance_swap_strike(params, ttm, regime=Regime.GROWTH, convention=VarianceConvention.QUADRATIC_VARIATION, as_vol=False)`; `implied_crash_size_from_var_swap(..., enforce_crash_premium_sign=True)` |

| Symbol | Meaning | Units and convention |
|---|---|---|
| $\mathcal{T}^{[1]}$, $\mathcal{T}^{[2]}$ | Expected time in growth and in stress over $[0, T]$ | Years, sum to $T$ |
| $[X]_T$ | Quadratic variation of the log-price | Log units squared |
| $K_{\mathrm{var}}$ | Fair quadratic-variation strike $\mathbb{E}^{Q}[[X]_T]/T$ | Per year |
| $K_{\mathrm{log}}$ | Log-contract strike $(2/T)\mathbb{E}^{Q}[\int_0^T dS/S - \ln(S_T/S_0)]$ | Per year |
| $\Lambda_{\mathrm{s}}$ | $\lambda^{[12]} + \lambda^{[21]}$ | Per year |
| $\eta_Q^{[1]}$, $\eta_P^{[1]}$ | Risk-neutral and physical mean crash size | Log units |

The regime parameters follow the [conventions page](conventions.md#reserved-notation). The code calls
the stationary probabilities `pi_0` and `pi_1`; they are $p_1$ and $p_2$ here.

## Methodology

### Expected occupation times

**Proposition (occupation times).** From growth, with $p_1$ and $p_2$ the stationary probabilities,

$$
\mathcal{T}^{[1]} = p_1T + p_2\frac{1 - e^{-\Lambda_{\mathrm{s}}T}}{\Lambda_{\mathrm{s}}}, \qquad \mathcal{T}^{[2]} = T - \mathcal{T}^{[1]},
$$

and from stress the roles of the regimes are exchanged.

**Proof.** The probability of being in growth at time $t$ from growth is
$p_1 + p_2e^{-\Lambda_{\mathrm{s}}t}$, the solution of the forward equation of a two-state chain, whose
generator has eigenvalues $0$ and $-\Lambda_{\mathrm{s}}$. Integrate over $[0, T]$. $\square$

As $T$ grows, $\mathcal{T}^{[i]}/T$ tends to the stationary probability. The vector of occupation times
is the starting row of $\int_0^Te^{\Lambda t}dt$, which the worked example computes independently.

### The fair strike and its decomposition

**Proposition (fair variance-swap strike).** The fair quadratic-variation strike is

$$
K_{\mathrm{var}} = \frac{1}{T}\left[(\sigma^{[1]})^2\mathcal{T}^{[1]} + (\sigma^{[2]})^2\mathcal{T}^{[2]} + 2\lambda^{[12]}(\eta^{[1]})^2\mathcal{T}^{[1]} + 2\lambda^{[21]}(\eta^{[2]})^2\mathcal{T}^{[2]}\right] .
$$

**Proof.** The quadratic variation of $X$ is $\int_0^T(\sigma^{[\chi_t]})^2dt + \sum(\Delta X)^2$ over the
jumps. The expectation of the first term is the diffusion part. Jumps out of regime $i$ arrive at rate
$\lambda^{[i \to j]}$ while the chain is in $i$, so the expected sum of their squares is
$\lambda^{[i \to j]}\mathcal{T}^{[i]}\mathbb{E}[(J^{[i]})^2]$, and the second moment of an exponential with
mean $\eta$ is $2\eta^2$. $\square$

The jump part is affine in the compound second moments $\lambda^{[12]}(\eta^{[1]})^2$ and
$\lambda^{[21]}(\eta^{[2]})^2$: a variance swap sees the products, never intensity and size
separately.

### The log contract and the jump skew gap

**Proposition (log-contract strike).** The strike replicated by a static position in options and a
dynamic position in the asset is

$$
K_{\mathrm{log}} = \frac{2}{T}\left(rT - \mathbb{E}^{Q}[X_T - X_0]\right), \qquad
\mathbb{E}^{Q}[X_T - X_0] = \nu^{[1]}\mathcal{T}^{[1]} + \nu^{[2]}\mathcal{T}^{[2]} - \lambda^{[12]}\eta^{[1]}\mathcal{T}^{[1]} + \lambda^{[21]}\eta^{[2]}\mathcal{T}^{[2]} .
$$

**Proof.** Under the risk-neutral measure $\int_0^TdS/S$, including the relative jumps, has expectation
$rT$. The expected log-return is the expected diffusion drift plus the expected jumps, with
$\mathbb{E}[J^{[1]}] = -\eta^{[1]}$ and $\mathbb{E}[J^{[2]}] = \eta^{[2]}$. $\square$

With continuous paths $K_{\mathrm{log}} = K_{\mathrm{var}}$. Each jump contributes
$2(e^{\Delta X} - 1 - \Delta X)$ to the log contract instead of $(\Delta X)^2$, so

$$
K_{\mathrm{log}} - K_{\mathrm{var}} = \frac{2}{T}\sum_{i}\lambda^{[i \to j]}\mathcal{T}^{[i]}\thinspace \mathbb{E}\left[e^{J^{[i]}} - 1 - J^{[i]} - \frac{1}{2}(J^{[i]})^2\right],
$$

a pure jump quantity led by the third moment (Demeterfi et al., 1999; Carr and Wu, 2009). For
exponential jumps the expectations are $1/(1 + \eta^{[1]}) - 1 + \eta^{[1]} - (\eta^{[1]})^2$ for the crash
and $1/(1 - \eta^{[2]}) - 1 - \eta^{[2]} - (\eta^{[2]})^2$ for the recovery. The gap is negative when crashes
dominate, which makes it a probe of the crash magnitude.

### The variance risk premium and the implied crash size

**Definition.** The variance risk premium is $K_{\mathrm{var}}^{Q} - K_{\mathrm{var}}^{P}$, split into a
diffusion part from the volatilities and a jump part from the compound moments. With equal
volatilities under both measures it is all jump premium, intensity and size entangled.

**Proposition (crash size implied by one quote).** Holding the volatilities, both intensities and the
recovery size at their physical values, a quadratic-variation strike $K$ identifies

$$
\eta_Q^{[1]} = \sqrt{\frac{TK - (\sigma^{[1]})^2\mathcal{T}^{[1]} - (\sigma^{[2]})^2\mathcal{T}^{[2]} - 2\lambda^{[21]}(\eta^{[2]})^2\mathcal{T}^{[2]}}{2\lambda^{[12]}\mathcal{T}^{[1]}}},
$$

which exists exactly when $K$ exceeds the jump-free floor
$((\sigma^{[1]})^2\mathcal{T}^{[1]} + (\sigma^{[2]})^2\mathcal{T}^{[2]} + 2\lambda^{[21]}(\eta^{[2]})^2\mathcal{T}^{[2]})/T$.

**Proof.** Solve the affine strike formula for the only free term. $\square$

> **Pitfall.** The normalisation $\lambda_Q = \lambda_P$ attributes the whole crash premium to its size.
> If the market also prices a higher crash intensity, $\eta_Q^{[1]}$ absorbs it: prices are unaffected,
> because they depend on $\lambda(\eta)^2$, but the size premium ratio $\eta_Q^{[1]}/\eta_P^{[1]}$ then
> overstates the size premium. Read it as a total crash premium in size units.

The calibration leaves no free parameter, so the put skew of the calibrated model is a prediction.
`skew_overidentification_test` returns the put implied volatilities under the calibrated and the
physical parameters; a model skew systematically flatter than the market's indicates that the
normalisation is too tight.

## Worked example

The physical parameters are the equity regime process of Table 1 with $r = 2$%; the risk-neutral
parameters raise the mean crash size from $1/3$ to 0.4 and keep everything else. The first block checks
the occupation times against $\int_0^Te^{\Lambda t}dt$ from an augmented matrix exponential, for one and
ten years from both regimes, and then evaluates the one-year swap from growth:

- the physical strike is 0.04615, a fair volatility of 21.48%, of which 49.1% comes from jumps,
  against an at-the-money implied volatility of 17.68% in the [option chapter](european_options.md#worked-example);
- an exact simulation of the quadratic variation, without a time grid, gives 0.04574 with a standard
  error of 0.00038;
- the log-contract strike is 0.040976, and replicating it statically from 6,000 put and call prices of
  `price_vanilla` gives the same value within $4 \times 10^{-7}$; the jump skew gap of $-0.005174$ equals
  its closed form;
- the risk-neutral strike is 0.05558, a variance risk premium of 0.00943, all of it from jumps, and a
  ratio of 1.204;
- inverting the risk-neutral strike recovers $\eta_Q^{[1]} = 0.4$, a size premium ratio of 1.2, and
  jumps carry 57.7% of the risk-neutral variance;
- under the calibrated parameters the one-year put implied volatility at a strike of 70 is 27.4%,
  against 25.8% under the physical parameters.

```python
import dataclasses
import numpy as np
from scipy.linalg import expm
import goal_based_allocation as gba
from goal_based_allocation.variance_swap import variance_swap_strike_mc

equity = gba.create_paper_assets()['equity'].params
physical = gba.RiskNeutralParams(sigma_0=equity.sigma0, sigma_1=equity.sigma1,
                                 lambda_01=equity.lambda01, lambda_10=equity.lambda10,
                                 eta_0=equity.eta0, eta_1=equity.eta1, rate=0.02)
risk_neutral = dataclasses.replace(physical, eta_0=0.4)

# occupation times against the integral of the matrix exponential of the generator
generator = np.array([[-physical.lambda_01, physical.lambda_01], [physical.lambda_10, -physical.lambda_10]])
augmented = np.zeros((4, 4))
augmented[:2, :2], augmented[:2, 2:] = generator, np.eye(2)
for ttm in (1.0, 10.0):
    integral = expm(augmented * ttm)[:2, 2:]
    for regime in (gba.Regime.GROWTH, gba.Regime.STRESS):
        np.testing.assert_allclose(gba.occupation_times(physical, ttm, regime), integral[int(regime)], rtol=1e-12)

# one-year swap from growth: decomposition, fair volatility and exact simulation
ttm = 1.0
parts = gba.decompose_variance(physical, ttm)
strike = gba.variance_swap_strike(physical, ttm)
np.testing.assert_allclose(strike, parts.diffusion + parts.jump_crash + parts.jump_recovery, rtol=1e-12)
np.testing.assert_allclose([strike, parts.fair_vol, parts.jump_fraction], [0.04615, 0.2148, 0.4907], atol=5e-5)
np.testing.assert_allclose(gba.variance_swap_strike(physical, ttm, as_vol=True), parts.fair_vol, rtol=1e-12)
simulated, standard_error = variance_swap_strike_mc(physical, ttm, n_paths=200_000, seed=11)
np.testing.assert_allclose([simulated, standard_error], [0.04574, 0.00038], atol=5e-6)
assert abs(simulated - strike) < 3.0 * standard_error

# log contract: closed form, jump skew gap, and static replication from option prices
log_strike = gba.variance_swap_strike(physical, ttm, convention=gba.VarianceConvention.LOG_CONTRACT)
growth_time, stress_time = gba.occupation_times(physical, ttm)
cumulant_crash = 1.0 / (1.0 + physical.eta_0) - 1.0 + physical.eta_0 - physical.eta_0**2
cumulant_recovery = 1.0 / (1.0 - physical.eta_1) - 1.0 - physical.eta_1 - physical.eta_1**2
gap = 2.0 * (physical.lambda_01 * growth_time * cumulant_crash
             + physical.lambda_10 * stress_time * cumulant_recovery) / ttm
np.testing.assert_allclose(gba.jump_skew_gap(physical, ttm), gap, rtol=1e-10)
np.testing.assert_allclose([log_strike, gap], [0.040976, -0.005174], atol=5e-7)
forward = 100.0 * np.exp(physical.rate * ttm)
put_strikes, call_strikes = np.geomspace(0.5, forward, 3000), np.geomspace(forward, 5000.0, 3000)
puts = gba.price_vanilla(physical, 100.0, put_strikes, ttm, option_type='put')
calls = gba.price_vanilla(physical, 100.0, call_strikes, ttm, option_type='call')
replicated = 2.0 * np.exp(physical.rate * ttm) / ttm * (np.trapezoid(puts / put_strikes**2, put_strikes)
                                                        + np.trapezoid(calls / call_strikes**2, call_strikes))
assert abs(replicated - log_strike) < 4e-7

# variance risk premium: equal volatilities, so the premium is all jump premium
premium = gba.variance_risk_premium(physical, risk_neutral, ttm)
np.testing.assert_allclose([premium.risk_neutral_var, premium.total], [0.05558, 0.00943], atol=5e-6)
np.testing.assert_allclose(premium.ratio, 1.204, atol=5e-4)
assert premium.diffusion == 0.0 and abs(premium.jump - premium.total) < 1e-15

# one quote identifies the crash size, and the skew it implies is a prediction
calibration = gba.implied_crash_size_from_var_swap(physical, premium.risk_neutral_var, ttm)
np.testing.assert_allclose([calibration.eta_0_q, calibration.size_premium_ratio], [0.4, 1.2], rtol=1e-12)
np.testing.assert_allclose(calibration.jump_variance_share, 0.577, atol=5e-4)
skew = gba.skew_overidentification_test(physical, calibration, 100.0, np.array([70.0, 85.0, 100.0]), ttm)
np.testing.assert_allclose(skew[0], [0.2741, 0.2581], atol=5e-5)
assert np.all(skew[:, 0] > skew[:, 1])

# a quote below the jump-free floor has no non-negative crash size
try:
    gba.implied_crash_size_from_var_swap(physical, 0.01, ttm)
    raise AssertionError('expected a ValueError')
except ValueError as error:
    assert 'jump-free floor' in str(error)
```

## Implementation in goal_based_allocation

| Quantity | Formula | Entry point |
|---|---|---|
| Occupation times | $(\mathcal{T}^{[1]}, \mathcal{T}^{[2]})$ | `occupation_times(params, ttm, regime=Regime.GROWTH)` |
| Strike decomposition | Diffusion, crash and recovery parts | `decompose_variance(params, ttm, regime=Regime.GROWTH)` returns `VarianceDecomposition` |
| Fair strike | $K_{\mathrm{var}}$ or $K_{\mathrm{log}}$ | `variance_swap_strike(params, ttm, regime, convention, as_vol=False)` |
| Jump skew gap | $K_{\mathrm{log}} - K_{\mathrm{var}}$ | `jump_skew_gap(params, ttm, regime=Regime.GROWTH)` |
| Variance risk premium | $K_{\mathrm{var}}^{Q} - K_{\mathrm{var}}^{P}$ by source | `variance_risk_premium(params_p, params_q, ttm, regime)` returns `VarianceRiskPremium` |
| Implied crash size | $\eta_Q^{[1]}$ from one quote | `implied_crash_size_from_var_swap(params_p, market_var_strike, ttm, regime, enforce_crash_premium_sign=True)` returns `SizePremiumCalibration` |
| Predicted skew | Put implied volatilities under $Q$ and $P$ | `skew_overidentification_test(params_p, calibration, spot, strikes, ttm, regime)` |
| Exact simulation | Quadratic variation without a time grid | `variance_swap.variance_swap_strike_mc(params, ttm, regime, convention, n_paths=200_000, seed=11)` (module level) |

The module lives in
[`variance_swap.py`](https://github.com/ArturSepp/GoalBasedAllocation/blob/main/src/goal_based_allocation/variance_swap.py).
API pages: {doc}`variance_swap_strike <api/generated/goal_based_allocation.variance_swap_strike>`,
{doc}`decompose_variance <api/generated/goal_based_allocation.decompose_variance>` and
{doc}`implied_crash_size_from_var_swap <api/generated/goal_based_allocation.implied_crash_size_from_var_swap>`.

Contract details:

- Every function takes a `RiskNeutralParams`, also for physical parameters; only the volatilities,
  intensities and jump means enter the quadratic variation, while the log contract also uses the
  drifts implied by the rate.
- `VarianceDecomposition` exposes `diffusion`, `jump_crash`, `jump_recovery`, `jump_total`, `total`,
  `fair_vol` and `jump_fraction`; `VarianceRiskPremium` exposes `total`, `diffusion`, `jump`,
  `physical_var`, `risk_neutral_var` and `ratio`; `SizePremiumCalibration` exposes `eta_0_q`, the full
  `params_q`, `size_premium_ratio` and `jump_variance_share`.
- `occupation_times` raises `ValueError` for a non-positive maturity or a regime other than 0 or 1.
- `implied_crash_size_from_var_swap` raises `ValueError` for a non-positive quote, for a quote below
  the jump-free floor, and, with `enforce_crash_premium_sign=True`, for an implied crash size below the
  physical one.
- `skew_overidentification_test` prices puts with `implied_vol` and returns an array of shape
  `(len(strikes), 2)`: calibrated risk-neutral volatilities first, physical second.

## Interpretation and limitations

- The strike is the expected quadratic variation of a continuously monitored log-price. Traded
  variance swaps sample daily closes and may cap the payoff; neither is modelled.
- The variance swap identifies compound moments $\lambda\eta^2$ only. Separating intensity from size
  needs an estimate of the physical intensity, which is why the calibration fixes it.
- The decomposition attributes all diffusion variance to the two regime volatilities. A volatility
  premium, $\sigma_Q \ne \sigma_P$, appears in the diffusion part of the premium; the size calibration
  assumes there is none.
- The log-contract replication in the worked example truncates the strike integral at 0.5 and 5,000;
  for longer maturities or heavier jumps the truncation must widen.

## See also

- [European options under regime switching](european_options.md)
- [The regime-switching jump-diffusion](regime_switching_model.md)
- [Buy-and-hold moments](buy_and_hold_moments.md)
- [Papers and research projects](papers.md)
- [Bibliography](bibliography.md)

## References

1. Demeterfi, K., Derman, E., Kamal, M., and Zou, J. (1999). A Guide to Volatility and Variance Swaps. *The Journal of Derivatives*, 6(4), 9–32. [DOI: 10.3905/jod.1999.319129](https://doi.org/10.3905/jod.1999.319129). Replication of the log contract with options and the effect of jumps.
2. Carr, P., and Wu, L. (2009). Variance Risk Premiums. *The Review of Financial Studies*, 22(3), 1311–1341. [DOI: 10.1093/rfs/hhn038](https://doi.org/10.1093/rfs/hhn038). Synthetic variance-swap rates and the variance risk premium.
3. Sepp, A. (2026). Dynamic Mean-Variance Portfolio Allocation under Regime-Switching Jump-Diffusions with Absorbing Barriers and Distribution Matching. Working paper. [DOI: 10.2139/ssrn.6534579](https://doi.org/10.2139/ssrn.6534579). The regime-switching model with jumps at transitions.
4. Sepp, A. goal-based-allocation: Semi-analytical dynamic mean-variance allocation and terminal-wealth risk under regime-switching jump-diffusions. [Software citation metadata](https://github.com/ArturSepp/GoalBasedAllocation/blob/main/CITATION.cff).
