# European option pricing under regime switching

Option pricing is a secondary workflow that reuses the package's two-regime jump-diffusion and
Laplace inversion. It supports European calls and puts, both starting regimes, scalar or joint
strike arrays, and Black-Scholes implied volatility.

## Public entry points

- `RiskNeutralParams` defines diffusion volatilities, transition intensities, jump parameters,
  and the continuously compounded rate.
- `RiskNeutralParams.from_rates` accepts exponential jump **rates**. The reciprocal is the mean
  magnitude.
- `Regime` and `OptionType` avoid ambiguous regime/payoff strings.
- `price_vanilla` prices all supplied strikes through one maturity inversion.
- `implied_vol` maps prices to Black-Scholes implied volatility.

The source example
[`examples/regime_switch_smile.py`](https://github.com/ArturSepp/GoalBasedAllocation/blob/main/examples/regime_switch_smile.py)
compares the Laplace price with independent Fourier and Monte Carlo references.

## Boundary

This is not a general exotic-pricing library. It does not add path-dependent payoffs, a market
data/calibration service, or alternative stochastic-volatility models. For conventional vanilla
option models and fitters, see
[`vanilla-option-pricers`](https://github.com/ArturSepp/VanillaOptionPricers); for stochastic
volatility analytics, see [`stochvolmodels`](https://github.com/ArturSepp/StochVolModels).

## Validation

Tests cover put-call parity, Black-Scholes limits, strike monotonicity, scalar/array consistency,
input validation, Fourier agreement, and a seeded Monte Carlo cross-check. See
[validation](../validation.md) for the role of each reference.

API: [vanilla option pricer](../api/index.md).
