# Conventions

Read these conventions before comparing outputs with another model or paper.

## Time, rates, and wealth

- Time is measured in years.
- Rates, target returns, consumption, drifts, and transition intensities are annual quantities.
- Continuous compounding is used where wealth grows as $e^{rT}$.
- Wealth is in arbitrary consistent units. Paper examples use initial wealth $\Pi_0=100$.
- Volatility is an annualised diffusion volatility.

## Regimes and transitions

- Regime 0 is growth and regime 1 is stress in option-pricing APIs.
- $\lambda_{01}$ is the growth-to-stress intensity; $\lambda_{10}$ is stress-to-growth.
- Intensities are rates per year; their inverses are mean dwell times when positive.
- The allocation paper also writes the regimes as 1 and 2. Use the paper-to-code table in the
  repository README when mapping notation.

## Jump parameters

The package distinguishes exponential **rate** and **mean** conventions. In
`RiskNeutralParams.from_rates`, `eta_01` and `eta_10` are exponential rates, so mean magnitude is
their reciprocal. Lower-level model objects may store the mean convention. Do not pass a mean to a
rate parameter without inversion.

## Floor and survival

- The floor is absorbing for diffusion hits: after reaching it, wealth is converted to cash.
- A crash jump can cross the barrier discontinuously. This creates a jump-overshoot density below
  the floor rather than placing all stopped mass at the floor.
- Terminal probability decomposes into survived density, floor atom, and jump overshoot.
- “Survival” means the model path has not stopped at the absorbing barrier by the stated horizon.

## Returns and moments

- `r_impl` is the continuously compounded annual return implied by expected terminal wealth:
  $\log(E[\Pi_T]/\Pi_0)/T$.
- Standard deviations are in terminal-wealth units unless explicitly labelled as annual
  volatility.
- Buy-and-hold moments are exact under the same two-state regime-switching jump-diffusion and are
  computed by a 2×2 matrix exponential.

See [validation](validation.md) for numerical tolerances and independent checks.
