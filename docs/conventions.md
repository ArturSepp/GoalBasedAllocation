---
myst:
  html_meta:
    description: >-
      Notation and conventions of the goal-based-allocation handbook: the seven-row convention
      card, time and rate units, regime numbering in the paper and the code, the mean convention
      for exponential jumps, wealth and log-gap coordinates, reserved symbols and numerical
      defaults.
---

# Notation and conventions

*Author: [Artur Sepp](https://github.com/ArturSepp)*

Project: [GoalBasedAllocation](https://github.com/ArturSepp/GoalBasedAllocation).
Software citation: [CITATION.cff](https://github.com/ArturSepp/GoalBasedAllocation/blob/main/CITATION.cff).

Every chapter of the handbook states its conventions in a seven-row card and uses the symbols of
this page with one meaning throughout. Read this page before comparing a number with the
manuscript of Sepp (2026), with another model, or with the output of another package: the paper
numbers regimes from one and the code from zero, jump sizes are exponential means rather than
rates, and several functions fix the horizon, the initial wealth and the riskless rate internally.

## The convention card

The first table under *Inputs, notation, and assumptions* of every methodology chapter has the
header `| Convention | This article |` and exactly these rows, in this order.

| Row | What it states |
|---|---|
| Time and rates | The unit of time, the compounding of rates, and whether a coefficient is indexed by calendar time $t$ or by time to horizon $\tau = T - t$ |
| Regimes | How the regimes are numbered on the page and in the code, which starting regime a result conditions on, and whether a result averages over the stationary regime probabilities |
| Jump sizes | Whether jump parameters are exponential means or rates, and how they map to the crash loss and the recovery gain quoted as relative price changes |
| Wealth coordinate | The variable a density or a grid is expressed in: wealth $\Pi$, log-price, gap $Z$, log-gap $X$ or overshoot distance $d$, and the scale of initial wealth |
| Measure | Physical parameters, used for allocation and the terminal wealth distribution, or risk-neutral parameters, used for option and variance-swap prices |
| Numerical method | The inversion, ODE, quadrature or grid integration that computes the result, with its settings, and the independent method that validates it |
| Package default | The defaults of the functions on the page that change a reported number |

## Time, rates, and wealth

- Time is measured in years. The horizon is $T$, calendar time is $t \in [0, T]$, and the Riccati
  coefficients are indexed by the time to horizon $\tau = T - t$, so $\tau = 0$ is the horizon
  and $\tau = T$ is inception. `RiccatiSolution.derived_at_tau(T)` returns the quantities at
  $t = 0$.
- Rates, target returns, consumption, drifts and transition intensities are annual quantities.
  Rates are continuously compounded: cash grows as $e^{rT}$ and an implied return is
  $\ln(\mathbb{E}[\Pi_T]/\Pi_0)/T$.
- Volatilities are annualised diffusion volatilities of log-prices.
- Wealth is in arbitrary consistent units. The paper and every packaged mandate use initial
  wealth $\Pi_0 = 100$; standard deviations of terminal wealth are in the same units, not in
  return units.

## Regimes and transitions

- The paper numbers the regimes 1 (growth) and 2 (stress); the code numbers them 0 and 1, as in
  `Regime.GROWTH = 0` and `Regime.STRESS = 1`. The handbook uses the paper's numbers in formulas,
  $\lambda^{[12]}$ for the crash intensity and $\lambda^{[21]}$ for the recovery intensity, and the
  code's names in code: `lambda01` and `lambda10` in `RegimeSwitchParams`, `lambda_01` and
  `lambda_10` in `RiskNeutralParams`.
- Intensities are rates per year; their reciprocals are the mean dwell times. The paper's
  intensities $\lambda^{[12]} = 0.1$ and $\lambda^{[21]} = 1$ give one crash per decade on average
  and stress periods of one year.
- The stationary regime probabilities are
  $p_1 = \lambda^{[21]}/(\lambda^{[12]} + \lambda^{[21]})$ and $p_2 = 1 - p_1$, which are
  $10/11$ and $1/11$ for the paper's intensities.
- The Laplace-domain analytics and the opportunity set condition on starting in growth.
  `bh_moments_rsjd` reports the regime-conditional means and the moments averaged with the
  stationary probabilities. Each chapter's card states which one it uses.

## Jump parameters

Jumps occur only at regime transitions and their magnitudes are exponential. The package uses the
**mean convention**: $\eta^{[1]}$ is the mean size of the crash log-jump and $\eta^{[2]}$ the mean
size of the recovery log-jump,

$$
J^{[1]} = -E^{[1]}, \qquad J^{[2]} = E^{[2]}, \qquad E^{[i]} \sim \mathrm{Exp}\ \text{with mean}\ \eta^{[i]} .
$$

The paper writes $\mathrm{Exp}(1/\eta^{[i]})$, an exponential law with rate $1/\eta^{[i]}$. The
crash loss and the recovery gain quoted as relative price changes are

$$
1 - \mathbb{E}\left[e^{J^{[1]}}\right] = \frac{\eta^{[1]}}{1 + \eta^{[1]}}, \qquad
\mathbb{E}\left[e^{J^{[2]}}\right] - 1 = \frac{\eta^{[2]}}{1 - \eta^{[2]}},
$$

so a 25% crash loss is $\eta^{[1]} = 0.25/0.75 = 1/3$. `RiskNeutralParams.from_rates` is the only
entry point in the **rate convention**: its `eta_01` and `eta_10` are the rates $1/\eta^{[1]}$ and
$1/\eta^{[2]}$. Passing a mean where a rate is expected inverts the jump size without an error.

## Floor and survival

- The floor is $L_t = L_0 e^{r_c t}$ and is absorbing: once wealth reaches it, the strategy holds
  cash at the net floor rate $r_c$ until the horizon.
- A diffusion path reaches the floor continuously and ends at $L_T$, the **floor atom**. A crash
  jump can carry wealth through the floor, which ends the path below $L_T$, the **jump
  overshoot**. Survival means that the path has not been stopped by the horizon.
- Terminal probability therefore splits into survival $Q$, floor atom $F$ and overshoot mass
  $\mathcal{O}$ with $Q + F + \mathcal{O} = 1$. The package computes $Q$ and $\mathcal{O}$ and
  obtains $F$ as the remainder.

## Returns and moments

- `r_impl` is the continuously compounded annual return implied by expected terminal wealth,
  $\ln(\mathbb{E}[\Pi_T]/\Pi_0)/T$. It is not a median or a mode.
- The hurdle rate is $r_h = \max(r, c)$ and the net floor rate is $r_c = r_h - c = \max(0, r - c)$.
  Risk premia in the Riccati system are measured against $r_h$; cash and the floor grow at $r_c$.
- Buy-and-hold moments are exact under the same two-regime jump-diffusion and are computed with a
  $2 \times 2$ matrix exponential; see [buy-and-hold moments](buy_and_hold_moments.md).

## Reserved notation

These symbols keep one meaning in every chapter. A chapter declares any other symbol it uses and
does not reuse one of these.

| Symbol | Meaning | Code name |
|---|---|---|
| $t$, $T$, $\tau$ | Calendar time, horizon, time to horizon $T - t$ | `t`, `T`, `tau` |
| $\chi_t$ | Regime, 1 (growth) or 2 (stress) | `regime`, `state_init` (0 or 1) |
| $\lambda^{[12]}$, $\lambda^{[21]}$ | Crash and recovery intensities per year | `lambda01`, `lambda10` |
| $p_1$, $p_2$ | Stationary regime probabilities | `p1`, `p2` |
| $\sigma^{[i]}$ | Diffusion volatility in regime $i$ | `sigma0`, `sigma1` |
| $\bar\mu^{[i]}$ | Total expected return in regime $i$, a capital market assumption | `mu_growth`, `mu_stress` |
| $\mu^{[i]}$ | Diffusion drift between transitions | `mu_bar` |
| $\nu^{[i]}$ | Log drift $\mu^{[i]} - (\sigma^{[i]})^2/2$ | `nu0`, `nu1` |
| $\eta^{[1]}$, $\eta^{[2]}$ | Mean crash and recovery log-jump sizes | `eta0`, `eta1` |
| $\alpha^{[i]}$ | Jump compensator $\mathbb{E}[e^{J^{[i]}}] - 1$ | `alpha` |
| $r$, $c$ | Riskless rate and proportional consumption rate | `r`, `c` |
| $r_h$, $r_c$ | Hurdle rate and net floor rate | `r_h`, `r_c` |
| $\Pi_t$, $\Pi_0$ | Wealth and initial wealth | `Pi`, `pi0` |
| $\omega_t$ | Fraction of wealth in the risky asset | `omega` |
| $\ell$ | Lagrange multiplier of the mean-variance problem | `ell` |
| $a^{[i]}$, $b^{[i]}$, $\gamma^{[i]}$ | Value-function coefficients | `a`, `b`, `gamma` |
| $\tilde a^{[i]}$, $\tilde b^{[i]}$, $\Sigma^{[i]}$ | Coupled Riccati coefficients | `a_tilde`, `b_tilde`, `Sigma` |
| $\omega_a^{[i]}$ | Allocation coefficient $-\tilde a^{[i]}/\Sigma^{[i]}$ | `w_a` |
| $\Pi^{*}(t)$ | Target wealth of the policy; $\Pi_T^{*}$ at the horizon | `Pi_star`, `PiT` |
| $L_t$, $L_0$, $k$ | Floor, initial floor, floor distance in crash units | `pi_floor`, `L_T`, `k` |
| $\tau_L$ | First time wealth reaches the floor | `stop_time` |
| $Z_t$, $B_t$ | Gap $\Pi^{*}(t) - \Pi_t$ and buffer $\Pi^{*}(t) - L_t$ | `Z0`, `B0`, `B_T` |
| $X_t$, $x_0$ | State of a regime-switching jump-diffusion and its initial value | `x0` |
| $\tilde\eta^{[i]}$ | Effective mean log-jump of the gap process | `eta_eff` |
| $Q$, $\tilde Q(n)$ | Survival probability and tilted survival | `S`, `TS1`, `TS2` |
| $F$, $\mathcal{O}$ | Floor atom and overshoot mass | `F`, `O` |
| $p$, $\psi_k$ | Laplace variable and characteristic roots | `p`, `psi` |

The state $X_t$ is the log-price in the [model chapter](regime_switching_model.md) and the log-gap
$\ln(B_t/Z_t)$ from the [floor chapter](wealth_floor_gap_process.md) onwards; the Laplace
framework applies to either. The covariance matrix of asset diffusions is written
$\mathbf{C}^{[i]}$, because $\Sigma^{[i]}$ is reserved for the Riccati coefficient.

## Paper and code index

Statement and equation numbers on these pages refer to the manuscript of Sepp (2026) tracked in
[`papers/goal_based_allocation_2026/paper/`](https://github.com/ArturSepp/GoalBasedAllocation/tree/main/papers/goal_based_allocation_2026/paper),
dated 7 April 2026. Code comments in the package use LaTeX labels of the same manuscript, such as
`eq:RS_def`, rather than numbers.

| Manuscript | Subject | Implementation |
|---|---|---|
| Section 2, equations (2.1)–(2.11) | Model, compensators, hurdle rate, wealth dynamics | `RegimeSwitchParams`, `AssetSpecification` |
| Section 3, equations (3.1)–(3.16) | Mean-variance problem and Riccati system | `find_ell`, `riccati_solver.solve_riccati` |
| Section 4, equations (4.1)–(4.6) | Floor, gap process, effective jump size | `gap_process_asset` |
| Section 5, equations (5.1)–(5.16) | Arrow–Debreu prices, characteristic polynomial, survival | `compute_density`, `compute_survival`, `compute_tilted_survival` |
| Section 6, equations (6.1)–(6.24) | Terminal distribution, moments, consumption, floor cost | `compute_overshoot_density`, `compute_opportunity_point` |
| Section 7, equations (7.1)–(7.9) | Asset classes, mandate aggregation, floor calibration | `create_paper_assets`, `build_effective_asset` |
| Section 8, equations (8.1)–(8.11) | Opportunity set, investor selection, glide path | `AdvisorSpec`, `build_opportunity_set` |
| Proposition A.1 | Buy-and-hold moments | `bh_moments_rsjd` |

## Numerical defaults

| Calculation | Method and settings | Where |
|---|---|---|
| Laplace inversion | Abate–Whitt Euler summation with $N = 25$ terms, $M = 12$ Euler terms and contour parameter $10^{-8}$ | `laplace_inversion.laplace_invert_abate_whitt` |
| Riccati system | `scipy.integrate.solve_ivp`, RK45, `rtol=1e-10`, `atol=1e-12`, maximum step $T/1000$, 10,001 output points | `riccati_solver.solve_riccati` |
| Effective jump sizes | `scipy.integrate.quad` on one exponential, `nquad` on two or three, truncated at $\max(50\eta, 20)$ | `gap_process_asset`, `portfolio_eta_quadrature` |
| Terminal densities | Trapezoid rule on 600 log-gap points and 400 overshoot points | `compute_opportunity_point` |
| Quantiles | Linear interpolation of the distribution function on 1,500 wealth points | `compute_opportunity_point` |
| Full-investment calibration | Bisection on the target growth rate in $[0.001, 0.60]$, at most 80 steps, stopped within 0.001 of $\omega_0$ | `compute_opportunity_point` |

Several functions fix model constants internally rather than taking them as arguments.
`build_effective_asset`, `portfolio_sigma_unc`, `compute_opportunity_point` and
`build_opportunity_set` use $\Pi_0 = 100$, $T = 10$ years, $r = 2\%$, the asset classes of
`create_paper_assets` with intensities $\lambda^{[12]} = 0.1$ and $\lambda^{[21]} = 1$, and the
correlation matrix of equation (7.1). Their numbers apply to that configuration only; the
chapters that use them say so on the card.

## See also

- [Documentation home](index.md)
- [The regime-switching jump-diffusion](regime_switching_model.md)
- [The MV-optimal policy and the Riccati system](mv_optimal_policy.md)
- [Validation and numerical evidence](validation.md)
- [Bibliography](bibliography.md)

## References

- Sepp, A. (2026). Dynamic Mean-Variance Portfolio Allocation under Regime-Switching
  Jump-Diffusions with Absorbing Barriers and Distribution Matching. Working paper.
  [SSRN 6534579](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=6534579).
- [GoalBasedAllocation software citation](https://github.com/ArturSepp/GoalBasedAllocation/blob/main/CITATION.cff).
