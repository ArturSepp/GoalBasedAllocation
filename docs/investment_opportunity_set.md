---
myst:
  html_meta:
    description: >-
      The investment opportunity set of goal-based-allocation: the two-step advisor and investor
      framework, full-investment calibration of the mean-variance policy, implied return along the
      bond-weight curve, discounted consumption and the floor protection cost, and the expected
      de-risking glide path, reproducing Table 2 and Figures 4 and 6 of the manuscript.
---

# The investment opportunity set and investor selection

*Author: [Artur Sepp](https://github.com/ArturSepp)*

Implemented in [GoalBasedAllocation](https://github.com/ArturSepp/GoalBasedAllocation).
Software citation: [CITATION.cff](https://github.com/ArturSepp/GoalBasedAllocation/blob/main/CITATION.cff).

The investment opportunity set is the curve of floor-protected mandates that an advisor can offer once
four preferences are fixed: the initial risky allocation, the consumption rate, the split between
public and private equity, and the drawdown tolerance. Along the curve the bond weight is the only free
variable, every point has an analytically computed terminal wealth distribution, and the implied
return rises as the bond weight falls, so the investor's choice reduces to one number (Sepp, 2026,
Section 8). This chapter states the construction, shows that its calibration has a closed form, and
reproduces the mandates of Table 2 with their floor protection costs and glide paths.

## Overview

The construction has two steps. In the first, the advisor fixes `AdvisorSpec(omega_0, c, q, q_dd)`,
and `compute_opportunity_point(w_bd, spec)` evaluates one mandate: it builds the
[effective asset](mandate_aggregation.md), sets the floor from the drawdown tolerance, calibrates the
[mean-variance policy](mv_optimal_policy.md) to start with the allocation `omega_0`, and returns the
[terminal distribution](terminal_wealth_distribution.md), its implied return, a
[buy-and-hold](buy_and_hold_moments.md) comparison and the floor protection cost.
`build_opportunity_set(spec)` repeats this across bond weights. In the second step, the investor picks
an implied return on the curve, which fixes the mandate, its floor, its target and its distribution.

The quickstart of the package evaluates one point of this curve, the balanced mandate. The worked
example reproduces it, the income and growth mandates of Table 2, and the same balanced mandate with
consumption of 2.5% a year.

## Inputs, notation, and assumptions

| Convention | This article |
|---|---|
| Time and rates | Fixed inside the package: $T = 10$ years and $r = 2$%; $c$ from the advisor; $r_h = \max(r, c)$ and $r_c = r_h - c$; implied returns continuously compounded |
| Regimes | The policy, survival and expected wealth start in growth; the buy-and-hold benchmark averages over the stationary starting regime |
| Jump sizes | Mean-matched jumps of the effective asset and effective jumps of its gap process |
| Wealth coordinate | Initial wealth fixed at 100; floor $L_T = 100e^{-q_{\mathrm{dd}}\sigma_{\mathrm{unc}}}$ at the horizon |
| Measure | Physical |
| Numerical method | Bisection of the policy's target growth rate to the initial allocation, Laplace building blocks, trapezoidal integration of densities and of consumption on 20 interior dates |
| Package default | `AdvisorSpec(omega_0=1.0, c=0.0, q=2/3, q_dd=2.0)`; `build_opportunity_set(spec, w_vals=None)` uses bond weights from 0.95 to 0 in steps of 0.05 |

| Symbol | Meaning | Units and convention |
|---|---|---|
| $\omega_0$ | Initial risky allocation $\omega_0^{*}$ of the policy | Fraction of wealth, `omega_0` |
| $w$ | Bond weight, the free variable of the curve | Fraction, `w_bd` |
| $g$ | Target growth rate $\ln(\Pi_T^{*}/\Pi_0)/T$ | Per year, `g` |
| $\rho$ | Present-value growth rate $\ln(\Pi^{*}(0)/\Pi_0)/T$, the `target_return` of `find_ell` | Per year |
| $C_{\mathrm{disc}}$ | Discounted consumption $c\int_0^T\mathbb{E}[\Pi_t]e^{-r(T - t)}dt$ | Wealth units at $T$ |
| $\mathrm{TV}$, $\mathrm{TV}^{\mathrm{BH}}$ | Total value: expected terminal wealth plus discounted consumption | Wealth units at $T$ |
| $R_t$ | Dollar risky exposure $\lvert\omega_a^{[1]}\rvert B_te^{-X_t}$ of a surviving path | Wealth units |
| $\bar\omega(t)$ | Expected wealth-weighted allocation $\mathbb{E}[R_t]/\mathbb{E}[\Pi_t]$ | Fraction |

The equity share $q$, drawdown scale $q_{\mathrm{dd}}$ and unconditional volatility $\sigma_{\mathrm{unc}}$
are defined in the [mandate chapter](mandate_aggregation.md), and the building blocks $Q$, $\tilde Q(n)$,
$F$ and $\mathcal{O}$ in the [terminal wealth chapter](terminal_wealth_distribution.md).

## Methodology

### Step 1: the advisor's curve

For each bond weight $w$ the mandate holds $w$ in bonds, $q(1 - w)$ in equity and $(1 - q)(1 - w)$ in
private equity (equation (7.5)). Its effective asset gives the floor
$L_T = \Pi_0e^{-q_{\mathrm{dd}}\sigma_{\mathrm{unc}}(w)}$ and the Riccati system gives the allocation
magnitude $\lvert\omega_a^{[1]}(w)\rvert$. The policy is then calibrated to the initial allocation:
from the [policy formula](mv_optimal_policy.md#the-quadratic-value-function-and-the-coupled-riccati-system),
$\omega_0^{*} = \lvert\omega_a^{[1]}\rvert(\Pi^{*}(0)/\Pi_0 - 1)$, equation (8.3).

**Proposition (closed-form calibration).** The allocation coefficient does not depend on the
multiplier $\ell$, so the policy starts at $\omega_0$ exactly when

$$
\Pi^{*}(0) = \Pi_0\left(1 + \frac{\omega_0}{\lvert\omega_a^{[1]}\rvert}\right), \qquad
\rho = \frac{1}{T}\ln\left(1 + \frac{\omega_0}{\lvert\omega_a^{[1]}\rvert}\right), \qquad
\Pi_T^{*} = \Pi^{*}(0)e^{r_cT},
$$

equation (8.4).

**Proof.** The coefficients $\tilde a^{[i]}$ and $\Sigma^{[i]}$ depend only on $a^{[i]}$, whose equation
does not contain $\ell$; hence $\omega_a^{[i]} = -\tilde a^{[i]}/\Sigma^{[i]}$ is the same for every
$\ell$ (Remark 4.4). Solve (8.3) for $\Pi^{*}(0)$; the target grows at $r_c$ by the
[regime-independent target](mv_optimal_policy.md#the-target-is-common-to-both-regimes). $\square$

`compute_opportunity_point` finds $\rho$ by bisection on $[0.001, 0.60]$ instead, solving the Riccati
system at each step and stopping when the initial allocation is within 0.001 of $\omega_0$; the balanced
mandate starts at 99.97% rather than 100%. The target growth rate $g = \rho + r_c$ is an output: under
full investment it is set by the mandate's risk, not chosen by the investor.

### The implied return and the investor's choice

**Definition (implied return; equation (8.5)).** $r_{\mathrm{impl}}(w) = \ln(\mathbb{E}[\Pi_T(w)]/\Pi_0)/T$,
with $\mathbb{E}[\Pi_T]$ from the [terminal distribution](terminal_wealth_distribution.md#the-building-blocks-and-the-moments).

Proposition 8.2 of the manuscript states that $r_{\mathrm{impl}}$ decreases in $w$ when consumption is
below the stationary return of every mandate on the curve, because expected wealth is driven mainly by
the stationary return, which rises from bonds to private equity; the manuscript gives the argument, not
a proof, and the worked example confirms the ordering at three points. The investor then chooses
$r_{\mathrm{impl}}^{*}$, which determines $w^{*}$, the weights, the floor $L_0(w^{*})$, the target
growth rate $g(w^{*})$ and the whole distribution (equation (8.6)). A point is called feasible when
$r_{\mathrm{impl}} \ge 0$.

### Consumption, total value and the floor protection cost

Consumption withdraws $c\Pi_t\thinspace dt$. The manuscript measures its value at the horizon as the discounted
consumption $C_{\mathrm{disc}} = c\int_0^T\mathbb{E}[\Pi_t]e^{-r(T - t)}dt$ (equation (6.19)), and the
total value as $\mathrm{TV} = \mathbb{E}[\Pi_T] + C_{\mathrm{disc}}$ (equation (6.22)). The same
quantity for buy-and-hold, with the exact buy-and-hold mean at each date, is $\mathrm{TV}^{\mathrm{BH}}$.
The floor protection cost is the value given up for the floor, $\mathrm{TV}^{\mathrm{BH}} - \mathrm{TV}$
(equation (6.24)); `compute_opportunity_point` reports it relative to $\mathrm{TV}^{\mathrm{BH}}$ as
`floor_cost_pct`.

The package evaluates $\mathbb{E}[\Pi_t]$ at intermediate dates by equation (6.20),
$\Pi^{*}(t)Q(t) - B_t\tilde Q(t; 1) + L_t(1 - Q(t))$, which values every stopped path at the floor and so
ignores the overshoot below it, and integrates with the trapezoidal rule on 20 interior dates.

> **Insight.** Consumption above the riskless rate removes the growth of the floor: with $c = 2.5$% and
> $r = 2$%, $r_h = c$ and $r_c = 0$, so the floor stays at its terminal level from inception and the
> buffer to it is smaller. For the balanced mandate survival falls from 78.7% to 59.7%, as Section 8.4
> of the manuscript reports, although only 0.5% a year is consumed beyond the riskless rate.

### The expected glide path

On a surviving path the dollar risky exposure is $R_t = \lvert\omega_a^{[1]}\rvert(\Pi^{*}(t) - \Pi_t) = \lvert\omega_a^{[1]}\rvert B_te^{-X_t}$,
and zero after stopping. Its moments are tilted survivals,

$$
\mathbb{E}[R_t] = \lvert\omega_a^{[1]}\rvert B_t\tilde Q(t; 1), \qquad
\mathrm{Var}[R_t] = \lvert\omega_a^{[1]}\rvert^2B_t^2\left(\tilde Q(t; 2) - \tilde Q(t; 1)^2\right),
$$

and the expected wealth-weighted allocation is $\bar\omega(t) = \mathbb{E}[R_t]/\mathbb{E}[\Pi_t]$ with
the band $\bar\omega(t) \pm \mathrm{Std}[R_t]/\mathbb{E}[\Pi_t]$ (equations (8.8)–(8.11)). It falls over
time for two reasons: surviving wealth approaches the target, and stopped paths hold no risky asset. The
glide path is not a package function; the worked example computes it from the public building blocks.

![Four panels of the opportunity set with no consumption: implied return rising from 2.4% at 100% bonds to 3.7% at no bonds, the frontier of implied return against portfolio volatility, distribution functions of terminal wealth for four mandates with a jump at each floor, and the quantile fan of terminal wealth against implied return with the floor as a dashed line](../papers/goal_based_allocation_2026/paper/figures/opportunity_set_c0.png)

[Open full-resolution figure](../papers/goal_based_allocation_2026/paper/figures/opportunity_set_c0.png).

Figure 4 of the manuscript shows the opportunity set without consumption: the implied return rises
monotonically from 2.4% at 100% bonds to 3.7% with no bonds, every point is feasible, and the
distribution functions jump at their floors by the floor atom. The figure is produced by
`generate_paper_figures.py --figure 1` from the
[replication folder](https://github.com/ArturSepp/GoalBasedAllocation/tree/main/papers/goal_based_allocation_2026/replication).

![Four panels of the expected risky allocation over ten years for the income, conservative, balanced and growth mandates, each starting at 100% and falling, to 74%, 52%, 43% and 37% respectively, with a shaded band of one standard deviation that widens over time](../papers/goal_based_allocation_2026/paper/figures/risky_allocation_subplots_c0.png)

[Open full-resolution figure](../papers/goal_based_allocation_2026/paper/figures/risky_allocation_subplots_c0.png).

Figure 6 of the manuscript shows the glide paths of the four mandates of Table 2. All start fully
invested; the balanced mandate de-risks to 43% after ten years with a band from 14% to 72%, and the
growth mandate de-risks fastest, to 37%, because it approaches its target fastest in expectation. The
figure is produced by `generate_paper_figures.py --figure 5`.

## Worked example

The first block evaluates the balanced mandate, the point of the package quickstart, and checks its
outputs against the formulas of this chapter and of the terminal wealth chapter:

- the reported initial allocation, 0.99966, equals $\lvert\omega_a^{[1]}\rvert(\Pi^{*}(0)/100 - 1)$ with
  $\lvert\omega_a^{[1]}\rvert = 1.2377$, and the bisection's $\rho = 5.9204$% is within $2 \times 10^{-5}$ of
  the closed form $\ln(1 + 1/1.2377)/10 = 5.9219$%;
- expected wealth 139.040, standard deviation 46.087, survival 78.703%, floor atom 13.431% and overshoot
  7.866%, an implied return of 3.296% against a target growth rate of 7.920%;
- the buy-and-hold mean 150.637 and standard deviation 74.848, an implied return of 4.097%, and a floor
  protection cost of 7.699% of the buy-and-hold value, 80 basis points of implied return;
- the expected risky allocation of 91.6% after one year, 61.5% after five and 42.9% after 9.9, with a
  band from 13.8% to 71.9% at the end, as in Figure 6.

The exact overshoot mass of the same policy is 7.899%. The package integrates the overshoot density
from 0.001 and builds its distribution function below the floor on a coarse grid, so its 5% quantile
is 67.02, against 65.54 from the closed-form quantile $\Pi_T^{*} - B_T(\mathcal{O}/0.05)^{\tilde\eta^{[1]}}$.

```python
import numpy as np
import goal_based_allocation as gba

horizon, wealth0, r = 10.0, 100.0, 0.02
spec = gba.AdvisorSpec()
assert (spec.omega_0, spec.c, spec.q, spec.q_dd) == (1.0, 0.0, 2.0 / 3.0, 2.0)
point = gba.compute_opportunity_point(w_bd=0.35, spec=spec)
r_h, r_c = max(r, spec.c), max(r, spec.c) - spec.c

# calibration: the bisection against the closed form of equation (8.4)
present_target = point['PiT'] * np.exp(-r_c * horizon)
np.testing.assert_allclose(point['omega_0'], point['wa'] * (present_target / wealth0 - 1.0), rtol=1e-12)
rho = point['g'] - r_c
rho_closed_form = np.log(1.0 + spec.omega_0 / point['wa']) / horizon
np.testing.assert_allclose([point['wa'], point['omega_0']], [1.2377, 0.99966], atol=5e-5)
np.testing.assert_allclose([rho, rho_closed_form], [0.059204, 0.059219], atol=5e-7)

# terminal distribution, implied return and target growth
np.testing.assert_allclose([point['E'], point['Std'], point['S'], point['F'], point['O']],
                           [139.040, 46.087, 0.78703, 0.13431, 0.07866], atol=5e-4)
np.testing.assert_allclose(point['S'] + point['F'] + point['O'], 1.0, atol=1e-12)
np.testing.assert_allclose(point['r_impl'], np.log(point['E'] / wealth0) / horizon, rtol=1e-12)
np.testing.assert_allclose([point['r_impl'], point['g']], [0.03296, 0.07920], atol=5e-6)

# buy-and-hold benchmark and the floor protection cost
np.testing.assert_allclose([point['E_BH'], point['Std_BH'], point['r_impl_BH']], [150.637, 74.848, 0.04097],
                           atol=5e-4)
assert point['C_disc'] == 0.0 and point['TV'] == point['E']
np.testing.assert_allclose(point['floor_cost_pct'], (point['TV_BH'] - point['TV']) / point['TV_BH'], rtol=1e-12)
np.testing.assert_allclose([point['floor_cost_pct'], point['r_impl_BH'] - point['r_impl']], [0.07699, 0.0080],
                           atol=5e-5)

# rebuild the same policy from the public functions
balanced = gba.build_effective_asset(point['w_eq'], point['w_pe'], point['k'])
ell, ric = gba.find_ell(balanced, horizon, rho, r=r_h, c=spec.c)
gap = gba.gap_process_asset(ric)
np.testing.assert_allclose([ell / 2.0, balanced.pi_floor * np.exp(r_c * horizon)], [point['PiT'], point['L_T']],
                           rtol=1e-9)

# the 5% quantile: closed form below the floor against the package's grid
eta = gap.params.eta0
overshoot = eta * gba.compute_overshoot_density(horizon, np.array([0.0]), gap)[0]
buffer_T = point['PiT'] - point['L_T']
q5 = point['PiT'] - buffer_T * (overshoot / 0.05)**eta
np.testing.assert_allclose([overshoot, q5, point['q5']], [0.07899, 65.54, 67.02], atol=5e-3)

# expected glide path and its band, equations (8.8) to (8.11)
glide = []
for t in (1.0, 5.0, 9.9):
    target_t = ric.derived_at_tau(horizon - t)['Pi_star'][0]
    floor_t = balanced.pi_floor * np.exp(r_c * t)
    buffer_t = target_t - floor_t
    survival_t = gba.compute_survival(t, gap.x0, gap)
    tilted_1 = gba.compute_tilted_survival(t, gap.x0, gap, 1.0)
    tilted_2 = gba.compute_tilted_survival(t, gap.x0, gap, 2.0)
    risky = point['wa'] * buffer_t * tilted_1
    risky_std = point['wa'] * buffer_t * np.sqrt(tilted_2 - tilted_1**2)
    wealth_t = target_t * survival_t - buffer_t * tilted_1 + floor_t * (1.0 - survival_t)
    glide.append((risky / wealth_t, (risky - risky_std) / wealth_t, (risky + risky_std) / wealth_t))
np.testing.assert_allclose([g[0] for g in glide], [0.916, 0.615, 0.429], atol=5e-4)
np.testing.assert_allclose(glide[-1][1:], [0.138, 0.719], atol=5e-4)
```

The second block evaluates the two ends of the curve and the balanced mandate with consumption. The
income and growth mandates reproduce Table 2:

| Mandate | $\lvert\omega_a^{[1]}\rvert$ | $\Pi_T^{*}$ | Expected wealth | Std | Std given survival | $r_{\mathrm{impl}}$ | Survival | $r_{\mathrm{impl}}^{\mathrm{BH}}$ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Income (100/0/0) | 1.04 | 240 | 127.0 | 24.6 | 18.2 | 2.39% | 85.5% | 2.46% |
| Balanced (35/43/22) | 1.24 | 221 | 139.0 | 46.1 | 24.1 | 3.30% | 78.7% | 4.10% |
| Growth (0/67/33) | 0.92 | 254 | 144.5 | 61.9 | 31.5 | 3.68% | 73.8% | 5.01% |

The implied return falls with the bond weight, as Proposition 8.2 states, and the difference between the
buy-and-hold and the floor-protected implied returns grows from 7 basis points for income to 133 for
growth. With consumption of 2.5% the balanced mandate keeps its terminal floor of 79.04 from inception,
its survival falls to 59.7% and its implied return to 0.88%; the discounted consumption is 24.34, the
total value 133.53 against 141.94 for buy-and-hold, and the floor protection cost 5.93%.

```python
income = gba.compute_opportunity_point(w_bd=1.0, spec=spec)
growth = gba.compute_opportunity_point(w_bd=0.0, spec=spec)
table_2 = {'income': (income, (1.04, 240, 127.0, 24.6, 18.2, 0.0239, 0.855, 0.0246)),
           'growth': (growth, (0.92, 254, 144.5, 61.9, 31.5, 0.0368, 0.738, 0.0501))}
rounding = np.array([0.005, 0.5, 0.05, 0.05, 0.05, 0.00005, 0.0005, 0.00005])  # half a printed digit
for result, row in table_2.values():
    computed = np.array([result['wa'], result['PiT'], result['E'], result['Std'], result['Stds'],
                         result['r_impl'], result['S'], result['r_impl_BH']])
    assert np.all(np.abs(computed - np.array(row)) <= rounding), computed
assert income['r_impl'] < point['r_impl'] < growth['r_impl']
np.testing.assert_allclose([income['r_impl_BH'] - income['r_impl'], growth['r_impl_BH'] - growth['r_impl']],
                           [0.0007, 0.0133], atol=5e-5)

# consumption above the riskless rate: r_h = c and r_c = 0, so the floor does not grow
consuming = gba.compute_opportunity_point(w_bd=0.35, spec=gba.AdvisorSpec(c=0.025))
np.testing.assert_allclose(consuming['L_T'], point['L_T'], rtol=1e-12)
np.testing.assert_allclose([consuming['S'], consuming['r_impl'], consuming['C_disc'], consuming['TV'],
                            consuming['TV_BH'], consuming['floor_cost_pct']],
                           [0.5973, 0.0088, 24.34, 133.53, 141.94, 0.0593], atol=5e-3)
np.testing.assert_allclose(consuming['TV'], consuming['E'] + consuming['C_disc'], rtol=1e-12)
```

## Implementation in goal_based_allocation

| Quantity | Formula | Entry point |
|---|---|---|
| Advisor preferences | $\omega_0$, $c$, $q$, $q_{\mathrm{dd}}$ | `AdvisorSpec(omega_0=1.0, c=0.0, q=2/3, q_dd=2.0)` |
| One point of the curve | Effective asset, floor, calibrated policy, terminal distribution, benchmark | `compute_opportunity_point(w_bd, spec)` |
| The curve | Points sorted by implied return | `build_opportunity_set(spec, w_vals=None)` |
| Investor's choice | Bond weight interpolated to a target implied return, then re-evaluated | `opportunity_set.select_portfolio(opportunity_set, r_impl_target)` (module level) |

The functions live in
[`opportunity_set.py`](https://github.com/ArturSepp/GoalBasedAllocation/blob/main/src/goal_based_allocation/opportunity_set.py).
API pages: {doc}`AdvisorSpec <api/generated/goal_based_allocation.AdvisorSpec>`,
{doc}`compute_opportunity_point <api/generated/goal_based_allocation.compute_opportunity_point>` and
{doc}`build_opportunity_set <api/generated/goal_based_allocation.build_opportunity_set>`.

Contract details:

- The horizon of ten years, initial wealth of 100 and riskless rate of 2% are module constants; the
  asset classes, correlations and intensities are those of [mandate aggregation](mandate_aggregation.md).
- `compute_opportunity_point` returns a dictionary with the inputs `w_bd`, `w_eq`, `w_pe`, `c`,
  `q_dd`; the calibration `omega_0` (achieved), `g`, `wa` ($\lvert\omega_a^{[1]}\rvert$), `k`,
  `sig_unc`, `PiT`, `L_T`; the distribution `E`, `Std`, `Es`, `Stds`, `S`, `F`, `O`, `r_impl`, `q5`,
  `q25`, `q50`, `q75`, `q95`, `Pi_cdf` and `cdf`; the benchmark `E_BH`, `Std_BH`, `r_impl_BH`; and
  `C_disc`, `TV`, `TV_BH`, `floor_cost_pct`. It returns `None` if the final Riccati solve fails, and
  treats a failed solve during the bisection as an allocation above target.
- The bisection does not report non-convergence: if $\omega_0$ is not reachable with $\rho$ in
  $[0.001, 0.60]$ the last midpoint is used; check `omega_0` in the result.
- `build_opportunity_set` skips points that return `None` and sorts the rest by `r_impl`. Its default
  grid runs from 0.95 to 0 and does not include the income mandate, `w_bd=1.0`.
- `select_portfolio` interpolates linearly in the bond weight between the two points that bracket the
  requested implied return, rebuilds an `AdvisorSpec` from the lower point, and evaluates the
  interpolated bond weight; requests outside the curve return its end points.
- The consumption integral uses equation (6.20), which values stopped paths at the floor, and the
  benchmark's consumption uses the stationary buy-and-hold mean at each date.

> **Pitfall.** `floor_cost_pct` is the floor protection cost as a fraction of the buy-and-hold total
> value, not an annual fee or a realised cost. Its benchmark starts from the stationary regime, while
> the floor-protected values start in growth (see [buy-and-hold moments](buy_and_hold_moments.md)), and
> its quantiles below the floor are biased upward by the coarse overshoot grid: compute lower quantiles
> with the closed form of the [terminal wealth chapter](terminal_wealth_distribution.md#quantiles).

## Interpretation and limitations

- The curve is one-dimensional by construction: the equity share and the drawdown tolerance are fixed
  in step one. Proposition 8.1 of the manuscript shows that mixtures across asset classes span richer
  families of distributions; the package does not implement the mixture problem.
- The implied return depends on the conditioning conventions and approximations of the reduced gap
  process; its ordering along the curve is robust, its level to a few basis points.
- The target growth rate $g$ is the ceiling of the mean-variance policy, not a return promise: for the
  balanced mandate $g = 7.92$% against an implied return of 3.30%.
- The glide path is an expectation across paths. Individual paths can lever above 100% after losses,
  as the [floor chapter](wealth_floor_gap_process.md) shows; the band widens because stopped paths hold
  no risky asset.
- With consumption above the riskless rate the hurdle-rate reformulation of the
  [policy chapter](mv_optimal_policy.md) applies; it is a modelling choice of the manuscript.

## See also

- [The terminal wealth distribution](terminal_wealth_distribution.md)
- [Mandates as one effective asset](mandate_aggregation.md)
- [Buy-and-hold moments](buy_and_hold_moments.md)
- [The MV-optimal policy and the Riccati system](mv_optimal_policy.md)
- [Getting started](getting-started.md)
- [Bibliography](bibliography.md)

## References

1. Sepp, A. (2026). Dynamic Mean-Variance Portfolio Allocation under Regime-Switching Jump-Diffusions with Absorbing Barriers and Distribution Matching. Working paper. [DOI: 10.2139/ssrn.6534579](https://doi.org/10.2139/ssrn.6534579). Section 8 defines the two-step framework and the glide path; Section 6.2 the floor protection cost; Table 2 and Figures 4 and 6 report the mandates.
2. Das, S., Markowitz, H., Scheid, J., and Statman, M. (2010). Portfolio Optimization with Mental Accounts. *Journal of Financial and Quantitative Analysis*, 45(2), 311–334. [DOI: 10.1017/S0022109010000141](https://doi.org/10.1017/S0022109010000141). Goal-based portfolios with threshold probabilities.
3. Sepp, A., Ossa, I., and Kastenholz, M. (2026). Robust Optimization of Strategic and Tactical Asset Allocation for Multi-Asset Portfolios. *The Journal of Portfolio Management*, 52(4), 86–120. [DOI: 10.3905/jpm.2025.1.806](https://doi.org/10.3905/jpm.2025.1.806). Strategic mandates across risk profiles.
4. Sepp, A. goal-based-allocation: Semi-analytical dynamic mean-variance allocation and terminal-wealth risk under regime-switching jump-diffusions. [Software citation metadata](https://github.com/ArturSepp/GoalBasedAllocation/blob/main/CITATION.cff).
