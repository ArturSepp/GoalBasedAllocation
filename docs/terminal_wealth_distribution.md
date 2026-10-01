---
myst:
  html_meta:
    description: >-
      The terminal wealth distribution of the floor-protected mean-variance strategy in
      goal-based-allocation: survived density, floor atom and jump overshoot mapped from the log-gap
      to wealth, the six Laplace building blocks, closed-form moments, quantiles and the gap to
      target, reproduced for the balanced mandate and checked by exact simulation.
---

# The terminal wealth distribution

*Author: [Artur Sepp](https://github.com/ArturSepp)*

Implemented in [GoalBasedAllocation](https://github.com/ArturSepp/GoalBasedAllocation).
Software citation: [CITATION.cff](https://github.com/ArturSepp/GoalBasedAllocation/blob/main/CITATION.cff).

The terminal wealth of the floor-protected mean-variance strategy is a mixture of three parts: a
continuous law between the floor and the target for paths that survive, a point mass at the floor
for paths stopped by the diffusion, and a continuous law below the floor for paths stopped by a
crash (Sepp, 2026, Section 6). Each part is a function of the [gap process](wealth_floor_gap_process.md)
at the horizon, so its probabilities and moments follow from survival, tilted survival and the
overshoot mass of the [Laplace framework](laplace_barrier_framework.md). This chapter maps the three
parts to wealth, derives the moments and quantiles, and reproduces the balanced mandate of Table 2
of the manuscript.

## Overview

The question is what an investor following the policy ends with: the probability of each outcome,
the expected terminal wealth and its dispersion, and the quantiles a client is shown. Six building
blocks answer it: survival $Q$, the tilted survivals $\tilde Q(1)$ and $\tilde Q(2)$, the overshoot
mass $\mathcal{O}$, its exponential moments $\tilde{\mathcal{O}}(1)$ and $\tilde{\mathcal{O}}(2)$, and
the floor atom $F$ as the remainder. They are scalars from one Laplace inversion each; the density and
the quantiles need one further inversion on a grid.

`compute_opportunity_point` computes these quantities for the advisor configuration of the
[opportunity set](investment_opportunity_set.md). This chapter builds them from the public functions
for any calibrated policy, so that each step can be checked; the worked example obtains the
balanced mandate's expected wealth of 139.0, standard deviation of 46.1 and survival of 78.7% of
Table 2, and confirms them with an exact simulation of the gap process.

## Inputs, notation, and assumptions

| Convention | This article |
|---|---|
| Time and rates | Horizon $T = 10$ years; the floor, the target and stopped wealth grow at $r_c$; implied returns are continuously compounded |
| Regimes | Every quantity starts in growth and sums over the terminal regime |
| Jump sizes | Effective crash mean $\tilde\eta^{[1]}$ of the gap process; the overshoot moments need $n\tilde\eta^{[1]} \lt 1$ |
| Wealth coordinate | Terminal wealth $\Pi_T$ from initial wealth 100; the log-gap $X_T$ and the overshoot distance $d$ map to it through $\Pi_T^{*}$ and $B_T$ |
| Measure | Physical |
| Numerical method | Laplace inversion of $Q$, $\tilde Q(1)$, $\tilde Q(2)$ and the overshoot factor; trapezoidal integration of the density for the distribution function; checked by exact simulation of the gap process and Euler simulation of wealth |
| Package default | `compute_opportunity_point` integrates the overshoot from $d = 0.001$ on 400 points and the density on 600 points, and reads quantiles from 1,500 wealth points |

| Symbol | Meaning | Units and convention |
|---|---|---|
| $\tilde{\mathcal{O}}(n)$ | Overshoot moment $\int_0^{\infty}e^{nd}f_{\mathrm{over}}(d)\thinspace dd$ | Probability-weighted |
| $F_{\Pi}(\pi)$ | Distribution function of terminal wealth | Probability |
| $\pi$ | A level of terminal wealth | Wealth units |
| $\pi_u$ | Quantile of terminal wealth at probability $u$ | Wealth units |
| $r_{\mathrm{impl}}$ | Implied return $\ln(\mathbb{E}[\Pi_T]/\Pi_0)/T$ | Per year |

The target $\Pi_T^{*}$, the buffer $B_T = \Pi_T^{*} - L_T$, the floor $L_T$, the survival $Q$, the
tilted survival $\tilde Q(n)$, the floor atom $F$ and the overshoot mass $\mathcal{O}$ are reserved on
the [conventions page](conventions.md#reserved-notation). All are evaluated at the horizon and refer to
the reduced gap process; the [floor chapter](wealth_floor_gap_process.md) measures how far that process
is from the wealth it describes.

## Methodology

### From the gap to wealth

Because the target, the floor and stopped wealth all grow at $r_c$, the ratio of stopped wealth to the
floor is frozen at stopping, and each component of terminal wealth is a fixed function of the gap
process.

**Proposition (wealth of the three components).** At the horizon,

$$
\Pi_T = \Pi_T^{*} - B_Te^{-X_T} \ \text{if the path survives}, \qquad
\Pi_T = L_T \ \text{at the floor atom}, \qquad
\Pi_T = \Pi_T^{*} - B_Te^{d} \ \text{after an overshoot of depth} \ d .
$$

**Proof.** For a surviving path, $X_T = \ln(B_T/Z_T)$ gives $Z_T = B_Te^{-X_T}$ and
$\Pi_T = \Pi_T^{*} - Z_T$. A path stopped at time $s$ with log-gap $-d \le 0$ holds
$\Pi_s = \Pi^{*}(s) - B_se^{d}$ and grows at $r_c$ to $T$; since $\Pi^{*}$ and $B$ also grow at $r_c$,
$\Pi_T = \Pi_T^{*} - B_Te^{d}$. The floor atom is the case $d = 0$. $\square$

Surviving wealth lies in $(L_T, \Pi_T^{*})$, increasing in $X_T$; overshoot wealth lies below $L_T$,
decreasing in $d$. With $A(X)$ the surviving density of the log-gap, the distribution function of
terminal wealth is, by Theorem 6.1 of the manuscript,

$$
F_{\Pi}(\pi) = \mathcal{O}e^{-d(\pi)/\tilde\eta^{[1]}} \ \text{for} \ \pi \lt L_T, \qquad
F_{\Pi}(\pi) = \mathcal{O} + F + \int_0^{x(\pi)}A(X)\thinspace dX \ \text{for} \ L_T \le \pi \lt \Pi_T^{*},
$$

with $x(\pi) = \ln(B_T/(\Pi_T^{*} - \pi))$ and $d(\pi) = -x(\pi)$; its density in wealth is
$A(x(\pi))/(\Pi_T^{*} - \pi)$ above the floor.

### The building blocks and the moments

**Identity (overshoot moments).** For $n\tilde\eta^{[1]} \lt 1$,
$\tilde{\mathcal{O}}(n) = \mathcal{O}/(1 - n\tilde\eta^{[1]})$, equation (6.15) of the manuscript.

**Proof.** The overshoot depth is exponential with mean $\tilde\eta^{[1]}$ and total mass $\mathcal{O}$,
so $\int_0^{\infty}e^{nd}(\mathcal{O}/\tilde\eta^{[1]})e^{-d/\tilde\eta^{[1]}}dd = \mathcal{O}/(1 - n\tilde\eta^{[1]})$. $\square$

**Proposition (moments of terminal wealth; Sepp, 2026, equations (6.17)–(6.18)).**

$$
\mathbb{E}[\Pi_T] = \Pi_T^{*}Q - B_T\tilde Q(1) + L_TF + \Pi_T^{*}\mathcal{O} - B_T\tilde{\mathcal{O}}(1),
$$

$$
\mathbb{E}[\Pi_T^2] = (\Pi_T^{*})^2Q - 2\Pi_T^{*}B_T\tilde Q(1) + B_T^2\tilde Q(2) + L_T^2F + (\Pi_T^{*})^2\mathcal{O} - 2\Pi_T^{*}B_T\tilde{\mathcal{O}}(1) + B_T^2\tilde{\mathcal{O}}(2) .
$$

**Proof.** Expand $(\Pi_T^{*} - B_Te^{\mp y})^n$ for $n = 1, 2$ in each component and take expectations:
$\mathbb{E}[e^{-nX_T};\ \text{survived}] = \tilde Q(n)$ and $\mathbb{E}[e^{nd};\ \text{overshoot}] = \tilde{\mathcal{O}}(n)$.
$\square$

Dividing the surviving terms by $Q$ gives the moments conditional on survival,
$\mathbb{E}[\Pi_T \mid \text{survival}] = \Pi_T^{*} - B_T\tilde Q(1)/Q$.

**Identity (gap to target; equation (8.7)).**

$$
\Pi_T^{*} - \mathbb{E}[\Pi_T] = B_T\left(\tilde Q(1) + \tilde{\mathcal{O}}(1)\right) + (\Pi_T^{*} - L_T)F .
$$

**Proof.** Substitute $Q + F + \mathcal{O} = 1$ into the first moment. $\square$

The three terms are the shortfall of surviving paths below the target, the loss of overshoot paths and
the loss of paths stopped at the floor. The target growth rate $\ln(\Pi_T^{*}/\Pi_0)/T$ is therefore
not a return forecast: it exceeds the implied return by the annualised gap.

### Quantiles

**Proposition (quantiles).** For $0 \lt u \le \mathcal{O}$ the quantile is in closed form,
$\pi_u = \Pi_T^{*} - B_T(\mathcal{O}/u)^{\tilde\eta^{[1]}}$; for $\mathcal{O} \lt u \le \mathcal{O} + F$ it
is the floor $L_T$; above, it is the level at which the integral of the surviving density reaches
$u - \mathcal{O} - F$.

**Proof.** Below the floor, $\mathcal{O}e^{-d/\tilde\eta^{[1]}} = u$ gives $e^{d} = (\mathcal{O}/u)^{\tilde\eta^{[1]}}$.
The atom makes the distribution function jump by $F$ at $L_T$, and above it the function is
continuous and increasing. $\square$

Because of the atom, every quantile between $\mathcal{O}$ and $\mathcal{O} + F$ equals the floor.
Quantiles below $\mathcal{O}$ describe crash losses through the floor, and they are where the
exponential approximation of the gap jump matters most.

![Two panels for the conservative and balanced mandates comparing analytical terminal wealth densities with Monte Carlo histograms: the survived density between the floor and the target matches the histogram closely, a tall histogram bar marks the floor atom at the dashed floor line, and the analytical overshoot density below the floor lies close to the overshoot histogram](../papers/goal_based_allocation_2026/paper/figures/mandate_comparison.png)

[Open full-resolution figure](../papers/goal_based_allocation_2026/paper/figures/mandate_comparison.png).

Figure 3 of the manuscript overlays the analytical survived density, blue, and overshoot density, red,
on histograms of 200,000 simulated wealth paths for the conservative and balanced mandates of Table 2.
The survived density matches the simulation; the tall bar at the dashed floor line is the floor atom,
which has no density. The figure is produced by `generate_paper_figures.py --figure 10` from the
[replication folder](https://github.com/ArturSepp/GoalBasedAllocation/tree/main/papers/goal_based_allocation_2026/replication).

![Two panels of terminal wealth densities for four mandates: the upper panel shows the floor-protected densities concentrated between their floors and targets, with dashed overshoot tails below the floors, and the lower panel shows wider buy-and-hold densities with long tails on both sides](../papers/goal_based_allocation_2026/paper/figures/mandate_density_overlay_c0.png)

[Open full-resolution figure](../papers/goal_based_allocation_2026/paper/figures/mandate_density_overlay_c0.png).

Figure 2 of the manuscript compares the floor-protected densities of the income, conservative, balanced
and growth mandates with their [buy-and-hold](buy_and_hold_moments.md) densities. The protected
densities are capped by the target and cut at the floor, apart from the overshoot tails; the
buy-and-hold densities extend far below the floors and above the targets. The figure is produced by
`generate_paper_figures.py --figure 9`. Its buy-and-hold densities are lognormal curves with the
stationary total return and diffusion volatility of each mandate, without jumps, drawn by the
replication script to show the shape; they are not the regime-switching distribution, whose exact
mean and variance the [buy-and-hold chapter](buy_and_hold_moments.md) computes.

## Worked example

The balanced mandate of Table 2 holds 35% bonds, 43.3% equity and 21.7% private equity. With a drawdown
scale of 2 its floor parameter is $k = 1.8655$ and its floor $L_0 = 64.72$, growing at $r_c = 2$% to
$L_T = 79.04$. The allocation magnitude does not depend on $\ell$, so the policy starts fully invested
when $\Pi^{*}(0) = \Pi_0(1 + 1/\lvert\omega_a^{[1]}\rvert)$; with $\lvert\omega_a^{[1]}\rvert = 1.2377$ this is
a `target_return` of 5.92% and a target at the horizon of $\Pi_T^{*} = 220.82$. The building blocks
and results are:

| Quantity | Value | Table 2 |
|---|---:|---:|
| Survival $Q$ | 78.69% | 78.7% |
| Floor atom $F$, overshoot mass $\mathcal{O}$ | 13.40%, 7.90% | |
| Tilted survivals $\tilde Q(1)$, $\tilde Q(2)$ | 0.34419, 0.17334 | |
| Expected terminal wealth | 139.03 | 139 |
| Standard deviation, unconditional and given survival | 46.13, 24.13 | 46.1, 24.1 |
| Implied return | 3.30% | 3.30% |
| Quantiles at 5%, 25%, 50%, 75%, 95% | 65.5, 112.2, 153.1, 173.1, 190.9 | |

The gap to target of 81.79 splits into the three terms of identity (8.7) exactly.

```python
import dataclasses
import numpy as np
from scipy.integrate import cumulative_trapezoid
import goal_based_allocation as gba

horizon, r, c, wealth0, q_dd = 10.0, 0.02, 0.0, 100.0, 2.0
r_h, r_c = max(r, c), max(r, c) - c
w_bd, q = 0.35, 2.0 / 3.0
w_eq, w_pe = q * (1.0 - w_bd), (1.0 - q) * (1.0 - w_bd)

# floor from the drawdown scale, equation (7.9); one quadrature for the effective asset
base = gba.build_effective_asset(w_eq, w_pe, k=1.5)
k = (q_dd * gba.portfolio_sigma_unc(w_eq, w_pe) + r_c * horizon) / base.params.eta0
balanced = dataclasses.replace(base, pi_floor=wealth0 * np.exp(-k * base.params.eta0))
floor_T = balanced.pi_floor * np.exp(r_c * horizon)
np.testing.assert_allclose([k, balanced.pi_floor, floor_T], [1.8655, 64.72, 79.04], atol=5e-3)

# full investment at inception: the allocation magnitude does not depend on ell
_, probe = gba.find_ell(balanced, horizon, 0.05, r=r_h, c=c)
kappa = abs(probe.derived_at_tau(horizon)['w_a'][0])
rho = np.log(1.0 + 1.0 / kappa) / horizon
ell, ric = gba.find_ell(balanced, horizon, rho, r=r_h, c=c)
np.testing.assert_allclose(ric.omega_star(0.0, wealth0, 0), 1.0, atol=1e-12)
np.testing.assert_allclose([kappa, rho], [1.2377, 0.0592], atol=5e-5)

# the six building blocks
gap = gba.gap_process_asset(ric)
target_T = ell / 2.0
buffer_T = target_T - floor_T
eta = gap.params.eta0
survival = gba.compute_survival(horizon, gap.x0, gap)
tilted_1 = gba.compute_tilted_survival(horizon, gap.x0, gap, 1.0)
tilted_2 = gba.compute_tilted_survival(horizon, gap.x0, gap, 2.0)
overshoot = eta * gba.compute_overshoot_density(horizon, np.array([0.0]), gap)[0]
overshoot_1, overshoot_2 = overshoot / (1.0 - eta), overshoot / (1.0 - 2.0 * eta)
floor_atom = 1.0 - survival - overshoot
np.testing.assert_allclose([target_T, survival, floor_atom, overshoot], [220.82, 0.7869, 0.1340, 0.0790], atol=5e-3)
np.testing.assert_allclose([tilted_1, tilted_2], [0.34419, 0.17334], atol=5e-6)

# moments, equations (6.17) and (6.18), and the moments given survival
mean = (target_T * survival - buffer_T * tilted_1 + floor_T * floor_atom
        + target_T * overshoot - buffer_T * overshoot_1)
second = (target_T**2 * survival - 2.0 * target_T * buffer_T * tilted_1 + buffer_T**2 * tilted_2
          + floor_T**2 * floor_atom
          + target_T**2 * overshoot - 2.0 * target_T * buffer_T * overshoot_1 + buffer_T**2 * overshoot_2)
std = np.sqrt(second - mean**2)
mean_survived = target_T - buffer_T * tilted_1 / survival
std_survived = np.sqrt(target_T**2 - 2.0 * target_T * buffer_T * tilted_1 / survival
                       + buffer_T**2 * tilted_2 / survival - mean_survived**2)
implied = np.log(mean / wealth0) / horizon
np.testing.assert_allclose([mean, std, std_survived], [139.03, 46.13, 24.13], atol=5e-3)
np.testing.assert_allclose(implied, 0.0330, atol=5e-5)

# the gap to target, identity (8.7)
np.testing.assert_allclose(target_T - mean, buffer_T * (tilted_1 + overshoot_1) + (target_T - floor_T) * floor_atom,
                           rtol=1e-12)
np.testing.assert_allclose(target_T - mean, 81.79, atol=5e-3)

# distribution function and quantiles
x_max = min(8.0, gap.x0 + 6.0 * max(gap.params.sigma0, gap.params.sigma1) * np.sqrt(horizon))
x = np.linspace(1e-4, x_max, 4001)
cumulative = overshoot + floor_atom + cumulative_trapezoid(sum(gba.compute_density(horizon, x, gap)), x, initial=0.0)
np.testing.assert_allclose(cumulative[-1], 1.0, atol=1e-5)
levels = target_T - buffer_T * np.exp(-x)


def quantile(u):
    """Quantile of terminal wealth: closed form below the floor, the floor at the atom, then interpolation."""
    if u <= overshoot:
        return target_T - buffer_T * (overshoot / u)**eta
    if u <= overshoot + floor_atom:
        return floor_T
    return float(np.interp(u, cumulative, levels))


probabilities = [0.05, 0.25, 0.50, 0.75, 0.95]
quantiles = np.array([quantile(u) for u in probabilities])
np.testing.assert_allclose(quantiles, [65.52, 112.25, 153.11, 173.10, 190.93], atol=5e-3)
```

The second block checks the formulas with an exact simulation of the gap process, as in the
[barrier chapter](laplace_barrier_framework.md#worked-example), mapping each simulated outcome to
wealth by the first proposition. Over one million paths the simulated survival, floor atom and
overshoot are 78.73%, 13.38% and 7.89%, the mean is 139.06 with a standard error of 0.05, the standard
deviation is 46.00, and the quantiles are 65.5, 112.3, 153.1, 173.1 and 190.9.

The package's wealth simulator, `simulate_mv_optimal`, follows the policy itself in 40,000 daily paths.
It gives survival of 79.52%, a mean of 140.76, a standard deviation of 41.66 and quantiles of 70.3,
115.5, 153.2, 173.1 and 191.0. The median and the upper quantiles agree with the analytical ones; the
lower tail does not, because the exact gap jump is bounded while the reduced process has exponential
overshoots, as the [floor chapter](wealth_floor_gap_process.md#worked-example) explains.

```python
def simulate_stopping(asset, horizon, n_paths, seed):
    """Exact simulation from growth: status 0 survived, 1 stopped by diffusion, 2 by a crash.

    Returns the status and the state at the horizon or, for a crash stop, just after the jump.
    """
    pr = asset.params
    rng = np.random.default_rng(seed)
    nu = np.array([asset.nu0, asset.nu1])
    vol = np.array([pr.sigma0, pr.sigma1])
    rate = np.array([pr.lambda01, pr.lambda10])
    state = np.full(n_paths, asset.x0)
    clock = np.zeros(n_paths)
    regime = np.zeros(n_paths, dtype=int)
    status = np.zeros(n_paths, dtype=int)
    active = np.ones(n_paths, dtype=bool)
    while active.any():
        i = np.flatnonzero(active)
        now = regime[i]
        hold = rng.exponential(1.0 / rate[now])
        remaining = horizon - clock[i]
        step = np.minimum(hold, remaining)
        end = state[i] + nu[now] * step + vol[now] * np.sqrt(step) * rng.standard_normal(i.size)
        bridge = np.exp(-2.0 * state[i] * np.maximum(end, 0.0) / (vol[now]**2 * step))
        hit = (end <= 0.0) | (rng.uniform(size=i.size) < bridge)
        status[i[hit]] = 1
        state[i], clock[i] = end, clock[i] + step
        switch = i[~hit & (hold < remaining)]
        crash = regime[switch] == 0
        state[switch] += np.where(crash, -rng.exponential(pr.eta0, switch.size), rng.exponential(pr.eta1, switch.size))
        regime[switch] = 1 - regime[switch]
        status[switch[state[switch] <= 0.0]] = 2
        active = (status == 0) & (clock < horizon)
    return status, state


status, state = simulate_stopping(gap, horizon, 1_000_000, seed=3)
wealth = np.where(status == 1, floor_T, target_T - buffer_T * np.exp(-state))
shares = [(status == s).mean() for s in (0, 1, 2)]
for share, exact in zip(shares, (survival, floor_atom, overshoot)):
    assert abs(share - exact) < 3.0 * np.sqrt(exact * (1.0 - exact) / status.size)
assert abs(wealth.mean() - mean) < 3.0 * wealth.std() / np.sqrt(status.size)
np.testing.assert_allclose(wealth.std(), std, rtol=0.005)
np.testing.assert_allclose(np.quantile(wealth, probabilities), quantiles, atol=0.2)

# the wealth process under the regime-conditional policy, the package's own validator
from goal_based_allocation.riccati_solver import simulate_mv_optimal

paths = simulate_mv_optimal(ric, n_paths=40_000, steps_per_year=260, seed=42)
np.testing.assert_allclose([paths['survived'].mean(), paths['Pi_T'].mean(), paths['Pi_T'].std()],
                           [0.7952, 140.76, 41.66], atol=5e-3)
np.testing.assert_allclose(np.quantile(paths['Pi_T'], probabilities[2:]), quantiles[2:], atol=0.2)
assert np.quantile(paths['Pi_T'], 0.05) - quantiles[0] > 4.0
```

## Implementation in goal_based_allocation

| Quantity | Formula | Entry point |
|---|---|---|
| Survival, tilted survival | $Q$, $\tilde Q(1)$, $\tilde Q(2)$ of the gap process | `compute_survival(T, gap.x0, gap)`, `compute_tilted_survival(T, gap.x0, gap, n)` |
| Overshoot mass | $\mathcal{O} = \tilde\eta^{[1]}f_{\mathrm{over}}(0)$ | `compute_overshoot_density(T, d_grid, gap)` |
| Surviving density of the log-gap | $A(X)$ by terminal regime | `compute_density(T, x_grid, gap)` |
| Target, floor and buffer at the horizon | $\ell/2$, $L_0e^{r_cT}$, their difference | `ell / 2`, `asset.pi_floor * np.exp(ric.r_c * T)` |
| All of the above for the advisor configuration | Moments, quantiles, distribution function | `compute_opportunity_point(w_bd, spec)` |

The building blocks live in
[`regime_switch_paper.py`](https://github.com/ArturSepp/GoalBasedAllocation/blob/main/src/goal_based_allocation/regime_switch_paper.py)
and their assembly in
[`opportunity_set.py`](https://github.com/ArturSepp/GoalBasedAllocation/blob/main/src/goal_based_allocation/opportunity_set.py).
API page: {doc}`compute_opportunity_point <api/generated/goal_based_allocation.compute_opportunity_point>`.

Contract details:

- No package function returns the terminal distribution for an arbitrary asset and policy; the
  blocks above are the public route, and `compute_opportunity_point` assembles them for its fixed
  configuration of ten years, initial wealth 100 and a riskless rate of 2%.
- `compute_opportunity_point` reports `S`, `F`, `O`, `E`, `Std`, `Es`, `Stds`, `r_impl`, the quantiles
  `q5`, `q25`, `q50`, `q75` and `q95`, and the distribution function as `Pi_cdf` and `cdf`.
- It obtains $\mathcal{O}$ by integrating the overshoot density from $d = 0.001$ on 400 points, which
  misses about $0.001 f_{\mathrm{over}}(0)$ and adds it to $F$: 0.07868 against the exact 0.07901 for
  the balanced mandate. It clips $F$ at zero.
- Its distribution function below the floor integrates the overshoot density from the first grid
  point at or beyond each depth, on a step of 0.02, so it is too low by up to $0.02f_{\mathrm{over}}$
  there. The [opportunity-set chapter](investment_opportunity_set.md) shows the effect on the 5%
  quantile.

> **Pitfall.** The expected terminal wealth of the floor-protected strategy is neither the target
> $\Pi_T^{*}$ nor the expected wealth of the unconstrained policy. For the balanced mandate the target
> is 220.82 and the expected wealth 139.03, a target growth rate of 7.92% against an implied return of
> 3.30%. Quote the implied return, and the quantiles, as the strategy's outcome.

## Interpretation and limitations

- The formulas are exact for the reduced gap process. The body of the distribution, from the floor
  to the target, agrees with a simulation of wealth; the overshoot tail is too heavy, because the
  exact gap jump is bounded, so the analytical 5% quantile of 65.5 is conservative against 70.3 in the
  wealth simulation.
- The floor atom has probability but no density, and its quantiles are flat; a client distribution
  plot should show it as a mass, as Figure 3 does.
- The expected wealth averages over survival and stopping. Conditional on survival the standard
  deviation, 24.1, is about half the unconditional one, because the target caps surviving wealth and
  the stopped paths sit at or below the floor.
- The second overshoot moment needs $2\tilde\eta^{[1]} \lt 1$. For larger effective crash sizes the
  variance of terminal wealth is infinite under the reduced process.

## See also

- [The wealth floor and the flat-barrier reduction](wealth_floor_gap_process.md)
- [Survival, densities and overshoot in the Laplace domain](laplace_barrier_framework.md)
- [The investment opportunity set and investor selection](investment_opportunity_set.md)
- [Buy-and-hold moments](buy_and_hold_moments.md)
- [Mandates as one effective asset](mandate_aggregation.md)
- [Bibliography](bibliography.md)

## References

1. Sepp, A. (2026). Dynamic Mean-Variance Portfolio Allocation under Regime-Switching Jump-Diffusions with Absorbing Barriers and Distribution Matching. Working paper. [DOI: 10.2139/ssrn.6534579](https://doi.org/10.2139/ssrn.6534579). Section 6 derives the terminal distribution and its moments; Table 2 and Figures 2–3 report the mandates.
2. Farkas, W., Mathys, L., and Vasiljević, N. (2021). Intra-Horizon Expected Shortfall and Risk Structure in Models with Jumps. *Mathematical Finance*, 31(2), 772–823. [DOI: 10.1111/mafi.12302](https://doi.org/10.1111/mafi.12302). Diffusion and jump contributions to intra-horizon risk, related to the floor atom and the overshoot.
3. Cont, R., and Tankov, P. (2009). Constant Proportion Portfolio Insurance in the Presence of Jumps in Asset Prices. *Mathematical Finance*, 19(3), 379–401. [DOI: 10.1111/j.1467-9965.2009.00377.x](https://doi.org/10.1111/j.1467-9965.2009.00377.x). The loss distribution below a floor under jumps.
4. Sepp, A. goal-based-allocation: Semi-analytical dynamic mean-variance allocation and terminal-wealth risk under regime-switching jump-diffusions. [Software citation metadata](https://github.com/ArturSepp/GoalBasedAllocation/blob/main/CITATION.cff).
