---
myst:
  html_meta:
    description: >-
      Exact moments of buy-and-hold terminal wealth under the two-regime jump-diffusion: the 2x2
      linear ODE for each moment, its matrix-exponential solution, regime-conditional and
      stationary-weighted moments, consumption scaling, and the benchmark used for the floor
      protection cost, checked against exact simulation.
---

# Buy-and-hold moments

*Author: [Artur Sepp](https://github.com/ArturSepp)*

Implemented in [GoalBasedAllocation](https://github.com/ArturSepp/GoalBasedAllocation).
Software citation: [CITATION.cff](https://github.com/ArturSepp/GoalBasedAllocation/blob/main/CITATION.cff).

A buy-and-hold strategy keeps the whole wealth in the risky asset, with no floor and no dynamic
allocation. Under the regime-switching jump-diffusion every moment of its terminal wealth solves a
two-dimensional linear ODE, one equation per starting regime, so the mean and the variance are
exact matrix exponentials (Sepp, 2026, Proposition A.1). The package uses them as the benchmark
against which the [opportunity set](investment_opportunity_set.md) measures the cost of floor
protection.

## Overview

`bh_moments_rsjd` answers one question: what are the expected value and the standard deviation of
terminal wealth if the investor holds the asset, or a continuously rebalanced mandate reduced to
one [effective asset](mandate_aggregation.md), for $T$ years and consumes at a proportional rate
$c$? The answer conditions either on the starting regime or on a starting regime drawn from the
stationary distribution of the chain, and it requires no simulation and no Laplace inversion.

The chapter derives the moment equation, proves that consumption scales each moment by a
deterministic factor, and shows that the closed form $\Pi_0 e^{(\bar\mu_{\mathrm{stat}} - c)T}$ of
equation (6.23) of the manuscript is an approximation of the exact first moment that the package
does not use.

## Inputs, notation, and assumptions

| Convention | This article |
|---|---|
| Time and rates | Years; total returns, consumption and intensities are annual and continuously compounded |
| Regimes | Regime-conditional moments start in growth or in stress; the unconditional moment averages them with $p_1$ and $p_2$, the stationary probabilities |
| Jump sizes | Mean convention $\eta^{[i]}$; the $n$-th moment needs $n\eta^{[2]} \lt 1$ |
| Wealth coordinate | Terminal wealth $\Pi_T$ in units of initial wealth `Pi0`; standard deviations are in wealth units |
| Measure | Physical |
| Numerical method | `scipy.linalg.expm` of a $2 \times 2$ matrix; checked against exact simulation |
| Package default | `bh_moments_rsjd(T, Pi0, asset, c=0.0)`; the asset's floor and initial wealth are ignored |

| Symbol | Meaning | Units and convention |
|---|---|---|
| $\Pi_t^{\mathrm{BH}}$ | Buy-and-hold wealth | Wealth units, $\Pi_0^{\mathrm{BH}} = \Pi_0$ |
| $M_n^{[i]}$ | $\mathbb{E}[(\Pi_T^{\mathrm{BH}})^n \mid \chi_0 = i]$ | Wealth units to the power $n$ |
| $M_n$ | Stationary-weighted moment $p_1 M_n^{[1]} + p_2 M_n^{[2]}$ | As $M_n^{[i]}$ |
| $h_n^{[i]}(\tau)$ | Normalised moment $M_n^{[i]}/\Pi_0^n$ over horizon $\tau$ | Dimensionless |
| $\mathbf A_n$ | Matrix of the moment equation | Per year |
| $\bar\mu_{\mathrm{stat}}$ | Stationary total return $p_1\bar\mu^{[1]} + p_2\bar\mu^{[2]}$ | Per year |

The reserved symbols follow the [conventions page](conventions.md#reserved-notation). Buy-and-hold
wealth with consumption is $\Pi_t^{\mathrm{BH}} = \Pi_0 (S_t/S_0) e^{-ct}$: consumption withdraws
the fraction $c\thinspace dt$ of wealth in each instant.

## Methodology

### The moment equation

**Proposition (buy-and-hold moments; Sepp, 2026, Proposition A.1).** For $n \ge 1$ with
$n\eta^{[2]} \lt 1$, the normalised moments solve

$$
\frac{d}{d\tau}\begin{pmatrix} h_n^{[1]} \\ h_n^{[2]} \end{pmatrix} = \mathbf A_n \begin{pmatrix} h_n^{[1]} \\ h_n^{[2]} \end{pmatrix}, \qquad h_n^{[1]}(0) = h_n^{[2]}(0) = 1,
$$

with

$$
\mathbf A_n = \begin{pmatrix}
n(\mu^{[1]} - c) + \frac{n(n-1)}{2}(\sigma^{[1]})^2 - \lambda^{[12]} & \lambda^{[12]}\thinspace \mathbb{E}[e^{nJ^{[1]}}] \\
\lambda^{[21]}\thinspace \mathbb{E}[e^{nJ^{[2]}}] & n(\mu^{[2]} - c) + \frac{n(n-1)}{2}(\sigma^{[2]})^2 - \lambda^{[21]}
\end{pmatrix},
$$

$\mathbb{E}[e^{nJ^{[1]}}] = 1/(1 + n\eta^{[1]})$ and $\mathbb{E}[e^{nJ^{[2]}}] = 1/(1 - n\eta^{[2]})$.
Hence $(h_n^{[1]}(T), h_n^{[2]}(T))^{\top} = e^{\mathbf A_n T}(1, 1)^{\top}$ and
$M_n^{[i]} = \Pi_0^n h_n^{[i]}(T)$.

**Proof.** Write $u^{[i]}(t, \pi) = \mathbb{E}[(\Pi_T^{\mathrm{BH}})^n \mid \Pi_t = \pi, \chi_t = i]$.
By scaling, $u^{[i]} = \pi^n h_n^{[i]}(T - t)$. Between transitions, Itô's formula applied to
$\pi^n$ under $d\Pi/\Pi = (\mu^{[i]} - c)dt + \sigma^{[i]} dW$ gives the drift
$n(\mu^{[i]} - c) + n(n-1)(\sigma^{[i]})^2/2$ per unit of $\pi^n$. At rate $\lambda^{[i \to j]}$ the
regime switches and wealth is multiplied by $e^{J^{[i]}}$, which contributes
$\lambda^{[i \to j]}(\mathbb{E}[e^{nJ^{[i]}}] h_n^{[j]} - h_n^{[i]})$. Collecting terms gives
$\partial_\tau h_n = \mathbf A_n h_n$; at the horizon $u = \pi^n$, so $h_n(0) = (1, 1)^{\top}$. The
transforms are finite exactly when $n\eta^{[2]} \lt 1$, from the
[moment-generating function](regime_switching_model.md#jump-transforms-and-compensators). $\square$

For $n = 1$ the row sums of $\mathbf A_1$ are $\bar\mu^{[i]} - c$, the total expected returns net
of consumption, as in the [model chapter](regime_switching_model.md#total-return-diffusion-drift-and-log-drift).
For $n = 2$ the condition is $\eta^{[2]} \lt 1/2$, Assumption 2.2 of the manuscript.

### Consumption and stationary averaging

**Identity (consumption scaling).** With consumption rate $c$, every regime-conditional moment is
the moment without consumption times $e^{-ncT}$:
$M_n^{[i]}(c) = e^{-ncT} M_n^{[i]}(0)$. The implied return falls by exactly $c$, and the standard
deviation is scaled by $e^{-cT}$.

**Proof.** $\mathbf A_n(c) = \mathbf A_n(0) - nc\thinspace \mathbf{I}$, and the identity matrix commutes with
$\mathbf A_n(0)$, so $e^{\mathbf A_n(c)T} = e^{-ncT}e^{\mathbf A_n(0)T}$. $\square$

**Definition (unconditional moments).** The unconditional moments draw the starting regime from
the stationary distribution,

$$
M_n = p_1 M_n^{[1]} + p_2 M_n^{[2]}, \qquad
\mathrm{Std}\left[\Pi_T^{\mathrm{BH}}\right] = \sqrt{M_2 - M_1^2}, \qquad
r_{\mathrm{impl}}^{\mathrm{BH}} = \frac{1}{T}\ln\frac{M_1}{\Pi_0} .
$$

The variance is that of the mixture over starting regimes, so it includes the uncertainty about the
current regime as well as the dispersion within each regime.

### The stationary-return approximation

Equation (6.23) of the manuscript writes the buy-and-hold expected wealth as
$\Pi_0 e^{(\bar\mu_{\mathrm{stat}} - c)T}$. This is exact when $\bar\mu^{[1]} = \bar\mu^{[2]}$, by
the [proposition on total expected return](regime_switching_model.md#total-return-diffusion-drift-and-log-drift),
and an approximation otherwise: the exact first moment compounds each regime's return over its own
random duration, and the regimes are persistent. `bh_moments_rsjd` computes the exact moment of the
proposition above; the paper's Table 2 reports the exact values as well.

> **Insight.** For the balanced mandate of the worked example the approximation gives 149.34 against
> the exact 150.64, an implied return lower by 0.09 percentage points. The gap is small at a
> ten-year horizon but it is not a rounding difference, and it carries into any floor protection
> cost computed against it.

## Worked example

The balanced mandate of Table 2 of the manuscript holds 35% bonds and splits the rest two to one
between equity and private equity. [Mandate aggregation](mandate_aggregation.md) reduces it to one
effective asset with growth and stress volatilities of 11.14% and 16.71%, total returns of 4.34%
and 0.70%, and mean jumps $\eta^{[1]} = 0.2333$ and $\eta^{[2]} = 0.1154$; the floor parameter `k`
does not affect buy-and-hold moments. Over ten years from initial wealth 100:

| Starting regime | Expected wealth | Standard deviation | Implied return |
|---|---:|---:|---:|
| Growth | 151.04 | | |
| Stress | 146.62 | | |
| Stationary (10/11 growth) | 150.64 | 74.85 | 4.097% |

The stationary values are those of the buy-and-hold column of Table 2 of the manuscript, 151, 74.8
and 4.10%. The block recomputes both moments with its own matrix exponential, checks the
consumption identity at $c = 2.5$%, and evaluates the approximation of equation (6.23).

```python
import numpy as np
from scipy.linalg import expm
import goal_based_allocation as gba

w_bd, q = 0.35, 2.0 / 3.0
balanced = gba.build_effective_asset(w_eq=q * (1.0 - w_bd), w_pe=(1.0 - q) * (1.0 - w_bd), k=1.5)
p = balanced.params
np.testing.assert_allclose([p.sigma0, p.sigma1, balanced.mu_growth, balanced.mu_stress],
                           [0.1114, 0.1671, 0.04342, 0.0070], atol=5e-5)
np.testing.assert_allclose([p.eta0, p.eta1], [0.2333, 0.1154], atol=5e-5)

horizon, wealth0 = 10.0, 100.0
bh = gba.bh_moments_rsjd(horizon, wealth0, balanced, c=0.0)
np.testing.assert_allclose([bh['E_growth'], bh['E_stress'], bh['E'], bh['Std']],
                           [151.04, 146.62, 150.64, 74.85], atol=5e-3)
np.testing.assert_allclose(bh['r_impl'], 0.04097, atol=5e-6)


def moment_matrix(asset, n, consumption=0.0):
    """The matrix A_n of the moment equation, built from the model definitions."""
    pr = asset.params
    alpha = (-pr.eta0 / (1.0 + pr.eta0), pr.eta1 / (1.0 - pr.eta1))
    mu = (asset.mu_growth - pr.lambda01 * alpha[0] - consumption,
          asset.mu_stress - pr.lambda10 * alpha[1] - consumption)
    return np.array([
        [n * mu[0] + 0.5 * n * (n - 1) * pr.sigma0**2 - pr.lambda01, pr.lambda01 / (1.0 + n * pr.eta0)],
        [pr.lambda10 / (1.0 - n * pr.eta1), n * mu[1] + 0.5 * n * (n - 1) * pr.sigma1**2 - pr.lambda10],
    ])


stationary = np.array([bh['p1'], bh['p2']])
first = wealth0 * expm(moment_matrix(balanced, 1) * horizon) @ np.ones(2)
second = wealth0**2 * expm(moment_matrix(balanced, 2) * horizon) @ np.ones(2)
np.testing.assert_allclose(first, [bh['E_growth'], bh['E_stress']], rtol=1e-12)
np.testing.assert_allclose(stationary @ first, bh['E'], rtol=1e-12)
np.testing.assert_allclose(np.sqrt(stationary @ second - (stationary @ first)**2), bh['Std'], rtol=1e-12)

# consumption scales the mean and the standard deviation by exp(-c T)
bh_consuming = gba.bh_moments_rsjd(horizon, wealth0, balanced, c=0.025)
np.testing.assert_allclose([bh_consuming['E'] / bh['E'], bh_consuming['Std'] / bh['Std']],
                           [np.exp(-0.25), np.exp(-0.25)], rtol=1e-12)
np.testing.assert_allclose(bh_consuming['r_impl'], bh['r_impl'] - 0.025, rtol=1e-12)

# equation (6.23): exp of the stationary total return understates the exact first moment
mu_stat = stationary @ np.array([balanced.mu_growth, balanced.mu_stress])
approximation = wealth0 * np.exp(mu_stat * horizon)
np.testing.assert_allclose([mu_stat, approximation], [0.040106, 149.34], atol=5e-3)
np.testing.assert_allclose(np.log(bh['E'] / approximation) / horizon, 0.000864, atol=5e-7)
```

The second block simulates the same effective asset exactly, as in the
[model chapter](regime_switching_model.md#worked-example), from 400,000 starting regimes drawn from
the stationary distribution and from 400,000 starts in growth. The simulated means are 150.63 and
151.07 with standard errors of 0.12, against the exact 150.64 and 151.04, and the simulated
standard deviation is 74.94 against 74.85.

```python
def simulate_buy_and_hold(asset, horizon, start, seed):
    """Exact simulation of buy-and-hold wealth from given starting regimes, initial wealth 100."""
    pr = asset.params
    rng = np.random.default_rng(seed)
    nu = np.array([asset.nu0, asset.nu1])
    vol = np.array([pr.sigma0, pr.sigma1])
    rate = np.array([pr.lambda01, pr.lambda10])
    log_wealth, clock = np.zeros((2, start.size))
    regime = start.copy()
    active = np.ones(start.size, dtype=bool)
    while active.any():
        i = np.flatnonzero(active)
        now = regime[i]
        hold = rng.exponential(1.0 / rate[now])
        remaining = horizon - clock[i]
        step = np.minimum(hold, remaining)
        log_wealth[i] += nu[now] * step + vol[now] * np.sqrt(step) * rng.standard_normal(i.size)
        clock[i] += step
        switch = i[hold < remaining]
        log_wealth[switch] += np.where(regime[switch] == 0,
                                       -rng.exponential(pr.eta0, switch.size),
                                       rng.exponential(pr.eta1, switch.size))
        regime[switch] = 1 - regime[switch]
        active[i[hold >= remaining]] = False
    return 100.0 * np.exp(log_wealth)


n_paths = 400_000
stationary_start = (np.random.default_rng(5).uniform(size=n_paths) > bh['p1']).astype(int)
wealth = simulate_buy_and_hold(balanced, horizon, stationary_start, seed=7)
wealth_growth = simulate_buy_and_hold(balanced, horizon, np.zeros(n_paths, dtype=int), seed=8)
for sample, exact in ((wealth, bh['E']), (wealth_growth, bh['E_growth'])):
    standard_error = sample.std() / np.sqrt(n_paths)
    assert abs(sample.mean() - exact) < 3.0 * standard_error
np.testing.assert_allclose([wealth.mean(), wealth_growth.mean(), wealth.std()],
                           [150.63, 151.07, 74.94], atol=5e-3)
np.testing.assert_allclose(wealth.std(), bh['Std'], rtol=0.01)
```

## Implementation in goal_based_allocation

| Quantity | Formula | Entry point |
|---|---|---|
| Regime-conditional means | $M_1^{[1]}$, $M_1^{[2]}$ | `bh_moments_rsjd(T, Pi0, asset, c=0.0)['E_growth']`, `['E_stress']` |
| Stationary mean | $M_1 = p_1 M_1^{[1]} + p_2 M_1^{[2]}$ | `bh_moments_rsjd(...)['E']` |
| Variance and standard deviation | $\max(0, M_2 - M_1^2)$ and its root | `['Var']`, `['Std']` |
| Implied return | $\ln(M_1/\Pi_0)/T$ | `['r_impl']` |
| Stationary probabilities | $p_1$, $p_2$ | `['p1']`, `['p2']` |

The function lives in
[`regime_switch_paper.py`](https://github.com/ArturSepp/GoalBasedAllocation/blob/main/src/goal_based_allocation/regime_switch_paper.py).
API page: {doc}`bh_moments_rsjd <api/generated/goal_based_allocation.bh_moments_rsjd>`.

Contract details:

- The function reads `asset.params`, `mu_growth` and `mu_stress`; it ignores `asset.pi0` and
  `asset.pi_floor`. Initial wealth is the argument `Pi0`, and buy-and-hold has no floor.
- `c` is subtracted from the diffusion drift in both regimes, so the consumption identity holds
  exactly.
- Only the stationary-weighted variance is returned; the regime-conditional second moments are not.
- `r_impl` is zero when the mean is not positive or the horizon is below $10^{-10}$ years.
- The inputs are not validated. With $2\eta^{[2]} \ge 1$ the second-moment transform is infinite,
  and the formula returns a finite but meaningless number instead of an error.
- [`compute_opportunity_point`](investment_opportunity_set.md) reports `E_BH`, `Std_BH` and
  `r_impl_BH` from this function, with the stationary weights, and integrates its mean over time
  for the discounted consumption of the benchmark.

> **Pitfall.** The opportunity set compares the stationary buy-and-hold mean with floor-protected
> quantities computed from a start in growth. For the balanced mandate the buy-and-hold mean from
> growth is 151.04, 0.40 above the stationary 150.64, so the reported floor protection cost mixes
> two conditioning conventions. Use `E_growth` for a like-for-like comparison from growth.

## Interpretation and limitations

- Only the mean and the variance are returned. Buy-and-hold terminal wealth is right-skewed, so
  its median lies below its mean; the standard deviation alone does not describe the left tail
  that the floor removes.
- The moments are exact for the model, not for the mandate: a mandate enters through its effective
  asset, whose jump sizes are mean-matched approximations, as the
  [mandate chapter](mandate_aggregation.md) explains.
- The unconditional variance includes the uncertainty about the starting regime. An investor who
  knows the current regime should use the regime-conditional moment.
- Higher moments follow from the same equation for any $n$ with $n\eta^{[2]} \lt 1$, but the
  package computes only $n = 1$ and $n = 2$.

## See also

- [The regime-switching jump-diffusion](regime_switching_model.md)
- [Mandates as one effective asset](mandate_aggregation.md)
- [The investment opportunity set and investor selection](investment_opportunity_set.md)
- [The terminal wealth distribution](terminal_wealth_distribution.md)
- [Bibliography](bibliography.md)

## References

1. Sepp, A. (2026). Dynamic Mean-Variance Portfolio Allocation under Regime-Switching Jump-Diffusions with Absorbing Barriers and Distribution Matching. Working paper. [DOI: 10.2139/ssrn.6534579](https://doi.org/10.2139/ssrn.6534579). Proposition A.1 gives the moment equation; equation (6.23) the stationary-return approximation; Table 2 the benchmark values.
2. Sepp, A. goal-based-allocation: Semi-analytical dynamic mean-variance allocation and terminal-wealth risk under regime-switching jump-diffusions. [Software citation metadata](https://github.com/ArturSepp/GoalBasedAllocation/blob/main/CITATION.cff).
