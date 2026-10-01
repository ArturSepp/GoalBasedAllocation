---
myst:
  html_meta:
    description: >-
      Mandate aggregation in goal-based-allocation: a fixed-weight bond, equity and private-equity
      mandate reduced to one regime-switching jump-diffusion, with exact portfolio volatility and
      total return, mean-matched portfolio jump sizes by quadrature, and the floor set from a
      drawdown scale, reproducing Table 2 of the manuscript.
---

# Mandates as one effective asset

*Author: [Artur Sepp](https://github.com/ArturSepp)*

Implemented in [GoalBasedAllocation](https://github.com/ArturSepp/GoalBasedAllocation).
Software citation: [CITATION.cff](https://github.com/ArturSepp/GoalBasedAllocation/blob/main/CITATION.cff).

A mandate is a continuously rebalanced portfolio of asset classes with fixed weights. When the asset
classes share one regime chain, the mandate is again a regime-switching jump-diffusion, with an exact
diffusion volatility and total return in each regime but a crash jump that is not exponential. The
package replaces that jump by an exponential with the same mean log-size and so reduces the mandate to
one effective asset, on which the [mean-variance policy](mv_optimal_policy.md) and the
[Laplace framework](laplace_barrier_framework.md) operate (Sepp, 2026, Section 7.2). This chapter
derives the reduction, states which parts of it are exact, and reproduces the four mandates of Table 2
of the manuscript.

## Overview

`build_effective_asset(w_eq, w_pe, k)` takes the equity and private-equity weights of a mandate over
the three asset classes of [Table 1](regime_switching_model.md#the-papers-asset-classes), with bonds
taking the rest, and a floor parameter, and returns an `AssetSpecification`. `portfolio_sigma_unc`
returns the stationary-weighted diffusion volatility used to set the floor from a drawdown tolerance,
and `portfolio_eta_quadrature` the mean-matched jump size of any weighted portfolio.

The investment problem then has one risky asset and cash, and the dynamic allocation of the policy
moves wealth between them; the composition of the risky sleeve stays fixed. The
[opportunity set](investment_opportunity_set.md) scans the bond weight along the curve
$w_{\mathrm{eq}} = q(1 - w_{\mathrm{bd}})$, $w_{\mathrm{pe}} = (1 - q)(1 - w_{\mathrm{bd}})$.

## Inputs, notation, and assumptions

| Convention | This article |
|---|---|
| Time and rates | Years; total returns and volatilities annual; the floor uses $r_c = 2$% and $T = 10$ years inside the package |
| Regimes | The asset classes share the chain with $\lambda^{[12]} = 0.1$ and $\lambda^{[21]} = 1$; the stationary probabilities $10/11$ and $1/11$ weight the unconditional volatility |
| Jump sizes | Mean convention; the jump sizes of different asset classes are independent given a transition, and the portfolio jump is mean-matched in log terms |
| Wealth coordinate | Mandate value from initial wealth 100; the floor is $L_0 = 100e^{-k\eta_{\mathrm{port}}^{[1]}}$ |
| Measure | Physical |
| Numerical method | Exact formulas for volatility and total return; `scipy.integrate.quad` or `nquad` over the independent exponential jumps; checked by sampling the jumps |
| Package default | `build_effective_asset(w_eq, w_pe, k)` uses the classes of `create_paper_assets()`, the correlation matrix of equation (7.1) and $\Pi_0 = 100$; `portfolio_eta_quadrature(weights, etas, crash=True)` |

| Symbol | Meaning | Units and convention |
|---|---|---|
| $w = (w_{\mathrm{bd}}, w_{\mathrm{eq}}, w_{\mathrm{pe}})$ | Mandate weights | Non-negative, sum to one |
| $\sigma_j^{[i]}$, $\bar\mu_j^{[i]}$, $\eta_j^{[i]}$ | Parameters of asset class $j$ in regime $i$ | As in Table 1 |
| $\rho$ | Correlation matrix of the asset diffusions | Constant across regimes |
| $\mathbf{C}^{[i]}$ | Diffusion covariance $\rho_{jk}\sigma_j^{[i]}\sigma_k^{[i]}$ | Per year |
| $\sigma_{\mathrm{port}}^{[i]}$, $\bar\mu_{\mathrm{port}}^{[i]}$, $\eta_{\mathrm{port}}^{[i]}$ | Effective volatility, total return and mean jump of the mandate | Per year; log units |
| $\sigma_{\mathrm{unc}}$ | Unconditional diffusion volatility $\sqrt{p_1(\sigma_{\mathrm{port}}^{[1]})^2 + p_2(\sigma_{\mathrm{port}}^{[2]})^2}$ | Per year |
| $q_{\mathrm{dd}}$ | Drawdown scale, in units of $\sigma_{\mathrm{unc}}$ | Typically 1 to 3 |
| $q$ | Equity share of the risky sleeve, the manuscript's $g$ in equation (7.5) | Default $2/3$ |

The reserved symbols follow the [conventions page](conventions.md#reserved-notation). The manuscript
calls the equity share $g$; the handbook writes $q$, as `AdvisorSpec.q` does, because $g$ is the target
growth rate elsewhere.

## Methodology

### Volatility and total return are exact

**Proposition (exact effective diffusion and total return).** A continuously rebalanced mandate
with weights $w$ has, in regime $i$, diffusion volatility and total expected return

$$
\sigma_{\mathrm{port}}^{[i]} = \sqrt{w^{\top}\mathbf{C}^{[i]}w}, \qquad
\bar\mu_{\mathrm{port}}^{[i]} = \sum_jw_j\bar\mu_j^{[i]},
$$

equations (7.2)–(7.3) of the manuscript.

**Proof.** Between transitions the mandate's return is $\sum_jw_j\thinspace dS_j/S_j$, a Gaussian increment
with variance $w^{\top}\mathbf{C}^{[i]}w\thinspace dt$. Its instantaneous expected return, including the
expected relative change at a transition, is linear in the weights, and for each asset it is
$\bar\mu_j^{[i]}$ by the [proposition on total expected return](regime_switching_model.md#total-return-diffusion-drift-and-log-drift). $\square$

### The portfolio jump is mean-matched

At a crash every asset class jumps at once, by independent exponential log-sizes. The mandate's
relative change is $\sum_jw_j(e^{J_j^{[1]}} - 1)$ and its log-jump $\ln\sum_jw_je^{J_j^{[1]}}$, which is
not exponential.

**Definition (mean-matched portfolio jump; equation (7.4)).**

$$
\eta_{\mathrm{port}}^{[1]} = -\mathbb{E}\left[\ln\sum_jw_je^{J_j^{[1]}}\right], \qquad
\eta_{\mathrm{port}}^{[2]} = \mathbb{E}\left[\ln\sum_jw_je^{J_j^{[2]}}\right],
$$

with independent $J_j^{[1]} = -E_j$ and $J_j^{[2]} = E_j'$ exponential with means $\eta_j^{[1]}$ and
$\eta_j^{[2]}$. The effective asset jumps by exponential log-sizes with these means.

**Identity (crash loss of the mandate).** The exact expected relative loss of the mandate at a crash
is the weighted loss $\sum_jw_j\eta_j^{[1]}/(1 + \eta_j^{[1]})$; the effective asset's is
$\eta_{\mathrm{port}}^{[1]}/(1 + \eta_{\mathrm{port}}^{[1]})$, and the two differ.

**Proof.** The first follows from linearity and the
[compensator identity](regime_switching_model.md#jump-transforms-and-compensators);
the second is the compensator of an exponential log-jump with mean $\eta_{\mathrm{port}}^{[1]}$. The
mean is matched in log terms, not in relative terms, so the relative means coincide only for a single
asset class. $\square$

For the balanced mandate the exact expected loss is 20.13%, the effective asset's is 18.91%, and the
loss at the mean log-jump, $1 - e^{-\eta_{\mathrm{port}}^{[1]}}$, is 20.81%; Table 2 of the manuscript
reports the last as "crash loss", while Table 1 reports the first definition for the asset classes. The
effective asset keeps the exact total return $\bar\mu_{\mathrm{port}}^{[i]}$, so the error of the
reduction is confined to the split of that return between diffusion drift and jump compensation, and to
the shape of the jump law.

> **Pitfall.** The effective asset's diffusion drift is $\bar\mu_{\mathrm{port}}^{[i]} - \lambda\alpha_{\mathrm{port}}^{[i]}$
> with the mean-matched compensator, not the weighted diffusion drift of the asset classes. For the
> balanced mandate in growth the two are 6.23% and 6.36%. Compare effective assets with mandates through
> total returns, which are exact.

### The floor from a drawdown scale

The investor states a drawdown tolerance $q_{\mathrm{dd}}$ in units of the mandate's unconditional
volatility. The manuscript sets the terminal floor to

$$
L_T = x_{25} = \Pi_0e^{-q_{\mathrm{dd}}\sigma_{\mathrm{unc}}}, \qquad
k = \frac{q_{\mathrm{dd}}\sigma_{\mathrm{unc}} + r_cT}{\eta_{\mathrm{port}}^{[1]}}, \qquad
L_0 = \Pi_0e^{-k\eta_{\mathrm{port}}^{[1]}},
$$

equations (7.7)–(7.9).

**Identity (the floor does not depend on the jump size).** The initial log distance to the floor is
$x_0 = \ln(\Pi_0/L_0) = q_{\mathrm{dd}}\sigma_{\mathrm{unc}} + r_cT$.

**Proof.** Substitute $k$ into $L_0$; the jump mean cancels. $\square$

The floor parameter $k$ reparametrises the floor in crash units, which keeps the
[single-crash breach probability](regime_switching_model.md#the-floor-rule-of-the-papers-asset-classes)
$e^{-k}$ interpretable, but the floor itself is set by the volatility, the tolerance and the cash rate.

## Worked example

The four mandates of Table 2 lie on the bond-weight curve with $q = 2/3$: income (100/0/0),
conservative (65/23.3/11.7), balanced (35/43.3/21.7) and growth (0/66.7/33.3). With $q_{\mathrm{dd}} = 2$
the package reproduces every portfolio parameter of the table:

| Mandate | $\sigma_{\mathrm{port}}^{[1]}$ / $\sigma_{\mathrm{port}}^{[2]}$ | $\bar\mu_{\mathrm{port}}^{[1]}$ / $\bar\mu_{\mathrm{port}}^{[2]}$ | $\eta_{\mathrm{port}}^{[1]}$ / $\eta_{\mathrm{port}}^{[2]}$ | $L_0$ / $x_0$ |
|---|---|---|---|---|
| Income | 6.0 / 9.0% | 2.50 / 2.00% | 0.087 / 0.048 | 72.1 / 0.327 |
| Conservative | 7.7 / 11.6% | 3.49 / 1.30% | 0.161 / 0.085 | 69.6 / 0.363 |
| Balanced | 11.1 / 16.7% | 4.34 / 0.70% | 0.233 / 0.115 | 64.7 / 0.435 |
| Growth | 15.8 / 23.8% | 5.33 / 0.00% | 0.334 / 0.148 | 58.6 / 0.534 |

The block also checks the volatility against the covariance formula with the correlation matrix of
equation (7.1), the three definitions of the balanced crash loss, the identity for $x_0$, and the floor
parameter $k = 1.8655$ of the balanced mandate.

```python
import numpy as np
import goal_based_allocation as gba

assets = gba.create_paper_assets()
classes = [assets['bonds'], assets['equity'], assets['private_equity']]
correlation = np.array([[1.0, 0.3, 0.3], [0.3, 1.0, 0.8], [0.3, 0.8, 1.0]])
q, q_dd, r_c, horizon = 2.0 / 3.0, 2.0, 0.02, 10.0

table_2 = {  # bond weight: (sigma growth, sigma stress, mu growth, mu stress, eta crash, eta recovery, L0, x0)
    1.00: (6.0, 9.0, 2.50, 2.00, 0.087, 0.048, 72.1, 0.327),
    0.65: (7.7, 11.6, 3.49, 1.30, 0.161, 0.085, 69.6, 0.363),
    0.35: (11.1, 16.7, 4.34, 0.70, 0.233, 0.115, 64.7, 0.435),
    0.00: (15.8, 23.8, 5.33, 0.00, 0.334, 0.148, 58.6, 0.534),
}
rounding = np.array([0.05, 0.05, 0.005, 0.005, 0.0005, 0.0005, 0.05, 0.0005])  # half a printed digit
effective = {}
for w_bd, row in table_2.items():
    w = np.array([w_bd, q * (1.0 - w_bd), (1.0 - q) * (1.0 - w_bd)])
    asset = gba.build_effective_asset(w_eq=w[1], w_pe=w[2], k=1.5)
    effective[w_bd] = asset
    p = asset.params
    sigma_unc = gba.portfolio_sigma_unc(w[1], w[2])
    x0 = q_dd * sigma_unc + r_c * horizon
    computed = (100 * p.sigma0, 100 * p.sigma1, 100 * asset.mu_growth, 100 * asset.mu_stress,
                p.eta0, p.eta1, 100.0 * np.exp(-x0), x0)
    assert np.all(np.abs(np.array(computed) - np.array(row)) <= rounding), (w_bd, computed)
    # exact volatility from the covariance matrix, stationary-weighted unconditional volatility
    for regime, vol in enumerate((p.sigma0, p.sigma1)):
        sig = np.array([[a.params.sigma0, a.params.sigma1][regime] for a in classes])
        np.testing.assert_allclose(vol, np.sqrt(w @ (correlation * np.outer(sig, sig)) @ w), rtol=1e-12)
    np.testing.assert_allclose(sigma_unc, np.sqrt((10 * p.sigma0**2 + p.sigma1**2) / 11), rtol=1e-12)

# the balanced mandate: floor parameter, x0 independent of the jump size, three crash losses
balanced = effective[0.35]
w = np.array([0.35, q * 0.65, (1.0 - q) * 0.65])
eta_port = balanced.params.eta0
x0 = q_dd * gba.portfolio_sigma_unc(w[1], w[2]) + r_c * horizon
k = x0 / eta_port
np.testing.assert_allclose(k, 1.8655, atol=5e-5)
np.testing.assert_allclose(np.log(100.0 / (100.0 * np.exp(-k * eta_port))), x0, rtol=1e-12)
exact_loss = w @ np.array([a.params.eta0 / (1.0 + a.params.eta0) for a in classes])
effective_loss = eta_port / (1.0 + eta_port)
loss_at_mean = 1.0 - np.exp(-eta_port)
np.testing.assert_allclose([exact_loss, effective_loss, loss_at_mean], [0.2013, 0.1891, 0.2081], atol=5e-5)

# effective diffusion drift in growth against the weighted drift of the classes
weighted_drift = w @ np.array([a.mu_growth + a.params.lambda01 * a.params.eta0 / (1.0 + a.params.eta0)
                               for a in classes])
effective_drift = balanced.mu_growth + balanced.params.lambda01 * effective_loss
np.testing.assert_allclose([effective_drift, weighted_drift], [0.0623, 0.0636], atol=5e-5)
```

The second block checks the quadrature independently. Two million draws of the three independent
exponential crash and recovery jumps of the balanced mandate give a mean log-jump of 0.23325, with a
standard error of 0.00009, against the quadrature's 0.23327, and 0.11539 against 0.11536 for the
recovery; the sampled expected relative loss is 20.13%.

```python
rng = np.random.default_rng(7)
crash_means = np.array([a.params.eta0 for a in classes])
recovery_means = np.array([a.params.eta1 for a in classes])
crash_draws = rng.exponential(crash_means, size=(2_000_000, 3))
recovery_draws = rng.exponential(recovery_means, size=(2_000_000, 3))
crash_log = -np.log(np.exp(-crash_draws) @ w)
recovery_log = np.log(np.exp(recovery_draws) @ w)

quadrature = (gba.portfolio_eta_quadrature(w, crash_means, crash=True),
              gba.portfolio_eta_quadrature(w, recovery_means, crash=False))
np.testing.assert_allclose(quadrature, [balanced.params.eta0, balanced.params.eta1], rtol=1e-12)
for sample, value in zip((crash_log, recovery_log), quadrature):
    assert abs(sample.mean() - value) < 3.0 * sample.std() / np.sqrt(sample.size)
np.testing.assert_allclose([crash_log.mean(), recovery_log.mean()], [0.23325, 0.11539], atol=5e-6)
np.testing.assert_allclose(1.0 - (np.exp(-crash_draws) @ w).mean(), exact_loss, atol=3e-4)
```

## Implementation in goal_based_allocation

| Quantity | Formula | Entry point |
|---|---|---|
| Effective asset of a mandate | Equations (7.2)–(7.4), floor $L_0 = 100e^{-k\eta_{\mathrm{port}}^{[1]}}$ | `build_effective_asset(w_eq, w_pe, k)` |
| Unconditional diffusion volatility | $\sqrt{p_1(\sigma_{\mathrm{port}}^{[1]})^2 + p_2(\sigma_{\mathrm{port}}^{[2]})^2}$ | `portfolio_sigma_unc(w_eq, w_pe)` |
| Mean-matched jump of any portfolio | Equation (7.4) by quadrature | `portfolio_eta_quadrature(weights, etas, crash=True)` |
| Effective asset of a `MandateSpecification` | Same, floor the weighted asset floors | `mandate_utils.mandate_effective_asset(mandate)` (module level) |

The functions live in
[`client_solver.py`](https://github.com/ArturSepp/GoalBasedAllocation/blob/main/src/goal_based_allocation/client_solver.py)
and [`mandate_utils.py`](https://github.com/ArturSepp/GoalBasedAllocation/blob/main/src/goal_based_allocation/mandate_utils.py).
API pages: {doc}`build_effective_asset <api/generated/goal_based_allocation.build_effective_asset>`,
{doc}`portfolio_sigma_unc <api/generated/goal_based_allocation.portfolio_sigma_unc>` and
{doc}`portfolio_eta_quadrature <api/generated/goal_based_allocation.portfolio_eta_quadrature>`.

Contract details:

- `build_effective_asset` and `portfolio_sigma_unc` always use the three classes of
  `create_paper_assets()`, the correlation matrix of equation (7.1), intensities 0.1 and 1, and
  initial wealth 100; only the weights and `k` are inputs. The bond weight is `1 - w_eq - w_pe` and is
  not checked for sign. `portfolio_sigma_unc` uses the stationary probabilities $10/11$ and $1/11$ of
  those intensities.
- `build_effective_asset` names its result `'client'` and sets `pi_floor` to
  $100e^{-k\eta_{\mathrm{port}}^{[1]}}$. It evaluates a two- or three-dimensional `nquad` for each of
  the two jumps, which takes a few seconds for three classes; replacing only the floor of an existing
  effective asset, with `dataclasses.replace`, avoids recomputing them.
- `portfolio_eta_quadrature` ignores classes with a weight below $10^{-10}$ or a jump mean below
  $10^{-12}$ and treats their weight as a riskless remainder. With one active class and no remainder
  it returns that class's mean exactly. Integration is truncated at $\max(50\eta_j, 20)$ per class.
- `mandate_utils.mandate_effective_asset` builds the effective asset of a `MandateSpecification` over
  the first classes of the correlation matrix, takes the intensities of the first class, and sets the
  floor to the weighted average of the classes' own floors rather than from a drawdown scale. The
  module's `simulate_mandate_mc` and `compute_mandate_analytical` hard-code $r = 2$% and $c = 3$% and
  are not used by the public functions.

## Interpretation and limitations

- The reduction treats the mandate as continuously rebalanced to fixed weights; drift of the weights
  between rebalancing dates and transaction costs are not modelled. The dynamic allocation of the
  policy is between the whole mandate and cash.
- The jump of the effective asset is exact in its mean log-size only. Its relative mean and its tail
  differ from the mandate's, and the error grows with the dispersion of the classes' crash sizes;
  for single-class mandates it vanishes.
- The correlation matrix is the same in both regimes. The common crash and recovery jumps add
  co-movement that the diffusion correlations do not show, so realised correlations of the classes
  are higher than $\rho$, as the manuscript notes.
- The name $x_{25}$ suggests that the floor is the 25th percentile of terminal wealth; it is not. The
  probability of ending at or below the floor is $F + \mathcal{O}$, 21.3% for the balanced mandate in
  the [terminal wealth chapter](terminal_wealth_distribution.md#worked-example).

## See also

- [The regime-switching jump-diffusion](regime_switching_model.md)
- [The investment opportunity set and investor selection](investment_opportunity_set.md)
- [The terminal wealth distribution](terminal_wealth_distribution.md)
- [Buy-and-hold moments](buy_and_hold_moments.md)
- [Model boundaries](model-boundaries.md)
- [Bibliography](bibliography.md)

## References

1. Sepp, A. (2026). Dynamic Mean-Variance Portfolio Allocation under Regime-Switching Jump-Diffusions with Absorbing Barriers and Distribution Matching. Working paper. [DOI: 10.2139/ssrn.6534579](https://doi.org/10.2139/ssrn.6534579). Section 7.2 defines the effective asset and the floor calibration; Table 2 reports the mandates.
2. Sepp, A., Ossa, I., and Kastenholz, M. (2026). Robust Optimization of Strategic and Tactical Asset Allocation for Multi-Asset Portfolios. *The Journal of Portfolio Management*, 52(4), 86–120. [DOI: 10.3905/jpm.2025.1.806](https://doi.org/10.3905/jpm.2025.1.806). Strategic and tactical allocation across fixed-weight mandates.
3. Sepp, A. goal-based-allocation: Semi-analytical dynamic mean-variance allocation and terminal-wealth risk under regime-switching jump-diffusions. [Software citation metadata](https://github.com/ArturSepp/GoalBasedAllocation/blob/main/CITATION.cff).
