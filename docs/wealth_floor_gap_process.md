---
myst:
  html_meta:
    description: >-
      The absorbing wealth floor of goal-based-allocation: stopping at the floor, the gap and
      log-gap processes, the flat-barrier reduction to a regime-switching jump-diffusion, the
      mean-matched effective jump size and its Taylor expansion, the three components of stopped
      wealth, and a decomposition of the reduction error by exact simulation.
---

# The wealth floor and the flat-barrier reduction

*Author: [Artur Sepp](https://github.com/ArturSepp)*

Implemented in [GoalBasedAllocation](https://github.com/ArturSepp/GoalBasedAllocation).
Software citation: [CITATION.cff](https://github.com/ArturSepp/GoalBasedAllocation/blob/main/CITATION.cff).

A wealth floor is a level, growing at the net floor rate, below which the strategy stops investing
and holds cash to the horizon. Under the mean-variance policy the logarithm of the ratio of two
distances, from the target to the floor and from the target to wealth, follows a regime-switching
jump-diffusion with a fixed barrier at zero (Sepp, 2026, Section 4). This flat-barrier reduction is
what lets the [Laplace framework](laplace_barrier_framework.md) compute survival, the terminal
distribution and its moments without simulation. This chapter derives the reduction, shows which of
its steps are exact, and measures the error of the one that is not.

## Overview

`gap_process_asset` takes a `RiccatiSolution` from [`find_ell`](mv_optimal_policy.md) and returns
an `AssetSpecification` for the log-gap process. Its starting point `x0` is the initial log-gap, its
barrier is at zero, and its diffusion and jump parameters are the effective ones of Theorem 4.3 of
the manuscript. Every Laplace-domain function of the package then treats it like any other asset.

The reduction has three ingredients. The change of variables to the log-gap maps the moving floor
to a fixed barrier exactly. Between regime transitions the log-gap is exactly a Brownian motion with
drift once the allocation coefficient is held constant, because the target grows exactly at the net
floor rate. At a crash the log-gap jumps by a bounded amount whose law is not exponential; the
package replaces it by an exponential with the same mean. The worked example shows that the Laplace
computation is exact for the reduced process and that the exponential approximation of the jump is
the source of the difference between the analytical survival and a simulation of wealth.

## Inputs, notation, and assumptions

| Convention | This article |
|---|---|
| Time and rates | Years; the floor grows at $r_c$; the allocation coefficients are frozen at their values at inception, $\tau = T$ |
| Regimes | The gap process starts in growth; each regime uses the magnitude $\lvert\omega_a^{[i]}\rvert$ of its own coefficient |
| Jump sizes | Effective means $\tilde\eta^{[i]}$ of the gap jumps, by mean-matching; the crash formula is also applied to the recovery |
| Wealth coordinate | Log-gap $X_t = \ln(B_t/Z_t)$ with the barrier at $X = 0$; the gap asset stores $B_0$ as `pi0` and $Z_0$ as `pi_floor`, so `x0` is $\ln(B_0/Z_0)$ |
| Measure | Physical |
| Numerical method | `scipy.integrate.quad` for $\tilde\eta^{[i]}$; checked by exact simulation of the gap process with exponential and with exact jump laws, and by Euler simulation of wealth |
| Package default | `gap_process_asset(ric)`; `create_paper_assets(k_floor=1.5)`; `riccati_solver.simulate_mv_optimal(ric, n_paths=100_000, steps_per_year=260, seed=42)` |

| Symbol | Meaning | Units and convention |
|---|---|---|
| $\Pi_T^{\mathrm{stop}}$ | Terminal wealth of the stopped strategy | Wealth units |
| $\kappa^{[i]}$ | Frozen allocation magnitude $\lvert\omega_a^{[i]}(\tau = T)\rvert$ | Dimensionless |
| $\sigma_{\mathrm{eff}}^{[i]}$, $\nu_{\mathrm{eff}}^{[i]}$ | Diffusion volatility and log drift of the log-gap | Per year |
| $\Delta X$ | Jump of the log-gap at a transition | Log units |
| $\varepsilon$ | Deviation $\kappa^{[1]} - 1$ from the natural allocation level | Dimensionless |
| $g(\varepsilon)$ | $\mathbb{E}[\ln(1 + (1 + \varepsilon)(1 - e^{J^{[1]}}))]$ | Log units |
| $d$ | Overshoot distance below the barrier | Log units, positive |

The floor $L_t$, its initial value $L_0$, the stopping time $\tau_L$, the gap $Z_t$, the buffer
$B_t$, the log-gap $X_t$, its start $x_0$ and the effective jump means $\tilde\eta^{[i]}$ are
reserved on the [conventions page](conventions.md#reserved-notation). The chapter assumes that wealth
starts between the floor and the target, $L_0 \lt \Pi_0 \lt \Pi^{*}(0)$, so that $Z_0 \gt 0$ and
$x_0 \gt 0$.

## Methodology

### The floor and stopping

**Definition (Sepp, 2026, Definition 4.1).** The floor is $L_t = L_0e^{r_ct}$ with $L_0 \lt \Pi_0$,
and the stopping time is $\tau_L = \inf\lbrace t \gt 0 : \Pi_t \le L_t\rbrace$. After stopping, wealth
earns $r_c$ in cash:

$$
\Pi_T^{\mathrm{stop}} = \Pi_{\tau_L}e^{r_c(T - \tau_L)} \ \text{if} \ \tau_L \le T, \qquad \Pi_T^{\mathrm{stop}} = \Pi_T \ \text{otherwise}.
$$

The floor is calibrated in units of the expected crash size, $L_0 = \Pi_0e^{-k\eta^{[1]}}$
(Assumption 4.2); the [model chapter](regime_switching_model.md#the-floor-rule-of-the-papers-asset-classes)
gives the paper's $k = 1.5$ for single assets, and the [mandate chapter](mandate_aggregation.md)
derives $k$ from a drawdown tolerance.

### The gap process and the flat barrier

**Definition.** The gap is $Z_t = \Pi^{*}(t) - \Pi_t$, the buffer is $B_t = \Pi^{*}(t) - L_t$ and the
log-gap is $X_t = \ln(B_t/Z_t)$. Wealth at the floor, $\Pi_t = L_t$, is $Z_t = B_t$, which is $X_t = 0$
whatever the values of $\Pi^{*}(t)$ and $L_t$: the moving floor becomes a fixed barrier. Wealth above
the floor and below the target is $X_t \gt 0$.

**Proposition (exact gap dynamics under a frozen coefficient).** Suppose the policy holds the dollar
exposure $\omega_t\Pi_t = \kappa^{[i]}Z_t$ in regime $i$ with constants $\kappa^{[i]}$. Between
transitions the log-gap is a Brownian motion with drift,

$$
dX_t = \nu_{\mathrm{eff}}^{[i]}dt + \sigma_{\mathrm{eff}}^{[i]}d\tilde W_t, \qquad
\sigma_{\mathrm{eff}}^{[i]} = \kappa^{[i]}\sigma^{[i]}, \qquad
\nu_{\mathrm{eff}}^{[i]} = \kappa^{[i]}(\mu^{[i]} - r_h) + \frac{1}{2}\left(\sigma_{\mathrm{eff}}^{[i]}\right)^2,
$$

with $\tilde W = -W$, and at a transition out of regime $i$ it jumps by

$$
\Delta X = -\ln\left(1 - \kappa^{[i]}\left(e^{J^{[i]}} - 1\right)\right) .
$$

The net floor rate $r_c$ does not appear.

**Proof.** By the [regime-independent target](mv_optimal_policy.md#the-target-is-common-to-both-regimes),
$d\Pi^{*} = r_c\Pi^{*}dt$ exactly, and $dL = r_cL\thinspace dt$, so $d\ln B_t = r_c\thinspace dt$. Wealth follows
$d\Pi = r_c\Pi\thinspace dt + \kappa Z(\mu^{[i]} - r_h)dt + \kappa Z\sigma^{[i]}dW$ between transitions, hence
$dZ = r_cZ\thinspace dt - \kappa(\mu^{[i]} - r_h)Z\thinspace dt - \kappa\sigma^{[i]}Z\thinspace dW$, and by Itô's formula
$d\ln Z = (r_c - \kappa(\mu^{[i]} - r_h) - \kappa^2(\sigma^{[i]})^2/2)dt - \kappa\sigma^{[i]}dW$.
Subtracting from $d\ln B$ gives the drift and volatility of $X$. At a transition wealth changes by
$\kappa Z(e^{J} - 1)$, so $Z$ is multiplied by $1 - \kappa(e^{J} - 1)$ while $B$ does not move. $\square$

The parameters are those of Theorem 4.3 and equations (4.2)–(4.3) of the manuscript. Its proof
carries a correction from the time variation of the target, of order $\lambda^{[12]}$ (Appendix C);
by the regime-independent target that correction is zero, so the reduction is exact except for two
steps. The coefficient $\omega_a^{[i]}(\tau)$ varies over the horizon and is frozen at inception;
for the equity example it moves from $-0.756$ to $-0.769$ over ten years, by 1.7%. The jump law, below,
is replaced by an exponential.

> **Insight.** At a crash the dollar exposure $\kappa Z$ loses at most all of its value, so the
> log-gap falls by less than $\ln(1 + \kappa^{[1]})$: the exact gap jump is bounded. If the starting
> log-gap $x_0$ exceeds $\ln(1 + \kappa^{[1]})$, no single crash at inception can reach the floor.
> The exponential law that replaces the jump has unbounded support and assigns positive probability
> to any overshoot.

### The effective jump size

**Definition (mean-matching; Sepp, 2026, equation (4.5)).** The crash gap jump
$-\Delta X = \ln(1 + \kappa^{[1]}(1 - e^{J^{[1]}}))$ is replaced by an exponential variable with the
same mean,

$$
\tilde\eta^{[1]} = \mathbb{E}\left[\ln\left(1 + \kappa^{[1]}\left(1 - e^{J^{[1]}}\right)\right)\right]
= \int_0^{\infty}\ln\left(1 + \kappa^{[1]}\left(1 - e^{-s}\right)\right)\frac{e^{-s/\eta^{[1]}}}{\eta^{[1]}}ds .
$$

**Proposition (Taylor expansion around the natural allocation level; equation (4.6)).** With
$u = 1 - e^{J^{[1]}}$ and $\varepsilon = \kappa^{[1]} - 1$,

$$
\tilde\eta^{[1]} = g(0) + \varepsilon g'(0) + \frac{1}{2}\varepsilon^2g''(0) + O(\varepsilon^3), \qquad
g(0) = \mathbb{E}[\ln(1 + u)], \quad g'(0) = \mathbb{E}\left[\frac{u}{1 + u}\right], \quad g''(0) = -\mathbb{E}\left[\frac{u^2}{(1 + u)^2}\right] .
$$

**Proof.** $g(\varepsilon) = \mathbb{E}[\ln(1 + (1 + \varepsilon)u)]$ has derivatives
$\mathbb{E}[u/(1 + (1 + \varepsilon)u)]$ and $-\mathbb{E}[u^2/(1 + (1 + \varepsilon)u)^2]$, bounded
because $0 \le u \lt 1$, so differentiation under the expectation is justified; evaluate at zero.
$\square$

The level $\kappa^{[1]} = 1$ is natural because the gap then jumps by the log-return of the asset
itself, $\ln(2 - e^{J})$ in place of $-J$. The quadrature is exact to machine precision; the expansion
is an analytical alternative whose error grows with $\lvert\varepsilon\rvert$.

For the recovery, `gap_process_asset` applies the same formula to $\eta^{[2]}$ and $\kappa^{[2]}$.
The exact recovery gap jump is $-\ln(1 - \kappa^{[2]}(e^{J^{[2]}} - 1))$, a different function, with a
larger mean in the worked example; a recovery jump can even lift wealth above the target, where the
log-gap is undefined. Recoveries move the log-gap away from the barrier, so this choice affects
survival much less than the crash approximation.

### Three components of stopped wealth

**Proposition (Sepp, 2026, Proposition 4.5).** Stopped terminal wealth has three components:

1. survival, $\tau_L \gt T$: $\Pi_T^{\mathrm{stop}} = \Pi_T^{*} - Z_T$, a continuous law on
   $(L_T, \Pi_T^{*})$;
2. the floor atom: the diffusion reaches the barrier continuously, $\Pi_{\tau_L} = L_{\tau_L}$, and
   $\Pi_T^{\mathrm{stop}} = L_T$;
3. the overshoot: a crash carries the log-gap below zero, $\Pi_{\tau_L} \lt L_{\tau_L}$, and
   $\Pi_T^{\mathrm{stop}} \lt L_T$, a continuous law below the floor.

**Proof.** The three events partition the sample space: either the path is not stopped by $T$, or it
is stopped at a continuous crossing, where the barrier is hit exactly, or at a jump, where it is
crossed with a strictly negative log-gap. After stopping, wealth grows at $r_c$ like the floor, so the
ratio $\Pi/L$ at stopping is preserved to the horizon. $\square$

The third component does not exist for a diffusion monitored continuously. In the reduced process
the overshoot distance $d = -X_{\tau_L}$ is exponential with mean $\tilde\eta^{[1]}$ by the
memoryless property; the [Laplace chapter](laplace_barrier_framework.md) computes its mass and the
[terminal wealth chapter](terminal_wealth_distribution.md) maps all three components to wealth.

The stopped strategy is the unconstrained policy stopped at the floor, not the optimum of the
floor-constrained problem (Remark 4.6). Near the floor the goal-seeking policy raises its allocation,
since $\Pi^{*}/\Pi$ grows, where a constrained optimum would de-risk (Bielecki et al., 2005); the
overshoot component remains under any policy because a crash cannot be avoided by trading.

![Two simulated ten-year paths of the balanced mandate under the growth-regime policy: on the left wealth ends at 167 above the dashed expected-wealth line; on the right a crash after three and a half years carries wealth from about 100 through the floor at 69 to 64, after which wealth grows in cash, and the lower panels show the allocation falling to zero at stopping](../papers/goal_based_allocation_2026/paper/figures/path_dynamics_balanced.png)

[Open full-resolution figure](../papers/goal_based_allocation_2026/paper/figures/path_dynamics_balanced.png).

Figure 1 of the manuscript shows the three components on single paths of the balanced mandate of
Table 2, calibrated to start fully invested. The left path survives and ends at 167; its allocation
falls as wealth approaches the target. The right path is stopped by a crash near $t = 3.5$ that
carries wealth from about 100 through the floor at 69 to 64, an overshoot of 5, after which it grows
in cash to 73. The dash-dot lines are the growth and stress magnitudes, 124% and 24%. The figure is
produced by `generate_paper_figures.py --figure 6` from the
[replication folder](https://github.com/ArturSepp/GoalBasedAllocation/tree/main/papers/goal_based_allocation_2026/replication);
its paths are simulations, and its numbers are those reported in the manuscript.

## Worked example

The equity class of Table 1 with its floor at 60.65, a horizon of ten years, $r = 2$% and no
consumption, and `find_ell` with a `target_return` of 4% give the frozen magnitudes
$\kappa = (0.7563, 0.2129)$. The first block checks the gap asset against the proposition:

- $\sigma_{\mathrm{eff}} = (11.34\%, 4.79\%)$ and $\nu_{\mathrm{eff}} = (4.43\%, -3.50\%)$;
- $B_0 = 149.18 - 60.65 = 88.53$, $Z_0 = 49.18$ and $x_0 = \ln(B_0/Z_0) = 0.5878$;
- $\tilde\eta^{[1]} = 0.1660$ by quadrature, against a mean of 0.1659 over two million sampled gap
  jumps; the first-order expansion is 0.88% too high and the second-order expansion 0.04%, at
  $\varepsilon = -0.244$;
- the bound $\ln(1 + \kappa^{[1]}) = 0.5632$ lies below $x_0$, so one crash at inception cannot stop
  the exact process;
- for the recovery the package uses 0.0240, while the exact gap jump has mean 0.0333.

```python
import numpy as np
from scipy.integrate import quad
import goal_based_allocation as gba
from goal_based_allocation.riccati_solver import simulate_mv_optimal

equity = gba.create_paper_assets()['equity']
horizon, r, c = 10.0, 0.02, 0.0
ell, ric = gba.find_ell(equity, horizon, target_return=0.04, r=r, c=c)
gap = gba.gap_process_asset(ric)
start = ric.derived_at_tau(horizon)
kappa = np.abs(np.array(start['w_a']))
mu = np.array(ric.dp['mu_bar'])
sigma = np.array([equity.params.sigma0, equity.params.sigma1])
np.testing.assert_allclose(kappa, [0.7563, 0.2129], atol=5e-5)

# effective diffusion parameters of the proposition
np.testing.assert_allclose([gap.params.sigma0, gap.params.sigma1], kappa * sigma, rtol=1e-12)
np.testing.assert_allclose([gap.nu0, gap.nu1], kappa * (mu - ric.r_h) + 0.5 * (kappa * sigma)**2, rtol=1e-12)
np.testing.assert_allclose([gap.params.sigma0, gap.params.sigma1, gap.nu0, gap.nu1],
                           [0.1134, 0.0479, 0.0443, -0.0350], atol=5e-5)

# the barrier: x0 = ln(B0 / Z0) with B0 = Pi*(0) - L0 and Z0 = Pi*(0) - Pi0
buffer_0 = start['Pi_star'][0] - equity.pi_floor
gap_0 = start['Pi_star'][0] - equity.pi0
np.testing.assert_allclose([gap.pi0, gap.pi_floor], [buffer_0, gap_0], rtol=1e-12)
np.testing.assert_allclose([buffer_0, gap_0, gap.x0], [88.53, 49.18, 0.5878], atol=5e-3)

# effective crash size: quadrature, sampled exact gap jumps, and the Taylor expansion
eta = equity.params.eta0
np.testing.assert_allclose(gap.params.eta0, 0.1660, atol=5e-5)
draws = np.random.default_rng(3).exponential(eta, 2_000_000)
sampled = np.log1p(kappa[0] * -np.expm1(-draws))
assert abs(sampled.mean() - gap.params.eta0) < 3.0 * sampled.std() / np.sqrt(draws.size)


def expectation(function):
    """Expectation over the crash size s with density exp(-s/eta)/eta, u = 1 - e^-s."""
    return quad(lambda s: function(-np.expm1(-s)) * np.exp(-s / eta) / eta, 0.0, np.inf)[0]


g0 = expectation(np.log1p)
g1 = expectation(lambda u: u / (1.0 + u))
g2 = -expectation(lambda u: (u / (1.0 + u))**2)
eps = kappa[0] - 1.0
first, second = g0 + eps * g1, g0 + eps * g1 + 0.5 * eps**2 * g2
np.testing.assert_allclose([first / gap.params.eta0 - 1.0, second / gap.params.eta0 - 1.0],
                           [0.0088, 0.0004], atol=5e-5)

# the exact crash gap jump is bounded by ln(1 + kappa), below the starting log-gap
assert np.log1p(kappa[0]) < gap.x0 and sampled.max() < np.log1p(kappa[0])
np.testing.assert_allclose(np.log1p(kappa[0]), 0.5632, atol=5e-5)

# recovery: the package applies the crash formula; the exact gap jump has a larger mean
eta_up = equity.params.eta1
ceiling = np.log1p(1.0 / kappa[1])   # a larger recovery lifts wealth above the target
exact_up = quad(lambda s: -np.log(1.0 - kappa[1] * np.expm1(s)) * np.exp(-s / eta_up) / eta_up,
                0.0, ceiling)[0]
np.testing.assert_allclose([gap.params.eta1, exact_up], [0.0240, 0.0333], atol=5e-5)
```

The second block measures the error of the reduction in three steps, from the computation the
package performs towards the strategy it describes. An exact simulator of a regime-switching
jump-diffusion with an absorbing barrier draws exponential holding times, exact Gaussian increments
and, for the time the diffusion spends between two observed points $x, y \gt 0$, the probability
$e^{-2xy/(\sigma^2h)}$ that it touched zero, so that the barrier is monitored continuously.

| Computation | Survival to ten years |
|---|---:|
| Laplace survival of the gap asset, `compute_survival` | 92.92% |
| Exact simulation of the gap asset, exponential jumps | 92.95% |
| Exact simulation of the gap diffusion with the exact gap-jump laws | 94.26% |
| `simulate_mv_optimal`: daily Euler simulation of wealth under the regime-conditional policy | 94.55% |

The first two rows agree within one standard error of 0.04 points: the Laplace computation is exact
for the process it is given. Replacing the exponential jumps by the bounded exact jumps raises
survival by 1.3 points; this is the error of the mean-matching approximation, and it is
conservative for survival here. Simulating wealth itself, with the coefficient varying over time,
the signed stress policy, the clipping of the allocation to $[-2, 5]$ and daily monitoring of the
floor, adds 0.3 points. The total difference of 1.6 points between the analytical and the simulated
survival lies in the range of one to two points that Section 4 of the manuscript reports.

```python
def survival_by_simulation(asset, horizon, crash_jump, recovery_jump, n_paths, seed):
    """Exact survival of a regime-switching jump-diffusion started in growth at asset.x0 > 0.

    Holding times and Gaussian increments are exact; the probability that the diffusion touched
    zero between two points x, y > 0 over a time h is exp(-2 x y / (sigma^2 h)). The jump laws
    are callables of (rng, size) returning the change of X at a crash and at a recovery.
    """
    p = asset.params
    rng = np.random.default_rng(seed)
    nu = np.array([asset.nu0, asset.nu1])
    vol = np.array([p.sigma0, p.sigma1])
    rate = np.array([p.lambda01, p.lambda10])
    x = np.full(n_paths, asset.x0)
    clock = np.zeros(n_paths)
    regime = np.zeros(n_paths, dtype=int)
    alive = np.ones(n_paths, dtype=bool)
    active = alive.copy()
    while active.any():
        i = np.flatnonzero(active)
        now = regime[i]
        hold = rng.exponential(1.0 / rate[now])
        remaining = horizon - clock[i]
        step = np.minimum(hold, remaining)
        end = x[i] + nu[now] * step + vol[now] * np.sqrt(step) * rng.standard_normal(i.size)
        bridge = np.exp(-2.0 * x[i] * np.maximum(end, 0.0) / (vol[now]**2 * step))
        hit = (end <= 0.0) | (rng.uniform(size=i.size) < bridge)
        alive[i[hit]] = False
        x[i], clock[i] = end, clock[i] + step
        switch = i[~hit & (hold < remaining)]
        crash = regime[switch] == 0
        x[switch] += np.where(crash, crash_jump(rng, switch.size), recovery_jump(rng, switch.size))
        regime[switch] = 1 - regime[switch]
        alive[switch[x[switch] <= 0.0]] = False
        active = alive & (clock < horizon)
    return alive.mean()


survival = gba.compute_survival(horizon, gap.x0, gap)
n_paths = 400_000
standard_error = np.sqrt(survival * (1.0 - survival) / n_paths)

# the gap asset itself: exponential jumps with the effective means
exponential_crash = lambda rng, n: -rng.exponential(gap.params.eta0, n)
exponential_recovery = lambda rng, n: rng.exponential(gap.params.eta1, n)
reduced = survival_by_simulation(gap, horizon, exponential_crash, exponential_recovery, n_paths, seed=11)

# the same diffusion with the exact gap jumps of the frozen-coefficient policy
exact_crash = lambda rng, n: -np.log1p(kappa[0] * -np.expm1(-rng.exponential(eta, n)))
exact_recovery = lambda rng, n: -np.log(np.maximum(1.0 - kappa[1] * np.expm1(rng.exponential(eta_up, n)),
                                                   1e-300))
exact_jumps = survival_by_simulation(gap, horizon, exact_crash, exact_recovery, n_paths, seed=12)

# wealth simulated under the regime-conditional policy, the package's own validator
wealth_paths = simulate_mv_optimal(ric, n_paths=40_000, steps_per_year=260, seed=42)

np.testing.assert_allclose([survival, reduced, exact_jumps, wealth_paths['survived'].mean()],
                           [0.9292, 0.9295, 0.9426, 0.9455], atol=5e-5)
assert abs(reduced - survival) < 3.0 * standard_error
assert exact_jumps - survival > 25.0 * standard_error
```

## Implementation in goal_based_allocation

| Quantity | Formula | Entry point |
|---|---|---|
| Gap asset | $\sigma_{\mathrm{eff}}^{[i]}$, $\nu_{\mathrm{eff}}^{[i]}$, $\tilde\eta^{[i]}$, $x_0 = \ln(B_0/Z_0)$ | `gap_process_asset(ric)` |
| Frozen magnitudes | $\kappa^{[i]} = \lvert\omega_a^{[i]}(\tau = T)\rvert$ | `ric.derived_at_tau(ric.T)['w_a']` |
| Effective jump mean | Equation (4.5) by quadrature | `riccati_solver._eta_eff_quadrature(eta, w)` (internal) |
| Floor at the horizon | $L_T = L_0e^{r_cT}$ | `asset.pi_floor * np.exp(ric.r_c * T)` |
| Wealth simulation | Euler, daily, regime-conditional policy, floor stopping | `riccati_solver.simulate_mv_optimal(ric, n_paths=100_000, steps_per_year=260, seed=42)` |

The functions live in
[`riccati_solver.py`](https://github.com/ArturSepp/GoalBasedAllocation/blob/main/src/goal_based_allocation/riccati_solver.py).
API pages: {doc}`gap_process_asset <api/generated/goal_based_allocation.gap_process_asset>` and
{doc}`simulate_mv_optimal <api/generated/goal_based_allocation.riccati_solver.simulate_mv_optimal>`.

Contract details:

- `gap_process_asset` evaluates the coefficients at `ric.T`, inception, and uses their magnitudes in
  both regimes, so the analytical distribution describes a policy with positive exposure
  $\kappa^{[i]}Z_t$ in each regime. It does not take a time argument.
- The returned `AssetSpecification` is named `'<asset>_gap'`. Its `mu_growth` and `mu_stress` are
  back-solved so that `nu0` and `nu1` equal $\nu_{\mathrm{eff}}^{[i]}$; they are not total returns of
  any asset. Its `pi0` is $\max(B_0, 0.01)$ and its `pi_floor` is $\max(Z_0, 0.001)$, so wealth at or
  above the target, $Z_0 \le 0$, is silently replaced by a tiny gap.
- The effective jump means are truncated quadratures on $[0, \max(50\eta, 20)]$ and are zero when
  the corresponding $\eta$ is zero.
- `simulate_mv_optimal` steps the regime chain with probability $\lambda\thinspace dt$ per step, clips the
  allocation to $[-2, 5]$, stops paths with $\Pi_t \le L_t$ at the end of a step, and returns a
  dictionary with `'Pi_T'`, `'L_T'`, `'survived'`, `'is_overshoot'` (stopped at least 0.1% below
  the floor), `'regime_T'`, `'n_paths'`, `'stop_time'` and `'Pi_at_stop'`. It is a validator with
  time-discretisation and monitoring bias, not an exact simulation.

> **Pitfall.** The analytical gap process and the package's wealth simulator do not describe the
> same policy in stress. `gap_process_asset` uses $\lvert\omega_a^{[2]}\rvert$, a positive exposure,
> while `simulate_mv_optimal` applies the signed regime-conditional policy, which is short below
> target in stress for the paper's assets. The manuscript's figures use the growth coefficient in
> both regimes, a third policy. State which policy a reported survival probability refers to.

## Interpretation and limitations

- The analytical survival, floor atom and overshoot refer to the reduced process. Its error against
  the strategy is dominated by the exponential approximation of the gap jump, 1.3 points of survival
  in the worked example; the error depends on $\kappa^{[1]}$, $\eta^{[1]}$ and $x_0$ and is not
  bounded uniformly.
- The coefficient is frozen at inception. The goal-seeking policy itself is not frozen: the
  exposure $\kappa Z_t$ changes with the gap, which the log-gap coordinates absorb exactly.
- Wealth above the target, $Z_t \lt 0$, is outside the gap coordinates. The unconstrained policy
  would short there; the reduction ignores it, which is harmless while the probability of a
  recovery lifting wealth above the target is negligible, about $2 \times 10^{-6}$ per recovery in
  the worked example.
- The floor is imposed by stopping. The stopped strategy is not optimal for a floor-constrained
  problem, and its expected wealth is lower than that of the unconstrained policy.

## See also

- [The MV-optimal policy and the Riccati system](mv_optimal_policy.md)
- [Survival, densities and overshoot in the Laplace domain](laplace_barrier_framework.md)
- [The terminal wealth distribution](terminal_wealth_distribution.md)
- [Mandates as one effective asset](mandate_aggregation.md)
- [Validation and numerical evidence](validation.md)
- [Bibliography](bibliography.md)

## References

1. Sepp, A. (2026). Dynamic Mean-Variance Portfolio Allocation under Regime-Switching Jump-Diffusions with Absorbing Barriers and Distribution Matching. Working paper. [DOI: 10.2139/ssrn.6534579](https://doi.org/10.2139/ssrn.6534579). Section 4 and Appendix C define the floor, the gap process and the effective jump size; Figure 1 shows stopped and survived paths.
2. Cont, R., and Tankov, P. (2009). Constant Proportion Portfolio Insurance in the Presence of Jumps in Asset Prices. *Mathematical Finance*, 19(3), 379–401. [DOI: 10.1111/j.1467-9965.2009.00377.x](https://doi.org/10.1111/j.1467-9965.2009.00377.x). Gap risk of cushion-proportional strategies under jumps.
3. Grossman, S. J., and Zhou, Z. (1993). Optimal Investment Strategies for Controlling Drawdowns. *Mathematical Finance*, 3(3), 241–276. [DOI: 10.1111/j.1467-9965.1993.tb00044.x](https://doi.org/10.1111/j.1467-9965.1993.tb00044.x). Investment proportional to the surplus over a floor.
4. Bielecki, T. R., Jin, H., Pliska, S. R., and Zhou, X. Y. (2005). Continuous-Time Mean-Variance Portfolio Selection with Bankruptcy Prohibition. *Mathematical Finance*, 15(2), 213–244. [DOI: 10.1111/j.0960-1627.2005.00218.x](https://doi.org/10.1111/j.0960-1627.2005.00218.x). The mean-variance problem with a wealth constraint inside the optimisation.
5. Sepp, A. goal-based-allocation: Semi-analytical dynamic mean-variance allocation and terminal-wealth risk under regime-switching jump-diffusions. [Software citation metadata](https://github.com/ArturSepp/GoalBasedAllocation/blob/main/CITATION.cff).
