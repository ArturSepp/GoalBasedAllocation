---
myst:
  html_meta:
    description: >-
      The two-regime jump-diffusion of goal-based-allocation: the regime chain, exponential jumps
      at regime transitions, jump compensators, total return against diffusion drift, the floor
      rule of the paper's asset classes and their parameters, with an exact-simulation check.
---

# The regime-switching jump-diffusion

*Author: [Artur Sepp](https://github.com/ArturSepp)*

Implemented in [GoalBasedAllocation](https://github.com/ArturSepp/GoalBasedAllocation).
Software citation: [CITATION.cff](https://github.com/ArturSepp/GoalBasedAllocation/blob/main/CITATION.cff).

A regime-switching jump-diffusion is a price process whose drift and volatility are set by a
hidden two-state Markov chain and which jumps exactly when the chain changes state. In the model of
Sepp (2026, Section 2) the chain alternates between a growth regime and a stress regime; the entry
into stress is a crash, a downward exponential jump of the log-price, and the return to growth is
a recovery, an upward exponential jump. This chapter defines the model, derives the quantities the
rest of the handbook builds on, and states the parameters of the paper's three asset classes.

## Overview

The same process describes three objects in the package. With physical parameters it is the price
of an asset class or of a mandate, the input to the [mean-variance policy](mv_optimal_policy.md)
and to the [buy-and-hold benchmark](buy_and_hold_moments.md). Under the optimal policy the log of
the distance between target wealth and wealth is again such a process, the
[gap process](wealth_floor_gap_process.md), whose barrier problems the
[Laplace framework](laplace_barrier_framework.md) solves. With risk-neutral parameters it prices
[European options](european_options.md) and [variance swaps](variance_swaps.md).

Tying jumps to transitions is the modelling choice that distinguishes this process from regime
switching with independent jumps (Shi and Xu, 2025). A crash is the regime switch, not a separate
event, so the jump intensity in each regime equals the transition intensity out of it, and the
characteristic equation of the Laplace framework has degree six rather than higher. The chapter
answers three questions that every later number depends on: how a quoted crash loss maps to the
jump parameter, how the total expected return of a capital market assumption maps to the drift of
the diffusion, and where the paper places the wealth floor of each asset class.

## Inputs, notation, and assumptions

| Convention | This article |
|---|---|
| Time and rates | Years; total returns, volatilities and intensities are annual and continuously compounded |
| Regimes | Paper regimes 1 (growth) and 2 (stress) are code indices 0 and 1; statements hold from either starting regime unless one is named; the simulation starts in growth |
| Jump sizes | Mean convention: `eta0` $= \eta^{[1]}$ and `eta1` $= \eta^{[2]}$ are mean absolute log-jumps; a crash loss $L$ gives $\eta^{[1]} = L/(1 - L)$ and a recovery gain $G$ gives $\eta^{[2]} = G/(1 + G)$ |
| Wealth coordinate | Price $S_t$ and log-price $X_t = \ln S_t$; `AssetSpecification.x0` is the log distance $\ln(\Pi_0/L_0)$ from initial wealth to the floor |
| Measure | Physical; the risk-neutral version is in [European options](european_options.md) |
| Numerical method | Closed forms; the worked example checks them against an exact simulation with exponential holding times and exact Gaussian increments |
| Package default | `RegimeSwitchParams(eta0=0.0, eta1=0.0)` has no jumps; `AssetSpecification(pi0=100.0, pi_floor=80.0)`; `create_paper_assets(pi0=100.0, k_floor=1.5)` |

| Symbol | Meaning | Units and convention |
|---|---|---|
| $\Lambda$ | Generator of the regime chain | Rates per year |
| $S_t$ | Price of the risky asset | Positive, any scale |
| $J^{[1]}$, $J^{[2]}$ | Log-jump at a crash and at a recovery | $J^{[1]} \lt 0$, $J^{[2]} \gt 0$ |
| $f^{[1]}$, $f^{[2]}$ | Densities of $J^{[1]}$ and $J^{[2]}$ | Per unit of log-price |
| $\Phi$ | Argument of a moment-generating function | Complex, within the strip of convergence |
| $\sigma_J^{[i]}$ | Root mean square of $e^{J^{[i]}} - 1$ | Decimal |
| $L$, $G$ | Crash loss and recovery gain as relative price changes | Decimal, $L = -\alpha^{[1]}$ and $G = \alpha^{[2]}$ |
| $W_t$ | Standard Brownian motion independent of the chain | Time in years |

The reserved symbols $\chi_t$, $\lambda^{[12]}$, $\lambda^{[21]}$, $p_1$, $p_2$, $\sigma^{[i]}$,
$\bar\mu^{[i]}$, $\mu^{[i]}$, $\nu^{[i]}$, $\eta^{[i]}$ and $\alpha^{[i]}$ have the meanings of the
[conventions page](conventions.md#reserved-notation). The model assumes that the Brownian motion and
the chain are independent, that the intensities are positive, and that $\eta^{[2]} \lt 1/2$
(Assumption 2.2 of the manuscript), which gives the recovery factor $e^{J^{[2]}}$ a finite second
moment.

## Methodology

### The regime chain

**Definition.** The regime $\chi_t \in \lbrace 1, 2 \rbrace$ is a continuous-time Markov chain
with generator

$$
\Lambda = \begin{pmatrix} -\lambda^{[12]} & \lambda^{[12]} \\ \lambda^{[21]} & -\lambda^{[21]} \end{pmatrix},
$$

equation (2.1) of the manuscript. Regime 1 is growth and regime 2 is stress; the transition from 1
to 2 is a crash and the transition from 2 to 1 a recovery. Holding times are exponential with
means $1/\lambda^{[12]}$ in growth and $1/\lambda^{[21]}$ in stress.

**Identity (stationary probabilities).** The stationary distribution of the chain is

$$
p_1 = \frac{\lambda^{[21]}}{\lambda^{[12]} + \lambda^{[21]}}, \qquad
p_2 = \frac{\lambda^{[12]}}{\lambda^{[12]} + \lambda^{[21]}} .
$$

**Proof.** The row vector $(p_1, p_2)$ solves $(p_1, p_2)\Lambda = 0$ because
$-p_1\lambda^{[12]} + p_2\lambda^{[21]} = 0$, and $p_1 + p_2 = 1$. $\square$

The expected time spent in stress over $[0, T]$ from growth is
$p_2 (T - (1 - e^{-(\lambda^{[12]} + \lambda^{[21]})T})/(\lambda^{[12]} + \lambda^{[21]}))$,
derived in the [variance-swap chapter](variance_swaps.md), where it weights the regime variances.

### Asset dynamics with jumps at transitions

**Definition (regime-switching jump-diffusion).** Between transitions the price follows a geometric
Brownian motion with the parameters of the current regime; at a transition it jumps:

$$
\frac{dS_t}{S_{t^-}} = \mu^{[\chi_t]} dt + \sigma^{[\chi_t]} dW_t \ \ \text{between transitions}, \qquad
S_t = S_{t^-} e^{J^{[1]}} \ \text{at a crash}, \qquad
S_t = S_{t^-} e^{J^{[2]}} \ \text{at a recovery}.
$$

The log-jumps are exponential in the mean convention, with densities

$$
f^{[1]}(J) = \frac{1}{\eta^{[1]}} e^{J/\eta^{[1]}} \ \text{for} \ J \lt 0, \qquad
f^{[2]}(J) = \frac{1}{\eta^{[2]}} e^{-J/\eta^{[2]}} \ \text{for} \ J \gt 0,
$$

equations (2.2)–(2.4) of the manuscript. The log-price $X_t = \ln S_t$ follows
$dX_t = \nu^{[\chi_t]} dt + \sigma^{[\chi_t]} dW_t$ between transitions, with
$\nu^{[i]} = \mu^{[i]} - (\sigma^{[i]})^2/2$, and $X_t = X_{t^-} + J^{[i]}$ at a transition out of
regime $i$.

> **Insight.** Because a crash can only happen in growth and a recovery only in stress, crashes and
> recoveries alternate. The process cannot crash twice in a row, and a long stress period ends with
> an upward jump. This is the model's version of the empirical observation that market dislocations
> coincide with changes of regime (Hamilton, 1989).

### Jump transforms and compensators

**Identity (moment-generating functions).** For complex $\Phi$ in the strips of convergence,

$$
\mathbb{E}\left[e^{\Phi J^{[1]}}\right] = \frac{1}{1 + \eta^{[1]}\Phi}, \ \ \mathrm{Re}(\Phi) \gt -\frac{1}{\eta^{[1]}}; \qquad
\mathbb{E}\left[e^{\Phi J^{[2]}}\right] = \frac{1}{1 - \eta^{[2]}\Phi}, \ \ \mathrm{Re}(\Phi) \lt \frac{1}{\eta^{[2]}} .
$$

**Proof.** For the crash,
$\int_{-\infty}^{0} e^{\Phi J} e^{J/\eta^{[1]}} dJ/\eta^{[1]} = 1/(\eta^{[1]}(\Phi + 1/\eta^{[1]}))$, which
converges when $\mathrm{Re}(\Phi + 1/\eta^{[1]}) \gt 0$. The recovery is the mirror image. $\square$

Both transforms are rational in $\Phi$. This is what gives the Laplace framework its polynomial
characteristic equation, and it is why the package needs exponential, or more generally rational,
jump laws.

**Identity (compensators, crash loss and recovery gain).** The expected relative price changes at
the two transitions are

$$
\alpha^{[1]} = \mathbb{E}\left[e^{J^{[1]}}\right] - 1 = -\frac{\eta^{[1]}}{1 + \eta^{[1]}}, \qquad
\alpha^{[2]} = \mathbb{E}\left[e^{J^{[2]}}\right] - 1 = \frac{\eta^{[2]}}{1 - \eta^{[2]}},
$$

so a crash loss $L$ and a recovery gain $G$ correspond to $\eta^{[1]} = L/(1 - L)$ and
$\eta^{[2]} = G/(1 + G)$. The mean squared relative change of the crash is

$$
\left(\sigma_J^{[1]}\right)^2 = \mathbb{E}\left[\left(e^{J^{[1]}} - 1\right)^2\right] = \frac{1}{1 + 2\eta^{[1]}} - \frac{2}{1 + \eta^{[1]}} + 1 .
$$

**Proof.** Set $\Phi = 1$ and $\Phi = 2$ in the transforms and expand the square. Solving
$L = \eta^{[1]}/(1 + \eta^{[1]})$ and $G = \eta^{[2]}/(1 - \eta^{[2]})$ for the jump parameters gives
the inverses. $\square$

The recovery gain is finite only for $\eta^{[2]} \lt 1$, and its square has a finite mean only for
$\eta^{[2]} \lt 1/2$, which the Riccati system needs for $(\sigma_J^{[2]})^2$.

### Total return, diffusion drift and log drift

**Definition (regime-conditional drift).** A capital market assumption quotes the total expected
return $\bar\mu^{[i]}$ of regime $i$, which includes the expected jump at the exit from the regime.
The diffusion drift and the log drift are

$$
\mu^{[i]} = \bar\mu^{[i]} - \lambda^{[i \to j]} \alpha^{[i]}, \qquad \nu^{[i]} = \mu^{[i]} - \frac{1}{2}\left(\sigma^{[i]}\right)^2,
$$

with $\lambda^{[1 \to 2]} = \lambda^{[12]}$ and $\lambda^{[2 \to 1]} = \lambda^{[21]}$, equations
(2.6) and (2.8) of the manuscript. Because $\alpha^{[1]} \lt 0 \lt \alpha^{[2]}$, the diffusion drift
exceeds the total return in growth, compensating for the expected crash, and falls short of it in
stress, discounting the expected recovery.

**Proposition (total expected return).** In regime $i$ the instantaneous expected return,
including the expected jump, is $\bar\mu^{[i]}$. If $\bar\mu^{[1]} = \bar\mu^{[2]} = \bar\mu$, then
$\mathbb{E}[S_T] = S_0 e^{\bar\mu T}$ from either starting regime.

**Proof.** The vector $m^{[i]}(\tau) = \mathbb{E}[S_T/S_t \mid \chi_t = i]$, with $\tau = T - t$,
solves the backward equation $m' = A m$, $m(0) = (1, 1)^{\top}$, where

$$
A = \begin{pmatrix} \mu^{[1]} - \lambda^{[12]} & \lambda^{[12]}(1 + \alpha^{[1]}) \\ \lambda^{[21]}(1 + \alpha^{[2]}) & \mu^{[2]} - \lambda^{[21]} \end{pmatrix} .
$$

The row sums of $A$ are $\mu^{[i]} + \lambda^{[i \to j]}\alpha^{[i]} = \bar\mu^{[i]}$, the
instantaneous expected returns. When both equal $\bar\mu$, the vector $e^{\bar\mu \tau}(1, 1)^{\top}$
solves the equation with the right initial value, and the solution is unique. $\square$

The matrix $A$ is the first-moment case of the buy-and-hold system of Proposition A.1, which the
[buy-and-hold chapter](buy_and_hold_moments.md) solves for the paper's unequal total returns.

> **Pitfall.** `mu_growth` and `mu_stress` are total expected returns $\bar\mu^{[i]}$, not diffusion
> drifts. For the paper's equity class the total returns are 4.5% and 0%, while the diffusion drifts
> are 7% and $-15$%. Supplying a diffusion drift as `mu_growth` counts the expected crash loss twice.

### The floor rule of the paper's asset classes

**Definition (floor calibration).** The floor of an asset class is placed $k$ expected crash sizes
below initial wealth in log terms, Assumption 4.2 and equation (7.6) of the manuscript:

$$
L_0 = \Pi_0 e^{-k \eta^{[1]}}, \qquad x_0 = \ln\frac{\Pi_0}{L_0} = k \eta^{[1]} .
$$

**Identity (single-crash breach).** A fully invested position at initial wealth breaches the floor
at a crash with probability $e^{-k}$, whatever the asset.

**Proof.** The position falls to $\Pi_0 e^{J^{[1]}}$, which is at or below $L_0$ exactly when
$J^{[1]} \le -k\eta^{[1]}$. The exponential tail gives
$\mathbb{P}(-J^{[1]} \ge k\eta^{[1]}) = e^{-k\eta^{[1]}/\eta^{[1]}} = e^{-k}$. $\square$

With the paper's $k = 1.5$ the breach probability is 22.3% for every asset class. It describes one
crash from initial wealth; it is not the probability of breaching the floor by the horizon, which
the [Laplace framework](laplace_barrier_framework.md) computes.

### The paper's asset classes

The three asset classes of Table 1 of the manuscript share the regime chain, with
$\lambda^{[12]} = 0.1$ (one crash per decade) and $\lambda^{[21]} = 1$ (stress lasts a year on
average), and a stress volatility one and a half times the growth volatility.

| Asset class | $\bar\mu^{[1]}$ | $\bar\mu^{[2]}$ | $\sigma^{[1]}$ | $\sigma^{[2]}$ | Crash loss | Recovery gain | $\eta^{[1]}$ | $\eta^{[2]}$ | $\nu^{[1]}$ | $\nu^{[2]}$ |
|---|---|---|---|---|---|---|---|---|---|---|
| Bonds | 2.5% | 2.0% | 6.0% | 9.0% | 8% | 5% | 0.087 | 0.048 | 3.12% | $-3.41$% |
| Equity | 4.5% | 0.0% | 15.0% | 22.5% | 25% | 15% | 0.333 | 0.130 | 5.88% | $-17.53$% |
| Private equity | 7.0% | 0.0% | 20.0% | 30.0% | 30% | 20% | 0.429 | 0.167 | 8.00% | $-24.50$% |

The log drifts in the last two columns are computed, not assumed: the worked example derives them
from the first seven columns. The diffusions of the three classes are correlated through the matrix
of equation (7.1), which enters only when they are combined into a
[mandate](mandate_aggregation.md).

## Worked example

The first block builds the paper's asset classes and checks the mappings of this chapter against
Table 1 of the manuscript: the jump parameters from the quoted crash loss and recovery gain, the
diffusion drifts of 7% and $-15$% for equity, the six log drifts of the table, and the floors of
87.77, 60.65 and 52.58 that the rule $L_0 = 100 e^{-1.5\eta^{[1]}}$ gives.

```python
import numpy as np
import goal_based_allocation as gba

assets = gba.create_paper_assets(pi0=100.0, k_floor=1.5)
equity = assets['equity']
params = equity.params

# crash loss 25% and recovery gain 15% in the mean convention
crash_loss = params.eta0 / (1.0 + params.eta0)
recovery_gain = params.eta1 / (1.0 - params.eta1)
np.testing.assert_allclose([params.eta0, params.eta1], [0.25 / 0.75, 0.15 / 1.15])
np.testing.assert_allclose([crash_loss, recovery_gain], [0.25, 0.15])

# diffusion drifts and log drifts of Table 1
alpha = np.array([-crash_loss, recovery_gain])
lam = np.array([params.lambda01, params.lambda10])
sigma = np.array([params.sigma0, params.sigma1])
mu_bar = np.array([equity.mu_growth, equity.mu_stress])
mu = mu_bar - lam * alpha
np.testing.assert_allclose(mu, [0.07, -0.15])
np.testing.assert_allclose([equity.nu0, equity.nu1], mu - 0.5 * sigma**2)
np.testing.assert_allclose(params.compute_drifts(equity.mu_growth, equity.mu_stress),
                           [equity.nu0, equity.nu1])

# all three asset classes against the log drifts printed in Table 1 (per cent)
table_1 = {'bonds': (3.12, -3.41), 'equity': (5.88, -17.53), 'private_equity': (8.00, -24.50)}
for name, (nu_growth, nu_stress) in table_1.items():
    asset = assets[name]
    assert [round(100 * asset.nu0, 2), round(100 * asset.nu1, 2)] == [nu_growth, nu_stress]

# floor rule: x0 = k * eta, and a single crash breaches a fully invested floor with probability e^-k
np.testing.assert_allclose(equity.pi_floor, 100.0 * np.exp(-1.5 * params.eta0))
np.testing.assert_allclose(equity.x0, 1.5 * params.eta0)
assert [round(a.pi_floor, 2) for a in assets.values()] == [87.77, 60.65, 52.58]

# stationary probabilities 10/11 and 1/11
p1 = params.lambda10 / (params.lambda01 + params.lambda10)
np.testing.assert_allclose([p1, 1.0 - p1], [10.0 / 11.0, 1.0 / 11.0])
```

The second block simulates the model exactly: holding times are drawn from their exponential laws,
the Gaussian increment over each holding time is drawn in one step, and each transition adds an
exponential log-jump, so there is no time-discretisation error. It uses the equity regime process
with equal total returns of 4% in both regimes. Over 200,000 ten-year paths from growth:

- the mean growth factor $S_T/S_0$ is 1.4924 with a standard error of 0.0022, against
  $e^{0.4} = 1.4918$ from the proposition on total expected return;
- the share of time spent in stress is 8.262%, against 8.264% from the closed form;
- the 183,738 simulated crashes have a mean relative price change of $-24.996$%, against
  $\alpha^{[1]} = -25$%;
- 22.28% of them exceed $1.5\eta^{[1]}$ in size, against $e^{-1.5} = 22.31$%.

Every comparison is within one third of a standard error.

```python
def simulate_regime_paths(asset, horizon, n_paths, seed):
    """Exact simulation from growth: log-price at the horizon, time in stress, crash log-jumps.

    Holding times are exponential, the Gaussian increment over each holding time is exact, and
    the price jumps by an exponential log-size at each transition.
    """
    p = asset.params
    rng = np.random.default_rng(seed)
    nu = np.array([asset.nu0, asset.nu1])
    vol = np.array([p.sigma0, p.sigma1])
    rate = np.array([p.lambda01, p.lambda10])
    log_price, stress_time, clock = np.zeros((3, n_paths))
    regime = np.zeros(n_paths, dtype=int)
    crash_jumps = []
    active = np.ones(n_paths, dtype=bool)
    while active.any():
        i = np.flatnonzero(active)
        now = regime[i]
        hold = rng.exponential(1.0 / rate[now])
        remaining = horizon - clock[i]
        step = np.minimum(hold, remaining)
        log_price[i] += nu[now] * step + vol[now] * np.sqrt(step) * rng.standard_normal(i.size)
        stress_time[i] += np.where(now == 1, step, 0.0)
        clock[i] += step
        switch = i[hold < remaining]
        crash = regime[switch] == 0
        jump = np.where(crash, -rng.exponential(p.eta0, switch.size),
                        rng.exponential(p.eta1, switch.size))
        crash_jumps.append(jump[crash])
        log_price[switch] += jump
        regime[switch] = 1 - regime[switch]
        active[i[hold >= remaining]] = False
    return log_price, stress_time, np.concatenate(crash_jumps)


# equal total returns of 4% in both regimes: E[S_T] = exp(0.04 T) from either regime
flat = gba.AssetSpecification('flat', params, mu_growth=0.04, mu_stress=0.04)
horizon, n_paths = 10.0, 200_000
log_price, stress_time, crash_jumps = simulate_regime_paths(flat, horizon, n_paths, seed=20261001)
growth_factor = np.exp(log_price)
standard_error = growth_factor.std() / np.sqrt(n_paths)
np.testing.assert_allclose([growth_factor.mean(), standard_error], [1.4924, 0.0022], atol=5e-5)
assert abs(growth_factor.mean() - np.exp(0.04 * horizon)) < 3.0 * standard_error

# expected share of time in stress over [0, T] from growth, against its closed form
total = params.lambda01 + params.lambda10
share = (params.lambda01 / total) * (1.0 - (1.0 - np.exp(-total * horizon)) / (total * horizon))
simulated_share = (stress_time / horizon).mean()
np.testing.assert_allclose([simulated_share, share], [0.08262, 0.08264], atol=5e-6)
assert abs(simulated_share - share) < 3.0 * (stress_time / horizon).std() / np.sqrt(n_paths)

# compensator and single-crash floor breach over all simulated crashes
relative_change = np.exp(crash_jumps) - 1.0
assert crash_jumps.size == 183_738
np.testing.assert_allclose(relative_change.mean(), -0.24996, atol=5e-6)
assert abs(relative_change.mean() - alpha[0]) < 3.0 * relative_change.std() / np.sqrt(crash_jumps.size)
breach_share = (crash_jumps <= -1.5 * params.eta0).mean()
np.testing.assert_allclose(breach_share, 0.2228, atol=5e-5)
assert abs(breach_share - np.exp(-1.5)) < 3.0 * np.sqrt(np.exp(-1.5) * (1.0 - np.exp(-1.5)) / crash_jumps.size)
```

## Implementation in goal_based_allocation

| Quantity | Formula | Entry point |
|---|---|---|
| Regime and jump parameters | $\sigma^{[i]}$, $\lambda^{[12]}$, $\lambda^{[21]}$, $\eta^{[1]}$, $\eta^{[2]}$ | `RegimeSwitchParams(sigma0, sigma1, lambda01, lambda10, eta0=0.0, eta1=0.0)` |
| Log drifts | $\nu^{[i]} = \bar\mu^{[i]} - \lambda^{[i \to j]}\alpha^{[i]} - (\sigma^{[i]})^2/2$ | `RegimeSwitchParams.compute_drifts(mu_growth, mu_stress)` |
| Jumps present | $\eta^{[1]} \gt 10^{-12}$ or $\eta^{[2]} \gt 10^{-12}$ | `RegimeSwitchParams.has_jumps` |
| Asset class | Parameters, total returns, initial wealth and floor | `AssetSpecification(name, params, mu_growth, mu_stress, pi0=100.0, pi_floor=80.0)` |
| Log drifts of an asset | $\nu^{[1]}$, $\nu^{[2]}$ | `AssetSpecification.nu0`, `AssetSpecification.nu1` |
| Distance to the floor | $x_0 = \ln(\Pi_0/L_0)$, infinite without a floor | `AssetSpecification.x0`, `AssetSpecification.has_barrier` |
| Paper asset classes | Table 1 with $L_0 = \Pi_0 e^{-k\eta^{[1]}}$ | `create_paper_assets(pi0=100.0, k_floor=1.5)` |
| Fixed-weight mandates | Named weight dictionaries over the asset classes | `MandateSpecification(name, allocations, T=10.0, r=0.02, c=0.03)`, `create_paper_mandates(assets)` |

The containers live in
[`regime_switch_paper.py`](https://github.com/ArturSepp/GoalBasedAllocation/blob/main/src/goal_based_allocation/regime_switch_paper.py).
API pages: {doc}`RegimeSwitchParams <api/generated/goal_based_allocation.RegimeSwitchParams>`,
{doc}`AssetSpecification <api/generated/goal_based_allocation.AssetSpecification>` and
{doc}`create_paper_assets <api/generated/goal_based_allocation.create_paper_assets>`.

Contract details:

- `RegimeSwitchParams` and `AssetSpecification` are frozen dataclasses, so an asset can key the
  `allocations` dictionary of a `MandateSpecification`. Neither validates its inputs: a
  non-positive intensity, a negative jump parameter or $\eta^{[2]} \ge 1$ is accepted and fails, or
  returns a meaningless number, only when a function uses it. `RiskNeutralParams` is the validated
  container for pricing.
- `compute_drifts` adds the compensator of a jump only when its parameter is positive, so
  `eta0=0.0` and `eta1=0.0` give a pure regime-switching diffusion with $\mu^{[i]} = \bar\mu^{[i]}$.
- `AssetSpecification.x0` is infinite and `has_barrier` false when `pi_floor` is zero or negative;
  the density routines then solve the unbounded problem.
- `create_paper_assets` returns the classes in the order `'bonds'`, `'equity'`,
  `'private_equity'`, with $\eta^{[1]}$ and $\eta^{[2]}$ computed from the crash losses and recovery
  gains of Table 1 and floors set by `k_floor`.
- `create_paper_mandates` returns three fixed-weight mandates, `'conservative'` (70/20/10),
  `'balanced'` (40/40/20) and `'growth'` (10/60/30) in bonds, equity and private equity, which the
  wealth-path and terminal-distribution examples use. The mandates of Table 2 of the manuscript,
  and of the paper's figures, lie on the bond-weight curve of equation (7.5) instead, 65/23/12 for
  conservative and 35/43/22 for balanced; [mandate aggregation](mandate_aggregation.md) builds
  those.
- No package function reads the `T`, `r` and `c` fields of a `MandateSpecification`; the horizon
  and rates are passed to the functions that use them.

## Interpretation and limitations

- The model has exactly two regimes, and jumps occur only at transitions. Intra-regime jumps,
  more regimes and stochastic intensities are outside the published model; the characteristic
  polynomial of the Laplace framework has degree $3N$ for $N$ regimes.
- The regime is not observable in practice. The package computes regime-conditional quantities;
  filtering the current regime from data is left to the user.
- Exponential jump laws are unbounded: a crash log-jump can exceed any size, although the price
  stays positive. The quoted crash loss is a mean, $\mathbb{E}[1 - e^{J^{[1]}}]$, and individual
  crashes are larger or smaller.
- All asset classes of a mandate share the regime chain, so their crashes coincide. The jump sizes
  of different classes are independent given the crash.
- The parameters of Table 1 are capital market assumptions of the manuscript, not estimates; the
  intensities match one crash per decade and one-year stress periods and are illustrative.

## See also

- [Notation and conventions](conventions.md)
- [Buy-and-hold moments](buy_and_hold_moments.md)
- [The MV-optimal policy and the Riccati system](mv_optimal_policy.md)
- [The wealth floor and the flat-barrier reduction](wealth_floor_gap_process.md)
- [Mandates as one effective asset](mandate_aggregation.md)
- [European options under regime switching](european_options.md)
- [Bibliography](bibliography.md)

## References

1. Sepp, A. (2026). Dynamic Mean-Variance Portfolio Allocation under Regime-Switching Jump-Diffusions with Absorbing Barriers and Distribution Matching. Working paper. [DOI: 10.2139/ssrn.6534579](https://doi.org/10.2139/ssrn.6534579). Section 2 defines the model; Table 1 gives the asset classes.
2. Hamilton, J. D. (1989). A New Approach to the Economic Analysis of Nonstationary Time Series and the Business Cycle. *Econometrica*, 57(2), 357–384. [DOI: 10.2307/1912559](https://doi.org/10.2307/1912559). Markov regime switching in economic time series.
3. Shi, X., and Xu, Z. Q. (2025). Optimal Mean-Variance Portfolio Selection under Regime-Switching-Induced Stock Price Shocks. *Systems & Control Letters*, 204, 106200. [DOI: 10.1016/j.sysconle.2025.106200](https://doi.org/10.1016/j.sysconle.2025.106200). Regime switching with independent jumps, the alternative to jumps at transitions.
4. Kou, S. G. (2002). A Jump-Diffusion Model for Option Pricing. *Management Science*, 48(8), 1086–1101. [DOI: 10.1287/mnsc.48.8.1086.166](https://doi.org/10.1287/mnsc.48.8.1086.166). Exponential jump magnitudes and their rational transforms.
5. Sepp, A. goal-based-allocation: Semi-analytical dynamic mean-variance allocation and terminal-wealth risk under regime-switching jump-diffusions. [Software citation metadata](https://github.com/ArturSepp/GoalBasedAllocation/blob/main/CITATION.cff).
