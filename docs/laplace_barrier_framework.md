---
myst:
  html_meta:
    description: >-
      The Laplace transform framework of goal-based-allocation for barrier problems under the
      two-regime jump-diffusion: Arrow-Debreu state prices, the sixth-order characteristic
      polynomial and its roots, unbounded and bounded solutions, survival, tilted survival and the
      overshoot density, verified against closed forms and exact simulation.
---

# Survival, densities and overshoot in the Laplace domain

*Author: [Artur Sepp](https://github.com/ArturSepp)*

Implemented in [GoalBasedAllocation](https://github.com/ArturSepp/GoalBasedAllocation).
Software citation: [CITATION.cff](https://github.com/ArturSepp/GoalBasedAllocation/blob/main/CITATION.cff).

The Laplace transform framework computes the transition density of a regime-switching
jump-diffusion killed at a barrier, its integral, the survival probability, exponentially weighted
integrals of it, and the law of the overshoot below the barrier, all from the roots of one
polynomial (Sepp, 2026, Section 5). Exponential jumps make the transform of the forward equation
rational, so its solution in the Laplace domain is a finite sum of exponentials; one
[numerical inversion](laplace_inversion.md) in time returns each quantity. The method extends the
Laplace-transform approach to barrier problems with exponential jumps of Lipton (2002) and Sepp
(2004) to regime switching with jumps at the transitions.

## Overview

Four public functions implement the framework for an `AssetSpecification` whose state starts at
`x0` above a barrier at zero, or for an unbounded process when the asset has no floor:

1. `compute_density` returns the transition density of the state at the horizon, split by the
   terminal regime and restricted to paths that survive;
2. `compute_survival` returns the probability that the barrier has not been reached;
3. `compute_tilted_survival` returns $\mathbb{E}[e^{-oX_T}\thinspace \mathbb{1}\lbrace\tau_L \gt T\rbrace]$,
   the exponential moments of the surviving state;
4. `compute_overshoot_density` returns the density of the distance below the barrier of paths
   stopped by a crash.

Applied to the [gap process](wealth_floor_gap_process.md), they give the three components of
stopped terminal wealth; applied to an asset with a floor, they give the probability that a fully
invested position reaches its floor. The framework starts in growth. The worked example checks each
function against a closed form or an exact simulation.

## Inputs, notation, and assumptions

| Convention | This article |
|---|---|
| Time and rates | Horizon $T$ in years; the Laplace variable $p$ is conjugate to time |
| Regimes | Every quantity starts in growth, regime 1; densities are split by the terminal regime; `state_init` of `compute_density` is ignored |
| Jump sizes | Mean convention; the transforms need the exact exponential laws, so the gap process uses its effective means $\tilde\eta^{[i]}$ |
| Wealth coordinate | The state $X$ of the asset, started at $x_0$ = `asset.x0` with the barrier at zero; densities are per unit of $X$, overshoot distances $d = -X_{\tau_L} \gt 0$ |
| Measure | Physical |
| Numerical method | `numpy.roots` of the sixth-order polynomial and dense linear solves at each of 38 Laplace arguments, Abate–Whitt inversion; checked against the reflection formula and exact simulation |
| Package default | `compute_density(T, x_grid, asset, state_init=0)`, `compute_survival(T, x0, asset)`, `compute_tilted_survival(T, x0, asset, o)`, `compute_overshoot_density(T, x_grid, asset)` |

| Symbol | Meaning | Units and convention |
|---|---|---|
| $A^{[i \mid s]}(\tau, X; x)$ | Arrow–Debreu price: density of the state at $X$ in regime $i$ at time $\tau$, from $x$ in regime $s$ | Per unit of $X$ |
| $\hat A^{[i \mid s]}(X; x, p)$ | Its Laplace transform in time | Per unit of $X$ and of time |
| $G^{[1]}(\psi)$, $G^{[2]}(\psi)$ | Factors of the characteristic polynomial | Polynomials in $\psi$ |
| $Q^{[i]}(\psi)$ | Quadratic part $-(\sigma^{[i]})^2\psi^2/2 + \nu^{[i]}\psi + \lambda^{[i \to j]} + p$ | |
| $\beta_k$ | Coupling ratio of the regimes at root $\psi_k$ | Dimensionless |
| $o$ | Tilt of the tilted survival | Per unit of $X$ |
| $C_{\mathrm{over}}(T)$ | Overshoot mass by the horizon | Probability |

The Laplace variable $p$, the roots $\psi_k$, survival $Q$, tilted survival $\tilde Q(o)$, floor atom
$F$ and overshoot mass $\mathcal{O}$ are reserved on the [conventions page](conventions.md#reserved-notation).
The model is the [regime-switching jump-diffusion](regime_switching_model.md) of the state $X$; the
framework requires positive intensities and positive volatilities in both regimes.

## Methodology

### Arrow–Debreu prices and the forward equation

**Definition (Sepp, 2026, Definition 5.1).** The regime-conditional Arrow–Debreu prices
$A^{[i \mid s]}(\tau, X; x)$ solve the forward Fokker–Planck system

$$
\partial_\tau A^{[1 \mid s]} = -\nu^{[1]}\partial_XA^{[1 \mid s]} + \frac{(\sigma^{[1]})^2}{2}\partial_{XX}A^{[1 \mid s]} - \lambda^{[12]}A^{[1 \mid s]} + \lambda^{[21]}\int_0^{\infty}A^{[2 \mid s]}(X - J)f^{[2]}(J)\thinspace dJ,
$$

and the mirror equation for regime 2 with the crash density $f^{[1]}$, started from a point mass at
$x$ in regime $s$, equations (5.1)–(5.3). With an absorbing barrier at zero the prices vanish for
$X \le 0$. Integrating them against a payoff values it (Proposition 5.2): the survival probability is
the integral over $X \gt 0$, the transition density is the price itself.

### The characteristic polynomial

**Theorem (Sepp, 2026, Theorem 5.3).** The Laplace-transformed prices are sums of exponentials
$e^{\psi X}$ whose exponents solve

$$
G^{[1]}(\psi)G^{[2]}(\psi) - \lambda^{[12]}\lambda^{[21]} = 0, \qquad
G^{[1]}(\psi) = (1 - \eta^{[1]}\psi)\thinspace Q^{[1]}(\psi), \qquad
G^{[2]}(\psi) = (1 + \eta^{[2]}\psi)\thinspace Q^{[2]}(\psi),
$$

a polynomial of degree six, equations (5.6)–(5.8).

**Proof.** Transform the forward system in time; the point mass at $x$ becomes a source term. Away
from the source, try $\hat A^{[1]} = a e^{\psi X}$ and $\hat A^{[2]} = b e^{\psi X}$. The jump integrals
are rational: $\int_0^{\infty}e^{\psi(X - J)}f^{[2]}(J)dJ = e^{\psi X}/(1 + \eta^{[2]}\psi)$ and, for the
crash, $e^{\psi X}/(1 - \eta^{[1]}\psi)$. The two equations become
$-Q^{[1]}(\psi)a + \lambda^{[21]}b/(1 + \eta^{[2]}\psi) = 0$ and
$-Q^{[2]}(\psi)b + \lambda^{[12]}a/(1 - \eta^{[1]}\psi) = 0$; multiplying out the denominators, a
non-zero solution exists exactly when the determinant vanishes. $\square$

**Lemma (root structure; Sepp, 2026, Lemma 5.4).** For real $p \gt 0$ the six roots are real, three
negative and three positive: $\psi_1 \lt \psi_2 \lt \psi_3 \lt 0 \lt \psi_4 \lt \psi_5 \lt \psi_6$.

The negative roots give the solution above the source, which must decay as $X \to \infty$, and the
positive roots the solution below it. The inversion evaluates the transforms at complex $p$, where the
roots are complex; the package sorts them by real part and assigns the first three to the decaying
side. The coupling ratio $b_k/a_k$ at each root is
$\beta_k = Q^{[1]}(\psi_k)(1 + \eta^{[2]}\psi_k)/\lambda^{[21]}$, equation (5.11).

### Unbounded and bounded solutions

**Theorem (Sepp, 2026, Theorems 5.5 and 5.6).** Without a barrier,

$$
\hat A^{[1 \mid s]}(X; x, p) = \sum_{k=4}^{6}a_k^{[s]}e^{\psi_k(X - x)} \ \text{for} \ X \le x, \qquad
\hat A^{[1 \mid s]}(X; x, p) = \sum_{k=1}^{3}a_k^{[s]}e^{\psi_k(X - x)} \ \text{for} \ X \gt x,
$$

and $\hat A^{[2 \mid s]}$ has coefficients $\beta_ka_k^{[s]}$. The six coefficients solve a linear
system: both components are continuous at the source, the derivative of the starting component jumps
by $-2/(\sigma^{[s]})^2$, and two spurious poles introduced by clearing the jump denominators are
removed. With an absorbing barrier at zero, a correction $\sum_{k=1}^{3}d_k^{[s]}(x)e^{\psi_kX}$ built
from the decaying roots is added, whose three coefficients make both components and the recovery
in-flow vanish at the barrier.

Integrating these sums of exponentials over $X$ is elementary, which gives the survival probability
in closed form in the Laplace domain (Corollary 5.7), and with the weight $e^{-oX}$ the tilted
survival: each $1/\psi_k$ becomes $1/(\psi_k - o)$.

**Definition (tilted survival; equation (6.13)).** For $o \ge 0$,

$$
\tilde Q(o) = \int_0^{\infty}e^{-oX}A(T, x_0; X)\thinspace dX = \mathbb{E}\left[e^{-oX_T}\mathbb{1}\lbrace\tau_L \gt T\rbrace\right],
$$

summed over the terminal regimes. $\tilde Q(0) = Q$, and on the gap process $\tilde Q(1)$ and
$\tilde Q(2)$ give the first two moments of surviving wealth, because $\Pi_T = \Pi_T^{*} - B_Te^{-X_T}$.

### The overshoot below the barrier

**Proposition (overshoot density; Sepp, 2026, Theorem 6.1(c)).** For exponential crash jumps with
mean $\eta^{[1]}$, the distance $d$ below the barrier of paths stopped by a crash has density

$$
f_{\mathrm{over}}(d) = \frac{C_{\mathrm{over}}(T)}{\eta^{[1]}}e^{-d/\eta^{[1]}}, \qquad
C_{\mathrm{over}}(T) = \lambda^{[12]}\int_0^T\mathbb{E}\left[e^{-X_t/\eta^{[1]}}\mathbb{1}\lbrace\tau_L \gt t, \chi_t = 1\rbrace\right]dt,
$$

and its transform is $\lambda^{[12]}\hat{\tilde Q}^{[1]}(p; 1/\eta^{[1]})/p$, the growth component of the
tilted survival with $o = 1/\eta^{[1]}$, divided by $p$.

**Proof.** Crashes occur in growth at rate $\lambda^{[12]}$. A crash at time $t$ from state $X_t \gt 0$
crosses the barrier when its size $E$ exceeds $X_t$, and then $d = E - X_t$. By the memoryless property
of the exponential law, $\mathbb{P}(E \gt X_t + d \mid X_t) = e^{-X_t/\eta^{[1]}}e^{-d/\eta^{[1]}}$, so
the rate of overshoots beyond $d$ at time $t$ is
$\lambda^{[12]}\mathbb{E}[e^{-X_t/\eta^{[1]}};\ \text{alive}, \chi_t = 1]\thinspace e^{-d/\eta^{[1]}}$. Integrating
over $t \in [0, T]$ and differentiating in $d$ gives the density; division by $p$ is the Laplace
transform of the time integral. $\square$

The overshoot mass is $\mathcal{O} = C_{\mathrm{over}}(T) = \eta^{[1]}f_{\mathrm{over}}(0)$, and the
floor atom is the remainder $F = 1 - Q - \mathcal{O}$: paths that reached the barrier continuously.

> **Insight.** The overshoot law is exactly exponential with the crash mean, whatever the horizon,
> the regime parameters or the starting point: only its mass depends on them. Recoveries cannot
> overshoot because they move the state away from the barrier, and the floor atom needs no transform
> of its own.

## Worked example

The first block checks the polynomial and the unbounded solution on the equity class of Table 1.
At $p = 0.01$, 0.5 and 5 the six roots of the characteristic polynomial are real with three of each
sign, as the lemma states. Without a floor, the density of the log-price after ten years from
growth integrates to one, its mean is 0.19607, equal to the exact mean
$\nu^{[1]}\mathcal{T}^{[1]} + \nu^{[2]}\mathcal{T}^{[2]} - \lambda^{[12]}\eta^{[1]}\mathcal{T}^{[1]} + \lambda^{[21]}\eta^{[2]}\mathcal{T}^{[2]}$
with $\mathcal{T}^{[i]}$ the expected occupation times, and its stress component has mass 0.09091,
the probability $p_2(1 - e^{-(\lambda^{[12]} + \lambda^{[21]})T})$ of ending in stress. In the
jump-free limit with identical regimes the process is a Brownian motion with drift, and survival and
density match the reflection formula and the method of images to $10^{-8}$.

```python
import numpy as np
from scipy.stats import norm
import goal_based_allocation as gba

equity = gba.create_paper_assets()['equity']
p = equity.params


def characteristic_polynomial(asset, lap):
    """Coefficients of G1(psi) G2(psi) - lambda12 lambda21, highest power first."""
    pr = asset.params
    q1 = [-0.5 * pr.sigma0**2, asset.nu0, pr.lambda01 + lap]
    q2 = [-0.5 * pr.sigma1**2, asset.nu1, pr.lambda10 + lap]
    poly = np.polymul(np.polymul(q1, [-pr.eta0, 1.0]), np.polymul(q2, [pr.eta1, 1.0]))
    poly[-1] -= pr.lambda01 * pr.lambda10
    return poly


for lap in (0.01, 0.5, 5.0):
    roots = np.roots(characteristic_polynomial(equity, lap))
    assert roots.size == 6 and np.abs(roots.imag).max() == 0.0
    assert (roots.real < 0).sum() == 3 and (roots.real > 0).sum() == 3

# unbounded density of the log-price from growth after ten years
horizon = 10.0
unbounded = gba.AssetSpecification('unbounded', p, equity.mu_growth, equity.mu_stress, pi_floor=0.0)
x = np.linspace(-6.0, 4.0, 4001)
growth_part, stress_part = gba.compute_density(horizon, x, unbounded)
density = growth_part + stress_part
total = p.lambda01 + p.lambda10
occupation_stress = (p.lambda01 / total) * (horizon - (1.0 - np.exp(-total * horizon)) / total)
occupation_growth = horizon - occupation_stress
exact_mean = (equity.nu0 * occupation_growth + equity.nu1 * occupation_stress
              - p.lambda01 * p.eta0 * occupation_growth + p.lambda10 * p.eta1 * occupation_stress)
np.testing.assert_allclose(np.trapezoid(density, x), 1.0, atol=1e-6)
np.testing.assert_allclose(np.trapezoid(x * density, x), exact_mean, atol=1e-5)
np.testing.assert_allclose(np.trapezoid(stress_part, x), (p.lambda01 / total) * (1.0 - np.exp(-total * horizon)),
                           atol=1e-6)
np.testing.assert_allclose([exact_mean, np.trapezoid(stress_part, x)], [0.19607, 0.09091], atol=5e-6)

# jump-free limit: two identical regimes are a Brownian motion with drift
drift, vol, start = 0.03, 0.2, 0.5
twin = gba.AssetSpecification('twin', gba.RegimeSwitchParams(sigma0=vol, sigma1=vol, lambda01=0.5, lambda10=0.5),
                              mu_growth=drift + 0.5 * vol**2, mu_stress=drift + 0.5 * vol**2,
                              pi0=100.0, pi_floor=100.0 * np.exp(-start))
np.testing.assert_allclose([twin.nu0, twin.x0], [drift, start], rtol=1e-12)
for t in (1.0, 5.0, 10.0):
    s = vol * np.sqrt(t)
    reflection = norm.cdf((start + drift * t) / s) - np.exp(-2.0 * drift * start / vol**2) * norm.cdf((-start + drift * t) / s)
    assert abs(gba.compute_survival(t, twin.x0, twin) - reflection) < 1e-8
grid = np.linspace(0.001, 3.0, 600)
s = vol * np.sqrt(5.0)
images = (norm.pdf((grid - start - drift * 5.0) / s)
          - np.exp(-2.0 * drift * start / vol**2) * norm.pdf((grid + start - drift * 5.0) / s)) / s
assert np.abs(sum(gba.compute_density(5.0, grid, twin)) - images).max() < 1e-8
```

The second block puts the equity class at its floor of 60.65, $x_0 = 0.5$, and asks how likely a
fully invested position is to reach it. The Laplace framework gives survival of 96.60%, 92.23%,
80.47% and 68.61% over one, two, five and ten years. At ten years the bounded density integrates to
the survival probability, its exponential moment equals the tilted survival 0.26698, and the
overshoot mass is 11.84% with a mean overshoot of $\eta^{[1]} = 1/3$, leaving a floor atom of 19.55%.
An exact simulation of one million paths, with the Brownian-bridge crossing probability between
jumps, gives 68.61%, 19.53%, 11.86% and 0.26696, and a mean overshoot of 0.334.

```python
def simulate_stopping(asset, horizon, n_paths, seed):
    """Exact simulation from growth: status 0 survived, 1 stopped by diffusion, 2 by a crash.

    Holding times and Gaussian increments are exact, and the diffusion touches zero between two
    points x, y > 0 over a time h with probability exp(-2 x y / (sigma^2 h)).
    """
    pr = asset.params
    rng = np.random.default_rng(seed)
    nu = np.array([asset.nu0, asset.nu1])
    vol_r = np.array([pr.sigma0, pr.sigma1])
    rate = np.array([pr.lambda01, pr.lambda10])
    x = np.full(n_paths, asset.x0)
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
        end = x[i] + nu[now] * step + vol_r[now] * np.sqrt(step) * rng.standard_normal(i.size)
        bridge = np.exp(-2.0 * x[i] * np.maximum(end, 0.0) / (vol_r[now]**2 * step))
        hit = (end <= 0.0) | (rng.uniform(size=i.size) < bridge)
        status[i[hit]] = 1
        x[i], clock[i] = end, clock[i] + step
        switch = i[~hit & (hold < remaining)]
        crash = regime[switch] == 0
        x[switch] += np.where(crash, -rng.exponential(pr.eta0, switch.size), rng.exponential(pr.eta1, switch.size))
        regime[switch] = 1 - regime[switch]
        status[switch[x[switch] <= 0.0]] = 2
        active = (status == 0) & (clock < horizon)
    return status, x


survival_path = [gba.compute_survival(t, equity.x0, equity) for t in (1.0, 2.0, 5.0, 10.0)]
np.testing.assert_allclose(survival_path, [0.9660, 0.9223, 0.8047, 0.6861], atol=5e-5)
assert all(earlier > later for earlier, later in zip(survival_path, survival_path[1:]))
survival = survival_path[-1]

level = np.linspace(1e-4, 6.0, 3000)
bounded = sum(gba.compute_density(horizon, level, equity))
tilted = gba.compute_tilted_survival(horizon, equity.x0, equity, 1.0)
np.testing.assert_allclose(np.trapezoid(bounded, level), survival, atol=1e-6)
np.testing.assert_allclose(np.trapezoid(np.exp(-level) * bounded, level), tilted, atol=1e-6)

distance = np.linspace(0.0, 8.0, 4001)
overshoot_density = gba.compute_overshoot_density(horizon, distance, equity)
overshoot = overshoot_density[0] * p.eta0                      # exact mass C_over = eta f(0)
np.testing.assert_allclose(np.trapezoid(overshoot_density, distance), overshoot, rtol=1e-4)
np.testing.assert_allclose(np.trapezoid(distance * overshoot_density, distance) / overshoot, p.eta0, rtol=1e-4)
floor_atom = 1.0 - survival - overshoot
np.testing.assert_allclose([tilted, overshoot, floor_atom], [0.26698, 0.1184, 0.1955], atol=5e-5)

status, terminal = simulate_stopping(equity, horizon, 1_000_000, seed=1)
simulated = [(status == 0).mean(), (status == 1).mean(), (status == 2).mean()]
for value, exact in zip(simulated, (survival, floor_atom, overshoot)):
    assert abs(value - exact) < 3.0 * np.sqrt(exact * (1.0 - exact) / status.size)
simulated_tilted = np.mean(np.where(status == 0, np.exp(-terminal), 0.0))
np.testing.assert_allclose(simulated_tilted, tilted, atol=3e-4)
np.testing.assert_allclose(-terminal[status == 2].mean(), p.eta0, rtol=0.01)

# the starting regime of compute_density is fixed to growth
assert np.array_equal(gba.compute_density(horizon, level, equity, state_init=1)[0],
                      gba.compute_density(horizon, level, equity)[0])
```

## Implementation in goal_based_allocation

| Quantity | Formula | Entry point |
|---|---|---|
| Density of the surviving state, by terminal regime | Inverse of the bounded or unbounded $\hat A^{[i \mid 1]}(X; x_0, p)$ | `compute_density(T, x_grid, asset, state_init=0)` returns `(d0, d1)` |
| Survival probability | $Q = \int_0^{\infty}A\thinspace dX$, both terminal regimes | `compute_survival(T, x0, asset)` |
| Tilted survival | $\tilde Q(o) = \int_0^{\infty}e^{-oX}A\thinspace dX$ | `compute_tilted_survival(T, x0, asset, o)` |
| Overshoot density | $(C_{\mathrm{over}}/\eta^{[1]})e^{-d/\eta^{[1]}}$ | `compute_overshoot_density(T, x_grid, asset)` |
| Density, survival and stopping probability together | | `regime_switch_paper.compute_wealth_density(T, x_grid, asset)` (module level) |

The framework lives in
[`regime_switch_paper.py`](https://github.com/ArturSepp/GoalBasedAllocation/blob/main/src/goal_based_allocation/regime_switch_paper.py).
API pages: {doc}`compute_density <api/generated/goal_based_allocation.compute_density>`,
{doc}`compute_survival <api/generated/goal_based_allocation.compute_survival>`,
{doc}`compute_tilted_survival <api/generated/goal_based_allocation.compute_tilted_survival>` and
{doc}`compute_overshoot_density <api/generated/goal_based_allocation.compute_overshoot_density>`.

Contract details:

- Each call solves the polynomial with `numpy.roots` and the linear systems with `numpy.linalg` at 38
  complex Laplace arguments and inverts with the default Abate–Whitt settings. A density call
  evaluates the whole grid from the same roots.
- `compute_density` solves the bounded problem when `asset.has_barrier`, with the source at
  `asset.x0`, and otherwise the unbounded problem with the source at zero. The grid is the terminal
  state; points at or below zero are not meaningful for the bounded problem.
- `compute_survival` and `compute_tilted_survival` take the starting state `x0` as an argument and use
  the barrier at zero; pass `asset.x0` for the asset's own floor. Both return a float, the sum over
  the terminal regimes.
- The tilted survival converges only for $o$ below the smallest positive root; the package uses
  $o = 1$ and $o = 2$ for moments and $o = 1/\tilde\eta^{[1]}$ for the overshoot.
- `compute_overshoot_density` returns zeros when the asset has no crash jump, and clips negative
  inversion noise at zero.
- With $\eta^{[1]} = \eta^{[2]} = 0$ the polynomial has degree four and the systems are $4 \times 4$
  and $2 \times 2$; the coupling ratio divides by $\lambda^{[21]}$, so both intensities must be positive.

> **Pitfall.** `compute_density` documents a `state_init` argument for the initial regime but does not
> use it: every density, like every survival probability of the package, starts in growth. The worked
> example confirms that `state_init=1` returns the growth-start density. Densities from stress need
> the right-hand side of the linear system for a stress source, which the package implements only in
> the [option pricer](european_options.md).

## Interpretation and limitations

- The framework is exact for regime-switching jump-diffusions with exponential jumps, up to the
  inversion and root-finding error, which the worked example bounds by $10^{-8}$ in the jump-free
  limit. Its error for a wealth process comes from the reduction to such a process, discussed in the
  [floor chapter](wealth_floor_gap_process.md).
- Only rational jump transforms keep the polynomial structure: exponential, hyperexponential or
  Erlang laws. Gamma jumps with non-integer shape or normal jumps do not.
- The overshoot mass is exact in the transform, but `compute_opportunity_point` integrates the
  density numerically from $d = 0.001$, which misses about $0.001\thinspace f_{\mathrm{over}}(0)$ of mass and
  moves it into the floor atom; the [terminal wealth chapter](terminal_wealth_distribution.md)
  quantifies it.
- First-passage time densities (Corollary 5.8) and survival-conditional moments of higher order
  (Theorem 6.4) follow from the same roots but are not implemented as functions.

## See also

- [Numerical Laplace inversion](laplace_inversion.md)
- [The wealth floor and the flat-barrier reduction](wealth_floor_gap_process.md)
- [The terminal wealth distribution](terminal_wealth_distribution.md)
- [European options under regime switching](european_options.md)
- [The regime-switching jump-diffusion](regime_switching_model.md)
- [Bibliography](bibliography.md)

## References

1. Sepp, A. (2026). Dynamic Mean-Variance Portfolio Allocation under Regime-Switching Jump-Diffusions with Absorbing Barriers and Distribution Matching. Working paper. [DOI: 10.2139/ssrn.6534579](https://doi.org/10.2139/ssrn.6534579). Section 5 and Appendix B derive the characteristic polynomial and the unbounded and bounded solutions; Theorem 6.1 the overshoot density.
2. Lipton, A. (2002). Assets with Jumps. *Risk*, 15(9), 149–153. [risk.net](https://www.risk.net/derivatives/1530269/assets-with-jumps). Laplace transforms for barrier problems with exponential jumps.
3. Sepp, A. (2004). Analytical Pricing of Double-Barrier Options under a Double-Exponential Jump Diffusion Process: Applications of Laplace Transform. *International Journal of Theoretical and Applied Finance*, 7(2), 151–175. [DOI: 10.1142/S0219024904002402](https://doi.org/10.1142/S0219024904002402). The Laplace-transform method for double-exponential jump-diffusions with barriers.
4. Kou, S. G., and Wang, H. (2003). First Passage Times of a Jump Diffusion Process. *Advances in Applied Probability*, 35(2), 504–531. [DOI: 10.1239/aap/1051201658](https://doi.org/10.1239/aap/1051201658). First passage and the memoryless overshoot of exponential jumps.
5. Abate, J., and Whitt, W. (1995). Numerical Inversion of Laplace Transforms of Probability Distributions. *ORSA Journal on Computing*, 7(1), 36–43. [DOI: 10.1287/ijoc.7.1.36](https://doi.org/10.1287/ijoc.7.1.36). The inversion algorithm.
6. Sepp, A. goal-based-allocation: Semi-analytical dynamic mean-variance allocation and terminal-wealth risk under regime-switching jump-diffusions. [Software citation metadata](https://github.com/ArturSepp/GoalBasedAllocation/blob/main/CITATION.cff).
