---
myst:
  html_meta:
    description: >-
      The pre-commitment mean-variance policy under the two-regime jump-diffusion: the quadratic
      value function, the coupled Riccati system, the Merton-Lipton policy, a regime-independent
      target, closed-form expected terminal wealth, variance and efficient frontier, and how
      find_ell sets the Lagrange multiplier, with Monte Carlo verification.
---

# The MV-optimal policy and the Riccati system

*Author: [Artur Sepp](https://github.com/ArturSepp)*

Implemented in [GoalBasedAllocation](https://github.com/ArturSepp/GoalBasedAllocation).
Software citation: [CITATION.cff](https://github.com/ArturSepp/GoalBasedAllocation/blob/main/CITATION.cff).

The dynamic mean-variance problem chooses a continuous-time allocation between a risky asset and
cash that minimises the variance of terminal wealth for a given expected terminal wealth. Under the
regime-switching jump-diffusion its value function is quadratic in wealth in each regime, its
coefficients solve a coupled system of Riccati equations, and the optimal policy keeps the
Merton–Lipton form: an allocation coefficient times the relative distance from wealth to a target
(Sepp, 2026, Section 3). This chapter derives the policy, proves that the target is the same in both
regimes, and gives the expected terminal wealth, its variance and the efficient frontier in closed
form from one coefficient of the Riccati solution.

## Overview

`find_ell` solves the Riccati system for an asset and a horizon and returns the Lagrange multiplier
$\ell$ with a `RiccatiSolution`, which evaluates the coefficients, the target $\Pi^{*}(t)$ and the
policy $\omega_t^{*}$ at any time. Every later chapter starts from this object: the
[floor chapter](wealth_floor_gap_process.md) turns the policy into a gap process, and the
[opportunity set](investment_opportunity_set.md) calibrates $\ell$ so that the policy starts fully
invested.

Three results make the solution transparent. First, the linear coefficient is proportional to the
quadratic one, so the target is $\Pi^{*}(t) = (\ell/2)e^{-r_c(T - t)}$ in both regimes. Second, the
constant coefficient follows from the quadratic one as well, so the expected value and the variance
of terminal wealth under the optimal policy depend on the Riccati solution only through
$a^{[i]}(T)$. Third, $-b(T)/(2a(T))$, which `find_ell` matches to its `target_return`, is the present
value of the target, not the expected terminal wealth; the chapter shows how to set $\ell$ for an
expected-wealth target.

## Inputs, notation, and assumptions

| Convention | This article |
|---|---|
| Time and rates | Years; $r$, $c$ and `target_return` are continuously compounded annual rates; Riccati coefficients are functions of $\tau = T - t$, from $\tau = 0$ at the horizon to $\tau = T$ at inception |
| Regimes | The system couples both regimes; expected wealth, variance and the policy at inception condition on the starting regime; the target is regime-independent |
| Jump sizes | Mean convention; $\alpha^{[i]}$ and $(\sigma_J^{[i]})^2$ follow from $\eta^{[i]}$ |
| Wealth coordinate | Wealth $\Pi$ in units of `asset.pi0`, 100 for the paper's assets |
| Measure | Physical |
| Numerical method | RK45 with `rtol=1e-10` and `atol=1e-12` on 10,001 points of $[0, T]$, linear interpolation in $\tau$; checked against the single-regime closed form and a Monte Carlo simulation of the policy |
| Package default | `find_ell(asset, T, target_return, r=0.02, c=0.02, regime=0)`; the default consumption equals the default rate, so $r_c = 0$ unless `c` is passed |

| Symbol | Meaning | Units and convention |
|---|---|---|
| $V^{[i]}(\tau, \Pi)$ | Value function in regime $i$ | Wealth squared |
| $\bar\Pi_T$ | Required expected terminal wealth of Problem 3.1 | Wealth units |
| $(\sigma_J^{[i]})^2$ | $\mathbb{E}[(e^{J^{[i]}} - 1)^2]$ | Decimal |
| $D$ | Initial gap $\Pi^{*}(0) - \Pi_0$ | Wealth units |
| $u^{[i]}$ | $e^{-2r_cT}a^{[i]}(T)$ | Dimensionless, in $(0, 1)$ |
| $\Theta^{[i]}$ | Dynamic information ratio $2r_cT - \ln a^{[i]}(T)$ | Dimensionless |
| $\theta$ | Sharpe ratio $(\mu - r_h)/\sigma$ of a single-regime asset without jumps | Per square-root year |

The value-function coefficients $a^{[i]}$, $b^{[i]}$, $\gamma^{[i]}$, the coupled coefficients
$\tilde a^{[i]}$, $\tilde b^{[i]}$, $\Sigma^{[i]}$, the allocation coefficient $\omega_a^{[i]}$, the
target $\Pi^{*}(t)$ and the multiplier $\ell$ are reserved on the
[conventions page](conventions.md#reserved-notation). The policy is unconstrained: it may lever and
short, and the floor is added only in the [next chapter](wealth_floor_gap_process.md).

## Methodology

### The pre-commitment mean-variance problem

The investor holds the fraction $\omega_t$ of wealth in the risky asset and the rest in cash, and
consumes at the proportional rate $c$. With the hurdle rate $r_h = \max(r, c)$ and the net floor rate
$r_c = r_h - c$, wealth follows equation (2.11) of the manuscript,

$$
\frac{d\Pi_t}{\Pi_{t^-}} = \left[r_c + \omega_t\left(\mu^{[\chi_t]} - r_h\right)\right]dt + \omega_t\sigma^{[\chi_t]} dW_t
$$

between transitions, and $\Pi_t = \Pi_{t^-}(1 + \omega_t(e^{J^{[i]}} - 1))$ at a transition out of
regime $i$. For $c \le r$ these are the budget dynamics, with cash earning $r - c$ net of consumption.
For $c \gt r$ the manuscript sets the cash drift to zero and measures the risk premium against $c$
(Assumption 2.5), a modelling choice rather than an identity.

**Problem (Sepp, 2026, Problem 3.1).** Minimise $\mathrm{Var}[\Pi_T]$ over admissible policies
subject to $\mathbb{E}[\Pi_T] = \bar\Pi_T$. With a Lagrange multiplier $\ell \gt 0$ the problem is
equivalent to

$$
V^{[i]}(\tau, \Pi) = \inf_{\omega}\ \mathbb{E}\left[\Pi_T^2 - \ell\thinspace \Pi_T \mid \Pi_t = \Pi, \chi_t = i\right] .
$$

The problem is time-inconsistent: re-optimising at a later date with the same expected-wealth
requirement changes the policy (Basak and Chabakauri, 2010). The package solves the pre-commitment
problem, in which the policy fixed at inception is followed to the horizon, as in Zhou and Li (2000)
and Zhou and Yin (2003).

### The quadratic value function and the coupled Riccati system

**Theorem (Sepp, 2026, Theorem 3.2, Proposition 3.6).** The value function is
$V^{[i]}(\tau, \Pi) = a^{[i]}(\tau)\Pi^2 + b^{[i]}(\tau)\Pi + \gamma^{[i]}(\tau)$. With

$$
\tilde a^{[i]} = (\mu^{[i]} - r_h)a^{[i]} + \lambda^{[i \to j]}\alpha^{[i]}a^{[j]}, \qquad
\tilde b^{[i]} = (\mu^{[i]} - r_h)b^{[i]} + \lambda^{[i \to j]}\alpha^{[i]}b^{[j]}, \qquad
\Sigma^{[i]} = (\sigma^{[i]})^2 a^{[i]} + \lambda^{[i \to j]}(\sigma_J^{[i]})^2 a^{[j]},
$$

equation (3.8), the coefficients solve

$$
\begin{aligned}
\frac{da^{[i]}}{d\tau} &= (2r_c - \lambda^{[i \to j]})a^{[i]} + \lambda^{[i \to j]}a^{[j]} - \frac{(\tilde a^{[i]})^2}{\Sigma^{[i]}}, \\
\frac{db^{[i]}}{d\tau} &= (r_c - \lambda^{[i \to j]})b^{[i]} + \lambda^{[i \to j]}b^{[j]} - \frac{\tilde a^{[i]}\tilde b^{[i]}}{\Sigma^{[i]}}, \\
\frac{d\gamma^{[i]}}{d\tau} &= -\lambda^{[i \to j]}\gamma^{[i]} + \lambda^{[i \to j]}\gamma^{[j]} - \frac{(\tilde b^{[i]})^2}{4\Sigma^{[i]}},
\end{aligned}
$$

with $a^{[i]}(0) = 1$, $b^{[i]}(0) = -\ell$ and $\gamma^{[i]}(0) = 0$, equations (3.10)–(3.12) and
(A.13)–(A.20). Under Assumption 2.2 the $a$-equations have a unique positive solution on any
horizon (Theorem 3.7).

The quadratic form survives the jumps because a quadratic function of post-jump wealth
$\Pi(1 + \omega(e^J - 1))$ is quadratic in both $\Pi$ and $\omega$; the regimes couple only through the
jump terms. Without transition jumps the $a$-equations become linear, as in Zhou and Yin (2003); the
nonlinearity $(\tilde a^{[i]})^2/\Sigma^{[i]}$ is entirely due to them.

**Corollary (optimal policy; Sepp, 2026, Corollary 3.4).** The optimal allocation in regime $i$ is

$$
\omega_t^{*} = \frac{\tilde a^{[i]}}{\Sigma^{[i]}}\left(\frac{\Pi^{*}(t)}{\Pi_t} - 1\right) = -\omega_a^{[i]}\left(\frac{\Pi^{*}(t)}{\Pi_t} - 1\right),
\qquad \Pi^{*}(t) = -\frac{\tilde b^{[i]}}{2\tilde a^{[i]}},
$$

evaluated at $\tau = T - t$. This is the Merton–Lipton structure: a regime-dependent effective Merton
ratio times a goal-seeking factor that grows as wealth falls below the target (Lipton, 2001).

### The target is common to both regimes

**Proposition (regime-independent target).** For every $\tau$ and both regimes,

$$
b^{[i]}(\tau) = -\ell e^{-r_c\tau}a^{[i]}(\tau), \qquad \tilde b^{[i]}(\tau) = -\ell e^{-r_c\tau}\tilde a^{[i]}(\tau), \qquad
\Pi^{*}(t) = \frac{\ell}{2}e^{-r_c(T - t)} .
$$

**Proof.** Try $b^{[i]} = \kappa(\tau)a^{[i]}$ with one scalar $\kappa$ for both regimes. Then
$\tilde b^{[i]} = \kappa\tilde a^{[i]}$, and the right-hand side of the $b$-equation is
$\kappa[(r_c - \lambda^{[i \to j]})a^{[i]} + \lambda^{[i \to j]}a^{[j]} - (\tilde a^{[i]})^2/\Sigma^{[i]}] = \kappa(da^{[i]}/d\tau - r_c a^{[i]})$
by the $a$-equation. The left-hand side is $\kappa' a^{[i]} + \kappa\thinspace da^{[i]}/d\tau$, so the
equation holds if and only if $\kappa' = -r_c\kappa$. With $\kappa(0) = -\ell$ this gives
$\kappa = -\ell e^{-r_c\tau}$, and the $b$-system is linear with a unique solution. The target is
$-\tilde b^{[i]}/(2\tilde a^{[i]}) = -\kappa/2$. $\square$

> **Insight.** The target is the bliss point $\ell/2$ discounted at the net floor rate. A portfolio
> exactly at the target holds no risky asset and reaches $\ell/2$ in cash at the horizon. The regime
> changes how hard the policy pursues the target, through $\omega_a^{[i]}$, but not the target itself.
> The manuscript's reduction to a flat barrier bounds the difference between the regime targets by a
> term of order $\lambda^{[12]}$ (Appendix C, Step 3); the proposition shows that it is zero.

**Proposition (constant coefficient).** For every $\tau$ and both regimes,
$\gamma^{[i]}(\tau) = (\ell^2/4)(e^{-2r_c\tau}a^{[i]}(\tau) - 1)$, so that

$$
V^{[i]}(T, \Pi_0) = a^{[i]}(T)\left(\Pi_0 - \Pi^{*}(0)\right)^2 - \left(\Pi_T^{*}\right)^2 .
$$

**Proof.** By the previous proposition $(\tilde b^{[i]})^2/(4\Sigma^{[i]}) = (\ell^2/4)e^{-2r_c\tau}(\tilde a^{[i]})^2/\Sigma^{[i]}$,
and the $a$-equation gives
$(\tilde a^{[i]})^2/\Sigma^{[i]} = (2r_c - \lambda^{[i \to j]})a^{[i]} + \lambda^{[i \to j]}a^{[j]} - da^{[i]}/d\tau$.
Substituting the candidate into the $\gamma$-equation, both sides equal
$(\ell^2/4)e^{-2r_c\tau}(da^{[i]}/d\tau - 2r_c a^{[i]} + \lambda^{[i \to j]}a^{[i]} - \lambda^{[i \to j]}a^{[j]})$,
and the candidate vanishes at $\tau = 0$. The value function follows by completing the square with
$\Pi^{*}(0) = (\ell/2)e^{-r_cT}$ and $\Pi_T^{*} = \ell/2$. $\square$

### Expected terminal wealth, variance and the efficient frontier

**Proposition (moments of optimal terminal wealth).** Let $D = \Pi^{*}(0) - \Pi_0$ and
$u^{[i]} = e^{-2r_cT}a^{[i]}(T)$. Under the optimal policy from regime $i$,

$$
\mathbb{E}[\Pi_T] = \Pi_T^{*} - e^{-r_cT}a^{[i]}(T)\thinspace D, \qquad
\mathrm{Var}[\Pi_T] = a^{[i]}(T)\thinspace D^2\left(1 - u^{[i]}\right),
$$

and the efficient frontier is

$$
\mathrm{Var}[\Pi_T] = \frac{\left(\mathbb{E}[\Pi_T] - \Pi_0 e^{r_cT}\right)^2}{e^{\Theta^{[i]}} - 1}, \qquad \Theta^{[i]} = 2r_cT - \ln a^{[i]}(T) .
$$

**Proof.** The value is the infimum over policies of $\mathbb{E}[\Pi_T^2] - \ell\thinspace \mathbb{E}[\Pi_T]$,
a family of functions affine in $\ell$, so by the envelope theorem its derivative in $\ell$ is
$-\mathbb{E}[\Pi_T]$ under the optimal policy. Differentiating
$V^{[i]}(T, \Pi_0) = a^{[i]}\Pi_0^2 - \ell e^{-r_cT}a^{[i]}\Pi_0 + (\ell^2/4)(u^{[i]} - 1)$, with
$a^{[i]} = a^{[i]}(T)$ independent of $\ell$, gives
$\mathbb{E}[\Pi_T] = e^{-r_cT}a^{[i]}\Pi_0 + (\ell/2)(1 - u^{[i]})$, which rearranges to the first
formula. Then $\mathbb{E}[\Pi_T^2] = V^{[i]} + \ell\thinspace \mathbb{E}[\Pi_T]$, and with $P = \Pi_T^{*} = \ell/2$,
$\mathrm{Var}[\Pi_T] = a^{[i]}D^2 - P^2 + 2P\thinspace \mathbb{E}[\Pi_T] - \mathbb{E}[\Pi_T]^2 = a^{[i]}D^2 - (e^{-r_cT}a^{[i]}D)^2$.
Finally $\mathbb{E}[\Pi_T] - \Pi_0 e^{r_cT} = De^{r_cT}(1 - u^{[i]})$, whose square divided by the
variance is $e^{2r_cT}(1 - u^{[i]})/a^{[i]} = e^{\Theta^{[i]}} - 1$. $\square$

Without regime switching or jumps, $a(T) = e^{(2r_c - \theta^2)T}$ (Proposition 3.9), so
$\Theta = \theta^2T$ and the frontier is the one of Zhou and Li (2000).

> **Pitfall.** `find_ell(asset, T, target_return)` sets $\ell$ so that $-b(T)/(2a(T)) = \Pi_0 e^{\rho T}$
> for $\rho$ = `target_return`, and `RiccatiSolution.expected_wealth` returns the same quantity.
> By the first proposition this is $\Pi^{*}(0)$, the present value of the target, and equation (3.13)
> of the manuscript identifies it with $\mathbb{E}[\Pi_T]$. The expected terminal wealth of the
> optimal policy is the formula above instead: in the worked example a `target_return` of 4% gives
> $\mathbb{E}[\Pi_T] = 131.80$, an implied return of 2.76%, which a Monte Carlo simulation confirms.
> For an expected-wealth target $\bar\Pi_T$, solve the first formula for the multiplier,
> $\ell = 2(\bar\Pi_T - e^{-r_cT}a^{[i]}(T)\Pi_0)/(1 - u^{[i]})$, which needs $a^{[i]}(T)$ from one Riccati
> solve.

Equation (3.15) of the manuscript writes the information ratio with $-2r_cT$; the frontier holds with
$+2r_cT$, as the proof and the single-regime limit $\Theta = \theta^2T$ show, and the worked example
confirms it numerically.

### The allocation coefficient

At the horizon $a^{[1]} = a^{[2]} = 1$, so $\tilde a^{[i]}(0) = \mu^{[i]} - r_h + \lambda^{[i \to j]}\alpha^{[i]} = \bar\mu^{[i]} - r_h$
and $\Sigma^{[i]}(0) = (\sigma^{[i]})^2 + \lambda^{[i \to j]}(\sigma_J^{[i]})^2$. The allocation
coefficient at the horizon is the jump-adjusted Merton ratio

$$
\omega_a^{[i]}(\tau = 0) = -\frac{\bar\mu^{[i]} - r_h}{(\sigma^{[i]})^2 + \lambda^{[i \to j]}(\sigma_J^{[i]})^2},
$$

which is Proposition 3.9 of the manuscript with regime switching. Earlier in the horizon the
coefficient blends both regimes through $a^{[j]}$ in $\tilde a^{[i]}$ and $\Sigma^{[i]}$.

> **Insight.** In stress the total return of every asset class of Table 1 is at or below the hurdle
> rate of 2%, so $\tilde a^{[2]} \le 0$ near the horizon: the regime-conditional policy is short the
> risky asset whenever wealth is below target in stress. The manuscript's numerical sections use the
> growth policy in both regimes because the regime is not observed (Section 7);
> `RiccatiSolution.omega_star` returns the regime-conditional policy, sign included, and the
> [gap process](wealth_floor_gap_process.md) uses the magnitude $\lvert\omega_a^{[i]}\rvert$ in each
> regime.

## Worked example

The equity class of Table 1 over ten years with $r = 2$%, no consumption and a `target_return` of 4%
gives $\ell = 364.42 = 2 \times 100 e^{(0.04 + 0.02) \times 10}$. The first block checks every
statement of the methodology on this solution:

- the identities $b = -\ell e^{-r_c\tau}a$ and $\gamma = (\ell^2/4)(e^{-2r_c\tau}a - 1)$ hold on all
  10,001 grid points to the ODE tolerance;
- the target is 149.18 at inception and 182.21 at the horizon in both regimes;
- the growth coefficient moves from $-0.7563$ at inception to $-0.7692$ at the horizon, the
  jump-adjusted Merton ratio $0.025/0.0325$; the stress coefficient at inception is $+0.2129$, so the
  policy at initial wealth 100 holds 37.2% in growth and $-10.5$% in stress;
- $a(T) = (1.2519, 1.2680)$ and $\Theta = (0.1753, 0.1625)$, so from growth the expected terminal
  wealth is 131.80 with a standard deviation of 22.07, an implied return of 2.76%, and the same
  values follow from the value function through the envelope theorem;
- for a single regime without jumps, $a(T) = e^{(2r_c - \theta^2)T}$ and the coefficient is the
  Merton ratio $-(\mu - r_h)/\sigma^2$.

```python
import numpy as np
import goal_based_allocation as gba
from goal_based_allocation.riccati_solver import solve_riccati

equity = gba.create_paper_assets()['equity']
horizon, r, c, rho = 10.0, 0.02, 0.0, 0.04
ell, ric = gba.find_ell(equity, horizon, target_return=rho, r=r, c=c)
r_c, tau = ric.r_c, ric.tau_grid
wealth0 = equity.pi0

# find_ell fixes the present value of the target: ell = 2 Pi0 exp((rho + r_c) T)
np.testing.assert_allclose(ell, 2.0 * wealth0 * np.exp((rho + r_c) * horizon), rtol=1e-12)
np.testing.assert_allclose(ell, 364.42, atol=5e-3)

# b = -ell exp(-r_c tau) a and gamma = ell^2/4 (exp(-2 r_c tau) a - 1) on the whole grid
np.testing.assert_allclose(ric.b, -ell * np.exp(-r_c * tau) * ric.a, rtol=1e-11)
np.testing.assert_allclose(ric.gamma, 0.25 * ell**2 * (np.exp(-2.0 * r_c * tau) * ric.a - 1.0),
                           rtol=1e-9, atol=1e-8)

# the target is the same in both regimes and grows at r_c to ell / 2
inception, horizon_end = ric.derived_at_tau(horizon), ric.derived_at_tau(0.0)
pi_star_0, pi_star_T = inception['Pi_star'][0], ell / 2.0
np.testing.assert_allclose(inception['Pi_star'], [pi_star_0, pi_star_0], rtol=1e-12)
np.testing.assert_allclose(horizon_end['Pi_star'], [pi_star_T, pi_star_T], rtol=1e-12)
np.testing.assert_allclose([pi_star_0, pi_star_T], [149.18, 182.21], atol=5e-3)
np.testing.assert_allclose(ric.expected_wealth(0), pi_star_0, rtol=1e-12)

# allocation coefficients: jump-adjusted Merton ratio at the horizon, regime signs at inception
sigma_j2 = 1.0 / (1.0 + 2.0 * equity.params.eta0) - 2.0 / (1.0 + equity.params.eta0) + 1.0
merton = (equity.mu_growth - r) / (equity.params.sigma0**2 + equity.params.lambda01 * sigma_j2)
np.testing.assert_allclose(-horizon_end['w_a'][0], merton, rtol=1e-10)
np.testing.assert_allclose([merton, inception['w_a'][0], inception['w_a'][1]],
                           [0.7692, -0.7563, 0.2129], atol=5e-5)
omega_0 = [ric.omega_star(0.0, wealth0, regime) for regime in (0, 1)]
np.testing.assert_allclose(omega_0, [0.3720, -0.1047], atol=5e-5)

# moments of the optimal terminal wealth from a(T) alone, and the efficient frontier
a_T = ric.a[:, -1]
gap_0 = pi_star_0 - wealth0
u = np.exp(-2.0 * r_c * horizon) * a_T
expected = pi_star_T - np.exp(-r_c * horizon) * a_T * gap_0
std = np.sqrt(a_T * gap_0**2 * (1.0 - u))
theta = 2.0 * r_c * horizon - np.log(a_T)
np.testing.assert_allclose(a_T, [1.2519, 1.2680], atol=5e-5)
np.testing.assert_allclose(theta, [0.1753, 0.1625], atol=5e-5)
np.testing.assert_allclose([expected[0], std[0]], [131.80, 22.07], atol=5e-3)
np.testing.assert_allclose(np.log(expected[0] / wealth0) / horizon, 0.0276, atol=5e-5)
np.testing.assert_allclose(std, np.abs(expected - wealth0 * np.exp(r_c * horizon)) / np.sqrt(np.exp(theta) - 1.0),
                           rtol=1e-12)

# the same moments from the value function V = a Pi0^2 + b Pi0 + gamma and the envelope theorem
value = ric.a[:, -1] * wealth0**2 + ric.b[:, -1] * wealth0 + ric.gamma[:, -1]
np.testing.assert_allclose(expected, np.exp(-r_c * horizon) * a_T * wealth0 - 2.0 * ric.gamma[:, -1] / ell,
                           rtol=1e-9)
np.testing.assert_allclose(std**2, value + ell * expected - expected**2, rtol=1e-8)

# single regime without jumps: a(T) = exp((2 r_c - theta^2) T) and the Merton ratio
single = gba.AssetSpecification('single', gba.RegimeSwitchParams(sigma0=0.15, sigma1=0.15,
                                                                lambda01=0.0, lambda10=0.0),
                                mu_growth=0.07, mu_stress=0.07)
lone = solve_riccati(single, horizon, ell=1.0, r=r, c=c)
sharpe = (0.07 - r) / 0.15
np.testing.assert_allclose(lone.a[0, -1], np.exp((2.0 * r_c - sharpe**2) * horizon), rtol=1e-9)
np.testing.assert_allclose(lone.derived_at_tau(horizon)['w_a'][0], -(0.07 - r) / 0.15**2, rtol=1e-9)
```

The second block simulates the regime-conditional policy without a floor or position limits: 40,000
paths with 250 Euler steps a year, regime switches and jumps drawn at each step, and the dollar
exposure $(\tilde a^{[i]}/\Sigma^{[i]})(\Pi^{*}(t) - \Pi_t)$ in the current regime. The simulated mean
is 131.93 with a standard error of 0.11, against 131.80 from the proposition, and the simulated
standard deviation is 22.07, against 22.07; the target of `find_ell`, 149.18, is more than 150
standard errors away. The block then sets the multiplier for an expected terminal wealth of
$100e^{0.4} = 149.18$: it is $\ell = 580.60$, the target at inception is 237.68, the policy starts at
104.1% in the risky asset, and `find_ell` reproduces the same multiplier with a `target_return` of
8.66%. The standard deviation of terminal wealth rises from 22.07 to 61.78.

```python
def simulate_unconstrained(ric, n_paths, steps_per_year, seed):
    """Euler simulation of the regime-conditional optimal policy without floor or position limits.

    The dollar exposure is (a_tilde / Sigma)(Pi* - Pi) in the current regime; a transition moves
    wealth by the exposure times (e^J - 1).
    """
    rng = np.random.default_rng(seed)
    p = ric.asset.params
    n_steps = int(ric.T * steps_per_year)
    dt = ric.T / n_steps
    excess = np.array(ric.dp['mu_bar']) - ric.r_h
    vol = np.array([p.sigma0, p.sigma1])
    rate = np.array([p.lambda01, p.lambda10])
    wealth = np.full(n_paths, ric.asset.pi0)
    regime = np.zeros(n_paths, dtype=int)
    for step in range(n_steps):
        coef = ric.derived_at_tau(ric.T - step * dt)
        exposure = (np.array(coef['a_tilde']) / np.array(coef['Sigma']))[regime] * (coef['Pi_star'][0] - wealth)
        wealth = (wealth + (ric.r_c * wealth + exposure * excess[regime]) * dt
                  + exposure * vol[regime] * np.sqrt(dt) * rng.standard_normal(n_paths))
        switch = rng.uniform(size=n_paths) < rate[regime] * dt
        jump = np.where(regime == 0, -rng.exponential(p.eta0, n_paths), rng.exponential(p.eta1, n_paths))
        wealth = wealth + np.where(switch, exposure * np.expm1(jump), 0.0)
        regime = np.where(switch, 1 - regime, regime)
    return wealth


terminal = simulate_unconstrained(ric, n_paths=40_000, steps_per_year=250, seed=1)
standard_error = terminal.std() / np.sqrt(terminal.size)
np.testing.assert_allclose([terminal.mean(), standard_error, terminal.std()], [131.93, 0.11, 22.07],
                           atol=5e-3)
assert abs(terminal.mean() - expected[0]) < 3.0 * standard_error
assert abs(terminal.mean() - pi_star_0) > 150.0 * standard_error
np.testing.assert_allclose(terminal.std(), std[0], rtol=0.01)

# the multiplier for an expected terminal wealth of 100 exp(0.4)
target_mean = wealth0 * np.exp(rho * horizon)
ell_mean = 2.0 * (target_mean - np.exp(-r_c * horizon) * a_T[0] * wealth0) / (1.0 - u[0])
mean_ric = solve_riccati(equity, horizon, ell_mean, r=r, c=c)
start = mean_ric.derived_at_tau(horizon)['Pi_star'][0]
np.testing.assert_allclose(mean_ric.a[:, -1], a_T, rtol=1e-12)
np.testing.assert_allclose(ell_mean / 2.0 - np.exp(-r_c * horizon) * a_T[0] * (start - wealth0),
                           target_mean, rtol=1e-12)
np.testing.assert_allclose([ell_mean, start, mean_ric.omega_star(0.0, wealth0, 0)],
                           [580.60, 237.68, 1.0413], atol=5e-3)
rho_equivalent = np.log(start / wealth0) / horizon
np.testing.assert_allclose(gba.find_ell(equity, horizon, rho_equivalent, r=r, c=c)[0], ell_mean, rtol=1e-10)
np.testing.assert_allclose(rho_equivalent, 0.0866, atol=5e-5)
np.testing.assert_allclose(np.sqrt(a_T[0] * (start - wealth0)**2 * (1.0 - u[0])), 61.78, atol=5e-3)
```

## Implementation in goal_based_allocation

| Quantity | Formula | Entry point |
|---|---|---|
| Multiplier and Riccati solution | $\ell$ with $-b(T)/(2a(T)) = \Pi_0e^{\rho T}$ | `find_ell(asset, T, target_return, r=0.02, c=0.02, regime=0)` |
| Riccati solution for a given multiplier | $a$, $b$, $\gamma$ on a grid of $\tau$ | `riccati_solver.solve_riccati(asset, T, ell, r=0.02, c=0.02, n_steps=10000)` |
| Coefficients at $\tau$ | linear interpolation of $a$ and $b$ | `RiccatiSolution.at_tau(tau)` |
| Coupled coefficients, allocation coefficient, target | $\tilde a$, $\tilde b$, $\Sigma$, $\omega_a$, $\Pi^{*}$ | `RiccatiSolution.derived_at_tau(tau)` |
| Regime-conditional policy | $-\omega_a^{[i]}(\Pi^{*}/\Pi - 1)$ | `RiccatiSolution.omega_star(t, Pi, regime)` |
| Present value of the target | $-b^{[i]}(T)/(2a^{[i]}(T)) = \Pi^{*}(0)$ | `RiccatiSolution.expected_wealth(regime=0)` |
| Hurdle and net floor rate | $\max(r, c)$, $\max(0, r - c)$ | `RiccatiSolution.r_h`, `RiccatiSolution.r_c` |

The solver lives in
[`riccati_solver.py`](https://github.com/ArturSepp/GoalBasedAllocation/blob/main/src/goal_based_allocation/riccati_solver.py).
API pages: {doc}`find_ell <api/generated/goal_based_allocation.find_ell>`,
{doc}`solve_riccati <api/generated/goal_based_allocation.riccati_solver.solve_riccati>` and
{doc}`RiccatiSolution <api/generated/goal_based_allocation.riccati_solver.RiccatiSolution>`.

Contract details:

- `find_ell` solves the system for $\ell = 1$ and $\ell = 10$, interpolates linearly to the
  multiplier whose $-b(T)/(2a(T))$ equals $\Pi_0e^{\rho T}$, and solves once more. The interpolation is
  exact because $-b(T)/(2a(T)) = (\ell/2)e^{-r_cT}$ is linear in $\ell$; its `regime` argument has no
  effect for the same reason. It returns `(ell, RiccatiSolution)`.
- `RiccatiSolution` stores `tau_grid`, the arrays `a`, `b` and `gamma` of shape `(2, n_steps + 1)`
  with rows for growth and stress, the asset, the derived parameters `dp` (compensators, diffusion
  drifts `mu_bar`, `sigJ_sq`), `ell`, `r` and `c`. Its `T` is the last grid point.
- `derived_at_tau` returns a dictionary of pairs, growth first, with keys `'a_tilde'`, `'b_tilde'`,
  `'Sigma'`, `'w_a'` and `'Pi_star'`. Where $\lvert\Sigma\rvert \lt 10^{-20}$ it sets `w_a` to zero, and where
  $\lvert\tilde a\rvert \lt 10^{-15}$ it sets `Pi_star` to zero. The second guard fires where the total
  return equals the hurdle rate at the horizon, for example bonds in stress with $c = 0$.
- `omega_star(t, Pi, regime)` takes calendar time, evaluates the coefficients at $\tau = T - t$, and
  returns zero for $\lvert\Pi\rvert \lt 10^{-12}$. It does not clip the allocation.
- The ODE solver raises `RuntimeError` if `solve_ivp` reports a failure.

## Interpretation and limitations

- The policy is pre-commitment and time-inconsistent: after a period of good returns the optimal
  policy of a fresh problem differs from the continuing one. The manuscript motivates the commitment
  by an investment mandate fixed at inception (Remark 3.5).
- The policy is unconstrained. It levers below target, shorts above target, and with jumps the
  unconstrained wealth can become negative after a crash. The floor of the
  [next chapter](wealth_floor_gap_process.md) is imposed by stopping, not inside the optimisation,
  which makes the stopped strategy sub-optimal for the floor-constrained problem (Remark 4.6).
- The expected wealth and variance of this chapter are those of the unconstrained policy. With the
  floor, terminal wealth has a different distribution, computed in the
  [terminal wealth chapter](terminal_wealth_distribution.md).
- For $c \gt r$ the wealth dynamics use $r_h = c$ and $r_c = 0$ by assumption; the
  [opportunity set](investment_opportunity_set.md) inherits that choice.
- The regime is not observed. Applying the growth coefficient in both regimes, as the manuscript
  does, is a different policy from the regime-conditional one that `omega_star` returns.

## See also

- [The regime-switching jump-diffusion](regime_switching_model.md)
- [The wealth floor and the flat-barrier reduction](wealth_floor_gap_process.md)
- [The terminal wealth distribution](terminal_wealth_distribution.md)
- [The investment opportunity set and investor selection](investment_opportunity_set.md)
- [Notation and conventions](conventions.md)
- [Bibliography](bibliography.md)

## References

1. Sepp, A. (2026). Dynamic Mean-Variance Portfolio Allocation under Regime-Switching Jump-Diffusions with Absorbing Barriers and Distribution Matching. Working paper. [DOI: 10.2139/ssrn.6534579](https://doi.org/10.2139/ssrn.6534579). Section 3 and Appendix A derive the quadratic value function, the Riccati system and the policy.
2. Zhou, X. Y., and Li, D. (2000). Continuous-Time Mean-Variance Portfolio Selection: A Stochastic LQ Framework. *Applied Mathematics and Optimization*, 42(1), 19–33. [DOI: 10.1007/s002450010003](https://doi.org/10.1007/s002450010003). The embedding of the mean-variance problem in a linear-quadratic control problem and the single-regime frontier.
3. Zhou, X. Y., and Yin, G. (2003). Markowitz's Mean-Variance Portfolio Selection with Regime Switching: A Continuous-Time Model. *SIAM Journal on Control and Optimization*, 42(4), 1466–1482. [DOI: 10.1137/S0363012902405583](https://doi.org/10.1137/S0363012902405583). Regime switching without transition jumps, where the Riccati equations are linear.
4. Lipton, A. (2001). *Mathematical Methods for Foreign Exchange: A Financial Engineer's Approach*. World Scientific. [DOI: 10.1142/4694](https://doi.org/10.1142/4694). The goal-seeking form of the mean-variance policy.
5. Merton, R. C. (1971). Optimum Consumption and Portfolio Rules in a Continuous-Time Model. *Journal of Economic Theory*, 3(4), 373–413. [DOI: 10.1016/0022-0531(71)90038-X](https://doi.org/10.1016/0022-0531(71)90038-X). The Merton ratio.
6. Basak, S., and Chabakauri, G. (2010). Dynamic Mean-Variance Asset Allocation. *The Review of Financial Studies*, 23(8), 2970–3016. [DOI: 10.1093/rfs/hhq028](https://doi.org/10.1093/rfs/hhq028). Time inconsistency of the pre-commitment policy.
7. Shi, X., and Xu, Z. Q. (2025). Optimal Mean-Variance Portfolio Selection under Regime-Switching-Induced Stock Price Shocks. *Systems & Control Letters*, 204, 106200. [DOI: 10.1016/j.sysconle.2025.106200](https://doi.org/10.1016/j.sysconle.2025.106200). Coupled nonlinear Riccati equations from regime-induced price shocks.
8. Sepp, A. goal-based-allocation: Semi-analytical dynamic mean-variance allocation and terminal-wealth risk under regime-switching jump-diffusions. [Software citation metadata](https://github.com/ArturSepp/GoalBasedAllocation/blob/main/CITATION.cff).
