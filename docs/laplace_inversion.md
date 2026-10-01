---
myst:
  html_meta:
    description: >-
      Numerical Laplace inversion in goal-based-allocation: the Bromwich integral, the
      Abate-Whitt trapezoidal rule with Euler summation and its aliasing error, the Gaver-Stehfest
      method and its failure on oscillating functions, checked on transforms with known inverses
      including Brownian first-passage survival.
---

# Numerical Laplace inversion

*Author: [Artur Sepp](https://github.com/ArturSepp)*

Implemented in [GoalBasedAllocation](https://github.com/ArturSepp/GoalBasedAllocation).
Software citation: [CITATION.cff](https://github.com/ArturSepp/GoalBasedAllocation/blob/main/CITATION.cff).

Numerical Laplace inversion recovers a function of time $f(t)$ at one point from its transform
$\hat f(p) = \int_0^{\infty}e^{-pt}f(t)dt$ evaluated at a finite set of complex arguments. Every
survival probability, density, tilted survival and option price of the package is a closed form in
the Laplace domain followed by one such inversion in time. The package uses the Euler-summation
algorithm of Abate and Whitt (1995) and also provides the Gaver–Stehfest method (Stehfest, 1970);
this chapter states both, derives the accuracy of the first, and shows where the second fails.

## Overview

The [Laplace framework](laplace_barrier_framework.md) produces transforms that are explicit in the
Laplace variable $p$ but whose time-domain inverses are not. Inverting them numerically, rather than
simulating the process, is what makes the analytics deterministic: the same inputs give the same
numbers to the last digit, and the error is controlled by the algorithm's parameters, not by a
sample size.

`laplace_invert_abate_whitt(g, t)` evaluates a vectorised transform `g` at 38 complex points on a
vertical line in the right half-plane and returns $f(t)$ for every column `g` returns, so a whole
density grid is inverted with one set of transform evaluations. `laplace_invert_stehfest(g, t)` uses
real arguments only. Neither is re-exported from the package root; they are the engine below the
model-level functions.

## Inputs, notation, and assumptions

| Convention | This article |
|---|---|
| Time and rates | The inversion time $t \gt 0$ is the horizon of the quantity being computed, in years |
| Regimes | Not applicable: the inversion acts on any transform, regime-conditional or not |
| Jump sizes | Not applicable |
| Wealth coordinate | Not applicable; one inversion returns one value per column of the transform |
| Measure | Not applicable |
| Numerical method | Abate–Whitt with $A = -\ln(10^{-8})$, step $\pi/t$, $N = 25$ terms and $M = 12$ Euler terms; Gaver–Stehfest with $N = 14$ precomputed weights; checked against known inverses |
| Package default | `laplace_invert_abate_whitt(g, t, N=25, M=12, tol=1e-8)`; `laplace_invert_stehfest(g, t, N=14)` with `N` in 10, 14 or 20 |

| Symbol | Meaning | Units and convention |
|---|---|---|
| $f(t)$, $\hat f(p)$ | Function of time and its Laplace transform | $\hat f$ defined for $\mathrm{Re}(p) \gt 0$ |
| $A$ | Contour parameter $-\ln(\mathrm{tol})$ | $A = 18.42$ for `tol=1e-8` |
| $s_n$ | Partial sum of the alternating series | Units of $f$ |
| $N$, $M$ | Number of plain and of Euler terms | Integers |
| $V_k$ | Gaver–Stehfest weights | Alternating in sign |
| $\Phi$ | Standard normal distribution function | Used in the worked example |

The transforms of the package are analytic for $\mathrm{Re}(p) \gt 0$ and their inverses are bounded
by one for probabilities. Discounting enters option transforms as a shift of $p$ by the riskless rate.

## Methodology

### The Bromwich integral and its trapezoidal rule

**Definition.** For $a \gt 0$ to the right of all singularities of $\hat f$, the inverse transform is
the Bromwich integral

$$
f(t) = \frac{1}{2\pi i}\int_{a - i\infty}^{a + i\infty}e^{pt}\hat f(p)\thinspace dp
= \frac{2e^{at}}{\pi}\int_0^{\infty}\mathrm{Re}\left(\hat f(a + iu)\right)\cos(ut)\thinspace du,
$$

the second form holding for real $f$. Abate and Whitt set $a = A/(2t)$ and apply the trapezoidal rule
with step $\pi/t$, which makes $\cos(ut)$ alternate between $+1$ and $-1$:

$$
f(t) \approx \frac{e^{A/2}}{t}\left[\frac{1}{2}\mathrm{Re}\thinspace \hat f\left(\frac{A}{2t}\right) + \sum_{k=1}^{\infty}(-1)^k\thinspace \mathrm{Re}\thinspace \hat f\left(\frac{A + 2k\pi i}{2t}\right)\right] .
$$

**Proposition (aliasing error; Abate and Whitt, 1995).** The infinite trapezoidal sum equals

$$
f(t) + \sum_{j=1}^{\infty}e^{-jA}f\left((2j + 1)t\right),
$$

so if $\lvert f\rvert \le C$ the discretisation error is at most $Ce^{-A}/(1 - e^{-A})$.

**Proof.** The trapezoidal sum is the Fourier series, with period $2t$, of the damped function
$e^{-as}f(s)$ extended by zero to $s \lt 0$. By the Poisson summation formula it equals the sum of the
translates $e^{-a(t + 2jt)}f(t + 2jt)$ over $j \ge 0$, multiplied by $e^{at}$, which is the stated sum
because $e^{-2ajt} = e^{-jA}$. Bounding each term by $Ce^{-jA}$ gives a geometric series. $\square$

The error is not a truncation error but aliasing: the method returns $f(t)$ plus damped copies of $f$
at $3t$, $5t$ and later. For a probability $C = 1$ and the default `tol=1e-8` the bound is $10^{-8}$.
For a growing function the copies grow too: the worked example inverts $1/p^2$, whose inverse is $t$,
and finds an error of $3t \times 10^{-8}$, the first copy exactly.

### Euler summation

The alternating series converges slowly. With partial sums $s_n$ of its first $n + 1$ terms, Euler
summation replaces the limit by a binomial average of $M + 1$ consecutive partial sums,

$$
f(t) \approx \frac{e^{A/2}}{t}\sum_{k=0}^{M}\binom{M}{k}2^{-M}s_{N + k},
$$

which needs $1 + N + M$ evaluations of $\hat f$, 38 with the defaults. The average cancels the
oscillation of the partial sums when their terms eventually alternate regularly, which holds for the
smooth transforms of the package. Unlike the aliasing error, the error of Euler summation has no
closed-form bound; the worked example measures it on functions with known inverses.

### The Gaver–Stehfest method

**Definition (Stehfest, 1970).** For even $N$,

$$
f(t) \approx \frac{\ln 2}{t}\sum_{k=1}^{N}V_k\thinspace \hat f\left(\frac{k\ln 2}{t}\right),
$$

with weights $V_k$ that alternate in sign and, for $N = 14$, reach $1.7 \times 10^{8}$ in magnitude.
The method uses real arguments only and is exact in the limit for smooth $f$, but the large weights
cancel to the result, so about eight of the sixteen significant digits of double precision are lost
for $N = 14$; a larger $N$ loses more to round-off than it gains in approximation, as the example
shows for $N = 20$.

> **Pitfall.** Gaver–Stehfest samples $\hat f$ only on the real axis and cannot represent oscillation.
> Inverting $1/(p^2 + 1)$, whose inverse is $\sin t$, it is wrong by 0.22 at $t = 5$ and by 0.55 at
> $t = 10$. The package therefore uses Abate–Whitt for every model quantity; Stehfest is kept for
> comparison.

## Worked example

The first block inverts four transforms with known inverses at several times: the exponential
$1/(p + 1)$, the ramp $1/p^2$, the sine $1/(p^2 + 1)$, and the survival probability of a Brownian
motion with drift $\nu = 3$% and volatility $\sigma = 20$% started at $x_0 = 0.5$ above an absorbing
barrier. The last transform is

$$
\hat Q(p) = \frac{1}{p}\left(1 - \exp\left(-x_0\frac{\nu + \sqrt{\nu^2 + 2\sigma^2p}}{\sigma^2}\right)\right),
$$

one minus the Laplace transform of the first-passage time, divided by $p$, with the time-domain
inverse $\Phi((x_0 + \nu t)/(\sigma\sqrt t)) - e^{-2\nu x_0/\sigma^2}\Phi((-x_0 + \nu t)/(\sigma\sqrt t))$.

| Inverse | Abate–Whitt error | Gaver–Stehfest error |
|---|---|---|
| $e^{-t}$ at $t = 1$ | $4.9 \times 10^{-10}$ | $-9.4 \times 10^{-7}$ |
| $t$ at $t = 1$ | $3.0 \times 10^{-8}$ | $-3.6 \times 10^{-7}$ |
| $\sin t$ at $t = 5$ | $6.5 \times 10^{-9}$ | $0.22$ |
| Brownian survival 0.82448 at $t = 5$ | $6.7 \times 10^{-9}$ | $-2.1 \times 10^{-6}$ |

Abate–Whitt is within $10^{-8}$ on every bounded inverse, and its error on the ramp is $3t$ times
$e^{-A} = 10^{-8}$ as the aliasing proposition predicts. The block also inverts two transforms at
once from one set of evaluations, and checks that Stehfest with 20 terms is less accurate than with
14.

```python
import numpy as np
from scipy.stats import norm
from goal_based_allocation.laplace_inversion import laplace_invert_abate_whitt, laplace_invert_stehfest

known = {
    'exponential': (lambda p: 1.0 / (p + 1.0), lambda t: np.exp(-t)),
    'ramp': (lambda p: 1.0 / p**2, lambda t: t),
    'sine': (lambda p: 1.0 / (p**2 + 1.0), np.sin),
}
times = np.array([0.5, 1.0, 5.0, 10.0])
errors = {}
for name, (transform, inverse) in known.items():
    errors[name] = np.array([[method(transform, t)[0] - inverse(t) for t in times]
                             for method in (laplace_invert_abate_whitt, laplace_invert_stehfest)])

# Abate-Whitt is within 1e-8 on bounded inverses; on the ramp its error is the first alias 3 t e^-A
assert np.abs(errors['exponential'][0]).max() < 1e-8 and np.abs(errors['sine'][0]).max() < 1e-8
np.testing.assert_allclose(errors['ramp'][0], 3.0 * times * 1e-8, rtol=1e-4)
# Gaver-Stehfest: adequate for smooth monotone inverses, wrong for the oscillating sine
assert np.abs(errors['exponential'][1]).max() < 1e-4
np.testing.assert_allclose(errors['sine'][1, 2:], [0.2247, 0.5505], atol=5e-5)

# survival of a Brownian motion with drift above an absorbing barrier
x0, drift, vol = 0.5, 0.03, 0.2


def survival_transform(p):
    """Laplace transform in time of P(no barrier hit by t), from the first-passage transform."""
    root = (drift + np.sqrt(drift**2 + 2.0 * vol**2 * p)) / vol**2
    return (1.0 - np.exp(-x0 * root)) / p


def survival(t):
    """Reflection-principle survival probability of x0 + drift t + vol W_t above zero."""
    s = vol * np.sqrt(t)
    return norm.cdf((x0 + drift * t) / s) - np.exp(-2.0 * drift * x0 / vol**2) * norm.cdf((-x0 + drift * t) / s)


for t in (1.0, 5.0, 10.0):
    assert abs(laplace_invert_abate_whitt(survival_transform, t)[0] - survival(t)) < 1e-8
    assert abs(laplace_invert_stehfest(survival_transform, t)[0] - survival(t)) < 1e-5
np.testing.assert_allclose(survival(5.0), 0.82448, atol=5e-6)

# one set of 38 transform evaluations inverts every column at once
both = laplace_invert_abate_whitt(lambda p: np.column_stack([1.0 / (p + 1.0), 1.0 / (p + 2.0)]), 1.0)
np.testing.assert_allclose(both, np.exp([-1.0, -2.0]), atol=1e-8)

# more Stehfest terms lose more to round-off than they gain
stehfest_error = [abs(laplace_invert_stehfest(survival_transform, 5.0, N=n)[0] - survival(5.0))
                  for n in (10, 14, 20)]
assert stehfest_error[1] < stehfest_error[0] and stehfest_error[1] < stehfest_error[2]
```

## Implementation in goal_based_allocation

| Quantity | Formula | Entry point |
|---|---|---|
| Euler-summation inversion | Trapezoidal Bromwich rule on $\mathrm{Re}(p) = A/(2t)$ with Euler averaging | `laplace_inversion.laplace_invert_abate_whitt(g, t, N=25, M=12, tol=1e-8)` |
| Gaver–Stehfest inversion | Weighted sum of $\hat f(k\ln 2/t)$ | `laplace_inversion.laplace_invert_stehfest(g, t, N=14)` |

The algorithms live in
[`laplace_inversion.py`](https://github.com/ArturSepp/GoalBasedAllocation/blob/main/src/goal_based_allocation/laplace_inversion.py).
API pages: {doc}`laplace_invert_abate_whitt <api/generated/goal_based_allocation.laplace_inversion.laplace_invert_abate_whitt>` and
{doc}`laplace_invert_stehfest <api/generated/goal_based_allocation.laplace_inversion.laplace_invert_stehfest>`.

Contract details:

- `g` receives a one-dimensional array of complex arguments, 38 for the defaults, and returns an
  array of shape `(len(p), n)` or `(len(p),)`. The result is a real array of shape `(n,)`; the
  imaginary parts of the transform are discarded.
- `tol` sets the contour, $A = -\ln(\mathrm{tol})$; it bounds the aliasing error of a bounded inverse,
  not the total error. `N` and `M` set the number of plain and Euler terms.
- `laplace_invert_stehfest` accepts only `N` in 10, 14 and 20, whose weights are stored, and raises
  `ValueError` otherwise; it evaluates `g` on real arguments and keeps the real part.
- Neither function validates `t`; $t = 0$ divides by zero.
- The model functions call `laplace_invert_abate_whitt` with its defaults; `price_vanilla` passes
  its `n_terms` and `n_euler` arguments as `N` and `M`.

## Interpretation and limitations

- The default accuracy, about $10^{-8}$ for probabilities, is far below the error of the model
  approximations elsewhere in the package, such as the [effective jump sizes](wealth_floor_gap_process.md).
  Tightening `tol` below about $10^{-10}$ gains little in double precision, because $e^{A/2}$
  multiplies the round-off of the partial sums.
- For an inverse that grows with $t$, such as an undiscounted moment, the aliasing error grows with
  it; express such quantities relative to a scale or check them against a known value.
- Euler summation assumes that the series terms eventually alternate regularly. A transform with
  a discontinuous inverse near $t$, such as a payoff paid at a fixed date, converges slowly; the
  package inverts in time only quantities that are smooth in $t$.
- The transform must be evaluated accurately at complex arguments with large imaginary parts,
  up to $37\pi/t$. The characteristic roots of the regime-switching model are computed with
  `numpy.roots` at each argument; the [barrier chapter](laplace_barrier_framework.md) checks the
  resulting survival against exact simulation.

## See also

- [Survival, densities and overshoot in the Laplace domain](laplace_barrier_framework.md)
- [European options under regime switching](european_options.md)
- [Validation and numerical evidence](validation.md)
- [Bibliography](bibliography.md)

## References

1. Abate, J., and Whitt, W. (1995). Numerical Inversion of Laplace Transforms of Probability Distributions. *ORSA Journal on Computing*, 7(1), 36–43. [DOI: 10.1287/ijoc.7.1.36](https://doi.org/10.1287/ijoc.7.1.36). The trapezoidal rule with Euler summation and its aliasing error.
2. Stehfest, H. (1970). Algorithm 368: Numerical Inversion of Laplace Transforms [D5]. *Communications of the ACM*, 13(1), 47–49. [DOI: 10.1145/361953.361969](https://doi.org/10.1145/361953.361969). The Gaver–Stehfest weights.
3. Sepp, A. (2026). Dynamic Mean-Variance Portfolio Allocation under Regime-Switching Jump-Diffusions with Absorbing Barriers and Distribution Matching. Working paper. [DOI: 10.2139/ssrn.6534579](https://doi.org/10.2139/ssrn.6534579). Section 5 inverts the barrier transforms with the Abate–Whitt algorithm.
4. Sepp, A. goal-based-allocation: Semi-analytical dynamic mean-variance allocation and terminal-wealth risk under regime-switching jump-diffusions. [Software citation metadata](https://github.com/ArturSepp/GoalBasedAllocation/blob/main/CITATION.cff).
