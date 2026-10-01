---
myst:
  html_meta:
    description: >-
      Documentation for goal-based-allocation: the handbook of dynamic mean-variance allocation
      under regime-switching jump-diffusions with an absorbing wealth floor, Laplace-transform
      survival and terminal wealth distributions, mandate opportunity sets, option and
      variance-swap pricing, runnable examples and API reference.
---

# goal-based-allocation

*Author: [Artur Sepp](https://github.com/ArturSepp)*

<a id="goalbasedallocation"></a>

[GoalBasedAllocation](https://github.com/ArturSepp/GoalBasedAllocation) is a Python library for
dynamic mean-variance allocation and terminal-wealth risk under a two-regime jump-diffusion with an
absorbing wealth floor. It solves the policy from a Riccati system, computes survival, the floor atom,
the jump overshoot and the terminal wealth distribution from Laplace transforms, aggregates
multi-asset mandates to one effective asset, and prices European options and variance swaps under the
same model. Monte Carlo validates the analytics; it does not implement them. The package is the
companion code to Sepp (2026), *Dynamic Mean-Variance Portfolio Allocation under Regime-Switching
Jump-Diffusions with Absorbing Barriers and Distribution Matching*
([SSRN 6534579](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=6534579)).

Software citation: [CITATION.cff](https://github.com/ArturSepp/GoalBasedAllocation/blob/main/CITATION.cff).

## Start here

1. [Install and run the quickstart](getting-started.md). The installation command is
   `python -m pip install goal-based-allocation`, and the quickstart evaluates one balanced mandate
   offline in seconds.
2. Read [Notation and conventions](conventions.md) before comparing a number with the manuscript:
   regimes are numbered from one in the paper and from zero in the code, jump sizes are exponential
   means, and several functions fix the horizon, initial wealth and riskless rate.
3. Check [Model boundaries](model-boundaries.md) to confirm that the question fits the published model.

## The goal-based allocation handbook

The methodology chapters form one book. Each chapter defines its method with formulas and concise
proofs, states its conventions in a seven-row card, works an example whose every quoted number the
test suite checks against an independent computation, and links to the functions that implement it.
Where the implementation and the manuscript differ, the chapter says so and shows the evidence.
Symbols keep one meaning throughout; see the [conventions](conventions.md) and the
[bibliography](bibliography.md).

### Part I: The model

- [The regime-switching jump-diffusion](regime_switching_model.md): the regime chain, exponential
  jumps at transitions, compensators, total return against diffusion drift, and the paper's asset
  classes and floors.
- [Buy-and-hold moments](buy_and_hold_moments.md): exact moments of terminal wealth by a $2 \times 2$
  matrix exponential, consumption scaling and the stationary benchmark.

### Part II: Dynamic mean-variance allocation

- [The MV-optimal policy and the Riccati system](mv_optimal_policy.md): the quadratic value function,
  the Merton–Lipton policy, a regime-independent target, and closed-form expected terminal wealth,
  variance and efficient frontier.
- [The wealth floor and the flat-barrier reduction](wealth_floor_gap_process.md): stopping at the
  floor, the log-gap process with a fixed barrier, mean-matched gap jumps and the three components of
  stopped wealth.

### Part III: The Laplace transform framework

- [Numerical Laplace inversion](laplace_inversion.md): the Abate–Whitt algorithm, its aliasing error,
  and the Gaver–Stehfest method.
- [Survival, densities and overshoot in the Laplace domain](laplace_barrier_framework.md):
  Arrow–Debreu prices, the sixth-order characteristic polynomial, survival, tilted survival and the
  exponential overshoot.

### Part IV: Terminal wealth and investor profiles

- [The terminal wealth distribution](terminal_wealth_distribution.md): survived density, floor atom
  and overshoot in wealth, closed-form moments, quantiles and the gap to target.
- [Mandates as one effective asset](mandate_aggregation.md): exact volatility and total return,
  mean-matched portfolio jumps, and the floor from a drawdown tolerance.
- [The investment opportunity set and investor selection](investment_opportunity_set.md): the
  two-step advisor framework, closed-form calibration to full investment, the floor protection cost
  and the expected glide path.

### Part V: Derivatives under the same model

- [European options under regime switching](european_options.md): the risk-neutral drifts, the
  closed-form payoff transform, one inversion for all strikes, and implied-volatility smiles from
  either regime.
- [Variance swaps and the crash-size premium](variance_swaps.md): occupation times, the strike
  decomposition, the log contract, the variance risk premium and the crash size implied by one quote.

## Reference

- [API reference](api/index.md): every public export, grouped by capability and linked to its
  chapters.
- [Bibliography](bibliography.md): every work cited by the handbook, in one style.
- [Validation and numerical evidence](validation.md): the test suites and the paper validator.
- [Model boundaries](model-boundaries.md): appropriate uses and intentional non-goals.
- [Papers and research projects](papers.md): the manuscript, its replication, and the KOSPI study.
- [Choosing the appropriate portfolio workflow](comparison.md): a dated comparison with related
  libraries.
- [Documentation standard](documentation_standard.md): chapter structure, notation, executed examples,
  bibliography and figures.

## Project resources

- [PyPI](https://pypi.org/project/goal-based-allocation/) and the
  [source repository](https://github.com/ArturSepp/GoalBasedAllocation).
- [Issue tracker](https://github.com/ArturSepp/GoalBasedAllocation/issues) and
  [contributor guide](https://github.com/ArturSepp/GoalBasedAllocation/blob/main/CONTRIBUTING.md).
- [Changelog](https://github.com/ArturSepp/GoalBasedAllocation/blob/main/CHANGELOG.md).
- The paper: [SSRN 6534579](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=6534579).

This software is research code distributed without warranty and does not provide investment advice.

```{toctree}
:hidden:
:maxdepth: 2
:caption: Start here

getting-started
conventions
model-boundaries
```

```{toctree}
:hidden:
:maxdepth: 2
:caption: Part I - The model

regime_switching_model
buy_and_hold_moments
```

```{toctree}
:hidden:
:maxdepth: 2
:caption: Part II - Dynamic mean-variance allocation

mv_optimal_policy
wealth_floor_gap_process
```

```{toctree}
:hidden:
:maxdepth: 2
:caption: Part III - The Laplace transform framework

laplace_inversion
laplace_barrier_framework
```

```{toctree}
:hidden:
:maxdepth: 2
:caption: Part IV - Terminal wealth and investor profiles

terminal_wealth_distribution
mandate_aggregation
investment_opportunity_set
```

```{toctree}
:hidden:
:maxdepth: 2
:caption: Part V - Derivatives under the same model

european_options
variance_swaps
```

```{toctree}
:hidden:
:maxdepth: 2
:caption: Reference

api/index
bibliography
validation
papers
comparison
documentation_standard
```
