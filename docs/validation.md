---
myst:
  html_meta:
    description: >-
      How goal-based-allocation is validated: the fast and full test suites, the executed worked
      examples of the handbook with their independent references, and the paper's integration
      validator.
---

# Validation and numerical evidence

*Author: [Artur Sepp](https://github.com/ArturSepp)*

Project: [GoalBasedAllocation](https://github.com/ArturSepp/GoalBasedAllocation).
Software citation: [CITATION.cff](https://github.com/ArturSepp/GoalBasedAllocation/blob/main/CITATION.cff).

Analytical and semi-analytical calculations are the implementation. Independent Monte Carlo and
alternative transforms are validators.

## Fast source suite

```bash
pytest -m "not slow" -q
```

This covers probability bounds/monotonicity, Riccati initial conditions, exact buy-and-hold
moments, option-pricer properties, Fourier agreement, metadata, paths, quickstart behavior, the
documentation structure and bibliography, and the handbook examples that run in seconds.

## Full suite

```bash
pytest -q
```

The `slow` tests add seeded Monte Carlo option-price cross-checks and the handbook chapters whose
examples calibrate mandates or simulate wealth paths. The option test accepts a four-standard-error
envelope rather than forcing a deterministic price match.

## Handbook examples

`tests/test_documentation_examples.py` runs the worked examples of every handbook chapter. Each quoted
number is asserted against a reference computed a different way:

| Chapter | Independent reference |
|---|---|
| [The regime-switching jump-diffusion](regime_switching_model.md) | Exact simulation with exponential holding times |
| [Buy-and-hold moments](buy_and_hold_moments.md) | Exact simulation and a separate matrix exponential |
| [The MV-optimal policy](mv_optimal_policy.md) | Single-regime closed form and Euler simulation of the unconstrained policy |
| [The wealth floor](wealth_floor_gap_process.md) | Exact barrier simulation with exponential and with exact gap jumps |
| [Numerical Laplace inversion](laplace_inversion.md) | Transforms with known inverses, including Brownian first passage |
| [Survival, densities and overshoot](laplace_barrier_framework.md) | Reflection formula, method of images and exact barrier simulation |
| [The terminal wealth distribution](terminal_wealth_distribution.md) | Exact simulation of the gap process mapped to wealth |
| [Mandates as one effective asset](mandate_aggregation.md) | Table 2 of the manuscript and sampled portfolio jumps |
| [The investment opportunity set](investment_opportunity_set.md) | Closed-form calibration, Table 2 and Figure 6 of the manuscript |
| [European options](european_options.md) | Lewis–Fourier pricer, put–call parity and the Black–Scholes limit |
| [Variance swaps](variance_swaps.md) | Matrix-exponential occupation times, exact quadratic variation and static replication |

The chapters also record where the package and the manuscript differ, with the evidence on the page.

## Paper validator

From a development install at repository root:

```bash
python papers/goal_based_allocation_2026/replication/generate_paper_figures.py --test
```

The current CLI runs nine assertions and then generates the ten figures. Output defaults to the external local runtime.
An explicit `--outdir` must be absolute and outside the checkout and OneDrive;
test mode honors the same destination. The assertions cover density normalization, barrier-density
and analytical-survival consistency, horizon monotonicity, asset comparisons, Riccati initial
conditions, a 100K-path Monte Carlo survival comparison, and Table 1 inputs.

## Numerical conventions

- Do not change Laplace inversion contours, quadrature nodes, or ODE tolerances to satisfy a test.
- Do not silently regenerate expected values or paper figures.
- When a plausible change can run but be numerically wrong, require an independent method: Monte
  Carlo for wealth-floor analytics and Fourier/Monte Carlo for option pricing.
- Record exact values and environment for migration/release gates.

See the repository's ignored `agents/` reports for dated local migration evidence; they are
operational records, not public package documentation.
