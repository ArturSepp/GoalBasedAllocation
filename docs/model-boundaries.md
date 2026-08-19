# Model boundaries

## Appropriate uses

Use this package when the research question matches the published model:

- dynamic pre-commitment mean-variance allocation with an absorbing wealth floor;
- two regimes with exponential jumps at regime transitions;
- terminal survival, density, shortfall decomposition, and moments;
- mandate aggregation to one effective risky asset; or
- European vanilla pricing under the same regime-switching jump-diffusion.

## Intentional non-goals

The package does not implement:

- discrete rolling optimisation with general asset-weight constraints;
- transaction-cost-aware rebalancing or execution;
- a broker, order-management, or production advice system;
- more than two regimes;
- general path-dependent or exotic option payoffs; or
- a pandas/DataFrame portfolio layer.

For discrete multi-asset construction and backtesting, use
[`optimalportfolios`](https://github.com/ArturSepp/OptimalPortfolios). See the
[choice guide](comparison.md) for other portfolio-optimisation workflows.

## Scientific boundary

The code accompanies Sepp (2026), *Dynamic Mean-Variance Portfolio Allocation under
Regime-Switching Jump-Diffusions with Absorbing Barriers and Distribution Matching*. Documentation
may explain and reproduce the implementation; it does not change the paper's claims, model,
parameters, or publication status.
