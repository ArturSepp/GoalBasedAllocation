# Choosing the appropriate portfolio workflow

Observed 2026-08-19 from each project's official documentation. This is a workflow comparison,
not a performance ranking. Versions identify the documentation inspected and should be refreshed
when this page is updated.

| Project | Primary documented workflow | Choose it when |
|---|---|---|
| GoalBasedAllocation 0.3.0 | analytical continuous-time MV policy, absorbing wealth floor, terminal distribution under two-regime jumps | the research question is survival, floor risk, glide paths, or mandate distributions under this published model |
| [optimalportfolios 6.6.0](https://github.com/ArturSepp/OptimalPortfolios) | rolling multi-asset construction/backtesting with optimisers, constraints, covariance/factor inputs, and drift-aware workflows | the task is asset-level portfolio weights, constraints, transaction costs, or historical backtesting |
| [PyPortfolioOpt 1.5.4](https://pyportfolioopt.readthedocs.io/en/latest/UserGuide.html) | expected-return/risk estimation and modular static optimisers including mean-variance, Black-Litterman, and HRP | a compact, accessible single-period allocation/prototyping workflow is the priority |
| [Riskfolio-Lib 7.3](https://riskfolio-lib.readthedocs.io/en/latest/riskfoliolib/portfolio.html) | convex portfolio optimisation across many risk measures, risk parity, factor, Black-Litterman, and related models | broad risk-measure/model coverage and weight/risk-contribution constraints are required |
| [CVXPortfolio 1.5.0](https://www.cvxportfolio.com/en/stable/manual.html) | single- and multi-period trade optimisation, transaction/holding costs, constraints, and market simulation | the decision is a sequence of trades with explicit costs and a backtest/simulator |

## Where workflows overlap

All five projects can inform portfolio allocation, but they optimise different objects.
GoalBasedAllocation solves one specialised continuous-time stochastic-control model and provides
the resulting terminal distribution analytically. The other projects primarily construct
asset-level weights or trades from user data/forecasts under broader constraints and objectives.

## Decision guide

- Choose **GoalBasedAllocation** for the paper's two-regime jump model, an absorbing floor,
  survival/overshoot decomposition, endogenous de-risking, and analytical terminal moments.
- Choose **optimalportfolios** when working inside Artur Sepp's package ecosystem with rolling
  multi-asset portfolios, factor/covariance inputs, constraints, and factsheet/backtest workflows.
- Choose **PyPortfolioOpt** for straightforward modular portfolio optimisation and educational or
  rapid prototyping workflows around expected returns, risk models, and optimisers.
- Choose **Riskfolio-Lib** when the key requirement is a wide menu of portfolio risk measures,
  risk parity/factor models, and optimisation constraints.
- Choose **CVXPortfolio** when transaction/holding costs and single- or multi-period trade
  decisions are central.

These tools can be complementary. For example, GoalBasedAllocation can study a strategic
mandate's floor-risk dynamics, while another tool constructs or backtests the asset-level sleeve.
Do not copy APIs or analytics between packages; exchange explicit inputs and conventions.

## Method and limitations

The table uses official project documentation and repositories accessed on 2026-08-19. It does
not benchmark speed, solver quality, popularity, or investment performance. Absence of a feature
from this table is not proof that no extension exists. Recheck versions, licenses, dependencies,
and supported features before making a project decision.
