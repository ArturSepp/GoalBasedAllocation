# AGENTS.md

Guidance for AI coding agents working in the **GoalBasedAllocation** repository.

## Project overview

`goal-based-allocation` implements dynamic mean-variance portfolio allocation under
regime-switching jump-diffusions with an absorbing wealth floor, solved analytically:
survival probabilities and conditional moments via Laplace transforms, the MV-optimal
policy via a Riccati ODE system, terminal wealth density decomposition, expected glide
paths, investment opportunity sets, and vanilla option pricing under the same model.
Monte Carlo exists only to validate the analytics.

It is the companion code to Sepp (2026), *Dynamic Mean-Variance Portfolio Allocation
under Regime-Switching Jump-Diffusions with Absorbing Barriers and Distribution
Matching* (SSRN 6534579). Distribution name `goal-based-allocation`; import name
`goal_based_allocation`. Licensed MIT (`LICENSE`).

## Ecosystem position

This package is one of eight open-source Python libraries maintained at
[github.com/ArturSepp](https://github.com/ArturSepp). Before implementing anything
non-trivial, check whether it already exists in one of these:

| Package | Repository | Purpose |
|---|---|---|
| `qis` | QuantInvestStrats | Performance analytics, factsheets, visualisation |
| `optimalportfolios` | OptimalPortfolios | Portfolio construction and backtesting |
| `factorlasso` | factorlasso | Sparse factor models and factor covariance estimation |
| `bbg-fetch` | BloombergFetch | Bloomberg data fetching |
| `trendfollowing` | TrendFollowingSystems | Trend-following systems: closed-form theory and replication |
| `goal-based-allocation` | GoalBasedAllocation | Dynamic MV allocation under regime-switching jump-diffusions |
| `stochvolmodels` | StochVolModels | Stochastic volatility pricing analytics |
| `vanilla-option-pricers` | VanillaOptionPricers | Vanilla option pricers and implied volatility fitters |

Actual package dependencies within the stack: `optimalportfolios` depends on `qis`
and `factorlasso`; `trendfollowing` depends on `qis`; `stochvolmodels` has an
optional `research` extra that pulls in `qis`. The others are independent.

Do not vendor or copy code between these packages. If functionality belongs in a
sibling package, say so rather than reimplementing it here.

## Repository layout

```
goal_based_allocation/
  laplace_inversion.py     Laplace transform inversion machinery
  riccati_solver.py        Riccati ODE system for the MV-optimal policy
  client_solver.py         mandate-level solver
  mandate_utils.py         portfolio mandate construction
  opportunity_set.py       investment opportunity set construction
  vanilla_option_pricer.py option pricing under the regime-switching model
  variance_swap.py         variance swap analytics
  regime_switch_paper.py   paper-facing entry points
tests/       4 test modules (top-level, test_*.py)
paper_code/  scripts reproducing the paper figures
examples/    runnable examples
```

## Commands

```bash
pip install -e ".[dev]"
pytest -q                    # as CI runs it
pytest -m "not slow"         # skip Monte Carlo cross-checks
pytest tests/test_framework.py -v
```

The `slow` pytest marker is declared for slower tests such as Monte Carlo cross-checks.
Runtime dependencies are numpy, scipy and matplotlib only. Supported Python is >= 3.10;
CI runs 3.10 - 3.12 via `.github/workflows/tests.yml`.

## Conventions

- Test files are named `test_*.py` and live in the top-level `tests/` directory.
- Mark slow tests with `@pytest.mark.slow` rather than deleting or skipping them.
- Dataclasses carry model parameters and results throughout the package — extend the
  existing containers rather than passing loose tuples.
- The package is numpy/scipy only: there is no pandas dependency and no DataFrame layer.
  Keep it that way.
- Analytical results are cross-checked against the Monte Carlo simulator; a new
  analytical routine should come with the corresponding validation test.

## Constraints — do not do these

- Do not replace analytical computations with Monte Carlo. The point of this package is
  that the quantities are available in closed or semi-closed form; Monte Carlo is the
  validator, not the implementation.
- Do not add pandas, or any dependency beyond numpy, scipy and matplotlib.
- Do not change the model specification (two regimes, exponential jumps at transitions,
  single effective asset after mandate aggregation) — it is the published model.
- Do not adjust Laplace inversion contours, quadrature nodes, or ODE solver tolerances
  to make a test pass; investigate the discrepancy instead.
- Do not commit generated figures.

## Replication contract

`paper_code/` reproduces the figures of Sepp (2026). Any change to the Laplace
inversion, the Riccati solver, or the mandate aggregation requires re-running those
scripts and confirming the figures and reported values are unchanged.

## Release checklist

A release touches three version locations. All three must agree:

1. `version` in `pyproject.toml`
2. `version` and `date-released` in `CITATION.cff`
3. the software BibTeX entry in `README.md` (if it pins a version)

Then: commit, tag `v<version>`, build and publish to PyPI, and cut a GitHub Release
with the same tag. Do not bump versions as part of an unrelated change, and do not
publish without the maintainer explicitly asking for a release.
