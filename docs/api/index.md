---
myst:
  html_meta:
    description: >-
      API reference for goal-based-allocation: every public export of goal_based_allocation,
      grouped by capability and linked to the handbook chapters that derive its formulas.
---

# Public API reference

*Author: [Artur Sepp](https://github.com/ArturSepp)*

Project: [GoalBasedAllocation](https://github.com/ArturSepp/GoalBasedAllocation).
Software citation: [CITATION.cff](https://github.com/ArturSepp/GoalBasedAllocation/blob/main/CITATION.cff).

Public means re-exported from
[`src/goal_based_allocation/__init__.py`](https://github.com/ArturSepp/GoalBasedAllocation/blob/main/src/goal_based_allocation/__init__.py).
Each group below lists its exports, links the handbook chapters that derive the formulas behind
them, and opens one page per object, generated from the docstrings on every build.
`tests/test_docs.py` checks that every export appears here exactly once, so the page cannot drift
from the package. The last group lists the module-level objects that the chapters use and that
are imported from their modules rather than from the package root.

## Model and asset specifications

Methodology: [The regime-switching jump-diffusion](../regime_switching_model.md) and
[Mandates as one effective asset](../mandate_aggregation.md).

```{eval-rst}
.. autosummary::
   :toctree: generated
   :nosignatures:

   goal_based_allocation.RegimeSwitchParams
   goal_based_allocation.AssetSpecification
   goal_based_allocation.MandateSpecification
   goal_based_allocation.create_paper_assets
   goal_based_allocation.create_paper_mandates
```

## Buy-and-hold benchmark

Methodology: [Buy-and-hold moments](../buy_and_hold_moments.md).

```{eval-rst}
.. autosummary::
   :toctree: generated
   :nosignatures:

   goal_based_allocation.bh_moments_rsjd
```

## MV-optimal policy and gap process

Methodology: [The MV-optimal policy and the Riccati system](../mv_optimal_policy.md) and
[The wealth floor and the flat-barrier reduction](../wealth_floor_gap_process.md).

```{eval-rst}
.. autosummary::
   :toctree: generated
   :nosignatures:

   goal_based_allocation.find_ell
   goal_based_allocation.gap_process_asset
```

## Laplace-domain barrier analytics

Methodology: [Survival, densities and overshoot in the Laplace domain](../laplace_barrier_framework.md)
and [The terminal wealth distribution](../terminal_wealth_distribution.md).

```{eval-rst}
.. autosummary::
   :toctree: generated
   :nosignatures:

   goal_based_allocation.compute_survival
   goal_based_allocation.compute_density
   goal_based_allocation.compute_tilted_survival
   goal_based_allocation.compute_overshoot_density
```

## Mandate aggregation

Methodology: [Mandates as one effective asset](../mandate_aggregation.md).

```{eval-rst}
.. autosummary::
   :toctree: generated
   :nosignatures:

   goal_based_allocation.build_effective_asset
   goal_based_allocation.portfolio_sigma_unc
   goal_based_allocation.portfolio_eta_quadrature
```

## Investment opportunity set

Methodology: [The investment opportunity set and investor selection](../investment_opportunity_set.md).

```{eval-rst}
.. autosummary::
   :toctree: generated
   :nosignatures:

   goal_based_allocation.AdvisorSpec
   goal_based_allocation.compute_opportunity_point
   goal_based_allocation.build_opportunity_set
```

## European options

Methodology: [European options under regime switching](../european_options.md).

```{eval-rst}
.. autosummary::
   :toctree: generated
   :nosignatures:

   goal_based_allocation.RiskNeutralParams
   goal_based_allocation.OptionType
   goal_based_allocation.Regime
   goal_based_allocation.price_vanilla
   goal_based_allocation.implied_vol
```

## Variance swaps and jump premia

Methodology: [Variance swaps and the crash-size premium](../variance_swaps.md).

```{eval-rst}
.. autosummary::
   :toctree: generated
   :nosignatures:

   goal_based_allocation.VarianceConvention
   goal_based_allocation.VarianceDecomposition
   goal_based_allocation.VarianceRiskPremium
   goal_based_allocation.SizePremiumCalibration
   goal_based_allocation.occupation_times
   goal_based_allocation.decompose_variance
   goal_based_allocation.variance_swap_strike
   goal_based_allocation.jump_skew_gap
   goal_based_allocation.variance_risk_premium
   goal_based_allocation.implied_crash_size_from_var_swap
   goal_based_allocation.skew_overidentification_test
```

## Module-level objects used by the handbook

These objects are not re-exported from the package root. Import them from their modules, for
example `from goal_based_allocation.laplace_inversion import laplace_invert_abate_whitt`. They
carry no stability promise beyond the
[changelog](https://github.com/ArturSepp/GoalBasedAllocation/blob/main/CHANGELOG.md). Prefer the
model-level functions above unless implementing or validating a transform calculation.

Methodology: [Numerical Laplace inversion](../laplace_inversion.md),
[The MV-optimal policy and the Riccati system](../mv_optimal_policy.md) and
[The wealth floor and the flat-barrier reduction](../wealth_floor_gap_process.md).

```{eval-rst}
.. autosummary::
   :toctree: generated
   :nosignatures:

   goal_based_allocation.laplace_inversion.laplace_invert_abate_whitt
   goal_based_allocation.laplace_inversion.laplace_invert_stehfest
   goal_based_allocation.riccati_solver.solve_riccati
   goal_based_allocation.riccati_solver.RiccatiSolution
   goal_based_allocation.riccati_solver.simulate_mv_optimal
```
