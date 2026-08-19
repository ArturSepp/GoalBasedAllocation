# Public API reference

The supported import surface is re-exported from `goal_based_allocation`. The catalogue below is
checked against the installed package so documentation cannot name a missing symbol. Follow the
module links for source and implementation docstrings.

## Wealth-floor model and analytics

| Public symbol | Purpose |
|---|---|
| `RegimeSwitchParams` | Two-regime diffusion, transition, and jump parameters. |
| `AssetSpecification` | Asset-level wealth-floor specification. |
| `MandateSpecification` | Named mandate specification. |
| `compute_density` | Bounded regime-conditional transition density. |
| `compute_survival` | Analytical survival probability. |
| `compute_tilted_survival` | Tilted survival transform for moments. |
| `compute_overshoot_density` | Jump-overshoot density below the barrier. |
| `create_paper_assets` | Asset specifications used by the paper. |
| `create_paper_mandates` | Named paper mandate specifications. |
| `bh_moments_rsjd` | Exact buy-and-hold moments by matrix exponential. |

Source: [`regime_switch_paper.py`](https://github.com/ArturSepp/GoalBasedAllocation/blob/main/src/goal_based_allocation/regime_switch_paper.py).

## MV-optimal policy

| Public symbol | Purpose |
|---|---|
| `find_ell` | Solve the Riccati policy for a target return. |
| `gap_process_asset` | Map a Riccati solution to the terminal gap process. |

Source: [`riccati_solver.py`](https://github.com/ArturSepp/GoalBasedAllocation/blob/main/src/goal_based_allocation/riccati_solver.py).

## Effective assets and opportunity sets

| Public symbol | Purpose |
|---|---|
| `build_effective_asset` | Aggregate the multi-asset mandate to one effective asset. |
| `portfolio_sigma_unc` | Unconditional portfolio volatility used by mandate construction. |
| `portfolio_eta_quadrature` | Deterministic effective-jump quadrature. |
| `AdvisorSpec` | Advisor-side opportunity-set assumptions. |
| `compute_opportunity_point` | Compute one analytical mandate and benchmark. |
| `build_opportunity_set` | Compute a bond-weight opportunity set. |

Sources: [`client_solver.py`](https://github.com/ArturSepp/GoalBasedAllocation/blob/main/src/goal_based_allocation/client_solver.py)
and [`opportunity_set.py`](https://github.com/ArturSepp/GoalBasedAllocation/blob/main/src/goal_based_allocation/opportunity_set.py).

## European options

| Public symbol | Purpose |
|---|---|
| `RiskNeutralParams` | Risk-neutral two-regime option parameters. |
| `OptionType` | Call/put selection. |
| `Regime` | Growth/stress starting regime. |
| `price_vanilla` | Joint-strike European call/put pricing. |
| `implied_vol` | Black-Scholes implied volatility inversion. |

Source: [`vanilla_option_pricer.py`](https://github.com/ArturSepp/GoalBasedAllocation/blob/main/src/goal_based_allocation/vanilla_option_pricer.py).

## Variance analytics

| Public symbol | Purpose |
|---|---|
| `VarianceConvention` | Variance-swap convention selection. |
| `VarianceDecomposition` | Diffusion/jump/regime variance decomposition. |
| `VarianceRiskPremium` | Variance risk-premium result. |
| `SizePremiumCalibration` | Size-premium calibration result. |
| `variance_swap_strike` | Closed-form variance-swap strike. |
| `decompose_variance` | Decompose total variance into model components. |
| `occupation_times` | Expected regime occupation times. |
| `jump_skew_gap` | Jump-induced skew-gap diagnostic. |
| `variance_risk_premium` | Compare physical and risk-neutral variance. |
| `implied_crash_size_from_var_swap` | Infer crash size from a variance-swap input. |
| `skew_overidentification_test` | Cross-check skew and variance restrictions. |

Source: [`variance_swap.py`](https://github.com/ArturSepp/GoalBasedAllocation/blob/main/src/goal_based_allocation/variance_swap.py).

## Advanced module

The inversion algorithms in
[`laplace_inversion.py`](https://github.com/ArturSepp/GoalBasedAllocation/blob/main/src/goal_based_allocation/laplace_inversion.py)
are lower-level numerical machinery. Prefer the model-level public functions above unless
implementing or validating a transform calculation.
