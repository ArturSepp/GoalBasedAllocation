# Mandates and investment opportunity sets

## Question answered

For a bond/equity/private-equity mandate, aggregate the risky sleeve to one effective asset,
calibrate the initial MV-optimal allocation, and compute terminal moments, survival, quantiles,
and the exact buy-and-hold benchmark.

## Workflow

1. Choose bond share `w_bd` and equity share `AdvisorSpec.q` of the non-bond sleeve.
2. Use `portfolio_sigma_unc` and `portfolio_eta_quadrature` through `build_effective_asset` to
   obtain the effective regime-switching asset.
3. `compute_opportunity_point` calibrates the target return to `AdvisorSpec.omega_0` and computes
   one analytical mandate.
4. `build_opportunity_set` repeats this over bond weights.
5. Compare `E`, `Std`, `S`, `F`, and `O` with `E_BH`, `Std_BH`, and `r_impl_BH`.

The [quickstart](../getting-started.md) executes exactly one balanced point. The full plotting
source is
[`examples/investment_opportunity_set.py`](https://github.com/ArturSepp/GoalBasedAllocation/blob/main/examples/investment_opportunity_set.py).

## Important outputs

| Key | Meaning |
|---|---|
| `E`, `Std` | unconditional terminal-wealth mean and standard deviation |
| `S`, `F`, `O` | survival, floor-atom, and overshoot probabilities |
| `Es`, `Stds` | moments conditional on survival |
| `q5` ... `q95` | terminal-wealth quantiles |
| `r_impl` | annual continuous return implied by expected terminal wealth |
| `E_BH`, `Std_BH`, `r_impl_BH` | exact buy-and-hold benchmark moments |
| `floor_cost_pct` | relative terminal-value difference from floor protection |

## Model boundary

Mandate aggregation reduces multiple assets to one effective risky process using a fixed
correlation/parameter specification. It is not a general constrained optimiser and does not model
rebalancing costs. Use the [choice guide](../comparison.md) when the task is discrete asset-level
portfolio construction.

## Common mistakes

- Treating `floor_cost_pct` as a charged fee or promised realised cost.
- Mixing percentage and decimal inputs (`0.02`, not `2`, for 2%).
- Changing paper asset assumptions while continuing to cite paper table values.
- Interpreting quantiles without the floor atom and overshoot decomposition.

API: [client solver and opportunity-set modules](../api/index.md).
