"""Development runner for regime-switching density diagnostics."""

from enum import Enum

import numpy as np

from goal_based_allocation.regime_switch_paper import (
    AssetSpecification,
    RegimeSwitchParams,
    compute_density,
    compute_survival,
    compute_wealth_density,
    create_paper_assets,
    create_paper_mandates,
)


class Locals(Enum):
    """Runnable regime-switch development cases."""

    PAPER_DIAGNOSTICS = 1


def run_local(local: Locals) -> None:
    """Run the paper density and survival diagnostics."""
    if local != Locals.PAPER_DIAGNOSTICS:
        raise NotImplementedError(local)

    np.set_printoptions(precision=8, linewidth=120)
    print("=" * 70)
    print("REGIME-SWITCHING PAPER — TESTS")
    print("=" * 70)

    assets = create_paper_assets()
    equity = assets['equity']
    print(f"\nEquity: nu0={equity.nu0:.6f}, nu1={equity.nu1:.6f}")
    print(f"  sigma0={equity.params.sigma0}, sigma1={equity.params.sigma1}")
    print(f"  lambda01={equity.params.lambda01}, lambda10={equity.params.lambda10}")
    print(f"  eta0={equity.params.eta0:.6f}, eta1={equity.params.eta1:.6f}")
    print(f"  x0={equity.x0:.6f} (pi0={equity.pi0}, floor={equity.pi_floor})")

    print("\n--- Test 1: Unbounded density normalization ---")
    equity_unbounded = AssetSpecification(
        'equity_unb', equity.params, equity.mu_growth, equity.mu_stress, pi0=100, pi_floor=0
    )
    x_grid = np.linspace(-5, 5, 600)
    density0, density1 = compute_density(5.0, x_grid, equity_unbounded)
    print(f"  integral(d0+d1) = {np.trapezoid(density0 + density1, x_grid):.8f} "
          "(expect 1.0)")
    print(f"  min(d0+d1) = {np.min(density0 + density1):.2e}")

    equity_no_jump = AssetSpecification(
        'equity_nj',
        RegimeSwitchParams(sigma0=0.15, sigma1=0.25, lambda01=0.1, lambda10=1.0),
        equity.mu_growth,
        equity.mu_stress,
        pi0=100,
        pi_floor=0,
    )
    density0_nj, density1_nj = compute_density(5.0, x_grid, equity_no_jump)
    print(f"  No-jump integral = {np.trapezoid(density0_nj + density1_nj, x_grid):.8f}")

    print("\n--- Test 2: Barrier density vs survival (equity, T=10) ---")
    barrier_grid = np.linspace(0.001, 4.0, 1000)
    result = compute_wealth_density(10.0, barrier_grid, equity)
    print(f"  survival (density):  {result['survival_density']:.8f}")
    print(f"  survival (analytic): {result['survival_analytic']:.8f}")
    print(f"  stopping prob:       {result['stopping_prob']:.8f}")
    print(f"  consistency gap:     "
          f"{abs(result['survival_density'] - result['survival_analytic']):.2e}")

    print("\n--- Test 3: Survival probability vs T (equity) ---")
    print(f"  {'T':>5s} {'survival':>12s} {'stopping':>12s}")
    for horizon in [1, 2, 5, 10]:
        survival = compute_survival(horizon, equity.x0, equity)
        print(f"  {horizon:5d} {survival:12.6f} {1 - survival:12.6f}")

    print("\n--- Test 4: Three assets, T=10 ---")
    for name, asset in assets.items():
        survival = compute_survival(10, asset.x0, asset)
        print(f"  {name:16s}: x0={asset.x0:.4f}, surv={survival:.6f}, "
              f"stop={1 - survival:.6f}")

    print("\n--- Test 5: Mandate specifications ---")
    for name, mandate in create_paper_mandates(assets).items():
        print(f"  {name}: ", end="")
        parts = [f"{asset.name}={weight:.0%}" for asset, weight in mandate.allocations.items()]
        print(", ".join(parts))

    print("\n" + "=" * 70)
    print("ALL TESTS COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    run_local(local=Locals.PAPER_DIAGNOSTICS)
