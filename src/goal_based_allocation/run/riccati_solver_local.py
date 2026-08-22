"""Development runner for Riccati and Monte Carlo diagnostics."""

from enum import Enum

import numpy as np

from goal_based_allocation.regime_switch_paper import (
    compute_survival,
    create_paper_assets,
)
from goal_based_allocation.riccati_solver import (
    find_ell,
    gap_process_asset,
    simulate_mv_optimal,
)


class Locals(Enum):
    """Runnable Riccati-solver development cases."""

    RICCATI_AND_MONTE_CARLO = 1


def run_local(local: Locals) -> None:
    """Run the Riccati, Monte Carlo, and gap-process diagnostics."""
    if local != Locals.RICCATI_AND_MONTE_CARLO:
        raise NotImplementedError(local)

    np.set_printoptions(precision=6, linewidth=120)
    print("=" * 70)
    print("RICCATI ODE + MV-OPTIMAL — TESTS")
    print("=" * 70)

    equity = create_paper_assets()['equity']
    horizon = 10.0
    rate = 0.02
    consumption = 0.03

    print("\n--- Test 1: Riccati ODE ---")
    ell, riccati = find_ell(equity, horizon, 0.05, rate, consumption)
    print(f"  ℓ = {ell:.4f}")
    print(f"  a(T): [{riccati.a[0, -1]:.6f}, {riccati.a[1, -1]:.6f}]")
    target = equity.pi0 * np.exp(0.05 * horizon)
    print(f"  E[Π_T] = {riccati.expected_wealth(0):.4f}  (target: {target:.4f})")
    print(f"  min(a): [{np.min(riccati.a[0]):.6f}, {np.min(riccati.a[1]):.6f}]")

    derived = riccati.derived_at_tau(horizon)
    print(f"  μ (diffusion drift): "
          f"[{riccati.dp['mu_bar'][0]:.6f}, {riccati.dp['mu_bar'][1]:.6f}]")
    print(f"  r_h = {riccati.r_h}, r_c = {riccati.r_c}")
    print(f"  ω*_a(t=0): [{derived['w_a'][0]:.4f}, {derived['w_a'][1]:.4f}]")
    print(f"  Π*(t=0):   [{derived['Pi_star'][0]:.4f}, {derived['Pi_star'][1]:.4f}]")
    print(f"  ω*(t=0, Π=100, reg=0) = {riccati.omega_star(0, 100, 0):.4f}")

    print("\n--- Test 2: MC (100K paths) ---")
    monte_carlo = simulate_mv_optimal(riccati, n_paths=100_000, seed=42)
    survival_mc = np.mean(monte_carlo['survived'])
    print(f"  Survival: {survival_mc:.4f}")
    print(f"  E[Π_T] (all): {np.mean(monte_carlo['Pi_T']):.4f}  "
          f"(target: {riccati.expected_wealth(0):.4f})")
    if survival_mc > 0.01:
        survived = monte_carlo['Pi_T'][monte_carlo['survived']]
        print(f"  E[Π_T|surv]: {np.mean(survived):.4f}")
        print(f"  std[Π_T|surv]: {np.std(survived):.4f}")

    print("\n--- Test 3: Gap process ---")
    gap = gap_process_asset(riccati)
    print(f"  Z₀={gap.pi0:.4f}, B₀={gap.pi_floor:.4f}, x₀={gap.x0:.4f}")
    print(f"  σ_eff: [{gap.params.sigma0:.4f}, {gap.params.sigma1:.4f}]")
    if 0 < gap.x0 < 20:
        survival_gap = compute_survival(horizon, gap.x0, gap)
        print(f"  Analytic surv (gap): {survival_gap:.4f}")
        print(f"  MC survival:         {survival_mc:.4f}")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    run_local(local=Locals.RICCATI_AND_MONTE_CARLO)
