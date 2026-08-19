"""Deterministic first-success workflow for a balanced goal-based mandate."""

from goal_based_allocation import AdvisorSpec, compute_opportunity_point


def main() -> None:
    """Compute one analytical mandate and its exact buy-and-hold benchmark."""
    point = compute_opportunity_point(w_bd=0.35, spec=AdvisorSpec())
    if point is None:
        raise RuntimeError("balanced-mandate calibration did not converge")

    print("GoalBasedAllocation quickstart")
    print("horizon=10y, initial_wealth=100, rates=continuous annual")
    print(
        "mandate weights: "
        f"bonds={point['w_bd']:.1%}, equity={point['w_eq']:.1%}, "
        f"private_equity={point['w_pe']:.1%}"
    )
    print(
        "MV-optimal: "
        f"expected_wealth={point['E']:.3f}, std={point['Std']:.3f}, "
        f"survival={point['S']:.3%}"
    )
    print(f"floor_atom={point['F']:.3%}, jump_overshoot={point['O']:.3%}")
    print(
        "buy-and-hold: "
        f"expected_wealth={point['E_BH']:.3f}, std={point['Std_BH']:.3f}, "
        f"implied_return={point['r_impl_BH']:.3%}"
    )
    print(f"floor_protection_cost={point['floor_cost_pct']:.3%} of terminal value")


if __name__ == "__main__":
    main()
