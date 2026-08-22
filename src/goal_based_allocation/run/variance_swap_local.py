"""Development checks and demonstrations for variance-swap analytics."""

from enum import Enum

import numpy as np

from goal_based_allocation.vanilla_option_pricer import Regime, RiskNeutralParams
from goal_based_allocation.variance_swap import (
    VarianceConvention,
    decompose_variance,
    implied_crash_size_from_var_swap,
    jump_skew_gap,
    occupation_times,
    skew_overidentification_test,
    variance_risk_premium,
    variance_swap_strike,
    variance_swap_strike_mc,
)


class Locals(Enum):
    """Available variance-swap development runs."""

    DEMO_STRIKE_DECOMPOSITION = 1
    DEMO_SIZE_PREMIUM = 2
    CHECK_OCCUPATION_TIMES = 3
    CHECK_NO_JUMP_LIMIT = 4
    TEST_MONTE_CARLO_QV = 5
    TEST_MONTE_CARLO_LOG = 6
    DEMO_IMPLIED_CRASH_SIZE = 7
    DEMO_SKEW_OVERIDENTIFICATION = 8


def _paper_params() -> RiskNeutralParams:
    """Return the risk-neutral base case in the paper's rate convention."""
    return RiskNeutralParams.from_rates(
        sigma_0=0.18,
        sigma_1=0.28,
        lambda_01=0.10,
        lambda_10=1.0,
        eta_01=3.0,
        eta_10=8.0,
        rate=0.03,
    )


def run_local(local: Locals) -> None:
    """Run the selected variance-swap development check or demonstration."""
    params = _paper_params()
    ttm = 1.0

    if local == Locals.DEMO_STRIKE_DECOMPOSITION:
        for regime in (Regime.GROWTH, Regime.STRESS):
            decomposition = decompose_variance(params, ttm, regime)
            print(f"\n{regime.name} regime, T={ttm:g}:")
            print(
                f"  fair vol            {100 * decomposition.fair_vol:6.2f}%   "
                f"(fair variance {decomposition.total:.5f})"
            )
            print(
                f"  diffusion           {decomposition.diffusion:.5f}   "
                f"({100 * decomposition.diffusion / decomposition.total:5.1f}%)"
            )
            print(
                f"  jump - crash        {decomposition.jump_crash:.5f}   "
                f"({100 * decomposition.jump_crash / decomposition.total:5.1f}%)"
            )
            print(
                f"  jump - recovery     {decomposition.jump_recovery:.5f}   "
                f"({100 * decomposition.jump_recovery / decomposition.total:5.1f}%)"
            )
            print(f"  jump fraction       {100 * decomposition.jump_fraction:5.1f}%")
            gap = jump_skew_gap(params, ttm, regime)
            print(f"  K_log - K_var       {gap:+.5f}   (third-cumulant / skew probe)")

    elif local == Locals.DEMO_SIZE_PREMIUM:
        physical = RiskNeutralParams.from_rates(
            sigma_0=0.18,
            sigma_1=0.28,
            lambda_01=0.10,
            lambda_10=1.0,
            eta_01=4.0,
            eta_10=8.0,
            rate=0.03,
        )
        risk_neutral = RiskNeutralParams.from_rates(
            sigma_0=0.18,
            sigma_1=0.28,
            lambda_01=0.15,
            lambda_10=1.0,
            eta_01=2.5,
            eta_10=8.0,
            rate=0.03,
        )
        for maturity in (1.0, 5.0):
            premium = variance_risk_premium(
                physical, risk_neutral, maturity, Regime.GROWTH
            )
            print(
                f"\nT={maturity:g}:  fair vol  "
                f"P={100 * np.sqrt(premium.physical_var):.2f}%  "
                f"Q={100 * np.sqrt(premium.risk_neutral_var):.2f}%   "
                f"ratio={premium.ratio:.2f}"
            )
            print(f"   VRP total      {premium.total:+.5f}")
            print(f"   from diffusion {premium.diffusion:+.5f}")
            print(
                f"   from jumps     {premium.jump:+.5f}   "
                "(intensity+size, entangled as l*eta^2)"
            )

    elif local == Locals.CHECK_OCCUPATION_TIMES:
        for regime in (Regime.GROWTH, Regime.STRESS):
            for maturity in (0.5, 1.0, 5.0, 50.0):
                time_0, time_1 = occupation_times(params, maturity, regime)
                assert abs(time_0 + time_1 - maturity) < 1e-10
            time_0, _ = occupation_times(params, 1e4, regime)
            total_intensity = params.lambda_01 + params.lambda_10
            stationary_0 = params.lambda_10 / total_intensity
            assert abs(time_0 / 1e4 - stationary_0) < 1e-3
        print("occupation times: sum to T and converge to stationary pi. PASS")

    elif local == Locals.CHECK_NO_JUMP_LIMIT:
        flat = RiskNeutralParams(
            sigma_0=0.2,
            sigma_1=0.2,
            lambda_01=0.05,
            lambda_10=0.05,
            eta_0=1e-4,
            eta_1=1e-4,
            rate=0.03,
        )
        quadratic = variance_swap_strike(
            flat, 1.0, Regime.GROWTH, VarianceConvention.QUADRATIC_VARIATION
        )
        log_contract = variance_swap_strike(
            flat, 1.0, Regime.GROWTH, VarianceConvention.LOG_CONTRACT
        )
        assert abs(quadratic - 0.04) < 1e-6
        assert abs(log_contract - 0.04) < 1e-6
        print(
            f"no-jump limit: K_var={quadratic:.6f}, K_log={log_contract:.6f}, "
            "both -> 0.04. PASS"
        )

    elif local in (Locals.TEST_MONTE_CARLO_QV, Locals.TEST_MONTE_CARLO_LOG):
        convention = (
            VarianceConvention.QUADRATIC_VARIATION
            if local == Locals.TEST_MONTE_CARLO_QV
            else VarianceConvention.LOG_CONTRACT
        )
        print(f"{'regime':>7} {'T':>5} {'closed':>10} {'MC':>10} {'MC se':>9} {'z':>7}")
        for regime in (Regime.GROWTH, Regime.STRESS):
            for maturity in (0.5, 2.0, 10.0):
                closed = variance_swap_strike(params, maturity, regime, convention)
                monte_carlo, standard_error = variance_swap_strike_mc(
                    params, maturity, regime, convention, n_paths=300_000
                )
                z_score = (closed - monte_carlo) / standard_error
                print(
                    f"{regime.name:>7} {maturity:5.1f} {closed:10.5f} "
                    f"{monte_carlo:10.5f} {standard_error:9.5f} {z_score:7.2f}"
                )
                assert abs(z_score) < 4.0
        print("PASS")

    elif local == Locals.DEMO_IMPLIED_CRASH_SIZE:
        physical = RiskNeutralParams.from_rates(
            sigma_0=0.18,
            sigma_1=0.28,
            lambda_01=0.10,
            lambda_10=1.0,
            eta_01=4.0,
            eta_10=8.0,
            rate=0.03,
        )
        physical_strike = variance_swap_strike(physical, ttm, Regime.GROWTH)
        print(
            f"physical fair vol: {100 * np.sqrt(physical_strike):.2f}%  "
            f"(eta_0^P = {physical.eta_0:.4f})"
        )
        for market_vol in (0.24, 0.26, 0.28):
            calibration = implied_crash_size_from_var_swap(
                physical, market_vol**2, ttm, Regime.GROWTH
            )
            print(
                f"market vol {100 * market_vol:.0f}%:  "
                f"eta_0^Q = {calibration.eta_0_q:.4f}  "
                f"(x{calibration.size_premium_ratio:.2f} physical), "
                f"jump share {100 * calibration.jump_variance_share:.1f}%"
            )
        try:
            implied_crash_size_from_var_swap(physical, 0.03**2, ttm, Regime.GROWTH)
        except ValueError as error:
            print(f"below-floor guard: {str(error)[:80]}...")
        try:
            implied_crash_size_from_var_swap(
                physical, physical_strike * 0.98, ttm, Regime.GROWTH
            )
        except ValueError as error:
            print(f"sign guard: {str(error)[:80]}...")

    elif local == Locals.DEMO_SKEW_OVERIDENTIFICATION:
        physical = RiskNeutralParams.from_rates(
            sigma_0=0.18,
            sigma_1=0.28,
            lambda_01=0.10,
            lambda_10=1.0,
            eta_01=4.0,
            eta_10=8.0,
            rate=0.03,
        )
        calibration = implied_crash_size_from_var_swap(
            physical, 0.26**2, ttm, Regime.GROWTH
        )
        strikes = np.array([60.0, 70.0, 80.0, 90.0, 100.0, 105.0])
        implied_vols = skew_overidentification_test(
            physical, calibration, 100.0, strikes, ttm, Regime.GROWTH
        )
        print(
            f"calibrated eta_0^Q = {calibration.eta_0_q:.4f} "
            "from the var swap alone;"
        )
        print("put skew below is a PREDICTION (no remaining freedom):\n")
        print(f"{'K':>6} {'iv Q (pred)':>12} {'iv P (bench)':>13} {'premium skew':>13}")
        for strike, (risk_neutral_iv, physical_iv) in zip(strikes, implied_vols):
            print(
                f"{strike:6.0f} {100 * risk_neutral_iv:11.2f}% "
                f"{100 * physical_iv:12.2f}% "
                f"{100 * (risk_neutral_iv - physical_iv):+12.2f}%"
            )
        print("\ncompare column 1 to the market skew: a systematic miss means the")
        print("no-intensity-premium normalisation is too tight (escape hatch: free")
        print("lambda_Q with shrinkage, or hyperexponential-2 severity).")

    else:
        raise NotImplementedError(local)


if __name__ == "__main__":
    run_local(local=Locals.DEMO_STRIKE_DECOMPOSITION)
