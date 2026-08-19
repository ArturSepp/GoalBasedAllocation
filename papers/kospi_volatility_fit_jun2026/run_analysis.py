"""
entry point for the KOSPI2 June-2026 volatility-fit study.

Self-contained example idiom: pick a case from `LocalTest` and dispatch via
`run_local_test`. Run from this folder with `goal_based_allocation` installed:

    python run_analysis.py            # runs the full term structure by default

Cases:
    SLICE_ONE      build + print one clean OTM slice, with the parity check
    MIXTURE        forward-constrained two-lognormal baseline on one tenor
    REGIME_SWITCH  shipped RS model, both start regimes, on one tenor
    TERM_STRUCTURE all three tenors: diagnostics table + 3-tenor figure
    BVOL_CHECK     BVOL fitted grid vs listed bid/ask, per tenor
"""
from __future__ import annotations

# packages
import os
from enum import Enum
# local
from vol_surface_utils import (PriceSource, read_chain, build_otm_slice,
                               fit_lognormal_mixture, parity_residual)
from regime_switch_calibration import calibrate_regime_switch, StartRegime
from term_structure import (TENORS, DATA_DIR, run_term_structure, analyze_tenor,
                            plot_three_tenor_figure)
from bvol_interpolation_check import compare_chain_to_grid, GRID_MATCH


class LocalTest(Enum):
    SLICE_ONE = 1
    MIXTURE = 2
    REGIME_SWITCH = 3
    TERM_STRUCTURE = 4
    BVOL_CHECK = 5


def run_local_test(case: LocalTest) -> None:
    if case == LocalTest.SLICE_ONE:
        meta, df = read_chain(os.path.join(DATA_DIR, TENORS[1]['csv']))
        max_resid, n = parity_residual(meta, df)
        print(f"forward={meta.forward:.2f}  T={meta.ttm:.4f}y  r_eff={meta.r_eff:+.4%}")
        print(f"put-call parity (last): {n} paired strikes, max|resid|={max_resid:.2f} pts")
        vs = build_otm_slice(meta, df, source=PriceSource.LAST, drop_strikes=[1500.0])
        print(f"ATM={100 * vs.atm_vol():.1f}  frown_depth(80/150)={vs.frown_depth_80_150():+.1f}"
              f"  skew_slope={vs.skew_slope():+.1f}")

    elif case == LocalTest.MIXTURE:
        meta, df = read_chain(os.path.join(DATA_DIR, TENORS[1]['csv']))
        vs = build_otm_slice(meta, df, source=PriceSource.LAST, drop_strikes=[1500.0])
        fit = fit_lognormal_mixture(vs)
        print(f"mixture RMSE={fit.rmse_volpts:.2f} volpts | "
              f"down w={fit.weight_down:.2f} F={fit.forward_down:.0f} s={100*fit.vol_down:.0f} | "
              f"up w={1-fit.weight_down:.2f} F={fit.forward_up:.0f} s={100*fit.vol_up:.0f}")
        print(f"RND skew={fit.rnd_skew:+.2f}  excess_kurt={fit.rnd_excess_kurtosis:+.2f}  "
              f"modes(S/F)={fit.rnd_modes}")

    elif case == LocalTest.REGIME_SWITCH:
        meta, df = read_chain(os.path.join(DATA_DIR, TENORS[1]['csv']))
        vs = build_otm_slice(meta, df, source=PriceSource.LAST, drop_strikes=[1500.0])
        for start in (StartRegime.GROWTH, StartRegime.STRESS):
            fit = calibrate_regime_switch(vs, start)
            print(f"start={start.name:6s} RMSE={fit.rmse_volpts:.2f}  {fit.params}  "
                  f"crash={fit.mean_crash_pct:+.1f}%  recovery={fit.mean_recovery_pct:+.1f}%")

    elif case == LocalTest.TERM_STRUCTURE:
        _table, diags = run_term_structure()
        print('figure ->', plot_three_tenor_figure(diags))

    elif case == LocalTest.BVOL_CHECK:
        for chain_csv in GRID_MATCH:
            print(f"\n=== {chain_csv} vs BVOL {GRID_MATCH[chain_csv]} ===")
            print(compare_chain_to_grid(chain_csv, 'bvol_moneyness_grid_20260724.csv').to_string(index=False))


if __name__ == '__main__':
    run_local_test(LocalTest.TERM_STRUCTURE)
