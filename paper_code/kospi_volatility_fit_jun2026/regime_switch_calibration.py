"""
calibrate the shipped GoalBasedAllocation regime-switch jump-diffusion to a KOSPI2
OTM vol slice, reusing `goal_based_allocation.vanilla_option_pricer` unchanged.

Model (risk-neutral): two regimes, growth (0) and stress (1). 0->1 carries a downward
crash jump -Exp(eta_0), 1->0 an upward recovery jump +Exp(eta_1). Drifts are pinned by
the per-state martingale condition. The market forward F is matched exactly by setting
the pricer carry to r_eff = ln(F/S)/T, so the model forward S*exp(r_eff*T) = F.

Objective: vega-weighted price-space least squares (one price_vanilla call per eval;
much faster than inverting model IVs inside the loop). Both start regimes are fitted:
from GROWTH the only jump is the downward crash, which produces a put skew naturally.
"""
from __future__ import annotations

# packages
import numpy as np
from dataclasses import dataclass
from enum import Enum
from typing import List, Tuple
from scipy.optimize import minimize
# qis / project
from goal_based_allocation import RiskNeutralParams, price_vanilla, implied_vol
# local
from vol_surface_utils import VolSlice, bs_forward, bs_vega_forward


PARAM_NAMES = ('sigma_0', 'sigma_1', 'lambda_01', 'lambda_10', 'eta_0', 'eta_1')
_LO = np.array([0.03, 0.03, 0.0, 0.0, 0.02, 0.02])
_HI = np.array([0.95, 2.00, 15.0, 25.0, 0.98, 0.98])   # eta_i < 1 for a finite jump MGF

# diverse Nelder-Mead starts per regime (growth: crash-jump-down; stress: recovery story)
_STARTS = {
    0: [np.array([0.55, 0.35, 2.0, 0.5, 0.30, 0.10]),
        np.array([0.60, 0.40, 3.0, 0.3, 0.20, 0.10]),
        np.array([0.45, 0.30, 1.0, 1.0, 0.40, 0.15]),
        np.array([0.65, 0.25, 4.0, 0.2, 0.15, 0.08])],
    1: [np.array([0.50, 0.50, 0.5, 2.0, 0.25, 0.30]),
        np.array([0.45, 0.55, 1.0, 1.5, 0.30, 0.40]),
        np.array([0.30, 0.65, 2.5, 2.0, 0.80, 0.10])],
}


class StartRegime(int, Enum):
    GROWTH = 0
    STRESS = 1


@dataclass(frozen=True)
class RegimeSwitchFit:
    """calibrated shipped-model parameters and diagnostics for one slice/start regime."""
    start_regime: StartRegime
    theta: np.ndarray                  # (sigma_0, sigma_1, lambda_01, lambda_10, eta_0, eta_1)
    rmse_volpts: float
    model_iv: np.ndarray               # aligned with the slice strikes
    nu_growth: float                   # martingale Q-drift, growth
    nu_stress: float                   # martingale Q-drift, stress
    mean_crash_pct: float              # 100*(exp(-eta_0)-1)
    mean_recovery_pct: float           # 100*(exp(+eta_1)-1)

    @property
    def params(self) -> dict:
        return dict(zip(PARAM_NAMES, np.round(self.theta, 4)))


def _make_params(theta: np.ndarray, r_eff: float) -> RiskNeutralParams:
    theta = np.asarray(theta, dtype=float).copy()
    theta[2:4] = np.clip(theta[2:4], 1e-6, None)     # intensities strictly positive
    return RiskNeutralParams(*theta, rate=r_eff)


def _model_prices(theta: np.ndarray,
                  vol_slice: VolSlice,
                  regime: int,
                  n_terms: int = 13,
                  n_euler: int = 7,
                  ) -> np.ndarray:
    """shipped-pricer forward-value prices at the slice strikes (put/call by moneyness)."""
    params = _make_params(theta, vol_slice.meta.r_eff)
    S, T = vol_slice.meta.spot, vol_slice.meta.ttm
    K, is_put = vol_slice.strike, vol_slice.is_put
    disc = np.exp(-vol_slice.meta.r_eff * T)
    out = np.empty(len(K))
    if is_put.any():
        out[is_put] = price_vanilla(params, S, K[is_put], T, regime=regime,
                                    option_type='put', n_terms=n_terms, n_euler=n_euler)
    if (~is_put).any():
        out[~is_put] = price_vanilla(params, S, K[~is_put], T, regime=regime,
                                     option_type='call', n_terms=n_terms, n_euler=n_euler)
    return out / disc     # forward value, to compare with forward-space market prices


def calibrate_regime_switch(vol_slice: VolSlice,
                            start_regime: StartRegime = StartRegime.GROWTH,
                            ) -> RegimeSwitchFit:
    """multistart Nelder-Mead in vega-weighted forward-price space; returns the best fit."""
    F, T = vol_slice.meta.forward, vol_slice.meta.ttm
    K, is_put, mkt_iv = vol_slice.strike, vol_slice.is_put, vol_slice.iv
    mkt_px = bs_forward(K, mkt_iv, F, T, is_put)
    vega = bs_vega_forward(K, mkt_iv, F, T)
    regime = int(start_regime)

    def objective(theta: np.ndarray) -> float:
        if np.any(theta <= _LO) or np.any(theta >= _HI):
            return 1e9
        try:
            px = _model_prices(theta, vol_slice, regime)
        except Exception:
            return 1e9
        if not np.all(np.isfinite(px)) or np.any(px < 0.0):
            return 1e9
        return 1e4 * float(np.mean(((px - mkt_px) / vega) ** 2))

    best = None
    for start in _STARTS[regime]:
        res = minimize(objective, start, method='Nelder-Mead',
                       options=dict(maxiter=400, fatol=1e-11, xatol=1e-9))
        if best is None or res.fun < best.fun:
            best = res

    theta = best.x
    params = _make_params(theta, vol_slice.meta.r_eff)
    model_iv = np.full(len(K), np.nan)
    if is_put.any():
        model_iv[is_put] = implied_vol(params, vol_slice.meta.spot, K[is_put], T,
                                       regime=regime, option_type='put')
    if (~is_put).any():
        model_iv[~is_put] = implied_vol(params, vol_slice.meta.spot, K[~is_put], T,
                                        regime=regime, option_type='call')
    rmse = 100.0 * float(np.sqrt(np.nanmean((model_iv - mkt_iv) ** 2)))
    return RegimeSwitchFit(start_regime=start_regime,
                           theta=theta,
                           rmse_volpts=rmse,
                           model_iv=model_iv,
                           nu_growth=float(params.nu[0]),
                           nu_stress=float(params.nu[1]),
                           mean_crash_pct=100.0 * (np.exp(-theta[4]) - 1.0),
                           mean_recovery_pct=100.0 * (np.exp(theta[5]) - 1.0))
