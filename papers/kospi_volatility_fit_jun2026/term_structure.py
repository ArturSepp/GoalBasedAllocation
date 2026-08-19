"""
multi-tenor term-structure analysis: build the OTM slice for each expiry, fit the
mixture baseline and the shipped regime-switch model, and assemble the frown-depth /
skew-slope / RND-kurtosis term structure that tests the regime-switch 'frown peaks at
3M' signature.

Verdict from this study: all tenors are downside skews (short tenor adds a call smile),
the skew flattens and the RND Gaussianises with maturity, and the frown depth never
turns positive. The shipped exponential-jump model fits every tenor from the GROWTH
regime with a plain crash jump; no Erlang or free-drift extension is needed.
"""
from __future__ import annotations

# packages
import os
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, List, Tuple
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
# local
from vol_surface_utils import (ChainMeta, PriceSource, VolSlice, read_chain,
                               build_otm_slice, fit_lognormal_mixture, parity_residual,
                               implied_vol_forward)
from regime_switch_calibration import calibrate_regime_switch, StartRegime
# qis / project
from goal_based_allocation import RiskNeutralParams, implied_vol

_HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(_HERE, 'data')
FIG_DIR = os.path.join(_HERE, 'figures')

# expiry file, label, days, and the price source that is cleanest at that tenor
TENORS = [
    dict(csv='kospi2_20260813_20d.csv', label='20d', days=20, source=PriceSource.MID,
         drop=None),
    dict(csv='kospi2_20260910_48d.csv', label='48d', days=48, source=PriceSource.LAST,
         drop=[1500.0]),   # C1500 last prints above its ask (stale)
    dict(csv='kospi2_20261008_76d.csv', label='76d', days=76, source=PriceSource.LAST,
         drop=None),
]

# README §6 regime-switch prediction, frown depth (80/150 definition), for contrast
README_PRED_YEARS = np.array([1 / 12, 0.25, 0.5, 1.0, 2.0])
README_PRED_DEPTH = np.array([-1.1, 5.4, 1.9, 0.8, 0.5])


@dataclass(frozen=True)
class TenorDiagnostics:
    label: str
    years: float
    atm_vol: float
    frown_depth_80_150: float
    frown_depth_wing: float
    skew_slope: float
    rnd_skew: float
    rnd_excess_kurtosis: float
    n_modes: int
    mixture_rmse: float
    rs_growth_rmse: float
    rs_growth_crash_pct: float


def analyze_tenor(cfg: dict) -> TenorDiagnostics:
    """build the slice, fit mixture + RS-growth, return the diagnostics for one tenor."""
    meta, df = read_chain(os.path.join(DATA_DIR, cfg['csv']))
    vol_slice = build_otm_slice(meta, df, source=cfg['source'], drop_strikes=cfg['drop'])
    mix = fit_lognormal_mixture(vol_slice, n_restarts=60)
    rs = calibrate_regime_switch(vol_slice, StartRegime.GROWTH)
    return TenorDiagnostics(
        label=cfg['label'], years=meta.ttm, atm_vol=100.0 * vol_slice.atm_vol(),
        frown_depth_80_150=vol_slice.frown_depth_80_150(),
        frown_depth_wing=vol_slice.frown_depth_wing(),
        skew_slope=vol_slice.skew_slope(),
        rnd_skew=mix.rnd_skew, rnd_excess_kurtosis=mix.rnd_excess_kurtosis,
        n_modes=len(mix.rnd_modes), mixture_rmse=mix.rmse_volpts,
        rs_growth_rmse=rs.rmse_volpts, rs_growth_crash_pct=rs.mean_crash_pct)


def run_term_structure() -> Tuple[pd.DataFrame, List[TenorDiagnostics]]:
    """analyze all tenors, print the term-structure table, return (table, diagnostics)."""
    diags = [analyze_tenor(cfg) for cfg in TENORS]
    table = pd.DataFrame([d.__dict__ for d in diags]).round(2)
    print(table.to_string(index=False))
    print('\nREADME §6 predicts frown depth peaking at +5.4 at 3M; the data stays <= 0.')
    return table, diags


def plot_three_tenor_figure(diags: List[TenorDiagnostics]) -> str:
    """3-panel: fanning skew, frown-depth vs README prediction, excess-kurtosis decay."""
    colors = {'20d': '#D55E00', '48d': '#0072B2', '76d': '#009E73'}
    ink, mut, grid, surf = '#1a1a1a', '#8a8a8a', '#e6e6e3', '#fcfcfb'
    plt.rcParams.update({'font.size': 10.5, 'figure.facecolor': surf, 'axes.facecolor': surf})
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5.6))
    for ax in (ax1, ax2, ax3):
        ax.grid(True, color=grid, lw=0.8, zorder=0)
        ax.set_axisbelow(True)

    for cfg in TENORS:
        meta, df = read_chain(os.path.join(DATA_DIR, cfg['csv']))
        vs = build_otm_slice(meta, df, source=cfg['source'], drop_strikes=cfg['drop'])
        c = colors[cfg['label']]
        ax1.plot(vs.moneyness, 100 * vs.iv, color=c, lw=1.3, alpha=0.5, zorder=3)
        ax1.scatter(vs.moneyness[vs.is_put], 100 * vs.iv[vs.is_put], s=32, color=c, zorder=5)
        ax1.scatter(vs.moneyness[~vs.is_put], 100 * vs.iv[~vs.is_put], s=32, marker='^', color=c, zorder=5)
        ax1.plot([], [], color=c, lw=6, label=f"{cfg['label']} ({cfg['days']}d)")
    ax1.axvline(1.0, color=mut, lw=1.0, ls=(0, (4, 3)))
    ax1.set_xlabel('moneyness  K / forward'); ax1.set_ylabel('implied volatility (%)')
    ax1.set_title('a. Skew fans out and flattens with maturity', fontweight='bold', loc='left')
    ax1.legend(frameon=False, fontsize=9)

    yrs = np.array([d.years for d in diags])
    ax2.axhline(0, color=mut, lw=1.0)
    ax2.plot(README_PRED_YEARS, README_PRED_DEPTH, color='#D55E00', lw=2, marker='D',
             label='regime-switch frown prediction (README §6)')
    ax2.plot(yrs, [d.frown_depth_80_150 for d in diags], color=ink, lw=1.6, marker='o', ms=9,
             label='real data (80/150 frown depth)')
    ax2.set_xlabel('maturity (years)'); ax2.set_ylabel('frown depth (vol pts)')
    ax2.set_title('b. The predicted 3M frown peak is absent', fontweight='bold', loc='left')
    ax2.legend(frameon=False, fontsize=8.5); ax2.set_xlim(0, 1.05)

    ax3.axhline(0, color=mut, lw=1.0)
    ax3.axhspan(-1.2, 0, color='#D55E00', alpha=0.06)
    ax3.plot(yrs, [d.rnd_excess_kurtosis for d in diags], color=ink, lw=1.6, marker='o', ms=9)
    ax3.text(0.10, -0.6, 'platykurtic — a frown needs EK < 0', color='#D55E00', fontsize=9, fontweight='bold')
    ax3.set_xlabel('maturity (years)'); ax3.set_ylabel('RND excess kurtosis')
    ax3.set_title('c. Tails go Gaussian, never platykurtic', fontweight='bold', loc='left')

    plt.tight_layout()
    path = os.path.join(FIG_DIR, 'kospi2_term_structure_3tenor.png')
    plt.savefig(path, dpi=150, bbox_inches='tight', facecolor=surf)
    plt.close(fig)
    return path


def _market_bidask_band(meta: ChainMeta, df: pd.DataFrame) -> np.ndarray:
    """listed OTM (moneyness, bid_vol, ask_vol) in vol points, from two-sided quotes."""
    rows = []
    for _, r in df.iterrows():
        k = float(r['opt_strike_px'])
        put = (r['cp'] == 'put')
        if not ((put and k < meta.forward) or ((not put) and k > meta.forward)):
            continue
        b, a = r['px_bid'], r['px_ask']
        if not (np.isfinite(b) and np.isfinite(a) and b > 0 and a >= b):
            continue
        vb = implied_vol_forward(b / meta.disc, k, meta.forward, meta.ttm, put)
        va = implied_vol_forward(a / meta.disc, k, meta.forward, meta.ttm, put)
        if np.isfinite(vb) and np.isfinite(va):
            rows.append((k / meta.forward, 100.0 * vb, 100.0 * va))
    return np.array(sorted(rows))


def _model_iv_curve(theta: np.ndarray, meta: ChainMeta, regime: int,
                    grid: np.ndarray) -> np.ndarray:
    """model implied vol (vol points) on a moneyness grid, priced from `regime`."""
    th = np.asarray(theta, float).copy()
    th[2:4] = np.clip(th[2:4], 1e-6, None)
    params = RiskNeutralParams(*th, rate=meta.r_eff)
    k = grid * meta.forward
    put = k < meta.forward
    iv = np.full(len(grid), np.nan)
    if put.any():
        iv[put] = implied_vol(params, meta.spot, k[put], meta.ttm, regime=regime, option_type='put')
    if (~put).any():
        iv[~put] = implied_vol(params, meta.spot, k[~put], meta.ttm, regime=regime, option_type='call')
    return 100.0 * iv


def plot_model_fit_by_regime() -> str:
    """per tenor: market bid/ask vs the model smile priced from regime 0 and regime 1.

    Uses the growth-regime calibrated parameters and prices the SAME parameters from each
    starting regime. From regime 0 the pending jump is the downward crash (0->1), giving
    the put skew that fits; from regime 1 the pending jump is the upward recovery (1->0),
    giving a flat/recovery-tilted shape that does not fit a downside-skewed market.
    """
    ink, mut, grid_c, surf = '#1a1a1a', '#8a8a8a', '#e6e6e3', '#fcfcfb'
    blu, grn, ver = '#0072B2', '#009E73', '#D55E00'
    plt.rcParams.update({'font.size': 10.5, 'figure.facecolor': surf, 'axes.facecolor': surf})
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.7))
    for ax, cfg in zip(axes, TENORS):
        meta, df = read_chain(os.path.join(DATA_DIR, cfg['csv']))
        vs = build_otm_slice(meta, df, source=cfg['source'], drop_strikes=cfg['drop'])
        fit = calibrate_regime_switch(vs, StartRegime.GROWTH)
        band = _market_bidask_band(meta, df)
        m, vb, va = band[:, 0], band[:, 1], band[:, 2]
        mid = 0.5 * (vb + va)
        gx = np.linspace(max(0.58, m.min() - 0.02), m.max() + 0.02, 44)
        iv0 = _model_iv_curve(fit.theta, meta, 0, gx)
        iv1 = _model_iv_curve(fit.theta, meta, 1, gx)
        ax.grid(True, color=grid_c, lw=0.8, zorder=0)
        ax.set_axisbelow(True)
        ax.axvline(1.0, color=mut, lw=1.0, ls=(0, (4, 3)), zorder=1)
        ax.errorbar(m, mid, yerr=[(mid - vb).clip(min=0), (va - mid).clip(min=0)], fmt='o', ms=4.5,
                    color=blu, ecolor=blu, elinewidth=1.1, capsize=2.5, alpha=0.9, zorder=5,
                    label='market mid + bid/ask')
        ax.plot(gx, iv0, color=grn, lw=2.4, zorder=6,
                label=f'model from regime 0 (growth) — fit, RMSE {fit.rmse_volpts:.1f}')
        ax.plot(gx, iv1, color=ver, lw=2.2, ls=(0, (6, 3)), zorder=4,
                label='model from regime 1 (stress) — same params')
        ax.set_title(f"{cfg['days']}-day expiry  (forward {meta.forward:.0f})", fontweight='bold', loc='left')
        ax.set_xlabel('moneyness  K / forward')
        ax.legend(frameon=False, fontsize=8.3, loc='upper center')
    axes[0].set_ylabel('implied volatility (%)')
    fig.suptitle('KOSPI2 model fit vs market bid/ask — smile viewed from regime 0 (growth) vs regime 1 (stress)',
                 fontsize=13, fontweight='bold', y=1.0)
    plt.tight_layout(rect=(0, 0.0, 1, 0.99))
    path = os.path.join(FIG_DIR, 'kospi2_model_fit_by_regime.png')
    plt.savefig(path, dpi=150, bbox_inches='tight', facecolor=surf)
    plt.close(fig)
    return path
