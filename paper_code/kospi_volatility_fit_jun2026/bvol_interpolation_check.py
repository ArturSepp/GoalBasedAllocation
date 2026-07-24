"""
compare the Bloomberg BVOL fitted-surface moneyness grid against the actual listed
bid/ask implied vols, at the tenor where a full listed chain nearly coincides with a
grid column (the 20d chain sits one day from the BVOL 3W column).

Finding: the interpolated surface sits OUTSIDE the tradeable market where checkable
(ATM cell 104 vs a market of 81-85; the 81-95% put strikes 5-10 vol below the bid), and
the frown's left wall lives below 80% moneyness where the grid has no node at all. So
the 'frown' is a property of the smoothing plus off-grid extrapolation, not the quotes.
"""
from __future__ import annotations

# packages
import os
import numpy as np
import pandas as pd
from typing import Dict, Tuple
from scipy.optimize import brentq
from scipy.stats import norm
# local
from vol_surface_utils import read_chain

_HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(_HERE, 'data')

# nearest BVOL grid tenor for each listed chain (days apart)
GRID_MATCH = {'kospi2_20260813_20d.csv': '3W',   # 20 vs 21 days
              'kospi2_20260910_48d.csv': '2M',   # 48 vs 61 days
              'kospi2_20261008_76d.csv': '3M'}   # 76 vs 91 days


def read_bvol_grid(csv_path: str) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """parse the transcribed OMON moneyness grid; return (moneyness, {tenor: vols})."""
    df = pd.read_csv(csv_path, comment='#')
    moneyness = np.array([float(c) for c in df.columns[1:]]) / 100.0
    grid = {row['tenor']: row.iloc[1:].to_numpy(dtype=float) for _, row in df.iterrows()}
    return moneyness, grid


def _quote_vol(price: float, strike: float, meta, is_put: bool) -> float:
    if not np.isfinite(price) or price <= 0:
        return np.nan
    fv = price / meta.disc
    intrinsic = max((strike - meta.forward) if is_put else (meta.forward - strike), 0.0)
    if fv <= intrinsic + 1e-9:
        return np.nan

    def bs(v: float) -> float:
        sd = v * np.sqrt(meta.ttm)
        d1 = (np.log(meta.forward / strike) + 0.5 * sd ** 2) / sd
        call = meta.forward * norm.cdf(d1) - strike * norm.cdf(d1 - sd)
        return (call - (meta.forward - strike)) if is_put else call
    try:
        return 100.0 * brentq(lambda v: bs(v) - fv, 1e-4, 8.0, xtol=1e-11)
    except ValueError:
        return np.nan


def compare_chain_to_grid(chain_csv: str, grid_csv: str) -> pd.DataFrame:
    """per-strike: market bid/ask vol vs BVOL-grid-interpolated vol (spot moneyness)."""
    meta, df = read_chain(os.path.join(DATA_DIR, chain_csv))
    gm, grid = read_bvol_grid(os.path.join(DATA_DIR, grid_csv))
    tenor = GRID_MATCH[chain_csv]
    rows = []
    for _, r in df.iterrows():
        k = float(r['opt_strike_px'])
        is_put = (r['cp'] == 'put')
        if not ((is_put and k < meta.forward) or ((not is_put) and k > meta.forward)):
            continue
        vb = _quote_vol(r['px_bid'], k, meta, is_put)
        va = _quote_vol(r['px_ask'], k, meta, is_put)
        if not (np.isfinite(vb) and np.isfinite(va)):
            continue
        m_spot = k / meta.spot
        on_grid = 0.80 <= m_spot <= 1.20
        gv = float(np.interp(m_spot, gm, grid[tenor])) if on_grid else np.nan
        rows.append(dict(strike=int(k), moneyness=round(m_spot, 3),
                         bid_vol=round(vb, 1), ask_vol=round(va, 1),
                         bvol_grid=round(gv, 1) if on_grid else np.nan,
                         status='on-grid' if on_grid else 'OFF-GRID (extrapolated)'))
    return pd.DataFrame(rows).sort_values('strike').reset_index(drop=True)
