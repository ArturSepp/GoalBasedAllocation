"""
reusable engine for building a clean OTM implied-vol slice from a bbg-fetch option
chain and fitting a forward-constrained two-lognormal mixture.

The chain CSV format is the one written by `bbg_fetch.option_chain.run().to_csv`:
a block of commented '# key=value' metadata (spot, year_fraction, forward, rate,
r2, num_strikes_used) followed by the option table.

Everything here is in forward (undiscounted) space and is discount-invariant: implied
vols depend only on (forward F, strike K, ttm T, forward option value), so the small
dividend/borrow carry never enters the vol fit. The forward reproduced by the fetcher
is matched exactly by setting the model carry to r_eff = ln(F/S)/T when this slice is
handed to the regime-switch pricer (see regime_switch_calibration.py).
"""
from __future__ import annotations

# packages
import numpy as np
import pandas as pd
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple
from scipy.optimize import brentq, minimize
from scipy.special import erfc
from scipy.signal import argrelextrema


class PriceSource(str, Enum):
    """which quote to price each option off."""
    LAST = 'last'      # parity-consistent (the fetcher recovers F from last)
    MID = 'mid'        # 0.5*(bid+ask); use when last prints are stale
    BID = 'bid'
    ASK = 'ask'


@dataclass(frozen=True)
class ChainMeta:
    """recovered pricing inputs from the chain header (immutable snapshot)."""
    spot: float
    ttm: float                 # year fraction
    forward: float             # parity-recovered forward
    rate: float                # parity-recovered funding rate (ill-conditioned at short T)
    r2: float
    num_strikes_used: int

    @property
    def r_eff(self) -> float:
        """carry that reproduces the market forward, r_eff = ln(F/S)/T."""
        return float(np.log(self.forward / self.spot) / self.ttm)

    @property
    def disc(self) -> float:
        """discount factor exp(-rate*T)."""
        return float(np.exp(-self.rate * self.ttm))


@dataclass(frozen=True)
class VolSlice:
    """a clean out-of-the-money implied-vol slice (immutable snapshot)."""
    meta: ChainMeta
    strike: np.ndarray         # OTM strikes, ascending
    moneyness: np.ndarray      # strike / forward
    is_put: np.ndarray         # bool, True where strike < forward
    iv: np.ndarray             # implied vol from the chosen price source
    source: PriceSource

    def atm_vol(self) -> float:
        """vol interpolated to moneyness 1.0."""
        return float(np.interp(1.0, self.moneyness, self.iv))

    def frown_depth_wing(self) -> float:
        """ATM minus the average of the two extreme wings, in vol points.

        > 0 => frown (ATM bid over both wings); < 0 => skew/smile.
        """
        return 100.0 * (self.atm_vol() - 0.5 * (self.iv[0] + self.iv[-1]))

    def frown_depth_80_150(self) -> float:
        """ATM minus mean of the 80% and 150% wings (the README §6 definition)."""
        w80 = float(np.interp(0.80, self.moneyness, self.iv))
        w150 = float(np.interp(1.50, self.moneyness, self.iv))
        return 100.0 * (self.atm_vol() - 0.5 * (w80 + w150))

    def skew_slope(self) -> float:
        """least-squares d(iv)/d(ln moneyness) in vol points, over the whole slice."""
        return 100.0 * float(np.polyfit(np.log(self.moneyness), self.iv, 1)[0])


# ==============================================================================
# BLACK-SCHOLES IN FORWARD SPACE (erfc-based ncdf for optimiser inner loops)
# ==============================================================================

def _ncdf(x: np.ndarray) -> np.ndarray:
    return 0.5 * erfc(-np.asarray(x) / np.sqrt(2.0))


def _npdf(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x)
    return np.exp(-0.5 * x * x) / np.sqrt(2.0 * np.pi)


def bs_forward(strike: np.ndarray,
               vol: np.ndarray,
               forward: float,
               ttm: float,
               is_put: np.ndarray,
               ) -> np.ndarray:
    """Black forward (undiscounted) option value. Put via call-minus-forward parity."""
    sd = vol * np.sqrt(ttm)
    d1 = (np.log(forward / strike) + 0.5 * sd ** 2) / sd
    call = forward * _ncdf(d1) - strike * _ncdf(d1 - sd)
    return np.where(is_put, call - (forward - strike), call)


def bs_vega_forward(strike: np.ndarray,
                    vol: np.ndarray,
                    forward: float,
                    ttm: float,
                    ) -> np.ndarray:
    """Black forward vega, dV/dvol (per unit vol)."""
    sd = vol * np.sqrt(ttm)
    d1 = (np.log(forward / strike) + 0.5 * sd ** 2) / sd
    return forward * _npdf(d1) * np.sqrt(ttm)


def implied_vol_forward(fwd_value: float,
                        strike: float,
                        forward: float,
                        ttm: float,
                        is_put: bool,
                        ) -> float:
    """invert a forward option value to Black implied vol; nan if below intrinsic."""
    intrinsic = max((strike - forward) if is_put else (forward - strike), 0.0)
    if not np.isfinite(fwd_value) or fwd_value <= intrinsic + 1e-9:
        return np.nan
    k_arr, put_arr = np.array([strike]), np.array([is_put])
    try:
        return brentq(lambda v: float(bs_forward(k_arr, v, forward, ttm, put_arr)[0]) - fwd_value,
                      1e-4, 8.0, xtol=1e-11)
    except ValueError:
        return np.nan


# ==============================================================================
# CHAIN PARSING AND OTM SLICE CONSTRUCTION
# ==============================================================================

def read_chain(csv_path: str) -> Tuple[ChainMeta, pd.DataFrame]:
    """parse a bbg-fetch chain: '# key=value' header metadata + the option table."""
    meta: Dict[str, float] = {}
    with open(csv_path) as fh:
        for line in fh:
            if not line.startswith('#'):
                break
            if '=' in line:
                key, val = line[1:].split('=', 1)
                meta[key.strip()] = float(val)
    required = ('spot', 'year_fraction', 'forward', 'rate', 'r2', 'num_strikes_used')
    missing = [k for k in required if k not in meta]
    if missing:
        raise ValueError(f"chain header missing keys {missing!r} in {csv_path!r}")
    chain_meta = ChainMeta(spot=meta['spot'], ttm=meta['year_fraction'],
                           forward=meta['forward'], rate=meta['rate'], r2=meta['r2'],
                           num_strikes_used=int(meta['num_strikes_used']))
    df = pd.read_csv(csv_path, comment='#')
    df['cp'] = df['opt_put_call'].str.lower()
    return chain_meta, df


def parity_residual(meta: ChainMeta, df: pd.DataFrame) -> Tuple[float, int]:
    """put-call parity check on last prices: max|C-P - disc*(F-K)| and n paired strikes."""
    piv = df.pivot_table(index='opt_strike_px', columns='cp', values='px_last')
    piv = piv.dropna(subset=['call', 'put'])
    resid = (piv['call'] - piv['put']).values - meta.disc * (meta.forward - piv.index.values)
    return float(np.max(np.abs(resid))), int(len(piv))


def _pick_price(row: pd.Series, source: PriceSource) -> float:
    if source == PriceSource.LAST:
        px = row['px_last']
        b, a = row['px_bid'], row['px_ask']
        # drop last prints that fall outside the current two-sided market (stale)
        if np.isfinite(b) and np.isfinite(a) and np.isfinite(px) and (px < b - 1e-9 or px > a + 1e-9):
            return np.nan
        return px
    if source == PriceSource.BID:
        return row['px_bid']
    if source == PriceSource.ASK:
        return row['px_ask']
    b, a = row['px_bid'], row['px_ask']
    return 0.5 * (b + a) if np.isfinite(b) and np.isfinite(a) else np.nan


def build_otm_slice(meta: ChainMeta,
                    df: pd.DataFrame,
                    source: PriceSource = PriceSource.LAST,
                    drop_strikes: Optional[List[float]] = None,
                    ) -> VolSlice:
    """build a clean OTM slice: puts below the forward, calls above, IV per strike.

    Prices are converted to forward value (px / disc) before inversion, so the IV is
    independent of the funding rate. Strikes with no usable price, an uninvertible
    (sub-intrinsic) value, or in `drop_strikes` are skipped.
    """
    drop = set(drop_strikes or [])
    rows = []
    for _, r in df.iterrows():
        k = float(r['opt_strike_px'])
        put = (r['cp'] == 'put')
        is_otm = (put and k < meta.forward) or ((not put) and k > meta.forward)
        if not is_otm or k in drop:
            continue
        px = _pick_price(r, source)
        if not np.isfinite(px) or px <= 0:
            continue
        iv = implied_vol_forward(px / meta.disc, k, meta.forward, meta.ttm, put)
        if not np.isfinite(iv):
            continue
        rows.append((k, k / meta.forward, put, iv))
    if not rows:
        raise ValueError(f"no invertible OTM options for source={source.value!r}")
    rows.sort(key=lambda t: t[0])
    arr = np.array(rows, dtype=object)
    return VolSlice(meta=meta,
                    strike=np.array([x[0] for x in rows], dtype=float),
                    moneyness=np.array([x[1] for x in rows], dtype=float),
                    is_put=np.array([x[2] for x in rows], dtype=bool),
                    iv=np.array([x[3] for x in rows], dtype=float),
                    source=source)


# ==============================================================================
# FORWARD-CONSTRAINED TWO-LOGNORMAL MIXTURE (the benchmark of the study)
# ==============================================================================

@dataclass(frozen=True)
class MixtureFit:
    """result of the forward-constrained two-lognormal mixture fit."""
    weight_down: float         # p on the low-forward component
    forward_down: float
    vol_down: float
    forward_up: float
    vol_up: float
    rmse_volpts: float
    rnd_skew: float
    rnd_excess_kurtosis: float
    rnd_modes: np.ndarray      # S/F locations of the density modes


def fit_lognormal_mixture(vol_slice: VolSlice,
                          n_restarts: int = 120,
                          seed: int = 0,
                          ) -> MixtureFit:
    """fit p*LN(F1,s1) + (1-p)*LN(F2,s2) with p*F1 + (1-p)*F2 = F (martingale).

    Objective: vega-weighted price-space least squares. Frown <=> negative excess
    kurtosis (platykurtic, in practice bimodal); skew/smile <=> positive.
    """
    F, T = vol_slice.meta.forward, vol_slice.meta.ttm
    K, is_put, mkt_iv = vol_slice.strike, vol_slice.is_put, vol_slice.iv
    mkt_px = bs_forward(K, mkt_iv, F, T, is_put)
    vega = bs_vega_forward(K, mkt_iv, F, T)

    def mix_prices(theta: np.ndarray) -> Tuple[np.ndarray, float]:
        p, f1r, s1, s2 = theta
        f1 = f1r * F
        f2 = (F - p * f1) / (1.0 - p)
        px = (p * bs_forward(K, np.full_like(K, s1), f1, T, is_put)
              + (1.0 - p) * bs_forward(K, np.full_like(K, s2), f2, T, is_put))
        return px, f2

    def objective(theta: np.ndarray) -> float:
        p, f1r, s1, s2 = theta
        if not (0.02 < p < 0.98 and 0.3 < f1r < 2.5 and 0.02 < s1 < 4.0 and 0.02 < s2 < 4.0):
            return 1e9
        px, f2 = mix_prices(theta)
        if f2 <= 0.05 * F:
            return 1e9
        return 1e4 * float(np.mean(((px - mkt_px) / vega) ** 2))

    best, rng = None, np.random.default_rng(seed)
    for _ in range(n_restarts):
        x0 = [rng.uniform(0.15, 0.85), rng.uniform(0.6, 1.4),
              rng.uniform(0.1, 1.0), rng.uniform(0.1, 1.0)]
        res = minimize(objective, x0, method='Nelder-Mead',
                       options=dict(maxiter=4000, fatol=1e-13, xatol=1e-9))
        if best is None or res.fun < best.fun:
            best = res

    p, f1r, s1, s2 = best.x
    px, f2 = mix_prices(best.x)
    f1 = f1r * F
    fit_iv = np.array([implied_vol_forward(px[i], K[i], F, T, bool(is_put[i]))
                       for i in range(len(K))])
    rmse = 100.0 * float(np.sqrt(np.nanmean((fit_iv - mkt_iv) ** 2)))
    skew, ek, modes = _rnd_moments(p, f1, s1, f2, s2, F, T)
    # order components so 'down' is the lower forward
    if f1 <= f2:
        return MixtureFit(p, f1, s1, f2, s2, rmse, skew, ek, modes)
    return MixtureFit(1.0 - p, f2, s2, f1, s1, rmse, skew, ek, modes)


def _rnd_moments(p: float, f1: float, s1: float, f2: float, s2: float,
                 forward: float, ttm: float) -> Tuple[float, float, np.ndarray]:
    """skew, excess kurtosis and modal S/F locations of the log-return mixture RND."""
    x = np.linspace(-4.0, 3.0, 600001)

    def lognormal_pdf(fi: float, si: float) -> np.ndarray:
        m = np.log(fi / forward) - 0.5 * si * si * ttm
        return _npdf((x - m) / (si * np.sqrt(ttm))) / (si * np.sqrt(ttm))

    q = p * lognormal_pdf(f1, s1) + (1.0 - p) * lognormal_pdf(f2, s2)
    q /= np.trapezoid(q, x)
    m1 = np.trapezoid(x * q, x)
    central = lambda n: np.trapezoid((x - m1) ** n * q, x)
    sd = np.sqrt(central(2))
    modes = np.exp(x[argrelextrema(q, np.greater)[0]])
    return central(3) / sd ** 3, central(4) / sd ** 4 - 3.0, np.round(modes, 3)
