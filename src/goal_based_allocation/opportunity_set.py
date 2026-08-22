"""Investment Opportunity Set Construction.

Two-step client specification framework:

Step 1: Advisor/RM specifies:
  i.   ω₀ ∈ [0.5, 1.5] — initial risky allocation (1.0 = fully invested, >1 = leveraged)
  ii.  c ∈ [0%, 5%] — annual consumption rate
  iii. q ∈ [0, 1] — equity share of risky: w_eq = q*(1-w_bd), w_pe = (1-q)*(1-w_bd)
  iv.  q_dd ∈ [1, 3] — drawdown scale: L_T = Π₀·exp(-q_dd·σ_unc)

This constructs the investment opportunity set: r_impl(w_bd) curve and CDFs.

Step 2: Client selects a point on the curve (an r_impl value),
which fixes w_bd, allocation, L_T, g, and the full distribution.
"""
import numpy as np
from dataclasses import dataclass
from .riccati_solver import find_ell, gap_process_asset
from .regime_switch_paper import (compute_survival, compute_tilted_survival,
    compute_overshoot_density, compute_density, bh_moments_rsjd)
from .client_solver import build_effective_asset, portfolio_sigma_unc

PI0 = 100.0; T = 10.0; r = 0.02


@dataclass
class AdvisorSpec:
    """Step 1: Advisor/RM specification."""
    omega_0: float = 1.0     # initial risky allocation [0.5, 1.5]
    c: float = 0.0           # consumption rate [0%, 5%]
    q: float = 2/3           # equity share of risky [0, 1]
    q_dd: float = 2.0        # drawdown scale [1, 3]


def portfolio_weights(w_bd, q):
    """Compute (w_bd, w_eq, w_pe) from bond weight and equity share."""
    w_eq = q * (1 - w_bd)
    w_pe = (1 - q) * (1 - w_bd)
    return w_bd, w_eq, w_pe


def compute_opportunity_point(w_bd, spec: AdvisorSpec):
    """Compute one point on the investment opportunity set.
    
    For given bond weight, find g such that ω*(0) = spec.omega_0,
    then compute all analytical quantities.
    """
    _, w_eq, w_pe = portfolio_weights(w_bd, spec.q)
    c = spec.c; q_dd = spec.q_dd; omega_0_target = spec.omega_0
    r_h = max(r, c); r_c = r_h - c

    sig_unc = portfolio_sigma_unc(w_eq, w_pe)
    x_25 = PI0 * np.exp(-q_dd * sig_unc)
    eff_temp = build_effective_asset(w_eq, w_pe, 1.5)
    eta_port = eff_temp.params.eta0
    k = (np.log(PI0) + r_c * T - np.log(x_25)) / eta_port
    eff = build_effective_asset(w_eq, w_pe, k)

    # Bisect on target_return to achieve ω*(0) = omega_0_target
    # ω*(0) = |ω*_a| · (Π*(0)/Π₀ - 1) = omega_0_target
    lo, hi = 0.001, 0.60
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        try:
            ell, ric = find_ell(eff, T, mid, r_h, c)
            d = ric.derived_at_tau(T)
            wa = abs(d['w_a'][0])
            Pi_star_0 = d['Pi_star'][0]
            om = wa * (Pi_star_0 / PI0 - 1)
            if om < omega_0_target:
                lo = mid
            else:
                hi = mid
            if abs(om - omega_0_target) < 0.001:
                break
        except:
            hi = mid

    try:
        ell, ric = find_ell(eff, T, mid, r_h, c)
    except:
        return None

    gap = gap_process_asset(ric)
    d0 = ric.derived_at_tau(T)
    dT = ric.derived_at_tau(0)
    wa = abs(d0['w_a'][0])
    Pi_star_0 = d0['Pi_star'][0]
    PiT = dT['Pi_star'][0]
    L_T = eff.pi_floor * np.exp(r_c * T)
    B_T = PiT - L_T
    g = np.log(PiT / PI0) / T
    omega_0_actual = wa * (Pi_star_0 / PI0 - 1)

    # Six Laplace building blocks
    S = compute_survival(T, gap.x0, gap)
    TS1 = compute_tilted_survival(T, gap.x0, gap, 1.0)
    TS2 = compute_tilted_survival(T, gap.x0, gap, 2.0)
    d_ov = np.linspace(0.001, 8, 400)
    f_ov = compute_overshoot_density(T, d_ov, gap)
    O = np.trapezoid(f_ov, d_ov)
    F = max(0, 1 - S - O)
    eta_eff = gap.params.eta0
    O1 = O / (1 - eta_eff)
    O2 = O / (1 - 2 * eta_eff)

    # Moments
    E1 = PiT * S - B_T * TS1 + L_T * F + PiT * O - B_T * O1
    E2 = (PiT**2 * S - 2*PiT*B_T*TS1 + B_T**2*TS2 +
          L_T**2 * F +
          PiT**2 * O - 2*PiT*B_T*O1 + B_T**2*O2)
    Var = E2 - E1**2
    Std = np.sqrt(max(0, Var))

    # Conditional on survival
    Es = PiT - B_T * TS1 / S
    E2s = PiT**2 - 2*PiT*B_T*TS1/S + B_T**2*TS2/S
    Stds = np.sqrt(max(0, E2s - Es**2))

    r_impl = np.log(E1 / PI0) / T

    # BH benchmark: exact RS-JD moments via matrix exponential
    bh = bh_moments_rsjd(T, PI0, eff, c=c)
    E_BH = bh['E']
    Std_BH = bh['Std']
    r_impl_BH = bh['r_impl']

    # Discounted consumption
    C_disc = 0.0
    if c > 0:
        n_q = 20
        tg = np.linspace(0.01, T - 0.01, n_q)
        E_Pi_t = np.zeros(n_q)
        for j, t in enumerate(tg):
            dt = ric.derived_at_tau(T - t)
            Pst = dt['Pi_star'][0]
            Lt = eff.pi_floor * np.exp(r_c * t)
            Bt = Pst - Lt
            St = compute_survival(t, gap.x0, gap)
            TS1t = compute_tilted_survival(t, gap.x0, gap, 1.0)
            E_Pi_t[j] = Pst*St - max(0, Bt)*TS1t + Lt*(1 - St)
        tf = np.concatenate([[0], tg, [T]])
        Ef = np.concatenate([[PI0], E_Pi_t, [E1]])
        C_disc = c * np.trapezoid(Ef * np.exp(-r*(T-tf)), tf)

    TV = E1 + C_disc
    if c > 0:
        Ef_BH = np.array([bh_moments_rsjd(t, PI0, eff, c=c)['E'] for t in tf])
        C_disc_BH = c * np.trapezoid(Ef_BH * np.exp(-r*(T-tf)), tf)
    else:
        C_disc_BH = 0.0
    TV_BH = E_BH + C_disc_BH

    # CDF
    x_max = min(8.0, gap.x0 + 6*max(gap.params.sigma0, gap.params.sigma1)*np.sqrt(T))
    x_grid = np.linspace(0.001, x_max, 600)
    d0_den, d1_den = compute_density(T, x_grid, gap)
    dg = d0_den + d1_den

    Pi_lo = max(0.1, PiT - B_T*np.exp(x_max))
    Pi_cdf = np.linspace(Pi_lo, PiT + 1, 1500)
    cdf = np.zeros_like(Pi_cdf)
    for j, pi in enumerate(Pi_cdf):
        if pi >= PiT: cdf[j] = 1.0; continue
        gap_arg = PiT - pi
        if gap_arg <= 0: cdf[j] = 1.0; continue
        xv = np.log(B_T / gap_arg)
        if xv <= 0:
            dv = -xv; m = d_ov >= dv
            cdf[j] = np.trapezoid(f_ov[m], d_ov[m]) if np.any(m) else O
        else:
            m = x_grid <= xv
            cdf[j] = O + F + (np.trapezoid(dg[m], x_grid[m]) if np.any(m) else 0)
    cdf = np.clip(cdf, 0, 1)

    def quantile(p):
        idx = np.searchsorted(cdf, p)
        if idx == 0: return Pi_cdf[0]
        if idx >= len(Pi_cdf): return Pi_cdf[-1]
        f0, f1 = cdf[idx-1], cdf[idx]
        if f1 > f0:
            return Pi_cdf[idx-1] + (p-f0)/(f1-f0)*(Pi_cdf[idx]-Pi_cdf[idx-1])
        return Pi_cdf[idx]

    return {
        'w_bd': w_bd, 'w_eq': w_eq, 'w_pe': w_pe,
        'c': c, 'q_dd': q_dd, 'omega_0': omega_0_actual,
        'g': g, 'wa': wa, 'k': k, 'sig_unc': sig_unc,
        'PiT': PiT, 'L_T': L_T, 'r_impl': r_impl,
        'E': E1, 'Std': Std, 'Es': Es, 'Stds': Stds,
        'S': S, 'F': F, 'O': O,
        'E_BH': E_BH, 'Std_BH': Std_BH, 'r_impl_BH': r_impl_BH,
        'C_disc': C_disc, 'TV': TV, 'TV_BH': TV_BH,
        'floor_cost_pct': (TV_BH - TV) / TV_BH if TV_BH > 0 else 0,
        'q5': quantile(0.05), 'q25': quantile(0.25),
        'q50': quantile(0.50), 'q75': quantile(0.75),
        'q95': quantile(0.95),
        'Pi_cdf': Pi_cdf, 'cdf': cdf,
    }


def build_opportunity_set(spec: AdvisorSpec, w_vals=None):
    """Build the full investment opportunity set.
    
    Returns list of dicts, one per bond weight, sorted by r_impl.
    """
    if w_vals is None:
        w_vals = np.arange(0.95, -0.01, -0.05)

    results = []
    for w in w_vals:
        res = compute_opportunity_point(w, spec)
        if res is not None:
            results.append(res)

    # Sort by r_impl
    results.sort(key=lambda x: x['r_impl'])
    return results


def select_portfolio(opportunity_set, r_impl_target):
    """Step 2: Client selects r_impl → interpolate to find portfolio."""
    r_impls = np.array([p['r_impl'] for p in opportunity_set])
    idx = np.searchsorted(r_impls, r_impl_target)
    if idx == 0:
        return opportunity_set[0]
    if idx >= len(opportunity_set):
        return opportunity_set[-1]
    # Linear interpolation on w_bd
    lo, hi = opportunity_set[idx-1], opportunity_set[idx]
    frac = (r_impl_target - lo['r_impl']) / (hi['r_impl'] - lo['r_impl'])
    w_bd_interp = lo['w_bd'] + frac * (hi['w_bd'] - lo['w_bd'])
    # Recompute at interpolated w_bd
    spec = AdvisorSpec(omega_0=lo['omega_0'], c=lo['c'],
                       q=lo['w_eq']/(lo['w_eq']+lo['w_pe']) if lo['w_eq']+lo['w_pe']>0 else 2/3,
                       q_dd=lo['q_dd'])
    return compute_opportunity_point(w_bd_interp, spec)
