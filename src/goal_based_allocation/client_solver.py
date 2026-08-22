"""Client Specification Solver v3.

Client inputs: (c, g, q_dd, x_50, x_75)
  c      = consumption rate
  g      = growth target (post consumption)
  q_dd   = drawdown scale (default 2): x_25 = Π₀·exp(-q_dd·σ_unc)
  x_50   = median wealth target
  x_75   = 75th percentile wealth target

Analytical reductions:
  r_h = max(r, c)               (hurdle rate)
  ℓ = g + c - r                 (internal Riccati target)
  Π*_T = Π₀·exp(g·T)            (target terminal wealth)
  x_25 = Π₀·exp(-q_dd·σ_unc)    (floor from drawdown scale, depends on weights)
  k = (ln(Π₀) + r_c·T - ln(x_25)) / η_port

Remaining 2×2: solve (w_eq, w_pe) from (x_50, x_75).
"""
import numpy as np
from dataclasses import dataclass
from scipy.optimize import minimize
from scipy.integrate import nquad, quad
from .regime_switch_paper import (RegimeSwitchParams, AssetSpecification,
    create_paper_assets, compute_density, compute_survival,
    compute_overshoot_density)
from .riccati_solver import find_ell, gap_process_asset
from .mandate_utils import mandate_effective_asset, RHO

PI0 = 100.0; T = 10.0; r = 0.02
p1 = 1.0 / 1.1; p2 = 0.1 / 1.1  # stationary regime probs


def portfolio_eta_quadrature(weights, etas, crash=True):
    """Compute portfolio jump size via numerical integration (paper eq:eta_port).

    For crash (J_j ~ -Exp(η_j)):
      η_port = -E[ln(Σ w_j exp(J_j))] = -E[ln(Σ w_j exp(-X_j))]
    For recovery (J_j ~ +Exp(η_j)):
      η_port = E[ln(Σ w_j exp(J_j))] = E[ln(Σ w_j exp(+X_j))]
    where X_j ~ Exp(1/η_j) are independent.

    Uses scipy.integrate for exact (deterministic) computation.
    """
    active = [(weights[i], etas[i]) for i in range(len(weights))
              if weights[i] > 1e-10 and etas[i] > 1e-12]
    w_rest = 1.0 - sum(wj for wj, _ in active)

    if len(active) == 0:
        return 0.0

    if len(active) == 1:
        wj, ej = active[0]
        if abs(w_rest) < 1e-10:
            return ej  # single asset, exact
        def integrand1(x):
            val = w_rest + wj * (np.exp(-x) if crash else np.exp(x))
            if val <= 0: return 0.0
            logval = -np.log(val) if crash else np.log(val)
            return logval * (1/ej) * np.exp(-x/ej)
        result, _ = quad(integrand1, 0, max(50*ej, 20), limit=200)
        return result

    ws = [wj for wj, _ in active]
    es = [ej for _, ej in active]
    n = len(active)

    def integrand(*xs):
        if crash:
            val = w_rest + sum(ws[i] * np.exp(-xs[i]) for i in range(n))
            if val <= 0: return 0.0
            logval = -np.log(val)
        else:
            val = w_rest + sum(ws[i] * np.exp(xs[i]) for i in range(n))
            logval = np.log(val)
        density = 1.0
        for i in range(n):
            density *= (1/es[i]) * np.exp(-xs[i]/es[i])
        return logval * density

    ranges = [[0, max(50*es[i], 20)] for i in range(n)]
    result, _ = nquad(integrand, ranges,
                      opts={'limit': 150 if n <= 2 else 80})
    return result


@dataclass
class ClientProfile:
    """Client-specified distribution targets."""
    name: str
    c: float            # consumption rate
    g: float            # growth target (post consumption)
    x_50: float         # median wealth target
    x_75: float         # 75th percentile wealth target
    q_dd: float = 2.0   # drawdown scale: x_25 = Π₀·exp(-q_dd·σ_unc)


def build_effective_asset(w_eq, w_pe, k):
    """Build mandate effective asset from (w_eq, w_pe, k)."""
    assets = create_paper_assets()
    al = [assets['bonds'], assets['equity'], assets['private_equity']]
    w = np.array([1.0 - w_eq - w_pe, w_eq, w_pe])

    sig_g = np.array([a.params.sigma0 for a in al])
    sig_s = np.array([a.params.sigma1 for a in al])
    sp_g = np.sqrt(w @ (np.outer(sig_g, sig_g) * RHO) @ w)
    sp_s = np.sqrt(w @ (np.outer(sig_s, sig_s) * RHO) @ w)
    mu_g = w @ np.array([a.mu_growth for a in al])
    mu_s = w @ np.array([a.mu_stress for a in al])

    rng = np.random.default_rng(42); n = 500_000
    etas0 = [a.params.eta0 for a in al]
    etas1 = [a.params.eta1 for a in al]
    eta0 = portfolio_eta_quadrature(w, etas0, crash=True)
    eta1 = portfolio_eta_quadrature(w, etas1, crash=False)

    par = RegimeSwitchParams(sigma0=sp_g, sigma1=sp_s,
        lambda01=0.1, lambda10=1.0, eta0=eta0, eta1=eta1)
    L0 = PI0 * np.exp(-k * eta0)

    return AssetSpecification('client', par, mu_growth=mu_g,
        mu_stress=mu_s, pi0=PI0, pi_floor=L0)


def portfolio_sigma_unc(w_eq, w_pe):
    """Compute unconditional portfolio volatility (diffusion only)."""
    assets = create_paper_assets()
    al = [assets['bonds'], assets['equity'], assets['private_equity']]
    w = np.array([1.0 - w_eq - w_pe, w_eq, w_pe])
    sig_g = np.array([a.params.sigma0 for a in al])
    sig_s = np.array([a.params.sigma1 for a in al])
    var_g = w @ (np.outer(sig_g, sig_g) * RHO) @ w
    var_s = w @ (np.outer(sig_s, sig_s) * RHO) @ w
    return np.sqrt(p1 * var_g + p2 * var_s)


def compute_distribution(w_eq, w_pe, k, ell, r_h, c_rate):
    """Compute full MV-optimal distribution and extract quantiles."""
    r_c = r_h - c_rate
    eff = build_effective_asset(w_eq, w_pe, k)
    _, ric = find_ell(eff, T, ell, r_h, c_rate)
    gap = gap_process_asset(ric)

    Pi_star_T = ric.derived_at_tau(0)['Pi_star'][0]
    Pi_star_0 = ric.derived_at_tau(T)['Pi_star'][0]
    B0 = Pi_star_0 - eff.pi_floor
    B_T = B0 * np.exp(r_c * T)
    L_T = eff.pi_floor * np.exp(r_c * T)

    surv = compute_survival(T, gap.x0, gap)

    x_max = min(8.0, gap.x0 + 6 * max(gap.params.sigma0,
                gap.params.sigma1) * np.sqrt(T))
    x_grid = np.linspace(0.001, x_max, 600)
    d0, d1 = compute_density(T, x_grid, gap)
    dg = d0 + d1

    d_ov_grid = np.linspace(0.001, min(8.0, x_max), 400)
    f_ov = compute_overshoot_density(T, d_ov_grid, gap)
    over_mass = np.trapezoid(f_ov, d_ov_grid)
    floor_mass = max(0, 1.0 - surv - over_mass)

    # Expected wealth
    Pi_survived = Pi_star_T - B_T * np.exp(-x_grid)
    E_surv = np.trapezoid(Pi_survived * dg, x_grid)
    E_floor = L_T * floor_mass
    Pi_over = Pi_star_T - B_T * np.exp(d_ov_grid)
    E_over = np.trapezoid(Pi_over * f_ov, d_ov_grid)
    mu = E_surv + E_floor + E_over

    # CDF on wealth grid
    Pi_lo = max(0.1, Pi_star_T - B_T * np.exp(x_max))
    Pi_grid = np.linspace(Pi_lo, Pi_star_T, 1000)
    cdf = np.zeros_like(Pi_grid)
    for i, pi in enumerate(Pi_grid):
        if pi >= Pi_star_T:
            cdf[i] = 1.0; continue
        gap_arg = Pi_star_T - pi
        if gap_arg <= 0:
            cdf[i] = 1.0; continue
        x_val = np.log(B_T / gap_arg)
        if x_val <= 0:
            d_val = -x_val
            m = d_ov_grid >= d_val
            cdf[i] = np.trapezoid(f_ov[m], d_ov_grid[m]) if np.any(m) else over_mass
        else:
            m = x_grid <= x_val
            surv_below = np.trapezoid(dg[m], x_grid[m]) if np.any(m) else 0
            cdf[i] = over_mass + floor_mass + surv_below
    cdf = np.clip(cdf, 0, 1)

    def quantile(p):
        idx = np.searchsorted(cdf, p)
        if idx == 0: return Pi_grid[0]
        if idx >= len(Pi_grid): return Pi_grid[-1]
        f0, f1 = cdf[idx-1], cdf[idx]
        if f1 > f0:
            return Pi_grid[idx-1] + (p - f0) / (f1 - f0) * (Pi_grid[idx] - Pi_grid[idx-1])
        return Pi_grid[idx]

    wa = abs(ric.derived_at_tau(T)['w_a'][0])

    return {
        'q10': quantile(0.10), 'q25': quantile(0.25),
        'q50': quantile(0.50), 'q75': quantile(0.75),
        'q90': quantile(0.90),
        'mu': mu, 'surv': surv, 'over': over_mass, 'floor': floor_mass,
        'Pi_star_T': Pi_star_T, 'L_T': L_T, 'B_T': B_T,
        'w_a': wa, 'eta_port': eff.params.eta0,
        'sigma_port': [eff.params.sigma0, eff.params.sigma1],
        'k': k, 'x_25_target': L_T,
    }


def solve_client(profile: ClientProfile, n_grid=8, verbose=True):
    """Solve for (w_eq, w_pe) given client profile.

    For each candidate (w_eq, w_pe):
      1. σ_unc = portfolio_sigma_unc(w_eq, w_pe)
      2. x_25 = Π₀·exp(-q_dd·σ_unc)
      3. k = (ln(Π₀) + r_c·T - ln(x_25)) / η_port
      4. Compute MV distribution → extract q50, q75
    Minimize |q50 - x_50|² + |q75 - x_75|²
    """
    c_rate = profile.c
    g = profile.g
    r_h = max(r, c_rate)
    r_c = r_h - c_rate
    ell = g + c_rate - r
    Pi_star_T = PI0 * np.exp(g * T)

    if verbose:
        print(f"Client: {profile.name}")
        print(f"  c={c_rate:.1%}, g={g:.1%}, r_h={r_h:.1%}, r_c={r_c:.1%}, ℓ={ell:.2%}")
        print(f"  Π*_T = {Pi_star_T:.0f}, q_dd = {profile.q_dd}")
        print(f"  Targets: x_50={profile.x_50}, x_75={profile.x_75}")
        print(f"  (x_25 determined by q_dd·σ_unc per weight combo)")

    def evaluate(w_eq, w_pe):
        if w_eq < 0.02 or w_pe < 0.0 or w_eq + w_pe > 0.93:
            return None
        # σ_unc → x_25 → k
        sig_unc = portfolio_sigma_unc(w_eq, w_pe)
        x_25 = PI0 * np.exp(-profile.q_dd * sig_unc)
        # k from x_25: L_T = x_25 when r_c = 0; L_T = x_25 · exp(-r_c·T) needs
        # L₀ = x_25 · exp(-r_c·T), and L₀ = Π₀·exp(-k·η)
        # So k = (ln(Π₀) - ln(x_25) + r_c·T) / η
        eff_temp = build_effective_asset(w_eq, w_pe, 1.5)
        eta_port = eff_temp.params.eta0
        k = (np.log(PI0) + r_c * T - np.log(x_25)) / eta_port
        if k < 0.1 or k > 6.0:
            return None
        try:
            result = compute_distribution(w_eq, w_pe, k, ell, r_h, c_rate)
            result['sig_unc'] = sig_unc
            result['x_25_input'] = x_25
            return result
        except:
            return None

    # Grid search
    w_eqs = np.linspace(0.05, 0.65, n_grid)
    w_pes = np.linspace(0.00, 0.50, n_grid)

    if verbose:
        print(f"\n{'Bd/Eq/PE':>12} {'k':>5} {'σ_unc':>6} {'x25':>5} | "
              f"{'q25':>5} {'q50':>5} {'q75':>5} {'surv':>5} {'|wa|':>5} {'err':>7}")
        print('-' * 75)

    best = None; best_err = np.inf
    for w_eq in w_eqs:
        for w_pe in w_pes:
            if w_eq + w_pe > 0.92:
                continue
            res = evaluate(w_eq, w_pe)
            if res is None:
                continue
            err = ((res['q50'] - profile.x_50)**2 +
                   (res['q75'] - profile.x_75)**2)

            if err < best_err:
                best_err = err
                best = (w_eq, w_pe, res)
                if verbose:
                    w_bd = 1 - w_eq - w_pe
                    print(f"{w_bd:3.0%}/{w_eq:3.0%}/{w_pe:3.0%} "
                          f"{res['k']:5.2f} {res['sig_unc']:5.1%} "
                          f"{res['x_25_input']:5.0f} | "
                          f"{res['q25']:5.1f} {res['q50']:5.1f} "
                          f"{res['q75']:5.1f} {res['surv']:5.1%} "
                          f"{res['w_a']:5.2f} {np.sqrt(err):7.2f} *")

    if best is None:
        if verbose:
            print("No feasible solution found!")
        return None

    # Refine with Nelder-Mead
    w0_eq, w0_pe = best[0], best[1]

    def objective(x):
        we, wp = x
        if we < 0.02 or wp < 0.0 or we + wp > 0.93:
            return 1e6
        res = evaluate(we, wp)
        if res is None:
            return 1e6
        return ((res['q50'] - profile.x_50)**2 +
                (res['q75'] - profile.x_75)**2)

    opt = minimize(objective, [w0_eq, w0_pe], method='Nelder-Mead',
                   options={'xatol': 0.003, 'fatol': 0.05, 'maxiter': 60})

    w_eq_opt = max(0.02, opt.x[0])
    w_pe_opt = max(0.0, opt.x[1])
    res_opt = evaluate(w_eq_opt, w_pe_opt)

    if res_opt is None:
        res_opt = best[2]
        w_eq_opt, w_pe_opt = best[0], best[1]

    w_bd_opt = 1 - w_eq_opt - w_pe_opt

    if verbose:
        print(f"\n→ Solution: {w_bd_opt:.1%}/{w_eq_opt:.1%}/{w_pe_opt:.1%} "
              f"(Bd/Eq/PE), k={res_opt['k']:.2f}")
        print(f"  σ_unc={res_opt['sig_unc']:.2%}, "
              f"x_25={res_opt['x_25_input']:.1f} "
              f"(= 100·exp(-{profile.q_dd}·{res_opt['sig_unc']:.2%}))")
        print(f"  Achieved: x_25/x_50/x_75 = "
              f"{res_opt['q25']:.1f} / {res_opt['q50']:.1f} / {res_opt['q75']:.1f}")
        print(f"  Target:                     "
              f"--- / {profile.x_50:.1f} / {profile.x_75:.1f}")
        print(f"  Full quantiles: q10={res_opt['q10']:.1f}, q25={res_opt['q25']:.1f}, "
              f"q50={res_opt['q50']:.1f}, q75={res_opt['q75']:.1f}, "
              f"q90={res_opt['q90']:.1f}")
        print(f"  E[Π_T]={res_opt['mu']:.1f}, Π*_T={res_opt['Pi_star_T']:.0f}, "
              f"L_T={res_opt['L_T']:.1f}")
        print(f"  Surv={res_opt['surv']:.1%}, Over={res_opt['over']:.1%}, "
              f"Floor={res_opt['floor']:.1%}")
        print(f"  |ω*_a|={res_opt['w_a']:.3f}, "
              f"σ_port=[{res_opt['sigma_port'][0]:.1%},"
              f"{res_opt['sigma_port'][1]:.1%}]")

        # Suitability checks
        print(f"\n  Suitability:")
        ok = True
        if res_opt['surv'] < 0.75:
            print(f"    ⚠ Survival {res_opt['surv']:.1%} < 75%")
            ok = False
        if w_bd_opt < 0.05:
            print(f"    ⚠ Bond weight {w_bd_opt:.0%} < 5%")
            ok = False
        if res_opt['k'] < 0.5:
            print(f"    ⚠ k={res_opt['k']:.2f} < 0.5")
            ok = False
        if res_opt['w_a'] > 2.0:
            print(f"    ⚠ |ω*_a|={res_opt['w_a']:.2f} > 2.0")
            ok = False
        q50_err = abs(res_opt['q50'] - profile.x_50)
        q75_err = abs(res_opt['q75'] - profile.x_75)
        if q50_err > 3:
            print(f"    ⚠ q50 error = {q50_err:.1f}")
            ok = False
        if q75_err > 3:
            print(f"    ⚠ q75 error = {q75_err:.1f}")
            ok = False
        if ok:
            print(f"    ✓ All checks passed")

    return {
        'w_bd': w_bd_opt, 'w_eq': w_eq_opt, 'w_pe': w_pe_opt,
        'k': res_opt['k'], 'ell': ell, 'r_h': r_h,
        **res_opt, 'profile': profile,
    }


def solve_client_fixed_bonds(profile: ClientProfile, w_bd: float, n_grid=20, verbose=True):
    """Solve with fixed bond weight. 1D search on w_eq, w_pe = (1-w_bd)-w_eq."""
    from scipy.optimize import minimize_scalar
    c_rate = profile.c; g = profile.g
    r_h = max(r, c_rate); r_c = r_h - c_rate
    ell = g + c_rate - r
    Pi_star_T = PI0 * np.exp(g * T)

    def evaluate(w_eq):
        w_pe = (1.0 - w_bd) - w_eq
        if w_pe < 0 or w_eq < 0.02: return None
        sig_unc = portfolio_sigma_unc(w_eq, w_pe)
        x_25 = PI0 * np.exp(-profile.q_dd * sig_unc)
        eff_temp = build_effective_asset(w_eq, w_pe, 1.5)
        eta_port = eff_temp.params.eta0
        k = (np.log(PI0) + r_c * T - np.log(x_25)) / eta_port
        if k < 0.1 or k > 6: return None
        try:
            res = compute_distribution(w_eq, w_pe, k, ell, r_h, c_rate)
            res['sig_unc'] = sig_unc
            res['x_25_input'] = x_25
            return res
        except: return None

    def obj(w_eq):
        res = evaluate(w_eq)
        if res is None: return 1e6
        return (res['q50'] - profile.x_50)**2 + (res['q75'] - profile.x_75)**2

    opt = minimize_scalar(obj, bounds=(0.05, 1.0 - w_bd - 0.01), method='bounded')
    w_eq_opt = opt.x; w_pe_opt = (1.0 - w_bd) - w_eq_opt
    res_opt = evaluate(w_eq_opt)

    if verbose and res_opt:
        print(f"→ Solution: {w_bd:.0%}/{w_eq_opt:.1%}/{w_pe_opt:.1%} (Bd/Eq/PE), k={res_opt['k']:.2f}")
        print(f"  σ_unc={res_opt['sig_unc']:.2%}, x_25={res_opt['x_25_input']:.1f}")
        print(f"  Achieved: x_25/x_50/x_75 = {res_opt['q25']:.1f} / {res_opt['q50']:.1f} / {res_opt['q75']:.1f}")
        print(f"  Target:                     --- / {profile.x_50:.1f} / {profile.x_75:.1f}")
        print(f"  Full: q10={res_opt['q10']:.1f}, q25={res_opt['q25']:.1f}, q50={res_opt['q50']:.1f}, "
              f"q75={res_opt['q75']:.1f}, q90={res_opt['q90']:.1f}")
        print(f"  E[Π_T]={res_opt['mu']:.1f}, Π*_T={res_opt['Pi_star_T']:.0f}, L_T={res_opt['L_T']:.1f}")
        print(f"  Surv={res_opt['surv']:.1%}, Over={res_opt['over']:.1%}, Floor={res_opt['floor']:.1%}")
        print(f"  |ω*_a|={res_opt['w_a']:.3f}")

    return {'w_bd': w_bd, 'w_eq': w_eq_opt, 'w_pe': w_pe_opt,
            'k': res_opt['k'], 'ell': ell, 'r_h': r_h,
            **res_opt, 'profile': profile}


def solve_client_liquidity(profile: ClientProfile, bd_range: tuple,
                           n_grid=10, verbose=True):
    """Solve with bond weight constrained to liquidity range [bd_lo, bd_hi].
    
    2D search on (w_eq, w_pe) subject to w_bd = 1 - w_eq - w_pe in [bd_lo, bd_hi].
    """
    from scipy.optimize import minimize
    c_rate = profile.c; g = profile.g
    r_h = max(r, c_rate); r_c = r_h - c_rate
    ell = g + c_rate - r
    Pi_star_T = PI0 * np.exp(g * T)
    bd_lo, bd_hi = bd_range

    if verbose:
        print(f"Client: {profile.name}")
        print(f"  c={c_rate:.1%}, g={g:.1%}, r_h={r_h:.1%}, r_c={r_c:.1%}, ℓ={ell:.2%}")
        print(f"  Π*_T = {Pi_star_T:.0f}, q_dd = {profile.q_dd}")
        print(f"  Targets: x_50={profile.x_50}, x_75={profile.x_75}")
        print(f"  Liquidity range: w_bd ∈ [{bd_lo:.0%}, {bd_hi:.0%}]")

    def evaluate(w_eq, w_pe):
        w_bd = 1.0 - w_eq - w_pe
        if w_bd < bd_lo - 0.001 or w_bd > bd_hi + 0.001:
            return None
        if w_eq < 0.02 or w_pe < 0.0 or w_eq < w_pe:
            return None
        sig_unc = portfolio_sigma_unc(w_eq, w_pe)
        x_25 = PI0 * np.exp(-profile.q_dd * sig_unc)
        eff_temp = build_effective_asset(w_eq, w_pe, 1.5)
        eta_port = eff_temp.params.eta0
        k = (np.log(PI0) + r_c * T - np.log(x_25)) / eta_port
        if k < 0.1 or k > 6.0:
            return None
        try:
            res = compute_distribution(w_eq, w_pe, k, ell, r_h, c_rate)
            res['sig_unc'] = sig_unc
            res['x_25_input'] = x_25
            return res
        except:
            return None

    # Grid search within liquidity range
    best = None; best_err = np.inf
    
    if verbose:
        print(f"\n{'Bd/Eq/PE':>12} {'k':>5} {'σ_unc':>6} {'x25':>5} | "
              f"{'q25':>5} {'q50':>5} {'q75':>5} {'surv':>5} {'|wa|':>5} {'err':>7}")
        print('-' * 75)

    risky_lo = 1.0 - bd_hi
    risky_hi = 1.0 - bd_lo
    for w_risky in np.linspace(risky_lo, risky_hi, n_grid):
        for eq_frac in np.linspace(0.1, 0.9, n_grid):
            w_eq = w_risky * eq_frac
            w_pe = w_risky * (1 - eq_frac)
            if w_eq < 0.02: continue
            res = evaluate(w_eq, w_pe)
            if res is None: continue
            err = ((res['q50'] - profile.x_50)**2 +
                   (res['q75'] - profile.x_75)**2)
            if err < best_err:
                best_err = err
                best = (w_eq, w_pe, res)
                if verbose:
                    w_bd = 1 - w_eq - w_pe
                    print(f"{w_bd:3.0%}/{w_eq:3.0%}/{w_pe:3.0%} "
                          f"{res['k']:5.2f} {res['sig_unc']:5.1%} "
                          f"{res['x_25_input']:5.0f} | "
                          f"{res['q25']:5.1f} {res['q50']:5.1f} "
                          f"{res['q75']:5.1f} {res['surv']:5.1%} "
                          f"{res['w_a']:5.2f} {np.sqrt(err):7.2f} *")

    if best is None:
        if verbose: print("No feasible solution found!")
        return None

    # Refine with Nelder-Mead within liquidity constraint
    w0_eq, w0_pe = best[0], best[1]
    def objective(x):
        we, wp = x
        w_bd = 1.0 - we - wp
        if we < 0.02 or wp < 0.0 or w_bd < bd_lo or w_bd > bd_hi or we < wp:
            return 1e6
        res = evaluate(we, wp)
        if res is None: return 1e6
        return ((res['q50'] - profile.x_50)**2 +
                (res['q75'] - profile.x_75)**2)

    opt = minimize(objective, [w0_eq, w0_pe], method='Nelder-Mead',
                   options={'xatol': 0.003, 'fatol': 0.05, 'maxiter': 60})
    w_eq_opt = max(0.02, opt.x[0])
    w_pe_opt = max(0.0, opt.x[1])
    w_bd_opt = 1 - w_eq_opt - w_pe_opt
    # Clamp to range
    if w_bd_opt < bd_lo:
        scale = (1 - bd_lo) / (w_eq_opt + w_pe_opt)
        w_eq_opt *= scale; w_pe_opt *= scale; w_bd_opt = bd_lo
    if w_bd_opt > bd_hi:
        scale = (1 - bd_hi) / (w_eq_opt + w_pe_opt)
        w_eq_opt *= scale; w_pe_opt *= scale; w_bd_opt = bd_hi

    res_opt = evaluate(w_eq_opt, w_pe_opt)
    if res_opt is None:
        res_opt = best[2]
        w_eq_opt, w_pe_opt = best[0], best[1]
        w_bd_opt = 1 - w_eq_opt - w_pe_opt

    if verbose:
        print(f"\n→ Solution: {w_bd_opt:.1%}/{w_eq_opt:.1%}/{w_pe_opt:.1%} "
              f"(Bd/Eq/PE), k={res_opt['k']:.2f}")
        print(f"  σ_unc={res_opt['sig_unc']:.2%}, "
              f"x_25={res_opt['x_25_input']:.1f}")
        print(f"  Achieved: x_25/x_50/x_75 = "
              f"{res_opt['q25']:.1f} / {res_opt['q50']:.1f} / {res_opt['q75']:.1f}")
        print(f"  Target:                     "
              f"--- / {profile.x_50:.1f} / {profile.x_75:.1f}")
        print(f"  Full: q10={res_opt['q10']:.1f}, q25={res_opt['q25']:.1f}, "
              f"q50={res_opt['q50']:.1f}, q75={res_opt['q75']:.1f}, "
              f"q90={res_opt['q90']:.1f}")
        print(f"  E[Π_T]={res_opt['mu']:.1f}, Π*_T={res_opt['Pi_star_T']:.0f}, "
              f"L_T={res_opt['L_T']:.1f}")
        print(f"  Surv={res_opt['surv']:.1%}, Over={res_opt['over']:.1%}, "
              f"Floor={res_opt['floor']:.1%}")
        print(f"  |ω*_a|={res_opt['w_a']:.3f}, "
              f"σ_port=[{res_opt['sigma_port'][0]:.1%},"
              f"{res_opt['sigma_port'][1]:.1%}]")

    return {'w_bd': w_bd_opt, 'w_eq': w_eq_opt, 'w_pe': w_pe_opt,
            'k': res_opt['k'], 'ell': ell, 'r_h': r_h,
            **res_opt, 'profile': profile}
