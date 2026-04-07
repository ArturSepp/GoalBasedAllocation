#!/usr/bin/env python3
"""
Generate all figures and run integration tests for the paper:
"Dynamic Mean-Variance Portfolio Allocation under Regime-Switching
Jump-Diffusions with Absorbing Barriers" (Sepp, 2026).

Usage:
    python -m paper_figures.generate_paper_figures [--outdir figures/] [--figure N]
    python -m paper_figures.generate_paper_figures --test

Seeds for reproducibility:
    Survived path: 370
    Stopped path:  1351
"""
import sys
import argparse
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import norm
from pathlib import Path

from goal_based_allocation import (
    create_paper_assets, create_paper_mandates,
    compute_density, compute_survival, compute_tilted_survival,
    compute_overshoot_density, AdvisorSpec, compute_opportunity_point,
    build_effective_asset, portfolio_sigma_unc,
)
from goal_based_allocation.riccati_solver import (
    find_ell, gap_process_asset, simulate_mv_optimal
)
from goal_based_allocation.mandate_utils import mandate_effective_asset
from goal_based_allocation.regime_switch_paper import (
    RegimeSwitchParams, AssetSpecification,
    compute_wealth_density,
    _solve_characteristic, _eval_density_unbounded
)
from goal_based_allocation.laplace_inversion import laplace_invert_abate_whitt

# ============================================================
# Constants
# ============================================================
PI0 = 100.0; T = 10.0; r = 0.02; Q_DD = 2.0
SEED_SURVIVED = 370
SEED_STOPPED = 1351
N_PATHS_MC = 200_000

ASSET_TARGETS = {'bonds': 0.005, 'equity': 0.04, 'private_equity': 0.06}
ASSET_LABELS = {'bonds': 'Bonds', 'equity': 'Equity', 'private_equity': 'Private Equity'}

MANDATE_TARGETS = {'conservative': 0.01, 'balanced': 0.02, 'growth': 0.04}
MANDATE_LABELS = {'conservative': 'Conservative (70/20/10)',
                  'balanced': 'Balanced (40/40/20)',
                  'growth': 'Growth (10/60/30)'}


# ============================================================
# Helpers
# ============================================================

def _calibrate_mandate(w_bd, c=0.0, omega_0=1.0, q=2/3, q_dd=Q_DD):
    """Calibrate a mandate: build effective asset, solve Riccati, return all objects."""
    w_eq = q * (1 - w_bd); w_pe = (1 - q) * (1 - w_bd)
    r_h = max(r, c); r_c = r_h - c
    sig_unc = portfolio_sigma_unc(w_eq, w_pe)
    x_25 = PI0 * np.exp(-q_dd * sig_unc)
    eff_temp = build_effective_asset(w_eq, w_pe, 1.5)
    k = (np.log(PI0) + r_c * T - np.log(x_25)) / eff_temp.params.eta0
    eff = build_effective_asset(w_eq, w_pe, k)
    lo, hi = 0.001, 0.50
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        try:
            ell, ric = find_ell(eff, T, mid, r_h, c)
            d = ric.derived_at_tau(T)
            wa = abs(d['w_a'][0])
            om = wa * (d['Pi_star'][0] / PI0 - 1)
            if om < omega_0: lo = mid
            else: hi = mid
            if abs(om - omega_0) < 0.001: break
        except:
            hi = mid
    ell, ric = find_ell(eff, T, mid, r_h, c)
    gap = gap_process_asset(ric)
    wa = abs(ric.derived_at_tau(T)['w_a'][0])
    PiT = ric.derived_at_tau(0)['Pi_star'][0]
    L_T = eff.pi_floor * np.exp(r_c * T)
    B_T = PiT - L_T
    return dict(eff=eff, ric=ric, gap=gap, wa=wa, PiT=PiT, L_T=L_T, B_T=B_T,
                w_eq=w_eq, w_pe=w_pe, r_h=r_h, r_c=r_c, sig_unc=sig_unc)

def figure_opportunity_set(spec, filename, outdir):
    """Generate 4-panel opportunity set figure."""
    w_vals = np.sort(np.unique(np.concatenate([
        np.arange(0.90, -0.01, -0.05), [1.0, 0.65, 0.35]
    ])))[::-1]
    opp = []
    for w in w_vals:
        res = compute_opportunity_point(w, spec)
        if res is not None and res['r_impl'] > -0.5:
            opp.append(res)
    opp.sort(key=lambda x: x['r_impl'])
    feasible = [p for p in opp if p['r_impl'] >= 0 and p['S'] >= 0.50 and p.get('wa', 1) >= 0.1]
    if not feasible:
        feasible = opp

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    r_f = [p['r_impl']*100 for p in feasible]
    w_f = [p['w_bd']*100 for p in feasible]
    s_f = [p['sig_unc']*100 for p in feasible]

    axes[0,0].plot(w_f, r_f, 'b-o', lw=2, ms=5)
    axes[0,0].set_xlabel('Bond weight (%)'); axes[0,0].set_ylabel('Implied return (%)')
    axes[0,0].set_title('Implied Return vs Bond Weight'); axes[0,0].invert_xaxis(); axes[0,0].grid(alpha=0.2)

    axes[0,1].plot(s_f, r_f, 'r-o', lw=2, ms=5)
    axes[0,1].set_xlabel('Portfolio volatility (%)'); axes[0,1].set_ylabel('Implied return (%)')
    axes[0,1].set_title('Efficient Frontier (MV with Floor)'); axes[0,1].grid(alpha=0.2)

    ax = axes[1,0]
    max_bd = max(p['w_bd'] for p in feasible)
    targets = [max_bd, 0.65, 0.35, 0.00]
    colors = ['C0','C1','C2','C3']
    for wb, col in zip(targets, colors):
        best = min(feasible, key=lambda p: abs(p['w_bd']-wb))
        lbl = f"{best['w_bd']:.0%}/{best['w_eq']:.0%}/{best['w_pe']:.0%}, r={best['r_impl']:.1%}"
        ax.plot(best['Pi_cdf'], best['cdf'], color=col, lw=2, label=lbl)
        for pq, mk in [(0.05,'d'),(0.25,'v'),(0.50,'o'),(0.75,'^'),(0.95,'s')]:
            idx = np.searchsorted(best['cdf'], pq)
            if 0 < idx < len(best['Pi_cdf']):
                piv = best['Pi_cdf'][idx]
                ax.plot(piv, pq, marker=mk, color=col, ms=6, zorder=5,
                        markeredgecolor='white', markeredgewidth=0.8)
                ax.annotate(f'{piv:.0f}', (piv, pq), textcoords='offset points',
                           xytext=(6,3), fontsize=7, color=col, fontweight='bold')
    for p in [0.05,0.25,0.50,0.75,0.95]:
        ax.axhline(p, color='gray', ls=':', alpha=0.3, lw=0.8)
    ax.set_yticks([0.05,0.25,0.50,0.75,0.95])
    ax.set_yticklabels(['5%','25%','50%','75%','95%'])
    ax.set_xlabel('Terminal wealth'); ax.set_ylabel('CDF')
    ax.set_title('Wealth Distribution'); ax.legend(fontsize=8, loc='center right')
    ax.set_xlim(30,280); ax.set_ylim(-0.02,1.02); ax.grid(alpha=0.2, axis='x')

    ax = axes[1,1]
    ax.fill_between(r_f, [p['q5'] for p in feasible], [p['q95'] for p in feasible],
                    alpha=0.15, color='C0', label='5th-95th')
    ax.fill_between(r_f, [p['q25'] for p in feasible], [p['q75'] for p in feasible],
                    alpha=0.3, color='C0', label='25th-75th')
    ax.plot(r_f, [p['q50'] for p in feasible], 'C0-', lw=2, label='Median')
    ax.plot(r_f, [p['L_T'] for p in feasible], 'r--', lw=1.5, label='Floor $L_T$')
    ax.axhline(PI0, color='gray', ls=':', alpha=0.5, label='Initial wealth')
    ax.set_xlabel('Implied return (%)'); ax.set_ylabel('Terminal wealth')
    ax.set_title('Terminal Wealth Quantiles'); ax.legend(fontsize=9); ax.grid(alpha=0.2)

    fig.suptitle(f'Investment Opportunity Set ($\\omega_0$={spec.omega_0:.0%}, '
                 f'c={spec.c:.1%}, q={spec.q:.2f}, $q_{{dd}}$={spec.q_dd})', fontsize=14)
    fig.tight_layout()
    fig.savefig(outdir / filename, dpi=150)
    plt.close('all')
    print(f'  Saved {filename}')


# ============================================================
# Figure: Allocation Paths
# ============================================================
def figure_allocation_paths(mandates, c_rate, filename, outdir):
    """Expected allocation glide path for multiple mandates."""
    r_h = max(r, c_rate); r_c = r_h - c_rate
    colors = ['C0','C1','C2','C3']
    t_grid = np.linspace(0.1, T-0.1, 40)
    fig, ax = plt.subplots(figsize=(12, 7))

    for idx, (name, w_bd, alloc_str) in enumerate(mandates):
        cal = _calibrate_mandate(w_bd, c=c_rate)
        alloc = np.zeros(len(t_grid))
        for j, t in enumerate(t_grid):
            d_t = cal['ric'].derived_at_tau(T-t)
            Pi_star_t = d_t['Pi_star'][0]
            L_t = cal['eff'].pi_floor * np.exp(r_c*t)
            B_t = Pi_star_t - L_t
            S_t = compute_survival(t, cal['gap'].x0, cal['gap'])
            TS1_t = compute_tilted_survival(t, cal['gap'].x0, cal['gap'], 1.0)
            E_Pi_t = Pi_star_t*S_t - B_t*TS1_t + L_t*(1-S_t)
            E_risky = cal['wa'] * B_t * TS1_t
            alloc[j] = E_risky / E_Pi_t if E_Pi_t > 0 else 0
        ax.plot(t_grid, alloc*100, color=colors[idx], lw=2, label=f"{name} ({alloc_str})")

    ax.axhline(100, color='gray', ls=':', alpha=0.4, lw=1, label='Fully invested')
    ax.set_xlabel('Time $t$ (years)'); ax.set_ylabel('Expected risky allocation (%)')
    c_str = f'{c_rate:.1%}'.replace('%','\\%')
    ax.set_title(f'Expected Allocation ($\\omega_0 = 1$, $c = {c_str}$, $q_{{dd}} = 2$)')
    ax.legend(fontsize=10); ax.set_xlim(0, 10); ax.set_ylim(0, 110); ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(outdir / filename, dpi=150)
    plt.close('all')
    print(f'  Saved {filename}')


# ============================================================
# Figure: Allocation ±1σ Subplots
# ============================================================
def figure_allocation_bands(mandates, c_rate, filename, outdir):
    """4-subplot figure: allocation ω̄(t) ± Std[R_t]/E[Π_t]."""
    r_h = max(r, c_rate); r_c = r_h - c_rate
    colors = ['C0','C1','C2','C3']
    t_grid = np.linspace(0.1, T-0.1, 40)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=True, sharey=True)

    for idx, (name, w_bd, alloc_str) in enumerate(mandates):
        cal = _calibrate_mandate(w_bd, c=c_rate)
        E_R = np.zeros(len(t_grid)); Std_R = np.zeros(len(t_grid)); E_Pi = np.zeros(len(t_grid))
        for j, t in enumerate(t_grid):
            d_t = cal['ric'].derived_at_tau(T-t)
            Pi_star_t = d_t['Pi_star'][0]
            L_t = cal['eff'].pi_floor * np.exp(r_c*t)
            B_t = Pi_star_t - L_t
            S_t = compute_survival(t, cal['gap'].x0, cal['gap'])
            TS1 = compute_tilted_survival(t, cal['gap'].x0, cal['gap'], 1.0)
            TS2 = compute_tilted_survival(t, cal['gap'].x0, cal['gap'], 2.0)
            E_R[j] = cal['wa'] * B_t * TS1
            Std_R[j] = np.sqrt(max(0, cal['wa']**2 * B_t**2 * TS2 - E_R[j]**2))
            E_Pi[j] = Pi_star_t*S_t - B_t*TS1 + L_t*(1-S_t)
        om_mean = E_R / E_Pi
        om_upper = (E_R + Std_R) / E_Pi
        om_lower = np.maximum(0, E_R - Std_R) / E_Pi

        ax = axes.flatten()[idx]
        ax.plot(t_grid, om_mean*100, color=colors[idx], lw=2.5, label='$\\bar{\\omega}(t)$')
        ax.fill_between(t_grid, om_lower*100, om_upper*100, alpha=0.2, color=colors[idx], label='$\\pm 1\\sigma$ band')
        ax.plot(t_grid, om_upper*100, color=colors[idx], lw=1, ls='--', alpha=0.6)
        ax.plot(t_grid, om_lower*100, color=colors[idx], lw=1, ls='--', alpha=0.6)
        ax.axhline(100, color='gray', ls=':', alpha=0.4, lw=1)
        ax.set_title(f"{name} ({alloc_str})"); ax.legend(fontsize=9, loc='upper right')
        ax.set_xlim(0,10); ax.set_ylim(0,120); ax.grid(alpha=0.2)
        ax.text(9.8, om_mean[-1]*100, f"{om_mean[-1]*100:.0f}%", fontsize=9,
                color=colors[idx], ha='right', va='bottom', fontweight='bold')

    axes[1,0].set_xlabel('Time $t$ (years)'); axes[1,1].set_xlabel('Time $t$ (years)')
    axes[0,0].set_ylabel('Risky allocation (%)'); axes[1,0].set_ylabel('Risky allocation (%)')
    fig.suptitle(r'Expected Risky Allocation $\bar{\omega}(t) \pm \mathrm{Std}[R_t]/\mathbb{E}[\Pi_t]$'
                 rf' ($\omega_0 = 1$, $c = {c_rate*100:.0f}\%$, $q_{{dd}} = 2$)')
    fig.tight_layout()
    fig.savefig(outdir / filename, dpi=150)
    plt.close('all')
    print(f'  Saved {filename}')


# ============================================================
# Figure: Floor vs Lipton (3 densities)
# ============================================================
def figure_floor_vs_lipton(w_bd, mandate_name, filename, outdir):
    """Three wealth distributions: BH, Lipton (no floor), MV with floor."""
    w_eq = (2/3)*(1-w_bd); w_pe = (1/3)*(1-w_bd)
    cal = _calibrate_mandate(w_bd)
    gap = cal['gap']; eff = cal['eff']; ric = cal['ric']
    PiT = cal['PiT']; L_T = cal['L_T']; B_T = cal['B_T']
    r_c = cal['r_c']

    # Bounded density
    x_max_b = min(8.0, gap.x0 + 6*max(gap.params.sigma0, gap.params.sigma1)*np.sqrt(T))
    x_grid_b = np.linspace(0.001, x_max_b, 800)
    d0_b, d1_b = compute_density(T, x_grid_b, gap); dg_b = d0_b + d1_b
    d_ov_grid = np.linspace(0.001, 8.0, 400)
    f_ov = compute_overshoot_density(T, d_ov_grid, gap)
    over_mass = np.trapezoid(f_ov, d_ov_grid)
    S = compute_survival(T, gap.x0, gap)
    Pi_surv = PiT - B_T*np.exp(-x_grid_b)
    f_Pi_surv = dg_b / (B_T*np.exp(-x_grid_b))
    Pi_over = PiT - B_T*np.exp(d_ov_grid)
    f_Pi_over = f_ov / (B_T*np.exp(d_ov_grid))

    # Unbounded density
    x0_true = gap.x0
    x_min_u = -np.log((PiT+50)/B_T)
    x_grid_u = np.linspace(x_min_u, x_max_b, 1200); Nx = len(x_grid_u)
    def g_unb(Ps):
        Np = len(Ps)
        G0 = np.zeros((Np,Nx), dtype=complex); G1 = np.zeros((Np,Nx), dtype=complex)
        for ip in range(Np):
            sol = _solve_characteristic(Ps[ip], gap)
            g0, g1 = _eval_density_unbounded(sol, x_grid_u, x0_true)
            G0[ip,:] = g0; G1[ip,:] = g1
        return np.hstack([G0, G1])
    res_u = laplace_invert_abate_whitt(g_unb, T)
    dg_u = res_u[:Nx] + res_u[Nx:]
    Pi_unb = PiT - B_T*np.exp(-x_grid_u)
    f_Pi_unb = dg_u / (B_T*np.exp(-x_grid_u))
    mask_neg = Pi_unb < 0
    prob_neg_lipton = np.trapezoid(f_Pi_unb[mask_neg], Pi_unb[mask_neg]) if np.any(mask_neg) else 0

    sort_idx = np.argsort(Pi_over); Po_s = Pi_over[sort_idx]; fo_s = f_Pi_over[sort_idx]
    mn = Po_s < 0
    prob_neg_floor = np.trapezoid(fo_s[mn], Po_s[mn]) if np.any(mn) else 0.0

    # BH
    p1 = 1/1.1
    mu_stat = p1*eff.mu_growth + (1-p1)*eff.mu_stress
    sig_stat = np.sqrt(p1*eff.params.sigma0**2 + (1-p1)*eff.params.sigma1**2)
    mu_log = np.log(PI0) + (mu_stat - 0.5*sig_stat**2)*T
    sig_log = sig_stat*np.sqrt(T)
    Pi_bh = np.linspace(1, 500, 800)
    f_bh = norm.pdf(np.log(Pi_bh), mu_log, sig_log) / Pi_bh

    alloc_str = f'{w_bd:.0%}/{w_eq:.0%}/{w_pe:.0%}'
    y_max = max(f_Pi_surv.max(), f_Pi_unb[f_Pi_unb>1e-6].max(), f_bh.max()) * 1.1

    fig, ax = plt.subplots(figsize=(12, 7))
    ax.plot(Pi_bh, f_bh, 'C2-', lw=2, alpha=0.8, label='Buy-and-hold')
    ax.fill_between(Pi_bh, f_bh, alpha=0.08, color='C2')
    valid_u = f_Pi_unb > 1e-6
    ax.plot(Pi_unb[valid_u], f_Pi_unb[valid_u], 'C1-', lw=2, label='MV without floor (Lipton)')
    ax.fill_between(Pi_unb[valid_u], f_Pi_unb[valid_u], alpha=0.1, color='C1')
    if np.any(mask_neg & valid_u):
        ax.fill_between(Pi_unb[mask_neg&valid_u], f_Pi_unb[mask_neg&valid_u], alpha=0.3, color='red')
    ax.plot(Pi_surv, f_Pi_surv, 'C0-', lw=2, label='MV with floor')
    ax.fill_between(Pi_surv, f_Pi_surv, alpha=0.15, color='C0')
    ax.plot(Pi_over[Pi_over>-55], f_Pi_over[Pi_over>-55], 'C0--', lw=1.5, alpha=0.7)
    ax.axvline(L_T, color='r', ls='--', lw=1.5, alpha=0.6)
    ax.axvline(0, color='k', ls='-', lw=1.5, alpha=0.5)
    ax.text(L_T+2, y_max*0.95, f'Floor $L_T = {L_T:.0f}$', fontsize=11, color='r', ha='left', va='top')
    ax.text(2, y_max*0.95, 'Zero\nwealth', fontsize=11, color='k', ha='left', va='top', alpha=0.7)
    fl_str = f'{prob_neg_floor:.1%}' if prob_neg_floor > 0.0005 else '0.0%'
    ax.text(-48, y_max*0.55, f'$P(\\Pi_T < 0)$:\n  Lipton: {prob_neg_lipton:.1%}\n  With floor: {fl_str}',
            fontsize=10, ha='left', va='top',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='lightyellow', edgecolor='red', alpha=0.9))
    ax.set_xlabel('Terminal wealth $\\Pi_T$'); ax.set_ylabel('Density')
    ax.set_title(f'{mandate_name} Mandate ({alloc_str}): Three Wealth Distributions')
    ax.legend(fontsize=11, loc='upper right')
    ax.set_xlim(-50, 300); ax.set_ylim(0, y_max); ax.grid(alpha=0.15)
    fig.savefig(outdir / filename, dpi=150, bbox_inches='tight')
    plt.close('all')
    print(f'  Saved {filename}')


# ============================================================
# Figure: Mandate Density Overlay (MV vs BH)
# ============================================================
def figure_mandate_density_overlay(filename, outdir):
    """MV vs BH density overlay for four mandates (c=0%)."""
    mandates = [
        ('Income',       1.00, '100%/0%/0%'),
        ('Conservative', 0.65, '65%/23%/12%'),
        ('Balanced',     0.35, '35%/43%/22%'),
        ('Growth',       0.00, '0%/67%/33%'),
    ]
    colors = ['C0', 'C1', 'C2', 'C3']
    data = []
    for name, w_bd, alloc_str in mandates:
        cal = _calibrate_mandate(w_bd)
        gap = cal['gap']; eff = cal['eff']
        PiT = cal['PiT']; L_T = cal['L_T']; B_T = cal['B_T']
        wa = cal['wa']; r_c = cal['r_c']

        x_max = min(8.0, gap.x0 + 6*max(gap.params.sigma0, gap.params.sigma1)*np.sqrt(T))
        x_grid = np.linspace(0.001, x_max, 800)
        d0, d1 = compute_density(T, x_grid, gap)
        dg = d0 + d1
        d_ov_grid = np.linspace(0.001, min(8.0, x_max), 400)
        f_ov = compute_overshoot_density(T, d_ov_grid, gap)

        Pi_surv = PiT - B_T * np.exp(-x_grid)
        f_Pi_surv = dg / (B_T * np.exp(-x_grid))
        Pi_over = PiT - B_T * np.exp(d_ov_grid)
        f_Pi_over = f_ov / (B_T * np.exp(d_ov_grid))

        # BH
        p1 = 1/1.1
        mu_stat = p1*eff.mu_growth + (1-p1)*eff.mu_stress
        sig_stat = np.sqrt(p1*eff.params.sigma0**2 + (1-p1)*eff.params.sigma1**2)
        mu_log = np.log(PI0) + (mu_stat - 0.5*sig_stat**2)*T
        sig_log = sig_stat * np.sqrt(T)
        Pi_bh = np.linspace(max(1, np.exp(mu_log-4*sig_log)), np.exp(mu_log+4*sig_log), 500)
        f_bh = norm.pdf(np.log(Pi_bh), mu_log, sig_log) / Pi_bh

        data.append(dict(name=name, alloc=alloc_str, PiT=PiT,
                         Pi_surv=Pi_surv, f_Pi_surv=f_Pi_surv,
                         Pi_over=Pi_over, f_Pi_over=f_Pi_over,
                         Pi_bh=Pi_bh, f_bh=f_bh))

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=False)
    for i, d in enumerate(data):
        ax1.plot(d['Pi_surv'], d['f_Pi_surv'], color=colors[i], lw=2, label=d['name'])
        ax1.fill_between(d['Pi_surv'], d['f_Pi_surv'], alpha=0.15, color=colors[i])
        valid = d['Pi_over'] > 20
        if np.any(valid):
            ax1.plot(d['Pi_over'][valid], d['f_Pi_over'][valid], color=colors[i], lw=1.2, ls='--', alpha=0.6)
    ax1.set_ylabel('Density'); ax1.set_title('MV-Optimal (fully invested, $\\omega^*(0)=1$)')
    ax1.legend(fontsize=9, loc='upper left'); ax1.set_xlim(20, 300); ax1.set_ylim(bottom=0); ax1.grid(alpha=0.15)

    for i, d in enumerate(data):
        ax2.plot(d['Pi_bh'], d['f_bh'], color=colors[i], lw=2, label=d['name'])
        ax2.fill_between(d['Pi_bh'], d['f_bh'], alpha=0.15, color=colors[i])
    ax2.set_xlabel('Terminal wealth $\\Pi_T$'); ax2.set_ylabel('Density')
    ax2.set_title('Buy-and-Hold (no floor, no dynamic allocation)')
    ax2.legend(fontsize=9, loc='upper left'); ax2.set_xlim(20, 300); ax2.set_ylim(bottom=0); ax2.grid(alpha=0.15)

    fig.suptitle('Terminal Wealth Density: MV-Optimal vs Buy-and-Hold ($T$=10y, $c$=0%)', fontsize=14)
    fig.tight_layout()
    fig.savefig(outdir / filename, dpi=150)
    plt.close('all')
    print(f'  Saved {filename}')


# ============================================================
# Figure: Path Dynamics (Survived vs Stopped)
# ============================================================
def figure_path_dynamics(filename, outdir):
    """4-panel figure: survived vs stopped path with S_t, Π_t, E[Π_t], L_t."""
    cal = _calibrate_mandate(0.35)  # Balanced
    ric = cal['ric']; gap = cal['gap']; eff = cal['eff']
    wa = cal['wa']; r_c = cal['r_c']
    dt = 1/252; n_steps = int(T/dt)

    eta0 = eff.params.eta0; eta1 = eff.params.eta1
    lam01 = eff.params.lambda01; lam10 = eff.params.lambda10
    mu_bar_g = eff.mu_growth - lam01*(-eta0/(1+eta0))
    mu_bar_s = eff.mu_stress - lam10*(eta1/(1-eta1))
    sig_g = eff.params.sigma0; sig_s = eff.params.sigma1
    wa_growth = abs(ric.derived_at_tau(T)['w_a'][0])
    wa_stress = abs(ric.derived_at_tau(T)['w_a'][1])

    def sim(seed):
        rng = np.random.RandomState(seed)
        Pi = np.zeros(n_steps+1); Pi[0] = PI0
        Sp = np.zeros(n_steps+1); Sp[0] = PI0
        omega = np.zeros(n_steps+1); regime = np.zeros(n_steps+1, dtype=int)
        stopped = False
        for i in range(n_steps):
            t = i*dt; reg = regime[i]
            mb = mu_bar_g if reg==0 else mu_bar_s
            sg = sig_g if reg==0 else sig_s
            lo = lam01 if reg==0 else lam10
            dW = rng.normal(0, np.sqrt(dt)); J = 0.0
            tr = rng.uniform() < lo*dt
            if tr:
                if reg==0: J=-rng.exponential(eta0); regime[i+1]=1
                else: J=rng.exponential(eta1); regime[i+1]=0
                Sp[i+1] = Sp[i]*np.exp((mb-0.5*sg**2)*dt + sg*dW + J)
            else:
                Sp[i+1] = Sp[i]*np.exp((mb-0.5*sg**2)*dt + sg*dW); regime[i+1]=reg
            if stopped: Pi[i+1] = Pi[i]*np.exp(r_c*dt); continue
            d_t = ric.derived_at_tau(T-t)
            ov = np.clip(wa*(d_t['Pi_star'][0]/Pi[i]-1), -2, 3); omega[i] = ov
            dr = r_c + ov*(mb-r); vl = ov*sg
            if tr: Pi[i+1] = Pi[i]*np.exp(dr*dt+vl*dW)*(1+ov*(np.exp(J)-1))
            else: Pi[i+1] = Pi[i]*np.exp(dr*dt+vl*dW)
            Lt1 = eff.pi_floor*np.exp(r_c*(t+dt))
            if Pi[i+1] <= Lt1: stopped = True
        if not stopped:
            dT = ric.derived_at_tau(0); omega[-1] = wa*(dT['Pi_star'][0]/Pi[-1]-1)
        return Pi, Sp, omega

    Pi1, Sp1, om1 = sim(SEED_SURVIVED)
    Pi2, Sp2, om2 = sim(SEED_STOPPED)

    # E[Π_t]
    ta = np.linspace(0.01, T-0.01, 50)
    EP = np.zeros(len(ta))
    for j, t in enumerate(ta):
        dt2 = ric.derived_at_tau(T-t)
        Pst = dt2['Pi_star'][0]; Lt = eff.pi_floor*np.exp(r_c*t); Bt = Pst-Lt
        St = compute_survival(t, gap.x0, gap)
        TS1t = compute_tilted_survival(t, gap.x0, gap, 1.0)
        EP[j] = Pst*St - Bt*TS1t + Lt*(1-St)
    ta = np.concatenate([[0], ta]); EP = np.concatenate([[PI0], EP])

    td = np.linspace(0, T, n_steps+1); Lp = eff.pi_floor*np.exp(r_c*td)
    sk = 5; ix = np.arange(0, n_steps+1, sk); tp = td[ix]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    for col, (Pi, Sp, om, clr, ttl) in enumerate([
        (Pi1, Sp1, om1, 'C2', 'Survived path'),
        (Pi2, Sp2, om2, 'C3', 'Stopped path'),
    ]):
        ax = axes[0, col]
        ax.plot(tp, Sp[ix], 'C1-', lw=1, alpha=0.7, label='Risky portfolio $S_t$')
        ax.plot(tp, Pi[ix], f'{clr}-', lw=1.5, label='Portfolio wealth $\\Pi_t$')
        ax.plot(ta, EP, 'k--', lw=1.5, alpha=0.6, label='Expected wealth $\\mathbb{E}[\\Pi_t]$')
        ax.plot(tp, Lp[ix], 'r--', lw=1.5, alpha=0.6, label='Floor $L_t$')
        ax.axhline(PI0, color='gray', ls=':', alpha=0.3)
        ax.set_ylabel('Wealth'); ax.set_title(f'{ttl} — wealth dynamics')
        ax.legend(fontsize=9, loc='upper left'); ax.set_xlim(0,10); ax.set_ylim(0,220); ax.grid(alpha=0.15)

        ax = axes[1, col]
        ax.plot(tp, om[ix]*100, f'{clr}-', lw=1.5, label='$\\omega^*(t)$ (growth-regime policy)')
        ax.axhline(wa_growth*100, color='C0', ls='-.', lw=1.5, alpha=0.6,
                   label=f'$|\\omega^{{*[1]}}_a|$ = {wa_growth:.0%} (growth)')
        ax.axhline(wa_stress*100, color='C3', ls='-.', lw=1.5, alpha=0.6,
                   label=f'$|\\omega^{{*[2]}}_a|$ = {wa_stress:.0%} (stress)')
        ax.axhline(100, color='gray', ls=':', alpha=0.4); ax.axhline(0, color='gray', ls=':', alpha=0.4)
        ax.set_xlabel('Time $t$ (years)'); ax.set_ylabel('Risky allocation (%)')
        ax.set_title(f'{ttl} — allocation'); ax.legend(fontsize=8, loc='upper right')
        ax.set_xlim(0,10); ax.set_ylim(-20,200); ax.grid(alpha=0.15)

    fig.suptitle('MV-Optimal Strategy: Balanced Mandate (35%/43%/22%), $c = 0\\%$, $\\omega_0 = 1$')
    fig.tight_layout()
    fig.savefig(outdir / filename, dpi=150)
    plt.close('all')
    print(f'  Saved {filename}')



# ============================================================
# Figure 10: Mandate Comparison (2 subplots: Conservative, Balanced)
# ============================================================
def figure_mandate_comparison(filename, outdir, n_paths=N_PATHS_MC):
    """Terminal wealth density for mandates: analytical vs MC.

    Two subplots showing survived density (blue) and overshoot density
    (red) overlaid on MC histograms. Uses _calibrate_mandate with
    c=0%, ω*(0)=1, q_dd=2 — matching Table 2 parameters.
    """
    mandate_specs = [
        ('Conservative',  0.65),
        ('Balanced',      0.35),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    for idx, (label, w_bd) in enumerate(mandate_specs):
        ax = axes[idx]
        cal = _calibrate_mandate(w_bd, c=0.0, omega_0=1.0)
        gap = cal['gap']; eff = cal['eff']; ric = cal['ric']
        PiT = cal['PiT']; L_T = cal['L_T']; B_T = cal['B_T']
        wa = cal['wa']

        # Analytical bounded density (survived component)
        x_max = min(10.0, gap.x0 + 8*max(gap.params.sigma0,
                                           gap.params.sigma1)*np.sqrt(T))
        x_grid = np.linspace(0.0001, x_max, 500)
        d0_b, d1_b = compute_density(T, x_grid, gap)
        dg_b = d0_b + d1_b
        S = compute_survival(T, gap.x0, gap)

        # Gap → wealth transform
        Pi_surv = PiT - B_T * np.exp(-x_grid)
        f_Pi_surv = dg_b / (B_T * np.exp(-x_grid))

        # Analytical overshoot density — extend to cover Pi > 0
        d_ov_max = np.log(B_T / 1.0) if B_T > 1 else 8.0  # Pi_over = 0 at d = ln(B_T)
        d_ov_grid = np.linspace(0.001, min(d_ov_max, 10.0), 300)
        f_ov = compute_overshoot_density(T, d_ov_grid, gap)
        over_mass = float(np.trapezoid(f_ov, d_ov_grid))
        Pi_over = PiT - B_T * np.exp(d_ov_grid)
        f_Pi_over = f_ov / (B_T * np.exp(d_ov_grid))

        # MC simulation
        mc = simulate_mv_optimal(ric, n_paths=n_paths, steps_per_year=260, seed=42)
        Pi_T_mc = mc['Pi_T']
        survived = mc['survived']
        is_overshoot = mc['is_overshoot']

        # MC histograms (density = counts / (N_total * bin_width))
        if np.sum(survived) > 100:
            vals_s = Pi_T_mc[survived]
            counts_s, edges_s = np.histogram(vals_s, bins=50)
            widths_s = np.diff(edges_s)
            density_s = counts_s / (n_paths * widths_s)
            centers_s = 0.5 * (edges_s[:-1] + edges_s[1:])
            ax.bar(centers_s, density_s, width=widths_s, alpha=0.35,
                   color='C0', edgecolor='none',
                   label='MC survived' if idx == 0 else None, zorder=2)

        if np.sum(is_overshoot) > 10:
            vals_o = Pi_T_mc[is_overshoot]
            counts_o, edges_o = np.histogram(vals_o, bins=25)
            widths_o = np.diff(edges_o)
            density_o = counts_o / (n_paths * widths_o)
            centers_o = 0.5 * (edges_o[:-1] + edges_o[1:])
            ax.bar(centers_o, density_o, width=widths_o, alpha=0.35,
                   color='C3', edgecolor='none',
                   label='MC overshoot' if idx == 0 else None, zorder=2)

        # Analytical survived density
        ax.plot(Pi_surv, f_Pi_surv, 'C0-', lw=2,
                label='Analytical (survived)' if idx == 0 else None, zorder=5)

        # Analytical overshoot density
        if over_mass > 0.001:
            valid_ov = (Pi_over > 0) & (f_Pi_over > 1e-8)
            ax.plot(Pi_over[valid_ov], f_Pi_over[valid_ov], 'C3-', lw=2,
                    label='Analytical (overshoot)' if idx == 0 else None, zorder=5)

        # Floor line
        ax.axvline(L_T, color='r', ls='--', lw=1.5, alpha=0.6,
                   label='Floor $L_T$' if idx == 0 else None)

        ax.set_title(label, fontsize=13, fontweight='bold')
        ax.set_xlabel(r'Terminal wealth $\Pi_T$', fontsize=10)
        if idx == 0:
            ax.set_ylabel('Density', fontsize=11)

        ax.set_xlim(0, 260)
        ax.set_ylim(bottom=0)
        ax.grid(alpha=0.15)

        print(f"    {label:20s}: S_an={S:.3f} O_an={over_mass:.3f} | "
              f"S_mc={np.mean(survived):.3f} O_mc={np.mean(is_overshoot):.3f} | "
              f"|w_a|={wa:.2f}")

    axes[0].legend(fontsize=8, loc='upper right')
    plt.subplots_adjust(left=0.07, right=0.98, bottom=0.14, top=0.92, wspace=0.22)
    fig.savefig(outdir / filename, dpi=150, bbox_inches='tight')
    plt.close('all')
    print(f'  Saved {filename}')


# ============================================================
# Integration Tests
# ============================================================

def local_integration_tests():
    """Run all integration tests for the package.

    Tests the core Laplace framework, Riccati solver, gap process,
    and MC simulation against analytical results. Then generates
    all paper figures.
    """
    np.set_printoptions(precision=8, linewidth=120)
    assets = create_paper_assets()
    eq = assets['equity']
    passed = 0; failed = 0

    print("=" * 70)
    print("INTEGRATION TESTS")
    print("=" * 70)

    # --- Test 1: Unbounded density normalization ---
    print("\n--- Test 1: Unbounded density normalization ---")
    eq_unb = AssetSpecification('equity_unb', eq.params,
                                 eq.mu_growth, eq.mu_stress,
                                 pi0=100, pi_floor=0)
    xg = np.linspace(-5, 5, 600)
    d0, d1 = compute_density(5.0, xg, eq_unb)
    norm_val = float(np.trapezoid(d0 + d1, xg))
    ok = abs(norm_val - 1.0) < 1e-4
    print(f"  integral(d0+d1) = {norm_val:.8f} (expect 1.0) {'✓' if ok else '✗'}")
    print(f"  min(d0+d1) = {np.min(d0+d1):.2e} (expect ≥ 0)")
    passed += ok; failed += (not ok)

    # No-jump test
    eq_nj = AssetSpecification('equity_nj',
                                RegimeSwitchParams(sigma0=0.15, sigma1=0.25,
                                                   lambda01=0.1, lambda10=1.0),
                                eq.mu_growth, eq.mu_stress,
                                pi0=100, pi_floor=0)
    d0nj, d1nj = compute_density(5.0, xg, eq_nj)
    norm_nj = float(np.trapezoid(d0nj + d1nj, xg))
    ok = abs(norm_nj - 1.0) < 1e-4
    print(f"  No-jump integral = {norm_nj:.8f} {'✓' if ok else '✗'}")
    passed += ok; failed += (not ok)

    # --- Test 2: Barrier density vs survival consistency ---
    print("\n--- Test 2: Barrier density vs survival (equity, T=10) ---")
    xb = np.linspace(0.001, 4.0, 1000)
    res = compute_wealth_density(10.0, xb, eq)
    gap_val = abs(res['survival_density'] - res['survival_analytic'])
    ok = gap_val < 1e-4
    print(f"  survival (density):  {res['survival_density']:.8f}")
    print(f"  survival (analytic): {res['survival_analytic']:.8f}")
    print(f"  consistency gap:     {gap_val:.2e} {'✓' if ok else '✗'}")
    passed += ok; failed += (not ok)

    # --- Test 3: Survival probability vs T ---
    print("\n--- Test 3: Survival probability vs T (equity) ---")
    print(f"  {'T':>5s} {'survival':>12s} {'stopping':>12s}")
    for Tv in [1, 2, 5, 10]:
        s = compute_survival(Tv, eq.x0, eq)
        print(f"  {Tv:5d} {s:12.6f} {1-s:12.6f}")
    ok = 0.5 < compute_survival(10, eq.x0, eq) < 1.0
    passed += ok; failed += (not ok)

    # --- Test 4: Three assets ---
    print("\n--- Test 4: Three assets, T=10 ---")
    for name, asset in assets.items():
        s = compute_survival(10, asset.x0, asset)
        print(f"  {name:16s}: x0={asset.x0:.4f}, surv={s:.6f}, stop={1-s:.6f}")

    # --- Test 5: Riccati ODE ---
    print("\n--- Test 5: Riccati ODE (equity, target=5%) ---")
    ell, ric = find_ell(eq, T, 0.04, r, r)
    d = ric.derived_at_tau(T)
    wa = abs(d['w_a'][0])
    Pi_star_0 = d['Pi_star'][0]
    print(f"  ℓ = {ell:.4f}")
    print(f"  a(0) = [{ric.a[0,0]:.4f}, {ric.a[1,0]:.4f}] (expect 1, 1)")
    print(f"  |ω*_a| = {wa:.4f}")
    print(f"  Π*(0)  = {Pi_star_0:.1f}")
    ok = abs(ric.a[0, 0] - 1.0) < 1e-6 and abs(ric.a[1, 0] - 1.0) < 1e-6
    print(f"  IC check: {'✓' if ok else '✗'}")
    passed += ok; failed += (not ok)

    # --- Test 6: Gap process vs MC ---
    print("\n--- Test 6: Gap process survival vs MC (equity) ---")
    gap = gap_process_asset(ric)
    surv_an = compute_survival(T, gap.x0, gap)
    mc = simulate_mv_optimal(ric, n_paths=100_000, seed=42)
    surv_mc = np.mean(mc['survived'])
    gap_pct = abs(surv_an - surv_mc) * 100
    ok = gap_pct < 5.0  # within 5 percentage points (discrete vs continuous)
    print(f"  Analytical survival: {surv_an:.4f}")
    print(f"  MC survival:         {surv_mc:.4f}")
    print(f"  Gap: {gap_pct:.1f} pp {'✓' if ok else '✗'}")
    passed += ok; failed += (not ok)

    # --- Test 7: Table 1 parameters ---
    print("\n--- Test 7: Table 1 parameter validation ---")
    expected = {
        'bonds':          (0.025, 0.020, 0.06, 0.09, 0.08, 0.05),
        'equity':         (0.045, 0.000, 0.15, 0.225, 0.25, 0.15),
        'private_equity': (0.070, 0.000, 0.20, 0.30, 0.30, 0.20),
    }
    for name, (mu1, mu2, s1, s2, crash, recov) in expected.items():
        a = assets[name]; p = a.params
        check = (abs(a.mu_growth - mu1) < 1e-6 and abs(a.mu_stress - mu2) < 1e-6
                 and abs(p.sigma0 - s1) < 1e-6 and abs(p.sigma1 - s2) < 1e-6)
        print(f"  {name:16s}: {'✓' if check else '✗'}")
        passed += check; failed += (not check)

    print(f"\n{'=' * 70}")
    print(f"RESULTS: {passed} passed, {failed} failed")
    print(f"{'=' * 70}")

    # --- Generate all figures ---
    print(f"\n{'=' * 70}")
    print("GENERATING ALL FIGURES")
    print(f"{'=' * 70}")

    outdir = Path('figures')
    outdir.mkdir(exist_ok=True)
    _generate_all_figures(outdir)

    print(f"\n{'=' * 70}")
    print(f"ALL TESTS AND FIGURES COMPLETE")
    print(f"{'=' * 70}")

    return passed, failed


# ============================================================
# Main entry point
# ============================================================

def _generate_all_figures(outdir):
    """Generate all 10 paper figures."""
    figures = {
        1: ('Opportunity set c=0%', lambda: figure_opportunity_set(
            AdvisorSpec(omega_0=1.0, c=0.0, q=2/3, q_dd=2.0),
            'opportunity_set_c0.png', outdir)),
        2: ('Opportunity set c=2.5%', lambda: figure_opportunity_set(
            AdvisorSpec(omega_0=1.0, c=0.025, q=2/3, q_dd=2.0),
            'opportunity_set_c25.png', outdir)),
        3: ('Allocation paths c=0%', lambda: figure_allocation_paths([
            ('Income', 1.00, '100%/0%/0%'), ('Conservative', 0.65, '65%/23%/12%'),
            ('Balanced', 0.35, '35%/43%/22%'), ('Growth', 0.00, '0%/67%/33%'),
            ], 0.0, 'allocation_paths_c0.png', outdir)),
        4: ('Allocation paths c=2.5%', lambda: figure_allocation_paths([
            ('Conservative', 0.85, '85%/10%/5%'), ('Moderate', 0.65, '65%/23%/12%'),
            ('Balanced', 0.35, '35%/43%/22%'), ('Growth', 0.00, '0%/67%/33%'),
            ], 0.025, 'allocation_paths_c25.png', outdir)),
        5: ('Allocation ±1σ bands', lambda: figure_allocation_bands([
            ('Income', 1.00, '100%/0%/0%'), ('Conservative', 0.65, '65%/23%/12%'),
            ('Balanced', 0.35, '35%/43%/22%'), ('Growth', 0.00, '0%/67%/33%'),
            ], 0.0, 'risky_allocation_subplots_c0.png', outdir)),
        6: ('Path dynamics (survived vs stopped)', lambda: figure_path_dynamics(
            'path_dynamics_balanced.png', outdir)),
        7: ('Floor vs Lipton (Growth)', lambda: figure_floor_vs_lipton(
            0.00, 'Growth', 'floor_vs_lipton.png', outdir)),
        8: ('Floor vs Lipton (Balanced)', lambda: figure_floor_vs_lipton(
            0.35, 'Balanced', 'floor_vs_lipton_balanced.png', outdir)),
        9: ('Mandate density overlay (MV vs BH)', lambda: figure_mandate_density_overlay(
            'mandate_density_overlay_c0.png', outdir)),
        10: ('Mandate comparison (analytical vs MC)', lambda: figure_mandate_comparison(
            'mandate_comparison.png', outdir)),
    }

    for num, (name, fn) in figures.items():
        print(f'\nFigure {num}: {name}')
        fn()
    print(f'\nAll {len(figures)} figures saved to {outdir}/')

    return figures


def main():
    parser = argparse.ArgumentParser(
        description='Generate paper figures and run integration tests')
    parser.add_argument('--outdir', type=str, default='figures',
                        help='Output directory for figures')
    parser.add_argument('--figure', type=int, default=None,
                        help='Generate only figure N (1-10)')
    parser.add_argument('--test', action='store_true',
                        help='Run integration tests and generate all figures')
    args = parser.parse_args()

    if args.test:
        local_integration_tests()
        return

    outdir = Path(args.outdir)
    outdir.mkdir(exist_ok=True)

    if args.figure is not None:
        # Build dict but don't execute; run only the requested figure
        figures = {
            1: ('Opportunity set c=0%', lambda: figure_opportunity_set(
                AdvisorSpec(omega_0=1.0, c=0.0, q=2/3, q_dd=2.0),
                'opportunity_set_c0.png', outdir)),
            2: ('Opportunity set c=2.5%', lambda: figure_opportunity_set(
                AdvisorSpec(omega_0=1.0, c=0.025, q=2/3, q_dd=2.0),
                'opportunity_set_c25.png', outdir)),
            3: ('Allocation paths c=0%', lambda: figure_allocation_paths([
                ('Income', 1.00, '100%/0%/0%'), ('Conservative', 0.65, '65%/23%/12%'),
                ('Balanced', 0.35, '35%/43%/22%'), ('Growth', 0.00, '0%/67%/33%'),
                ], 0.0, 'allocation_paths_c0.png', outdir)),
            4: ('Allocation paths c=2.5%', lambda: figure_allocation_paths([
                ('Conservative', 0.85, '85%/10%/5%'), ('Moderate', 0.65, '65%/23%/12%'),
                ('Balanced', 0.35, '35%/43%/22%'), ('Growth', 0.00, '0%/67%/33%'),
                ], 0.025, 'allocation_paths_c25.png', outdir)),
            5: ('Allocation ±1σ bands', lambda: figure_allocation_bands([
                ('Income', 1.00, '100%/0%/0%'), ('Conservative', 0.65, '65%/23%/12%'),
                ('Balanced', 0.35, '35%/43%/22%'), ('Growth', 0.00, '0%/67%/33%'),
                ], 0.0, 'risky_allocation_subplots_c0.png', outdir)),
            6: ('Path dynamics (survived vs stopped)', lambda: figure_path_dynamics(
                'path_dynamics_balanced.png', outdir)),
            7: ('Floor vs Lipton (Growth)', lambda: figure_floor_vs_lipton(
                0.00, 'Growth', 'floor_vs_lipton.png', outdir)),
            8: ('Floor vs Lipton (Balanced)', lambda: figure_floor_vs_lipton(
                0.35, 'Balanced', 'floor_vs_lipton_balanced.png', outdir)),
            9: ('Mandate density overlay (MV vs BH)', lambda: figure_mandate_density_overlay(
                'mandate_density_overlay_c0.png', outdir)),
            10: ('Mandate comparison (analytical vs MC)', lambda: figure_mandate_comparison(
                'mandate_comparison.png', outdir)),
        }
        if args.figure in figures:
            name, fn = figures[args.figure]
            print(f'Generating figure {args.figure}: {name}')
            fn()
        else:
            print(f'Unknown figure {args.figure}. Available: {list(figures.keys())}')
    else:
        _generate_all_figures(outdir)


if __name__ == '__main__':
    main()
