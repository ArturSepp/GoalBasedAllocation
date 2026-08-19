"""Mandate terminal wealth: analytical vs MC for all three mandates.
Produces:
  1. Per-mandate comparison (3 panels): MC histogram + analytical curves
  2. All mandates analytical density overlay
  3. All mandates analytical CDF overlay
"""
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .regime_switch_paper import (create_paper_assets, create_paper_mandates,
    RegimeSwitchParams, AssetSpecification, compute_density, compute_survival,
    compute_overshoot_density)

RHO = np.array([[1,0.3,0.3],[0.3,1,0.8],[0.3,0.8,1]])


def mandate_effective_asset(mandate):
    alloc = mandate.allocations
    assets_list = list(alloc.keys())
    w = np.array([alloc[a] for a in assets_list])
    n_a = len(assets_list)
    sig_g = np.array([a.params.sigma0 for a in assets_list])
    sig_s = np.array([a.params.sigma1 for a in assets_list])
    cov_g = np.outer(sig_g, sig_g) * RHO[:n_a,:n_a]
    cov_s = np.outer(sig_s, sig_s) * RHO[:n_a,:n_a]
    sig_port_g = np.sqrt(w @ cov_g @ w)
    sig_port_s = np.sqrt(w @ cov_s @ w)
    mu_g = sum(alloc[a]*a.mu_growth for a in assets_list)
    mu_s = sum(alloc[a]*a.mu_stress for a in assets_list)

    # Portfolio jump sizes via numerical integration (paper eq:eta_port)
    from .client_solver import portfolio_eta_quadrature
    etas0 = [a.params.eta0 for a in assets_list]
    etas1 = [a.params.eta1 for a in assets_list]
    eta0_p = portfolio_eta_quadrature(w, etas0, crash=True)
    eta1_p = portfolio_eta_quadrature(w, etas1, crash=False)

    lam01 = assets_list[0].params.lambda01
    lam10 = assets_list[0].params.lambda10
    par = RegimeSwitchParams(sigma0=sig_port_g, sigma1=sig_port_s,
        lambda01=lam01, lambda10=lam10, eta0=eta0_p, eta1=eta1_p)
    L0 = sum(alloc[a]*a.pi_floor for a in assets_list)
    return AssetSpecification(f'mandate_{mandate.name}', par,
        mu_growth=mu_g, mu_stress=mu_s, pi0=100.0, pi_floor=L0)


def simulate_mandate_mc(eff, T=10.0, n_paths=200_000, seed=42):
    rng = np.random.default_rng(seed)
    par = eff.params; r_c = 0.02-0.03
    n_steps = int(T*260); dt = T/n_steps; sqrt_dt = np.sqrt(dt)
    X = np.full(n_paths, eff.x0)
    regime = np.zeros(n_paths, dtype=int)
    stopped = np.zeros(n_paths, dtype=bool)
    X_at_stop = np.zeros(n_paths)
    stop_time = np.full(n_paths, np.inf)
    is_jump = np.zeros(n_paths, dtype=bool)
    for step in range(n_steps):
        alive = ~stopped
        nu = np.where(regime==0, eff.nu0, eff.nu1)
        sig = np.where(regime==0, par.sigma0, par.sigma1)
        dW = rng.standard_normal(n_paths)*sqrt_dt
        u = rng.uniform(size=n_paths)
        X_new = X + nu*dt + sig*dW
        p_sw = np.where(regime==0, par.lambda01*dt, par.lambda10*dt)
        sw = alive & (u < p_sw)
        crash = sw & (regime==0); recov = sw & (regime==1)
        if par.eta0>0 and np.any(crash):
            X_new[crash] = X[crash] - rng.exponential(par.eta0, size=np.sum(crash))
        if par.eta1>0 and np.any(recov):
            X_new[recov] = X[recov] + rng.exponential(par.eta1, size=np.sum(recov))
        regime = np.where(crash,1,np.where(recov,0,regime))
        X = np.where(alive, X_new, X)
        newly = alive & (X<=0)
        if np.any(newly):
            stopped[newly]=True; X_at_stop[newly]=X[newly]
            stop_time[newly]=(step+1)*dt; is_jump[newly]=crash[newly]
    L_T = eff.pi_floor*np.exp(-0.01*T)
    Pi_T = np.where(stopped, L_T*np.exp(X_at_stop), L_T*np.exp(X))
    return {'Pi_T': Pi_T, 'L_T': L_T, 'survived': ~stopped,
            'is_overshoot': stopped & is_jump, 'n_paths': n_paths}


def compute_mandate_analytical(eff, T=10.0):
    r_c = -0.01; L_T = eff.pi_floor*np.exp(r_c*T)
    surv = compute_survival(T, eff.x0, eff)

    # Survived density
    x_grid = np.linspace(0.001, 6.0, 800)
    d0, d1 = compute_density(T, x_grid, eff)
    d_total = d0 + d1
    Pi_surv = L_T * np.exp(x_grid)
    dens_surv = d_total / (L_T * np.exp(x_grid))

    # Overshoot density
    d_ov = np.linspace(0.001, 5.0, 400)
    f_ov = compute_overshoot_density(T, d_ov, eff)
    over_mass = np.trapezoid(f_ov, d_ov)
    Pi_ov = L_T * np.exp(-d_ov)
    dens_ov = f_ov / (L_T * np.exp(-d_ov))

    return {
        'surv': surv, 'over_mass': over_mass, 'floor': 1-surv-over_mass,
        'L_T': L_T,
        'Pi_surv': Pi_surv, 'dens_surv': dens_surv,
        'Pi_ov': Pi_ov, 'dens_ov': dens_ov,
        'x_grid': x_grid, 'd_total': d_total,
        'd_ov_grid': d_ov, 'f_ov': f_ov,
    }


def figure1_per_mandate():
    """Three-panel: per-mandate MC vs analytical comparison."""
    assets = create_paper_assets()
    mandates = create_paper_mandates(assets)
    T = 10.0; n_paths = 200_000
    fig, axes = plt.subplots(3, 1, figsize=(12, 14), dpi=150)

    mandate_order = ['conservative', 'balanced', 'growth']
    labels = {'conservative': '70/20/10', 'balanced': '40/40/20', 'growth': '10/45/45'}

    for idx, mname in enumerate(mandate_order):
        ax = axes[idx]; m = mandates[mname]
        eff = mandate_effective_asset(m)
        an = compute_mandate_analytical(eff, T)
        mc = simulate_mandate_mc(eff, T, n_paths, seed=42)

        surv_mc = np.mean(mc['survived'])
        over_mc = np.mean(mc['is_overshoot'])
        floor_mc = np.mean(~mc['survived'] & ~mc['is_overshoot'])
        Pi_surv_mc = mc['Pi_T'][mc['survived']]
        Pi_over_mc = mc['Pi_T'][mc['is_overshoot']]

        print(f"{mname}: An surv={an['surv']:.4f} MC={surv_mc:.4f}, "
              f"An over={an['over_mass']:.4f} MC={over_mc:.4f}")

        # Range: find where analytical density drops to ~0.1% of peak
        peak = np.max(an['dens_surv'])
        pi_hi = float(an['Pi_surv'][np.searchsorted(-an['dens_surv'][::-1], -peak*0.005)])
        pi_hi = max(pi_hi, np.percentile(Pi_surv_mc, 99.9)) * 1.05
        pi_lo = min(-10, np.min(Pi_over_mc)-5) if len(Pi_over_mc)>0 else -10

        bins = np.linspace(pi_lo, pi_hi, 60)
        dx = bins[1]-bins[0]; ctrs = 0.5*(bins[:-1]+bins[1:])
        h_s,_ = np.histogram(Pi_surv_mc, bins=bins)
        h_o,_ = np.histogram(Pi_over_mc, bins=bins)

        ax.bar(ctrs, h_s/(n_paths*dx), width=dx, alpha=0.3, color='steelblue',
               edgecolor='none', label='Survived (MC)')
        ax.bar(ctrs, h_o/(n_paths*dx), width=dx, alpha=0.4, color='salmon',
               edgecolor='none', label='Overshoot (MC)')

        mask = (an['Pi_surv']>pi_lo) & (an['Pi_surv']<pi_hi) & (an['dens_surv']>0)
        ax.plot(an['Pi_surv'][mask], an['dens_surv'][mask], 'k-', lw=1.8, label='Analytical')
        mask_o = (an['Pi_ov']>pi_lo) & (an['Pi_ov']<pi_hi) & (an['dens_ov']>0)
        if np.any(mask_o):
            ax.plot(an['Pi_ov'][mask_o], an['dens_ov'][mask_o], 'r-', lw=1.8, label='Overshoot (An)')

        ax.axvline(an['L_T'], color='red', ls='--', lw=1, alpha=0.5)
        ymax = max(np.max(h_s/(n_paths*dx)), np.max(h_o/(n_paths*dx))) * 1.15
        ax.set_ylim(0, ymax); ax.set_xlim(pi_lo, pi_hi)

        ax.text(an['L_T']+2, ymax*0.85,
                f'$L_T$={an["L_T"]:.0f}\nFloor: {floor_mc:.1%}\nOver: {over_mc:.1%}',
                fontsize=8, color='red', va='top')

        dname = mname.capitalize()
        ax.set_title(f'{dname} ({labels[mname]}) — '
                     f'$\\sigma$=[{eff.params.sigma0:.1%},{eff.params.sigma1:.1%}], '
                     f'$\\eta$={eff.params.eta0:.3f}', fontsize=11)
        ax.set_xlabel('Terminal wealth $\\Pi_T$', fontsize=10)
        ax.set_ylabel('Density', fontsize=10)
        ax.legend(fontsize=7, loc='upper right')
        ax.grid(True, alpha=0.3)
        ax.text(0.02, 0.97,
            f'Surv: An={an["surv"]:.3f} MC={surv_mc:.3f}\n'
            f'E[$\\Pi_T$|s]={np.mean(Pi_surv_mc):.1f}, σ={np.std(Pi_surv_mc):.1f}',
            transform=ax.transAxes, fontsize=8, va='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    fig.suptitle('Mandate Terminal Wealth: Analytical vs MC ($T$=10y, $\\rho_{eq,pe}$=0.8)',
                 fontsize=13, y=0.99)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    for ext in ['pdf','png']:
        fig.savefig(f'/mnt/user-data/outputs/mandate_comparison.{ext}', bbox_inches='tight', dpi=150)
    print("Saved mandate_comparison")


def figure2_density_overlay():
    """All three mandate densities on one plot."""
    assets = create_paper_assets()
    mandates = create_paper_mandates(assets)
    T = 10.0

    fig, ax = plt.subplots(1, 1, figsize=(12, 6), dpi=150)
    colors = {'conservative': 'C0', 'balanced': 'C1', 'growth': 'C2'}
    labels = {'conservative': 'Conservative (70/20/10)',
              'balanced': 'Balanced (40/40/20)',
              'growth': 'Growth (10/45/45)'}

    for mname in ['conservative', 'balanced', 'growth']:
        eff = mandate_effective_asset(mandates[mname])
        an = compute_mandate_analytical(eff, T)
        c = colors[mname]

        # Survived + overshoot on same axis
        mask = an['dens_surv'] > 1e-6
        ax.plot(an['Pi_surv'][mask], an['dens_surv'][mask], color=c, lw=2,
                label=labels[mname])
        ax.fill_between(an['Pi_surv'][mask], 0, an['dens_surv'][mask], color=c, alpha=0.1)

        # Overshoot
        mask_o = an['dens_ov'] > 1e-6
        if np.any(mask_o):
            ax.plot(an['Pi_ov'][mask_o], an['dens_ov'][mask_o], color=c, lw=1.5, ls='--')
            ax.fill_between(an['Pi_ov'][mask_o], 0, an['dens_ov'][mask_o], color=c, alpha=0.08)

        # Floor line
        ax.axvline(an['L_T'], color=c, ls=':', lw=0.8, alpha=0.5)

        # Annotation
        print(f"{mname}: surv={an['surv']:.3f}, over={an['over_mass']:.3f}, L_T={an['L_T']:.0f}")

    ax.set_title('Mandate Terminal Wealth Density ($T$=10y)', fontsize=13)
    ax.set_xlabel('Terminal wealth $\\Pi_T$', fontsize=11)
    ax.set_ylabel('Density', fontsize=11)
    ax.legend(fontsize=10)
    ax.set_xlim(-20, 350); ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    for ext in ['pdf','png']:
        fig.savefig(f'/mnt/user-data/outputs/mandate_density_overlay.{ext}', bbox_inches='tight', dpi=150)
    print("Saved mandate_density_overlay")


def figure3_cdf_overlay():
    """All three mandate CDFs on one plot."""
    assets = create_paper_assets()
    mandates = create_paper_mandates(assets)
    T = 10.0

    fig, ax = plt.subplots(1, 1, figsize=(12, 6), dpi=150)
    colors = {'conservative': 'C0', 'balanced': 'C1', 'growth': 'C2'}
    labels = {'conservative': 'Conservative (70/20/10)',
              'balanced': 'Balanced (40/40/20)',
              'growth': 'Growth (10/45/45)'}

    for mname in ['conservative', 'balanced', 'growth']:
        eff = mandate_effective_asset(mandates[mname])
        an = compute_mandate_analytical(eff, T)
        c = colors[mname]
        L_T = an['L_T']

        # Build CDF from overshoot + floor atom + survived density
        # Need a unified Pi grid
        Pi_grid = np.linspace(-20, 400, 2000)
        cdf = np.zeros_like(Pi_grid)

        # Overshoot CDF: ∫_{-∞}^{Π} f_over dΠ'
        # f_over is in terms of d = -ln(Π/L_T), so integrate in d
        # P(Π_over ≤ Π) = P(d ≥ -ln(Π/L_T)) = ∫_{d_min}^∞ f_ov dd
        for i, pi in enumerate(Pi_grid):
            if pi <= 0:
                # Overshoot tail: very low wealth
                d_val = -np.log(max(pi, 0.01)/L_T) if pi > 0 else 10.0
                # P(Π_ov ≤ pi) = mass in overshoot with d ≥ d_val
                mask = an['d_ov_grid'] >= d_val
                if np.any(mask):
                    cdf[i] = np.trapezoid(an['f_ov'][mask], an['d_ov_grid'][mask])
            elif pi < L_T:
                # Below floor: overshoot only
                d_val = -np.log(pi/L_T)
                mask = an['d_ov_grid'] >= d_val
                cdf[i] = np.trapezoid(an['f_ov'][mask], an['d_ov_grid'][mask]) if np.any(mask) else an['over_mass']
            else:
                # Above floor: overshoot + floor + survived up to pi
                x_val = np.log(pi/L_T)
                mask_x = an['x_grid'] <= x_val
                surv_below = np.trapezoid(an['d_total'][mask_x], an['x_grid'][mask_x]) if np.any(mask_x) else 0
                cdf[i] = an['over_mass'] + an['floor'] + surv_below

        ax.plot(Pi_grid, cdf, color=c, lw=2, label=labels[mname])
        ax.axvline(L_T, color=c, ls=':', lw=0.8, alpha=0.5)

        # Mark key quantiles
        for q in [0.05, 0.5]:
            idx = np.searchsorted(cdf, q)
            if idx < len(Pi_grid):
                ax.plot(Pi_grid[idx], q, 'o', color=c, markersize=4)
                ax.annotate(f'{q:.0%}: {Pi_grid[idx]:.0f}', xy=(Pi_grid[idx], q),
                           xytext=(Pi_grid[idx]+5, q+0.03), fontsize=7, color=c)

    ax.axhline(1.0, color='gray', ls='-', lw=0.5, alpha=0.3)
    ax.set_title('Mandate Terminal Wealth CDF ($T$=10y)', fontsize=13)
    ax.set_xlabel('Terminal wealth $\\Pi_T$', fontsize=11)
    ax.set_ylabel('$P(\\Pi_T \\leq \\Pi)$', fontsize=11)
    ax.legend(fontsize=10, loc='lower right')
    ax.set_xlim(-20, 350); ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    for ext in ['pdf','png']:
        fig.savefig(f'/mnt/user-data/outputs/mandate_cdf_overlay.{ext}', bbox_inches='tight', dpi=150)
    print("Saved mandate_cdf_overlay")


if __name__ == "__main__":
    print("="*60)
    figure1_per_mandate()
    print()
    figure2_density_overlay()
    print()
    figure3_cdf_overlay()
    print("\nAll done.")
