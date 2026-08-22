"""Analytical and Monte Carlo terminal-wealth helpers for portfolio mandates."""
import numpy as np
from .regime_switch_paper import (
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
