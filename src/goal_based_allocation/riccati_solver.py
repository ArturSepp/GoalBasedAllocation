"""
Riccati ODE solver for MV-optimal allocation under regime-switching.

Notation (paper → code):
  regime 1 (growth) = code index 0
  regime 2 (stress) = code index 1
  λ^[12] = lambda01,  λ^[21] = lambda10
  α^[1] = alpha0 = -η₀/(1+η₀) < 0  (expected crash loss)
  α^[2] = alpha1 = +η₁/(1-η₁) > 0  (expected recovery gain)
  μ̄^[i] = mu_growth/mu_stress       (total expected return, CMA observable)
  μ^[i]  = mu_bar  = μ̄ - λα         (diffusion drift between transitions)
  r_h    = max(r, c)                 (hurdle rate)
  r_c    = r_h - c = max(0, r-c)    (net floor growth rate)

Paper eq (RS_def):
  ã^[i] = (μ^[i] - r_h)·a^[i] + λ^[i→j]·α^[i]·a^[j]
  b̃^[i] = (μ^[i] - r_h)·b^[i] + λ^[i→j]·α^[i]·b^[j]
  Σ^[i] = (σ^[i])²·a^[i] + λ^[i→j]·(σ_J^[i])²·a^[j]
"""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from scipy.integrate import solve_ivp
from .regime_switch_paper import (AssetSpecification, RegimeSwitchParams,
                                  compute_density, compute_survival)


def _asset_derived_params(asset: AssetSpecification, r: float, c: float = None):
    """Precompute all derived parameters from asset specification.

    Stores both r and r_h = max(r, c) (hurdle rate). Risk premia in the
    Riccati system use μ^[i] - r_h (paper eq:RS_def).
    """
    if c is None:
        c = r  # default: no consumption drag
    par = asset.params
    alpha0 = -par.eta0 / (1 + par.eta0) if par.eta0 > 0 else 0.0
    alpha1 = par.eta1 / (1 - par.eta1) if par.eta1 > 0 else 0.0
    # mu_bar = diffusion drift = paper's μ^[i]
    # mu_bar = bar_mu - lambda * alpha  (paper Definition 2.3)
    mu_bar0 = asset.mu_growth - par.lambda01 * alpha0
    mu_bar1 = asset.mu_stress - par.lambda10 * alpha1
    sigJ0_sq = (1/(1 + 2*par.eta0) - 2/(1 + par.eta0) + 1) if par.eta0 > 0 else 0.0
    sigJ1_sq = (1/(1 - 2*par.eta1) - 2/(1 - par.eta1) + 1) if par.eta1 > 0 else 0.0
    r_h = max(r, c)  # hurdle rate (paper eq:hurdle)
    return {
        'alpha': (alpha0, alpha1),
        'mu_bar': (mu_bar0, mu_bar1),       # μ^[i]: diffusion drift
        'mu': (asset.mu_growth, asset.mu_stress),  # μ̄^[i]: total return
        'sigma': (par.sigma0, par.sigma1),
        'sigJ_sq': (sigJ0_sq, sigJ1_sq),
        'lam': (par.lambda01, par.lambda10),
        'eta': (par.eta0, par.eta1),
        'r': r,
        'r_h': r_h,
    }


def _compute_tildes(a0, a1, b0, b1, dp):
    """Compute ã, b̃, Σ, ω*_a, Π* from Riccati coefficients.

    Paper eq (RS_def):
      ã^[i] = (μ^[i] - r_h)·a^[i] + λ·α^[i]·a^[j]
      b̃^[i] = (μ^[i] - r_h)·b^[i] + λ·α^[i]·b^[j]
      Σ^[i] = σ²·a^[i] + λ·σ_J²·a^[j]
    where μ^[i] = mu_bar (diffusion drift) and r_h = max(r, c).
    """
    mu_bar0, mu_bar1 = dp['mu_bar']   # μ^[i]: diffusion drift
    alpha0, alpha1 = dp['alpha']
    sig0, sig1 = dp['sigma']
    sigJ0_sq, sigJ1_sq = dp['sigJ_sq']
    lam01, lam10 = dp['lam']
    r_h = dp['r_h']

    # Paper eq (RS_def): ã = (μ - r_h)·a + λ·α·a_j
    a_tilde0 = (mu_bar0 - r_h)*a0 + lam01*alpha0*a1
    a_tilde1 = (mu_bar1 - r_h)*a1 + lam10*alpha1*a0
    Sigma0 = sig0**2 * a0 + lam01 * sigJ0_sq * a1
    Sigma1 = sig1**2 * a1 + lam10 * sigJ1_sq * a0
    w_a0 = -a_tilde0 / Sigma0 if abs(Sigma0) > 1e-20 else 0.0
    w_a1 = -a_tilde1 / Sigma1 if abs(Sigma1) > 1e-20 else 0.0
    # Paper eq (RS_def): b̃ = (μ - r_h)·b + λ·α·b_j
    b_tilde0 = (mu_bar0 - r_h)*b0 + lam01*alpha0*b1
    b_tilde1 = (mu_bar1 - r_h)*b1 + lam10*alpha1*b0
    Pi_star0 = -b_tilde0 / (2*a_tilde0) if abs(a_tilde0) > 1e-15 else 0.0
    Pi_star1 = -b_tilde1 / (2*a_tilde1) if abs(a_tilde1) > 1e-15 else 0.0

    return {
        'a_tilde': (a_tilde0, a_tilde1),
        'b_tilde': (b_tilde0, b_tilde1),
        'Sigma': (Sigma0, Sigma1),
        'w_a': (w_a0, w_a1),
        'Pi_star': (Pi_star0, Pi_star1),
    }


@dataclass
class RiccatiSolution:
    tau_grid: np.ndarray
    a: np.ndarray          # shape (2, N)
    b: np.ndarray          # shape (2, N)
    gamma: np.ndarray      # shape (2, N)
    asset: AssetSpecification
    dp: dict
    ell: float
    r: float
    c: float

    @property
    def T(self): return self.tau_grid[-1]
    @property
    def r_c(self): return max(0.0, self.r - self.c)
    @property
    def r_h(self): return max(self.r, self.c)

    def at_tau(self, tau):
        a0 = np.interp(tau, self.tau_grid, self.a[0])
        a1 = np.interp(tau, self.tau_grid, self.a[1])
        b0 = np.interp(tau, self.tau_grid, self.b[0])
        b1 = np.interp(tau, self.tau_grid, self.b[1])
        return a0, a1, b0, b1

    def derived_at_tau(self, tau):
        return _compute_tildes(*self.at_tau(tau), self.dp)

    def omega_star(self, t, Pi, regime):
        d = self.derived_at_tau(self.T - t)
        if abs(Pi) < 1e-12: return 0.0
        # Paper eq (27): ω* = (ã/Σ)·(Π*/Π − 1) = -w_a·(Π*/Π − 1)
        return -d['w_a'][regime] * (d['Pi_star'][regime] / Pi - 1)

    def expected_wealth(self, regime=0):
        return -self.b[regime, -1] / (2 * self.a[regime, -1])


def solve_riccati(asset, T, ell, r=0.02, c=0.02, n_steps=10000):
    dp = _asset_derived_params(asset, r, c)
    par = asset.params
    r_h = dp['r_h']                  # max(r, c)
    r_c = max(0.0, r - c)            # r_h - c = max(0, r - c)
    mu_bar0, mu_bar1 = dp['mu_bar']  # μ^[i]: diffusion drift
    alpha0, alpha1 = dp['alpha']
    lam01, lam10 = dp['lam']

    def rhs(tau, y):
        a0, a1, b0, b1, g0, g1 = y
        d = _compute_tildes(a0, a1, b0, b1, dp)
        at0, at1 = d['a_tilde']
        bt0, bt1 = d['b_tilde']
        S0, S1 = d['Sigma']
        wa0, wa1 = d['w_a']
        # Paper eq (ode_a): a' = (2r_c - λ)a + λa_j - ã²/Σ
        da0 = (2*r_c - lam01)*a0 + lam01*a1 - at0**2/S0
        da1 = (2*r_c - lam10)*a1 + lam10*a0 - at1**2/S1
        # Paper eq (ode_b_matrix): B matrix with μ^[i] - r_h
        db0 = (r_c - lam01 + wa0*(mu_bar0 - r_h))*b0 + lam01*(1 + wa0*alpha0)*b1
        db1 = lam10*(1 + wa1*alpha1)*b0 + (r_c - lam10 + wa1*(mu_bar1 - r_h))*b1
        # Paper eq (ode_g): γ' = -λγ + λγ_j - b̃²/(4Σ)
        dg0 = -lam01*g0 + lam01*g1 - bt0**2/(4*S0)
        dg1 = -lam10*g1 + lam10*g0 - bt1**2/(4*S1)
        return [da0, da1, db0, db1, dg0, dg1]

    y0 = [1.0, 1.0, -ell, -ell, 0.0, 0.0]
    sol = solve_ivp(rhs, [0, T], y0, method='RK45',
                    t_eval=np.linspace(0, T, n_steps+1),
                    rtol=1e-10, atol=1e-12, max_step=T/1000)
    if not sol.success:
        raise RuntimeError(f"ODE failed: {sol.message}")
    return RiccatiSolution(
        tau_grid=sol.t, a=np.array([sol.y[0], sol.y[1]]),
        b=np.array([sol.y[2], sol.y[3]]),
        gamma=np.array([sol.y[4], sol.y[5]]),
        asset=asset, dp=dp, ell=ell, r=r, c=c)


def find_ell(asset, T, target_return, r=0.02, c=0.02, regime=0):
    Pi_target = asset.pi0 * np.exp(target_return * T)
    sol0 = solve_riccati(asset, T, 1.0, r, c)
    sol1 = solve_riccati(asset, T, 10.0, r, c)
    E0, E1 = sol0.expected_wealth(regime), sol1.expected_wealth(regime)
    ell_opt = 1.0 + (Pi_target - E0) / (E1 - E0) * 9.0
    return ell_opt, solve_riccati(asset, T, ell_opt, r, c)


def simulate_mv_optimal(ric, n_paths=100_000, steps_per_year=260, seed=42):
    """MC of wealth under MV-optimal allocation with overshoot tracking.

    The wealth SDE between transitions (paper eq:wealth_sde):
      dΠ/Π = [r_c + ω·(μ^[i] − r_h)]dt + ω·σ^[i]·dW
    where μ^[i] = mu_bar (diffusion drift) and r_h = max(r, c).
    """
    rng = np.random.default_rng(seed)
    par = ric.asset.params
    T = ric.T; n_steps = int(T*steps_per_year)
    dt = T / n_steps; sqrt_dt = np.sqrt(dt)
    mu_bar = ric.dp['mu_bar']  # μ^[i]: diffusion drift
    r_h = ric.r_h              # max(r, c)
    r_c = ric.r_c              # max(0, r - c)

    Pi = np.full(n_paths, float(ric.asset.pi0))
    regime = np.zeros(n_paths, dtype=int)
    stopped = np.zeros(n_paths, dtype=bool)
    stop_time = np.full(n_paths, np.inf)
    Pi_at_stop = np.zeros(n_paths)
    L_at_stop = np.zeros(n_paths)
    L = np.full(n_paths, float(ric.asset.pi_floor))

    for step in range(n_steps):
        t = step * dt
        alive = ~stopped
        if not np.any(alive): break

        d = ric.derived_at_tau(T - t)
        w_a = np.where(regime == 0, d['w_a'][0], d['w_a'][1])
        Pi_star = np.where(regime == 0, d['Pi_star'][0], d['Pi_star'][1])
        omega = np.where(alive & (Pi > 1e-10), -w_a*(Pi_star/Pi - 1), 0.0)
        omega = np.clip(omega, -2.0, 5.0)

        p_sw = np.where(regime == 0, par.lambda01*dt, par.lambda10*dt)
        switches = alive & (rng.uniform(size=n_paths) < p_sw)
        crash = switches & (regime == 0)
        recov = switches & (regime == 1)

        Z = rng.standard_normal(n_paths)
        sig = np.where(regime == 0, par.sigma0, par.sigma1)
        mb = np.where(regime == 0, mu_bar[0], mu_bar[1])
        log_ret = (r_c + omega*(mb - r_h) - 0.5*(omega*sig)**2)*dt + omega*sig*sqrt_dt*Z
        Pi_new = Pi * np.exp(log_ret)

        nc = np.sum(crash & alive)
        if nc > 0 and par.eta0 > 0:
            idx = np.where(crash & alive)[0]
            J = -rng.exponential(par.eta0, size=nc)
            Pi_new[idx] = Pi[idx] * np.maximum(1 + omega[idx]*(np.exp(J)-1), 1e-10)

        nr = np.sum(recov & alive)
        if nr > 0 and par.eta1 > 0:
            idx = np.where(recov & alive)[0]
            J = rng.exponential(par.eta1, size=nr)
            Pi_new[idx] = Pi[idx] * np.maximum(1 + omega[idx]*(np.exp(J)-1), 1e-10)

        regime = np.where(crash, 1, np.where(recov, 0, regime))
        Pi = np.where(alive, Pi_new, Pi)
        L *= np.exp(r_c * dt)

        # Stop: record actual wealth (may be < L for jump overshoot)
        newly = alive & (Pi <= L)
        if np.any(newly):
            stopped[newly] = True
            stop_time[newly] = t + dt
            Pi_at_stop[newly] = Pi[newly]
            L_at_stop[newly] = L[newly]

    # Terminal wealth: stopped paths grow at r_c from stop time
    Pi_terminal = Pi.copy()
    is_stopped = stopped
    remaining = T - stop_time[is_stopped]
    Pi_terminal[is_stopped] = Pi_at_stop[is_stopped] * np.exp(r_c * remaining)

    L_T = ric.asset.pi_floor * np.exp(r_c * T)

    # Classify: overshoot = stopped with Π_stop significantly below L_stop
    is_overshoot = stopped & (Pi_at_stop < L_at_stop * 0.999)

    return {'Pi_T': Pi_terminal, 'L_T': L_T, 'survived': ~stopped,
            'is_overshoot': is_overshoot, 'regime_T': regime, 'n_paths': n_paths,
            'stop_time': stop_time, 'Pi_at_stop': Pi_at_stop}


def _eta_eff_quadrature(eta0, w):
    """Exact effective gap jump size via numerical quadrature.

    The gap jump at crash is ΔX = -ln(1 + w·(1-e^J)) with J ~ -Exp(η₀).
    We approximate ΔX ~ -Exp(η_eff) by mean-matching:

      η_eff = E[ln(1 + w·(1-e^J))] = ∫₀^∞ ln(1+w(1-e^{-t})) · (1/η₀)e^{-t/η₀} dt

    This is a single well-behaved 1D integral, exact to machine precision.

    At w = 1, η_eff ≈ E[ln(2-e^J)] (a specific function of η₀).
    For small ε = w-1, the first-order correction is:
      η_eff ≈ η_eff(w=1) + ε · E[(1-e^J)/(2-e^J)]
    """
    from scipy.integrate import quad
    if eta0 <= 0 or w <= 0:
        return 0.0
    def integrand(t):
        return np.log(1 + w*(1 - np.exp(-t))) * (1/eta0) * np.exp(-t/eta0)
    result, _ = quad(integrand, 0, max(50*eta0, 20), limit=200)
    return max(result, 1e-10)


def gap_process_asset(ric):
    """Construct AssetSpecification for the gap process Z = Π* - Π.

    Under MV-optimal allocation, the gap Z = Π* - Π has RS-JD dynamics
    with constant coefficients (flat barrier after r_c drift removal).

    Paper eq (nu_eff):
      σ_eff = |ω*_a| · σ
      ν_eff = |ω*_a|·(μ^[i] − r_h) + ½σ²_eff
    where μ^[i] = mu_bar (diffusion drift) and r_h = max(r, c).
    """
    par = ric.asset.params

    # Steady-state ω*_a (use τ=T, approximately constant)
    d = ric.derived_at_tau(ric.T)
    w_a0, w_a1 = d['w_a']
    w0 = abs(w_a0); w1 = abs(w_a1)
    mu_bar0, mu_bar1 = ric.dp['mu_bar']  # μ^[i]: diffusion drift
    r_h = ric.r_h                         # max(r, c)

    # Effective vols
    sigma_eff0 = w0 * par.sigma0
    sigma_eff1 = w1 * par.sigma1

    # Effective jump sizes: exact quadrature (no approximation)
    eta_eff0 = _eta_eff_quadrature(par.eta0, w0) if par.eta0 > 0 else 0.0
    eta_eff1 = _eta_eff_quadrature(par.eta1, w1) if par.eta1 > 0 else 0.0

    # Target log-drift for gap process X = ln(B/Z):
    #   Paper eq (nu_eff): ν_eff = |ω*_a|·(μ − r_h) + ½σ²_eff
    # where μ = mu_bar (diffusion drift), r_h = max(r, c).
    # The r_c cancels in the log-gap because both B and Z grow at r_c.
    #
    # AssetSpecification.compute_drifts computes:
    #   nu = mu_growth + comp − ½σ²
    # so we back-compute mu_growth to get the target nu.
    nu_target0 = w0 * (mu_bar0 - r_h) + 0.5 * sigma_eff0**2
    nu_target1 = w1 * (mu_bar1 - r_h) + 0.5 * sigma_eff1**2

    gap_par = RegimeSwitchParams(
        sigma0=sigma_eff0, sigma1=sigma_eff1,
        lambda01=par.lambda01, lambda10=par.lambda10,
        eta0=eta_eff0, eta1=eta_eff1)

    # Compute comp that AssetSpec will add internally
    comp0 = par.lambda01 * eta_eff0 / (1 + eta_eff0) if eta_eff0 > 0 else 0.0
    comp1 = par.lambda10 * eta_eff1 / (1 - eta_eff1) if eta_eff1 > 0 else 0.0
    mu_growth = nu_target0 - comp0 + 0.5 * sigma_eff0**2
    mu_stress = nu_target1 + comp1 + 0.5 * sigma_eff1**2

    # Barrier: B = Π* - L, Z₀ = Π* - Π₀
    # Use terminal Π*(T) since Π*(t) = Π*(0)·exp(r_c·t), barrier is flat in log
    Pi_star0 = d['Pi_star'][0]
    B0 = Pi_star0 - ric.asset.pi_floor
    Z0 = Pi_star0 - ric.asset.pi0

    return AssetSpecification(
        name=f'{ric.asset.name}_gap', params=gap_par,
        mu_growth=mu_growth, mu_stress=mu_stress,
        pi0=max(B0, 0.01), pi_floor=max(Z0, 0.001))
