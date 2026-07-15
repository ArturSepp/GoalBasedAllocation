"""
Regime-Switching Jump-Diffusion: Analytical Density via Laplace Inversion.

Companion code to: "Dynamic Mean-Variance Portfolio Allocation under Regime-Switching
Jump-Diffusions with Absorbing Barriers and Distribution Matching" (Sepp, 2026).

Solves the forward Kolmogorov (Fokker-Planck) equation for the transition
density of a two-state regime-switching process with exponential jumps
at regime transitions, with an absorbing lower barrier.

Paper notation (η₀ > 0, η₁ > 0 throughout):
  Regime 0 = growth,  Regime 1 = stress
  Crash  (0→1):  log-wealth jump ~ −Exp(1/η₀),  mgf factor (1−η₀ψ)
  Recovery (1→0): log-wealth jump ~ +Exp(1/η₁),  mgf factor (1+η₁ψ)

  Forward characteristic exponents:
    G⁰(ψ) = (−½σ₀²ψ² + ν₀ψ + λ₀₁+p)(1 − η₀ψ)
    G¹(ψ) = (−½σ₁²ψ² + ν₁ψ + λ₁₀+p)(1 + η₁ψ)

  Characteristic polynomial: G⁰(ψ)·G¹(ψ) − λ₀₁λ₁₀ = 0

  Coupling ratio (forward): β_k = G⁰(ψ_k) / λ₁₀
"""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from .laplace_inversion import laplace_invert_abate_whitt

def _safe_exp(x):
    """Exp with overflow protection — returns 0 for large arguments."""
    return np.exp(np.clip(x, -700, 700))



# ==============================================================================
# DATA CLASSES
# ==============================================================================

@dataclass(frozen=True)
class RegimeSwitchParams:
    """Structural parameters for the two-state regime-switching jump-diffusion.

    These are the process parameters that do NOT depend on asset-level CMAs.
    Drifts are computed from CMAs via :meth:`compute_drifts`.
    """
    sigma0: float       # diffusion vol, growth regime
    sigma1: float       # diffusion vol, stress regime
    lambda01: float     # crash transition intensity (0→1)
    lambda10: float     # recovery transition intensity (1→0)
    eta0: float = 0.0   # crash jump size parameter (>0; 0 = no jumps)
    eta1: float = 0.0   # recovery jump size parameter (>0; 0 = no jumps)

    @property
    def has_jumps(self) -> bool:
        return self.eta0 > 1e-12 or self.eta1 > 1e-12

    def compute_drifts(self, mu_growth: float, mu_stress: float) -> tuple[float, float]:
        """Convert CMA total returns (μ̄) to compensated log-space drifts ν₀, ν₁.

        Paper eq (2.10):  ν^[i] = μ^[i] − ½(σ^[i])²
        where μ^[i] = μ̄^[i] − λ^[i→j]·α^[i] is the diffusion drift
        (paper Definition 2.3), and:
          comp₀ = λ₀₁·η₀/(1+η₀)  = −λ₀₁·α^[1]  (positive, crash compensation)
          comp₁ = λ₁₀·η₁/(1−η₁)  = +λ₁₀·α^[2]  (positive, recovery compensation)
        So: ν₀ = μ̄₀ + comp₀ − ½σ₀² = μ₀ − ½σ₀²
            ν₁ = μ̄₁ − comp₁ − ½σ₁² = μ₁ − ½σ₁²
        """
        comp0 = self.lambda01 * self.eta0 / (1 + self.eta0) if self.eta0 > 0 else 0.0
        comp1 = self.lambda10 * self.eta1 / (1 - self.eta1) if self.eta1 > 0 else 0.0
        nu0 = mu_growth + comp0 - 0.5 * self.sigma0**2
        nu1 = mu_stress - comp1 - 0.5 * self.sigma1**2
        return nu0, nu1


@dataclass(frozen=True)
class AssetSpecification:
    """Single asset class: process parameters + capital market assumptions.

    mu_growth, mu_stress are the total expected returns μ̄^[i] (CMA observables).
    The diffusion drifts μ^[i] = μ̄^[i] − λ·α^[i] are computed internally.
    """
    name: str
    params: RegimeSwitchParams
    mu_growth: float        # μ̄^[1]: total expected return in growth regime (CMA)
    mu_stress: float        # μ̄^[2]: total expected return in stress regime (CMA)
    pi0: float = 100.0      # initial wealth level
    pi_floor: float = 80.0  # absorbing barrier (wealth floor)

    @property
    def nu0(self) -> float:
        """Compensated drift, growth regime."""
        return self.params.compute_drifts(self.mu_growth, self.mu_stress)[0]

    @property
    def nu1(self) -> float:
        """Compensated drift, stress regime."""
        return self.params.compute_drifts(self.mu_growth, self.mu_stress)[1]

    @property
    def x0(self) -> float:
        """Log-gap to absorbing barrier: x₀ = ln(Π₀/Π_floor)."""
        if self.pi_floor <= 0:
            return np.inf
        return np.log(self.pi0 / self.pi_floor)

    @property
    def has_barrier(self) -> bool:
        return self.pi_floor > 0 and np.isfinite(self.x0)


@dataclass
class MandateSpecification:
    """Portfolio mandate: named collection of (asset, weight) pairs."""
    name: str
    allocations: dict[AssetSpecification, float]   # {asset: weight}
    T: float = 10.0     # investment horizon
    r: float = 0.02     # risk-free rate
    c: float = 0.03     # consumption rate


# ==============================================================================
# POLYNOMIAL SOLVER
# ==============================================================================

def _solve_characteristic(p: complex, asset: AssetSpecification) -> dict:
    """Solve the characteristic polynomial and matching system.

    For η₀,η₁ > 0: degree-6 polynomial, 6 roots, 6×6 system.
    For η₀=η₁=0:   degree-4 polynomial, 4 roots, 4×4 system.

    Returns dict with keys: psi, A, B, beta, n_neg, sigma02, sigma12
    """
    par = asset.params
    nu0, nu1 = asset.nu0, asset.nu1
    sigma02 = 0.5 * par.sigma0**2
    sigma12 = 0.5 * par.sigma1**2
    c0 = par.lambda01 + p
    c1 = par.lambda10 + p

    # --- Build characteristic polynomial via explicit multiplication ---
    # Quadratic factors:  Q⁰(ψ) = −½σ₀²ψ² + ν₀ψ + c₀
    #                     Q¹(ψ) = −½σ₁²ψ² + ν₁ψ + c₁
    Q0 = np.array([-sigma02, nu0, c0])   # descending powers
    Q1 = np.array([-sigma12, nu1, c1])

    if par.has_jumps:
        # G⁰(ψ) = Q⁰(ψ)·(1 − η₀ψ),  G¹(ψ) = Q¹(ψ)·(1 + η₁ψ)
        G0_poly = np.polymul(Q0, [-par.eta0, 1])
        G1_poly = np.polymul(Q1, [par.eta1, 1])
    else:
        G0_poly = Q0
        G1_poly = Q1

    # Characteristic polynomial: G⁰·G¹ − λ₀₁λ₁₀ = 0
    char_poly = np.polymul(G0_poly, G1_poly)
    char_poly[-1] -= par.lambda01 * par.lambda10

    # Solve and sort by real part
    psi = np.roots(char_poly)
    psi = psi[np.argsort(psi.real)]

    K = len(psi)
    n_neg = K // 2  # roots split evenly: n_neg negative, n_neg positive

    # --- Physical coupling ratio ---
    # Q₀(ψ_k) = −½σ₀²ψ² + ν₀ψ + (λ₀₁+p)  [quadratic part of G^[1]]
    Q0_vals = np.array([-sigma02*psi[k]**2 + nu0*psi[k] + c0 for k in range(K)])

    # β_k = Q₀(ψ_k)·(1+η₁ψ_k) / λ₁₀   [unique b_k/a_k from forward regime-1 OIDE]
    if par.has_jumps:
        beta = Q0_vals * (1 + par.eta1 * psi) / par.lambda10
    else:
        beta = Q0_vals / par.lambda10

    # --- Matching system ---
    neg = list(range(n_neg))
    pos = list(range(n_neg, K))
    sign = np.array([1]*n_neg + [-1]*n_neg)

    if par.has_jumps:
        # 6×6 system:
        # Row 0: crash spurious  — 1/(1−η₀ψ)
        # Row 1: A continuity    — 1
        # Row 2: A' delta source — ψ
        # Row 3: B continuity    — β
        # Row 4: B' delta source — β·ψ
        # Row 5: recovery spurious — β/(1+η₁ψ) = Q₀/λ₁₀
        M = np.zeros((K, K), dtype=complex)
        for j in range(K):
            s = sign[j]
            M[0, j] = s / (1 - par.eta0 * psi[j])
            M[1, j] = s * 1
            M[2, j] = s * psi[j]
            M[3, j] = s * beta[j]
            M[4, j] = s * beta[j] * psi[j]
            M[5, j] = s * Q0_vals[j] / par.lambda10  # = β/(1+η₁ψ)

        R = np.zeros(K, dtype=complex)
        R[2] = -1.0 / sigma02  # start in regime 0
        R[4] = 0.0
    else:
        M = np.zeros((K, K), dtype=complex)
        for j in range(K):
            s = sign[j]
            M[0, j] = s * 1
            M[1, j] = s * psi[j]
            M[2, j] = s * beta[j] * psi[j]
            M[3, j] = s * beta[j]

        R = np.zeros(K, dtype=complex)
        R[1] = -1.0 / sigma02
        R[2] = 0.0

    A = np.linalg.solve(M, R)
    B = beta * A

    return {
        'psi': psi, 'A': A, 'B': B, 'beta': beta, 'Q0_vals': Q0_vals,
        'n_neg': n_neg, 'sigma02': sigma02, 'sigma12': sigma12,
        'lambda10': par.lambda10,
    }


# ==============================================================================
# DENSITY EVALUATION (CT=0) — VECTORIZED
# ==============================================================================

def _eval_density_unbounded(sol: dict, x_grid: np.ndarray, x0: float) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate unbounded density at x_grid given solution data. Returns (d0, d1)."""
    psi, A, B = sol['psi'], sol['A'], sol['B']
    n_neg = sol['n_neg']
    neg, pos = slice(0, n_neg), slice(n_neg, None)
    Nx = len(x_grid)

    xn = x_grid - x0
    # exp_mat[k, n] = exp(ψ_k · (x_n - x₀)),  shape (K, Nx)
    exp_mat = _safe_exp(np.outer(psi, xn))

    mask_left = xn <= 0
    G0 = np.zeros(Nx, dtype=complex)
    G1 = np.zeros(Nx, dtype=complex)

    # x ≤ x₀: use positive roots (decay to the left)
    if np.any(mask_left):
        G0[mask_left] = A[pos] @ exp_mat[pos, :][:, mask_left]
        G1[mask_left] = B[pos] @ exp_mat[pos, :][:, mask_left]
    # x > x₀: use negative roots (decay to the right)
    if np.any(~mask_left):
        G0[~mask_left] = A[neg] @ exp_mat[neg, :][:, ~mask_left]
        G1[~mask_left] = B[neg] @ exp_mat[neg, :][:, ~mask_left]

    return G0, G1


def _eval_density_barrier(sol: dict, x_grid: np.ndarray, x0: float,
                           eta0: float, eta1: float) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate barrier density at x_grid. Returns (d0, d1).

    Barrier at X=0, source at x₀>0. Three conditions at barrier:
      (1) regime-1 density vanishes
      (2) regime-2 density vanishes
      (3) recovery jump flux vanishes (Q₀/λ₁₀ row)
    """
    psi, A, B, beta = sol['psi'], sol['A'], sol['B'], sol['beta']
    Q0_vals, lam10 = sol['Q0_vals'], sol['lambda10']
    n_neg = sol['n_neg']
    neg = np.arange(n_neg)
    pos = np.arange(n_neg, len(psi))
    Nx = len(x_grid)

    xn = x_grid - x0
    exp_mat = _safe_exp(np.outer(psi, xn))

    if n_neg == 3:
        # 3×3: [A vanish, B vanish, recovery flux vanish]
        M3 = np.zeros((3, 3), dtype=complex)
        for j in range(3):
            M3[0, j] = 1
            M3[1, j] = beta[neg[j]]
            M3[2, j] = Q0_vals[neg[j]] / lam10  # = β/(1+η₁ψ)
        exp_neg_x0 = np.exp(-psi[pos] * x0)
        rhs = np.array([
            -np.dot(A[pos], exp_neg_x0),
            -np.dot(B[pos], exp_neg_x0),
            -np.dot(Q0_vals[pos] / lam10 * A[pos], exp_neg_x0),
        ])
    else:
        M3 = np.zeros((2, 2), dtype=complex)
        for j in range(2):
            M3[0, j] = 1
            M3[1, j] = beta[neg[j]]
        exp_neg_x0 = np.exp(-psi[pos] * x0)
        rhs = np.array([
            -np.dot(A[pos], exp_neg_x0),
            -np.dot(B[pos], exp_neg_x0),
        ])

    C_raw = np.linalg.solve(M3, rhs)
    C = np.exp(psi[neg] * x0) * C_raw
    E = beta[neg] * C

    mask_left = xn <= 0
    G0 = np.zeros(Nx, dtype=complex)
    G1 = np.zeros(Nx, dtype=complex)

    if np.any(mask_left):
        G0[mask_left] = C @ exp_mat[neg][:, mask_left] + A[pos] @ exp_mat[pos][:, mask_left]
        G1[mask_left] = E @ exp_mat[neg][:, mask_left] + B[pos] @ exp_mat[pos][:, mask_left]
    if np.any(~mask_left):
        G0[~mask_left] = (C + A[neg]) @ exp_mat[neg][:, ~mask_left]
        G1[~mask_left] = (E + B[neg]) @ exp_mat[neg][:, ~mask_left]

    return G0, G1


# ==============================================================================
# SURVIVAL PROBABILITY (CT=2)
# ==============================================================================

def _eval_survival_barrier(sol: dict, x0_vals: np.ndarray,
                            eta0: float, eta1: float) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate survival probability in Laplace domain for array of x₀ values."""
    psi, A, B, beta = sol['psi'], sol['A'], sol['B'], sol['beta']
    Q0_vals, lam10 = sol['Q0_vals'], sol['lambda10']
    n_neg = sol['n_neg']
    neg = np.arange(n_neg)
    pos = np.arange(n_neg, len(psi))

    if n_neg == 3:
        M3 = np.zeros((3, 3), dtype=complex)
        for j in range(3):
            M3[0, j] = 1
            M3[1, j] = beta[neg[j]]
            M3[2, j] = Q0_vals[neg[j]] / lam10
    else:
        M3 = np.zeros((2, 2), dtype=complex)
        for j in range(2):
            M3[0, j] = 1
            M3[1, j] = beta[neg[j]]

    M3_inv = np.linalg.inv(M3)
    Nx0 = len(x0_vals)
    S0 = np.zeros(Nx0, dtype=complex)
    S1 = np.zeros(Nx0, dtype=complex)

    for nn in range(Nx0):
        x0n = x0_vals[nn]
        xn = -x0n

        exp_neg_x0 = np.exp(-psi[pos] * x0n)
        if n_neg == 3:
            rhs = np.array([
                -np.dot(A[pos], exp_neg_x0),
                -np.dot(B[pos], exp_neg_x0),
                -np.dot(Q0_vals[pos] / lam10 * A[pos], exp_neg_x0),
            ])
        else:
            rhs = np.array([
                -np.dot(A[pos], exp_neg_x0),
                -np.dot(B[pos], exp_neg_x0),
            ])

        C_raw = M3_inv @ rhs
        C = np.exp(psi[neg] * x0n) * C_raw
        E = beta[neg] * C

        exp_neg_xn = _safe_exp(psi[neg] * xn)
        exp_pos_xn = np.exp(psi[pos] * xn)

        S0[nn] = (-np.dot(C / psi[neg], exp_neg_xn)
                  + np.dot(A[pos] / psi[pos], 1 - exp_pos_xn)
                  - np.sum(A[neg] / psi[neg]))
        S1[nn] = (-np.dot(E / psi[neg], exp_neg_xn)
                  + np.dot(B[pos] / psi[pos], 1 - exp_pos_xn)
                  - np.sum(B[neg] / psi[neg]))

    return S0, S1


def _eval_tilted_survival_barrier(sol: dict, x0_vals: np.ndarray,
                                    eta0: float, eta1: float,
                                    o: float) -> tuple[np.ndarray, np.ndarray]:
    """Tilted survival: Q_o = ∫₀^∞ e^{-ox} · Â(x₀; x, p) dx.

    Derivation: splitting into [0,x₀] and [x₀,∞] parts with the bounded
    density Â = Â_unb + D, and integrating e^{-ox} against each piece:

      Q_o = -Σ C_k/(ψ⁻_k-o)·exp(ψ⁻_k·xn)
            + Σ A⁺_k/(ψ⁺_k-o)·[e^{-o·x₀} - exp(ψ⁺_k·xn)]
            - Σ A⁻_k/(ψ⁻_k-o)·e^{-o·x₀}

    where xn = -x₀. Converges when o < min(ψ⁺_k) (positive roots).
    """
    psi, A, B, beta = sol['psi'], sol['A'], sol['B'], sol['beta']
    Q0_vals, lam10 = sol['Q0_vals'], sol['lambda10']
    n_neg = sol['n_neg']
    neg = np.arange(n_neg)
    pos = np.arange(n_neg, len(psi))

    if n_neg == 3:
        M3 = np.zeros((3, 3), dtype=complex)
        for j in range(3):
            M3[0, j] = 1
            M3[1, j] = beta[neg[j]]
            M3[2, j] = Q0_vals[neg[j]] / lam10
    else:
        M3 = np.zeros((2, 2), dtype=complex)
        for j in range(2):
            M3[0, j] = 1
            M3[1, j] = beta[neg[j]]

    M3_inv = np.linalg.inv(M3)
    Nx0 = len(x0_vals)
    S0 = np.zeros(Nx0, dtype=complex)
    S1 = np.zeros(Nx0, dtype=complex)

    for nn in range(Nx0):
        x0n = x0_vals[nn]
        xn = -x0n
        e_ox0 = np.exp(-o * x0n)

        exp_neg_x0 = np.exp(-psi[pos] * x0n)
        if n_neg == 3:
            rhs = np.array([
                -np.dot(A[pos], exp_neg_x0),
                -np.dot(B[pos], exp_neg_x0),
                -np.dot(Q0_vals[pos] / lam10 * A[pos], exp_neg_x0),
            ])
        else:
            rhs = np.array([
                -np.dot(A[pos], exp_neg_x0),
                -np.dot(B[pos], exp_neg_x0),
            ])

        C_raw = M3_inv @ rhs
        C = np.exp(psi[neg] * x0n) * C_raw
        E = beta[neg] * C

        exp_neg_xn = _safe_exp(psi[neg] * xn)
        exp_pos_xn = np.exp(psi[pos] * xn)

        # Tilted formula: ψ → (ψ-o), and 1 → e^{-ox₀} in unbounded terms
        S0[nn] = (-np.dot(C / (psi[neg] - o), exp_neg_xn)
                  + np.dot(A[pos] / (psi[pos] - o), e_ox0 - exp_pos_xn)
                  - np.sum(A[neg] / (psi[neg] - o)) * e_ox0)
        S1[nn] = (-np.dot(E / (psi[neg] - o), exp_neg_xn)
                  + np.dot(B[pos] / (psi[pos] - o), e_ox0 - exp_pos_xn)
                  - np.sum(B[neg] / (psi[neg] - o)) * e_ox0)

    return S0, S1


# ==============================================================================
# PUBLIC API
# ==============================================================================

def compute_density(T: float, x_grid: np.ndarray, asset: AssetSpecification,
                    state_init: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Compute transition density via Laplace inversion.

    Parameters
    ----------
    T : float
        Time horizon.
    x_grid : ndarray
        Spatial evaluation points (log-wealth relative to barrier).
    asset : AssetSpecification
        Asset with process params, CMAs, barrier.
    state_init : int
        Initial regime (0 = growth, 1 = stress).

    Returns
    -------
    d0, d1 : ndarray
        Density contributions from regime 0 and regime 1.
    """
    is_barrier = asset.has_barrier
    x0 = asset.x0 if is_barrier else 0.0
    Nx = len(x_grid)

    def g(Ps):
        Np = len(Ps)
        G0_all = np.zeros((Np, Nx), dtype=complex)
        G1_all = np.zeros((Np, Nx), dtype=complex)
        for ip in range(Np):
            sol = _solve_characteristic(Ps[ip], asset)
            if is_barrier:
                g0, g1 = _eval_density_barrier(sol, x_grid, x0, asset.params.eta0, asset.params.eta1)
            else:
                g0, g1 = _eval_density_unbounded(sol, x_grid, x0)
            G0_all[ip, :] = g0
            G1_all[ip, :] = g1
        return np.hstack([G0_all, G1_all])

    result = laplace_invert_abate_whitt(g, T)
    return result[:Nx], result[Nx:]


def compute_survival(T: float, x0: float, asset: AssetSpecification) -> float:
    """Compute survival probability P(τ > T | X₀ = x₀).

    Parameters
    ----------
    T : float
        Time horizon.
    x0 : float
        Log-gap to barrier. If None, uses asset.x0.
    asset : AssetSpecification

    Returns
    -------
    float : survival probability (sum of regime 0 + regime 1 components).
    """
    x0_arr = np.array([x0])

    def g(Ps):
        Np = len(Ps)
        G = np.zeros((Np, 2), dtype=complex)
        for ip in range(Np):
            sol = _solve_characteristic(Ps[ip], asset)
            s0, s1 = _eval_survival_barrier(sol, x0_arr, asset.params.eta0, asset.params.eta1)
            G[ip, 0] = s0[0]
            G[ip, 1] = s1[0]
        return G

    result = laplace_invert_abate_whitt(g, T)
    return float(result[0] + result[1])


def compute_tilted_survival(T: float, x0: float, asset: AssetSpecification,
                             o: float) -> float:
    """Compute tilted survival ∫₀^∞ e^{-o·x} A(T, x₀; x) dx.

    Parameters
    ----------
    o : float > 0 — tilt parameter (typically 1/η for overshoot).
    """
    x0_arr = np.array([x0])

    def g(Ps):
        Np = len(Ps)
        G = np.zeros((Np, 2), dtype=complex)
        for ip in range(Np):
            sol = _solve_characteristic(Ps[ip], asset)
            s0, s1 = _eval_tilted_survival_barrier(
                sol, x0_arr, asset.params.eta0, asset.params.eta1, o)
            G[ip, 0] = s0[0]
            G[ip, 1] = s1[0]
        return G

    result = laplace_invert_abate_whitt(g, T)
    return float(result[0] + result[1])


def compute_overshoot_density(T: float, x_grid: np.ndarray,
                               asset: AssetSpecification) -> np.ndarray:
    """Compute the overshoot (sub-barrier) density in x-coordinates.

    At a crash (regime 0→1), the process may jump through the barrier.
    By the memoryless property, the conditional overshoot distance is
    Exp(1/η₀). The unconditional density involves the time-integrated
    regime-0 tilted survival:

      f_over(d, T) = (1/η₀) · exp(-d/η₀) · C_over(T)

    where C_over(T) = L⁻¹[λ₁₂ · Q̂_o^{[0]}(p; 1/η₀) / p](T)

    The /p gives the time integral (accumulating crash events over [0,T]),
    and only the regime-0 component contributes (crashes only from growth).

    Parameters
    ----------
    T : float — horizon
    x_grid : array — overshoot distances d > 0 (distance below barrier)
    asset : AssetSpecification

    Returns
    -------
    overshoot_density : array — density at each x_grid point
    """
    eta0 = asset.params.eta0
    if eta0 <= 0:
        return np.zeros_like(x_grid)

    x0 = asset.x0
    o = 1.0 / eta0
    lam12 = asset.params.lambda01
    x0_arr = np.array([x0])

    def g(Ps):
        Np = len(Ps)
        G = np.zeros((Np, 1), dtype=complex)
        for ip in range(Np):
            sol = _solve_characteristic(Ps[ip], asset)
            t0, t1 = _eval_tilted_survival_barrier(
                sol, x0_arr, asset.params.eta0, asset.params.eta1, o)
            # λ₁₂ · Q̂_o^{[0]}(p) / p — regime-0 only, with time integral
            G[ip, 0] = lam12 * t0[0] / Ps[ip]
        return G

    result = laplace_invert_abate_whitt(g, T)
    C_over = float(result[0])

    # f(d) = (1/η₀) · exp(-d/η₀) · C_over
    density = (1.0 / eta0) * np.exp(-x_grid / eta0) * C_over
    return np.maximum(density, 0.0)


def compute_wealth_density(T: float, x_grid: np.ndarray,
                           asset: AssetSpecification) -> dict:
    """Compute the full wealth density with barrier stopping.

    Returns dict with:
        density0, density1, density_total : arrays on x_grid
        survival_density  : integral of density (numerical check)
        survival_analytic : from CT=2 Laplace formula
        stopping_prob     : 1 − survival
    """
    d0, d1 = compute_density(T, x_grid, asset)
    d_total = d0 + d1
    survival_density = float(np.trapezoid(d_total, x_grid))

    survival_analytic = compute_survival(T, asset.x0, asset)

    return {
        'density0': d0,
        'density1': d1,
        'density_total': d_total,
        'survival_density': survival_density,
        'survival_analytic': survival_analytic,
        'stopping_prob': 1.0 - survival_analytic,
        'x_grid': x_grid,
    }


def bh_moments_rsjd(T: float, Pi0: float, asset: AssetSpecification,
                    c: float = 0.0) -> dict:
    """Exact moments of buy-and-hold terminal wealth under RS-JD.

    The n-th moment satisfies a 2×2 linear ODE (backward Kolmogorov):
      h_n'(τ) = A_n · h_n(τ),   h_n(0) = [1, 1],
    where A_n^[ij] encodes regime drifts, volatilities, jump MGFs, and
    switching rates. The solution is h_n(T) = expm(A_n T) · [1,1].

    The unconditional moment is M_n = Π₀ⁿ · (p₁ h_n^[1] + p₂ h_n^[2])
    with pᵢ the stationary regime probabilities.

    Parameters
    ----------
    T : horizon
    Pi0 : initial wealth
    asset : effective single-asset specification (portfolio-level)
    c : consumption rate (reduces drift by c)

    Returns dict with E, Var, Std, r_impl, and regime-conditional values.
    """
    from scipy.linalg import expm

    par = asset.params
    # Diffusion drifts μ^[i] (paper Definition 2.3)
    alpha0 = -par.eta0 / (1 + par.eta0) if par.eta0 > 0 else 0.0
    alpha1 = par.eta1 / (1 - par.eta1) if par.eta1 > 0 else 0.0
    mu0 = asset.mu_growth - par.lambda01 * alpha0 - c
    mu1 = asset.mu_stress - par.lambda10 * alpha1 - c

    # Stationary regime probabilities
    p1 = par.lambda10 / (par.lambda01 + par.lambda10)
    p2 = par.lambda01 / (par.lambda01 + par.lambda10)

    def _moment(n):
        EenJ0 = 1.0 / (1 + n * par.eta0) if par.eta0 > 0 else 1.0
        EenJ1 = 1.0 / (1 - n * par.eta1) if par.eta1 > 0 else 1.0
        A = np.array([
            [n*mu0 + n*(n-1)/2*par.sigma0**2 - par.lambda01,
             par.lambda01 * EenJ0],
            [par.lambda10 * EenJ1,
             n*mu1 + n*(n-1)/2*par.sigma1**2 - par.lambda10]
        ])
        h = expm(A * T) @ np.array([1.0, 1.0])
        M_unc = Pi0**n * (p1 * h[0] + p2 * h[1])
        return M_unc, Pi0**n * h[0], Pi0**n * h[1]

    M1, M1_g, M1_s = _moment(1)
    M2, M2_g, M2_s = _moment(2)
    Var = max(0.0, M2 - M1**2)

    return {
        'E': M1, 'E_growth': M1_g, 'E_stress': M1_s,
        'Var': Var, 'Std': np.sqrt(Var),
        'r_impl': np.log(M1 / Pi0) / T if M1 > 0 and T > 1e-10 else 0.0,
        'p1': p1, 'p2': p2,
    }


# ==============================================================================
# PAPER PARAMETERS
# ==============================================================================

def create_paper_assets(pi0: float = 100.0,
                        k_floor: float = 1.5,
                        ) -> dict[str, AssetSpecification]:
    """Create the three asset classes from the paper's Table 1.

    Common regime dynamics: λ₀₁ = 0.1, λ₁₀ = 1.0

    Floor rule: L₀ = Π₀ · exp(-k · η₀), ensuring uniform single-crash
    breach probability P(breach|crash) = exp(-k) for all assets.
    With k=1.5: P = 22.3%.
    """
    common = dict(lambda01=0.1, lambda10=1.0)

    bonds_params = RegimeSwitchParams(
        sigma0=0.06, sigma1=0.09,
        eta0=0.08 / (1 - 0.08),        # crash loss 8%
        eta1=0.05 / (1 + 0.05),        # recovery gain 5%
        **common)

    equity_params = RegimeSwitchParams(
        sigma0=0.15, sigma1=0.225,
        eta0=0.25 / (1 - 0.25),         # crash loss 25%
        eta1=0.15 / (1 + 0.15),         # recovery gain 15%
        **common)

    pe_params = RegimeSwitchParams(
        sigma0=0.20, sigma1=0.30,
        eta0=0.30 / (1 - 0.30),         # crash loss 30%
        eta1=0.20 / (1 + 0.20),         # recovery gain 20%
        **common)

    return {
        'bonds': AssetSpecification('bonds', bonds_params,
                                     mu_growth=0.025, mu_stress=0.02,
                                     pi0=pi0,
                                     pi_floor=pi0 * np.exp(-k_floor * bonds_params.eta0)),
        'equity': AssetSpecification('equity', equity_params,
                                      mu_growth=0.045, mu_stress=0.00,
                                      pi0=pi0,
                                      pi_floor=pi0 * np.exp(-k_floor * equity_params.eta0)),
        'private_equity': AssetSpecification('private_equity', pe_params,
                                              mu_growth=0.070, mu_stress=0.00,
                                              pi0=pi0,
                                              pi_floor=pi0 * np.exp(-k_floor * pe_params.eta0)),
    }


def create_paper_mandates(assets: dict[str, AssetSpecification]) -> dict[str, MandateSpecification]:
    """Create the three mandates from the paper's Table 2."""
    b, e, pe = assets['bonds'], assets['equity'], assets['private_equity']
    return {
        'conservative': MandateSpecification('conservative',
                                              {b: 0.70, e: 0.20, pe: 0.10}),
        'balanced': MandateSpecification('balanced',
                                          {b: 0.40, e: 0.40, pe: 0.20}),
        'growth': MandateSpecification('growth',
                                        {b: 0.10, e: 0.60, pe: 0.30}),
    }


# ==============================================================================
# TESTS
# ==============================================================================

if __name__ == "__main__":
    np.set_printoptions(precision=8, linewidth=120)

    print("=" * 70)
    print("REGIME-SWITCHING PAPER — TESTS")
    print("=" * 70)

    assets = create_paper_assets()
    eq = assets['equity']

    print(f"\nEquity: nu0={eq.nu0:.6f}, nu1={eq.nu1:.6f}")
    print(f"  sigma0={eq.params.sigma0}, sigma1={eq.params.sigma1}")
    print(f"  lambda01={eq.params.lambda01}, lambda10={eq.params.lambda10}")
    print(f"  eta0={eq.params.eta0:.6f}, eta1={eq.params.eta1:.6f}")
    print(f"  x0={eq.x0:.6f} (pi0={eq.pi0}, floor={eq.pi_floor})")

    # --- Test 1: Unbounded normalization ---
    print("\n--- Test 1: Unbounded density normalization ---")
    # Temporarily create asset with no barrier for unbounded test
    eq_unb = AssetSpecification('equity_unb', eq.params,
                                 eq.mu_growth, eq.mu_stress,
                                 pi0=100, pi_floor=0)
    xg = np.linspace(-5, 5, 600)
    d0, d1 = compute_density(5.0, xg, eq_unb)
    print(f"  integral(d0+d1) = {np.trapezoid(d0+d1, xg):.8f} (expect 1.0)")
    print(f"  min(d0+d1) = {np.min(d0+d1):.2e}")

    # No-jump test
    eq_nj = AssetSpecification('equity_nj',
                                RegimeSwitchParams(sigma0=0.15, sigma1=0.25,
                                                   lambda01=0.1, lambda10=1.0),
                                eq.mu_growth, eq.mu_stress,
                                pi0=100, pi_floor=0)
    d0nj, d1nj = compute_density(5.0, xg, eq_nj)
    print(f"  No-jump integral = {np.trapezoid(d0nj+d1nj, xg):.8f}")

    # --- Test 2: Barrier consistency ---
    print("\n--- Test 2: Barrier density vs survival (equity, T=10) ---")
    xb = np.linspace(0.001, 4.0, 1000)
    res = compute_wealth_density(10.0, xb, eq)
    print(f"  survival (density):  {res['survival_density']:.8f}")
    print(f"  survival (analytic): {res['survival_analytic']:.8f}")
    print(f"  stopping prob:       {res['stopping_prob']:.8f}")
    print(f"  consistency gap:     {abs(res['survival_density'] - res['survival_analytic']):.2e}")

    # --- Test 3: Survival vs T ---
    print("\n--- Test 3: Survival probability vs T (equity) ---")
    print(f"  {'T':>5s} {'survival':>12s} {'stopping':>12s}")
    for T in [1, 2, 5, 10]:
        s = compute_survival(T, eq.x0, eq)
        print(f"  {T:5d} {s:12.6f} {1-s:12.6f}")

    # --- Test 4: Three assets ---
    print("\n--- Test 4: Three assets, T=10 ---")
    for name, asset in assets.items():
        s = compute_survival(10, asset.x0, asset)
        print(f"  {name:16s}: x0={asset.x0:.4f}, surv={s:.6f}, stop={1-s:.6f}")

    # --- Test 5: Mandate summary ---
    print("\n--- Test 5: Mandate specifications ---")
    mandates = create_paper_mandates(assets)
    for mname, mandate in mandates.items():
        print(f"  {mname}: ", end="")
        parts = [f"{a.name}={w:.0%}" for a, w in mandate.allocations.items()]
        print(", ".join(parts))

    print("\n" + "=" * 70)
    print("ALL TESTS COMPLETE")
    print("=" * 70)
