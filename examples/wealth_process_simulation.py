#!/usr/bin/env python3
"""Reproduce Figure 1: simulated wealth paths under the MV-optimal strategy.

A handful of Monte Carlo paths of the portfolio wealth Pi_t for the balanced
mandate, driven by the MV-optimal allocation omega*(t). Paths that touch the
absorbing floor L_t stop and grow at the net floor rate r_c. Overlaid are the
target trajectory Pi*(t), the expected wealth E[Pi_t], and the floor L_t.
"""
# packages
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
# project
from goal_based_allocation import (
    create_paper_assets, create_paper_mandates,
    compute_survival, compute_tilted_survival,
)
from goal_based_allocation.mandate_utils import mandate_effective_asset
from goal_based_allocation.riccati_solver import find_ell, gap_process_asset

T = 10.0            # horizon, years
R = 0.02            # risk-free rate
C = 0.0             # consumption rate
TARGET_RETURN = 0.04
STEPS_PER_YEAR = 252
N_PATHS = 14
SEED = 7


def simulate_path(ric, seed: int) -> tuple[np.ndarray, np.ndarray, bool]:
    """simulate one MV-optimal wealth path; return (t_grid, Pi, stopped).

    Mirrors the wealth SDE of ``simulate_mv_optimal``:
      dPi/Pi = [r_c + omega*(mu_bar^[i] - r_h)]dt + omega*sigma^[i] dW
    with exponential jumps at regime transitions and an absorbing floor L_t.
    """
    rng = np.random.default_rng(seed)
    par = ric.asset.params
    mu_bar = ric.dp['mu_bar']
    r_h, r_c = ric.r_h, ric.r_c
    n_steps = int(T * STEPS_PER_YEAR)
    dt, sqrt_dt = T / n_steps, np.sqrt(T / n_steps)

    t_grid = np.linspace(0.0, T, n_steps + 1)
    Pi = np.empty(n_steps + 1)
    Pi[0] = ric.asset.pi0
    regime, stopped = 0, False
    L = float(ric.asset.pi_floor)

    for i in range(n_steps):
        L *= np.exp(r_c * dt)
        if stopped:
            Pi[i + 1] = Pi[i] * np.exp(r_c * dt)
            continue
        omega = float(np.clip(ric.omega_star(t_grid[i], Pi[i], regime), -2.0, 5.0))
        sig = par.sigma0 if regime == 0 else par.sigma1
        mb = mu_bar[0] if regime == 0 else mu_bar[1]
        z = rng.standard_normal()
        log_ret = (r_c + omega * (mb - r_h) - 0.5 * (omega * sig) ** 2) * dt + omega * sig * sqrt_dt * z
        pi_next = Pi[i] * np.exp(log_ret)

        p_sw = par.lambda01 * dt if regime == 0 else par.lambda10 * dt
        if rng.uniform() < p_sw:
            if regime == 0:                       # growth -> stress: crash jump
                jump = -rng.exponential(par.eta0)
                regime = 1
            else:                                 # stress -> growth: recovery jump
                jump = rng.exponential(par.eta1)
                regime = 0
            pi_next = Pi[i] * max(1 + omega * (np.exp(jump) - 1), 1e-10)

        Pi[i + 1] = pi_next
        if Pi[i + 1] <= L:
            stopped = True
    return t_grid, Pi, stopped


def expected_wealth(ric, gap, eff) -> tuple[np.ndarray, np.ndarray]:
    """analytical E[Pi_t] over the horizon via survival and tilted survival."""
    t = np.linspace(0.05, T, 40)
    e_pi = np.empty_like(t)
    for j, tj in enumerate(t):
        d = ric.derived_at_tau(T - tj)
        pi_star = d['Pi_star'][0]
        l_t = eff.pi_floor * np.exp(ric.r_c * tj)
        b_t = pi_star - l_t
        s_t = compute_survival(tj, gap.x0, gap)
        ts1 = compute_tilted_survival(tj, gap.x0, gap, 1.0)
        e_pi[j] = pi_star * s_t - b_t * ts1 + l_t * (1 - s_t)
    return np.r_[0.0, t], np.r_[ric.asset.pi0, e_pi]


def main() -> None:
    eff = mandate_effective_asset(create_paper_mandates(create_paper_assets())['balanced'])
    ell, ric = find_ell(eff, T, TARGET_RETURN, R, C)
    gap = gap_process_asset(ric)

    t_full = np.linspace(0.0, T, int(T * STEPS_PER_YEAR) + 1)
    floor = eff.pi_floor * np.exp(ric.r_c * t_full)
    target = np.array([ric.derived_at_tau(T - t)['Pi_star'][0] for t in t_full])
    t_e, e_pi = expected_wealth(ric, gap, eff)

    fig, ax = plt.subplots(figsize=(10, 6))
    for k in range(N_PATHS):
        t_grid, Pi, stopped = simulate_path(ric, SEED + k)
        colour = 'C3' if stopped else 'C2'
        ax.plot(t_grid, Pi, color=colour, lw=0.9, alpha=0.7)
    ax.plot(t_full, target, 'k-', lw=2, label=r'Target $\Pi^*(t)$')
    ax.plot(t_e, e_pi, 'k--', lw=1.8, label=r'Expected $\mathbb{E}[\Pi_t]$')
    ax.plot(t_full, floor, 'r--', lw=1.8, label=r'Floor $L_t$')
    ax.axhline(ric.asset.pi0, color='gray', ls=':', alpha=0.5, label='Initial wealth')
    ax.plot([], [], color='C2', lw=1.5, label='Survived paths')
    ax.plot([], [], color='C3', lw=1.5, label='Stopped paths')

    ax.set_xlabel('Time $t$ (years)')
    ax.set_ylabel('Wealth $\\Pi_t$')
    ax.set_title('Figure 1 — MV-optimal wealth process (balanced mandate, $c=0\\%$)')
    ax.set_xlim(0, T)
    ax.set_ylim(0, max(260, 1.05 * target.max()))
    ax.grid(alpha=0.2)
    ax.legend(fontsize=9, loc='upper left')
    fig.tight_layout()

    out = Path(__file__).with_name('wealth_process_simulation.png')
    fig.savefig(out, dpi=150)
    print(f'saved {out}')


if __name__ == '__main__':
    main()
