#!/usr/bin/env python3
"""Reproduce Figure 2: terminal wealth distribution under the MV-optimal strategy.

The terminal wealth of the stopped MV-optimal strategy has three components:
  (a) a survived density above the floor,
  (b) a floor atom of probability F at L_T (diffusion paths that hit the floor),
  (c) an overshoot density below the floor (crash jumps that gap through it).
All three follow analytically from the Laplace framework. A Monte Carlo
histogram is overlaid for validation.
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
    compute_density, compute_survival, compute_overshoot_density,
)
from goal_based_allocation.mandate_utils import mandate_effective_asset
from goal_based_allocation.riccati_solver import (
    find_ell, gap_process_asset, simulate_mv_optimal,
)

T = 10.0
R = 0.02
C = 0.0
TARGET_RETURN = 0.04
N_PATHS_MC = 200_000


def wealth_density(ric, gap, eff) -> dict:
    """map the gap-process density to the terminal wealth density in Pi-space."""
    pi_target = ric.derived_at_tau(0)['Pi_star'][0]
    l_t = eff.pi_floor * np.exp(ric.r_c * T)
    buffer = pi_target - l_t

    x_max = min(8.0, gap.x0 + 6 * max(gap.params.sigma0, gap.params.sigma1) * np.sqrt(T))
    x_grid = np.linspace(1e-3, x_max, 800)
    d0, d1 = compute_density(T, x_grid, gap)
    dg = d0 + d1
    pi_surv = pi_target - buffer * np.exp(-x_grid)
    f_surv = dg / (buffer * np.exp(-x_grid))

    x_ov = np.linspace(1e-3, x_max, 400)
    f_ov = compute_overshoot_density(T, x_ov, gap)
    pi_over = pi_target - buffer * np.exp(x_ov)
    f_over = f_ov / (buffer * np.exp(x_ov))

    survival = compute_survival(T, gap.x0, gap)
    overshoot_mass = float(np.trapezoid(f_ov, x_ov))
    floor_atom = max(0.0, 1.0 - survival - overshoot_mass)
    return dict(pi_target=pi_target, l_t=l_t,
                pi_surv=pi_surv, f_surv=f_surv,
                pi_over=pi_over, f_over=f_over,
                survival=survival, floor_atom=floor_atom, overshoot_mass=overshoot_mass)


def main() -> None:
    eff = mandate_effective_asset(create_paper_mandates(create_paper_assets())['balanced'])
    ell, ric = find_ell(eff, T, TARGET_RETURN, R, C)
    gap = gap_process_asset(ric)
    d = wealth_density(ric, gap, eff)

    mc = simulate_mv_optimal(ric, n_paths=N_PATHS_MC, seed=42)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(mc['Pi_T'], bins=200, range=(0, d['pi_target'] + 20), density=True,
            color='0.8', edgecolor='none', label='Monte Carlo')

    ax.plot(d['pi_surv'], d['f_surv'], 'C0-', lw=2, label='Survived density')
    ax.fill_between(d['pi_surv'], d['f_surv'], alpha=0.2, color='C0')
    below = d['pi_over'] > 0
    ax.plot(d['pi_over'][below], d['f_over'][below], 'C3-', lw=2, label='Overshoot density')
    ax.fill_between(d['pi_over'][below], d['f_over'][below], alpha=0.2, color='C3')

    ax.axvline(d['l_t'], color='r', ls='--', lw=1.5, label=f"Floor $L_T$ = {d['l_t']:.0f}")
    ax.annotate(f"Floor atom\nF = {d['floor_atom']:.1%}",
                xy=(d['l_t'], 0), xytext=(d['l_t'] + 12, 0.6 * ax.get_ylim()[1]),
                fontsize=9, color='r',
                arrowprops=dict(arrowstyle='->', color='r', lw=1))

    ax.set_xlabel(r'Terminal wealth $\Pi_T$')
    ax.set_ylabel('Density')
    ax.set_title('Figure 2 — Terminal wealth distribution (balanced mandate, $c=0\\%$)')
    ax.set_xlim(0, d['pi_target'] + 20)
    ax.set_ylim(bottom=0)
    ax.grid(alpha=0.2)
    ax.legend(fontsize=9, loc='upper left')

    txt = (f"survival = {d['survival']:.1%}\n"
           f"floor atom = {d['floor_atom']:.1%}\n"
           f"overshoot = {d['overshoot_mass']:.1%}")
    ax.text(0.98, 0.60, txt, transform=ax.transAxes, ha='right', va='top',
            fontsize=9, bbox=dict(boxstyle='round', fc='white', ec='0.7'))
    fig.tight_layout()

    out = Path(__file__).with_name('terminal_wealth_distribution.png')
    fig.savefig(out, dpi=150)
    print(f'saved {out}')
    print(f"survival={d['survival']:.4f}  floor_atom={d['floor_atom']:.4f}  "
          f"overshoot={d['overshoot_mass']:.4f}")


if __name__ == '__main__':
    main()
