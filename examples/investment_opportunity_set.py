#!/usr/bin/env python3
"""Reproduce Figure 4: the investment opportunity set.

The advisor fixes the risk profile (omega_0, consumption c, equity share q,
drawdown scale q_dd). Sweeping the bond weight then traces a one-parameter
opportunity set. Two views:
  left  - efficient frontier: implied return vs portfolio volatility;
  right - terminal wealth quantile fan vs implied return, with the floor.
The client's choice reduces to a single point on this curve.
"""
# packages
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
# project
from goal_based_allocation import AdvisorSpec, build_opportunity_set

PI0 = 100.0


def main() -> None:
    spec = AdvisorSpec(omega_0=1.0, c=0.0, q=2 / 3, q_dd=2.0)
    opp = build_opportunity_set(spec)
    opp = [p for p in opp if p['r_impl'] >= 0 and p['S'] >= 0.5]

    r_impl = np.array([p['r_impl'] * 100 for p in opp])
    sig = np.array([p['sig_unc'] * 100 for p in opp])
    w_bd = np.array([p['w_bd'] * 100 for p in opp])
    q05 = np.array([p['q5'] for p in opp])
    q25 = np.array([p['q25'] for p in opp])
    q50 = np.array([p['q50'] for p in opp])
    q75 = np.array([p['q75'] for p in opp])
    q95 = np.array([p['q95'] for p in opp])
    floor = np.array([p['L_T'] for p in opp])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))

    # Left: efficient frontier, coloured by bond weight
    sc = ax1.scatter(sig, r_impl, c=w_bd, cmap='viridis', s=40, zorder=3)
    ax1.plot(sig, r_impl, 'k-', lw=1, alpha=0.4, zorder=2)
    cbar = fig.colorbar(sc, ax=ax1)
    cbar.set_label('Bond weight (%)')
    ax1.set_xlabel('Portfolio volatility $\\sigma$ (%)')
    ax1.set_ylabel('Implied return $r_{\\mathrm{impl}}$ (%)')
    ax1.set_title('Efficient frontier (MV with floor)')
    ax1.grid(alpha=0.2)

    # Right: terminal wealth quantile fan
    ax2.fill_between(r_impl, q05, q95, alpha=0.15, color='C0', label='5th–95th pct')
    ax2.fill_between(r_impl, q25, q75, alpha=0.30, color='C0', label='25th–75th pct')
    ax2.plot(r_impl, q50, 'C0-', lw=2, label='Median')
    ax2.plot(r_impl, floor, 'r--', lw=1.5, label='Floor $L_T$')
    ax2.axhline(PI0, color='gray', ls=':', alpha=0.6, label='Initial wealth')
    ax2.set_xlabel('Implied return $r_{\\mathrm{impl}}$ (%)')
    ax2.set_ylabel('Terminal wealth $\\Pi_T$')
    ax2.set_title('Terminal wealth quantiles')
    ax2.grid(alpha=0.2)
    ax2.legend(fontsize=9)

    fig.suptitle('Figure 4 — Investment opportunity set '
                 '($\\omega_0=1$, $c=0\\%$, $q=2/3$, $q_{dd}=2$)', fontsize=13)
    fig.tight_layout()

    out = Path(__file__).with_name('investment_opportunity_set.png')
    fig.savefig(out, dpi=150)
    print(f'saved {out}')
    print(f'{len(opp)} portfolios, '
          f'r_impl in [{r_impl.min():.2f}%, {r_impl.max():.2f}%]')


if __name__ == '__main__':
    main()
