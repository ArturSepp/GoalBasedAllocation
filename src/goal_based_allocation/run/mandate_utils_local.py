"""Development runner for analytical and Monte Carlo mandate figures."""

from enum import Enum

import matplotlib.pyplot as plt
import numpy as np

from goal_based_allocation.mandate_utils import (
    compute_mandate_analytical,
    mandate_effective_asset,
    simulate_mandate_mc,
)
from goal_based_allocation.regime_switch_paper import (
    create_paper_assets,
    create_paper_mandates,
)

plt.switch_backend("Agg")


class Locals(Enum):
    """Runnable mandate-figure development cases."""

    PER_MANDATE = 1
    DENSITY_OVERLAY = 2
    CDF_OVERLAY = 3
    ALL_FIGURES = 4


def _figure_per_mandate() -> None:
    """Plot per-mandate Monte Carlo and analytical comparisons."""
    mandates = create_paper_mandates(create_paper_assets())
    horizon = 10.0
    n_paths = 200_000
    fig, axes = plt.subplots(3, 1, figsize=(12, 14), dpi=150)
    labels = {'conservative': '70/20/10', 'balanced': '40/40/20', 'growth': '10/45/45'}

    for index, name in enumerate(['conservative', 'balanced', 'growth']):
        ax = axes[index]
        effective = mandate_effective_asset(mandates[name])
        analytical = compute_mandate_analytical(effective, horizon)
        monte_carlo = simulate_mandate_mc(effective, horizon, n_paths, seed=42)

        survival_mc = np.mean(monte_carlo['survived'])
        overshoot_mc = np.mean(monte_carlo['is_overshoot'])
        floor_mc = np.mean(~monte_carlo['survived'] & ~monte_carlo['is_overshoot'])
        wealth_survived = monte_carlo['Pi_T'][monte_carlo['survived']]
        wealth_overshoot = monte_carlo['Pi_T'][monte_carlo['is_overshoot']]
        print(f"{name}: An surv={analytical['surv']:.4f} MC={survival_mc:.4f}, "
              f"An over={analytical['over_mass']:.4f} MC={overshoot_mc:.4f}")

        peak = np.max(analytical['dens_surv'])
        upper = float(analytical['Pi_surv'][
            np.searchsorted(-analytical['dens_surv'][::-1], -peak * 0.005)
        ])
        upper = max(upper, np.percentile(wealth_survived, 99.9)) * 1.05
        lower = min(-10, np.min(wealth_overshoot) - 5) if len(wealth_overshoot) > 0 else -10
        bins = np.linspace(lower, upper, 60)
        width = bins[1] - bins[0]
        centers = 0.5 * (bins[:-1] + bins[1:])
        survived_histogram, _ = np.histogram(wealth_survived, bins=bins)
        overshoot_histogram, _ = np.histogram(wealth_overshoot, bins=bins)

        ax.bar(centers, survived_histogram / (n_paths * width), width=width,
               alpha=0.3, color='steelblue', edgecolor='none', label='Survived (MC)')
        ax.bar(centers, overshoot_histogram / (n_paths * width), width=width,
               alpha=0.4, color='salmon', edgecolor='none', label='Overshoot (MC)')
        mask = ((analytical['Pi_surv'] > lower) & (analytical['Pi_surv'] < upper)
                & (analytical['dens_surv'] > 0))
        ax.plot(analytical['Pi_surv'][mask], analytical['dens_surv'][mask],
                'k-', lw=1.8, label='Analytical')
        overshoot_mask = ((analytical['Pi_ov'] > lower) & (analytical['Pi_ov'] < upper)
                          & (analytical['dens_ov'] > 0))
        if np.any(overshoot_mask):
            ax.plot(analytical['Pi_ov'][overshoot_mask],
                    analytical['dens_ov'][overshoot_mask],
                    'r-', lw=1.8, label='Overshoot (An)')
        ax.axvline(analytical['L_T'], color='red', ls='--', lw=1, alpha=0.5)
        maximum = max(np.max(survived_histogram / (n_paths * width)),
                      np.max(overshoot_histogram / (n_paths * width))) * 1.15
        ax.set_ylim(0, maximum)
        ax.set_xlim(lower, upper)
        ax.text(analytical['L_T'] + 2, maximum * 0.85,
                f'$L_T$={analytical["L_T"]:.0f}\nFloor: {floor_mc:.1%}\n'
                f'Over: {overshoot_mc:.1%}', fontsize=8, color='red', va='top')
        ax.set_title(f'{name.capitalize()} ({labels[name]}) — '
                     f'$\\sigma$=[{effective.params.sigma0:.1%},'
                     f'{effective.params.sigma1:.1%}], '
                     f'$\\eta$={effective.params.eta0:.3f}', fontsize=11)
        ax.set_xlabel('Terminal wealth $\\Pi_T$', fontsize=10)
        ax.set_ylabel('Density', fontsize=10)
        ax.legend(fontsize=7, loc='upper right')
        ax.grid(True, alpha=0.3)
        ax.text(0.02, 0.97,
                f'Surv: An={analytical["surv"]:.3f} MC={survival_mc:.3f}\n'
                f'E[$\\Pi_T$|s]={np.mean(wealth_survived):.1f}, '
                f'σ={np.std(wealth_survived):.1f}',
                transform=ax.transAxes, fontsize=8, va='top',
                bbox={'boxstyle': 'round', 'facecolor': 'wheat', 'alpha': 0.5})

    fig.suptitle('Mandate Terminal Wealth: Analytical vs MC '
                 '($T$=10y, $\\rho_{eq,pe}$=0.8)', fontsize=13, y=0.99)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    for extension in ['pdf', 'png']:
        fig.savefig(f'/mnt/user-data/outputs/mandate_comparison.{extension}',
                    bbox_inches='tight', dpi=150)
    print("Saved mandate_comparison")


def _figure_density_overlay() -> None:
    """Plot all three mandate densities together."""
    mandates = create_paper_mandates(create_paper_assets())
    fig, ax = plt.subplots(1, 1, figsize=(12, 6), dpi=150)
    colors = {'conservative': 'C0', 'balanced': 'C1', 'growth': 'C2'}
    labels = {'conservative': 'Conservative (70/20/10)',
              'balanced': 'Balanced (40/40/20)', 'growth': 'Growth (10/45/45)'}

    for name in ['conservative', 'balanced', 'growth']:
        analytical = compute_mandate_analytical(mandate_effective_asset(mandates[name]), 10.0)
        color = colors[name]
        mask = analytical['dens_surv'] > 1e-6
        ax.plot(analytical['Pi_surv'][mask], analytical['dens_surv'][mask],
                color=color, lw=2, label=labels[name])
        ax.fill_between(analytical['Pi_surv'][mask], 0, analytical['dens_surv'][mask],
                        color=color, alpha=0.1)
        overshoot_mask = analytical['dens_ov'] > 1e-6
        if np.any(overshoot_mask):
            ax.plot(analytical['Pi_ov'][overshoot_mask],
                    analytical['dens_ov'][overshoot_mask],
                    color=color, lw=1.5, ls='--')
            ax.fill_between(analytical['Pi_ov'][overshoot_mask], 0,
                            analytical['dens_ov'][overshoot_mask],
                            color=color, alpha=0.08)
        ax.axvline(analytical['L_T'], color=color, ls=':', lw=0.8, alpha=0.5)
        print(f"{name}: surv={analytical['surv']:.3f}, "
              f"over={analytical['over_mass']:.3f}, L_T={analytical['L_T']:.0f}")

    ax.set_title('Mandate Terminal Wealth Density ($T$=10y)', fontsize=13)
    ax.set_xlabel('Terminal wealth $\\Pi_T$', fontsize=11)
    ax.set_ylabel('Density', fontsize=11)
    ax.legend(fontsize=10)
    ax.set_xlim(-20, 350)
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    for extension in ['pdf', 'png']:
        fig.savefig(f'/mnt/user-data/outputs/mandate_density_overlay.{extension}',
                    bbox_inches='tight', dpi=150)
    print("Saved mandate_density_overlay")


def _figure_cdf_overlay() -> None:
    """Plot all three mandate cumulative distributions together."""
    mandates = create_paper_mandates(create_paper_assets())
    fig, ax = plt.subplots(1, 1, figsize=(12, 6), dpi=150)
    colors = {'conservative': 'C0', 'balanced': 'C1', 'growth': 'C2'}
    labels = {'conservative': 'Conservative (70/20/10)',
              'balanced': 'Balanced (40/40/20)', 'growth': 'Growth (10/45/45)'}

    for name in ['conservative', 'balanced', 'growth']:
        analytical = compute_mandate_analytical(mandate_effective_asset(mandates[name]), 10.0)
        floor = analytical['L_T']
        wealth_grid = np.linspace(-20, 400, 2000)
        cdf = np.zeros_like(wealth_grid)
        for index, wealth in enumerate(wealth_grid):
            if wealth <= 0:
                distance = -np.log(max(wealth, 0.01) / floor) if wealth > 0 else 10.0
                mask = analytical['d_ov_grid'] >= distance
                if np.any(mask):
                    cdf[index] = np.trapezoid(
                        analytical['f_ov'][mask], analytical['d_ov_grid'][mask]
                    )
            elif wealth < floor:
                distance = -np.log(wealth / floor)
                mask = analytical['d_ov_grid'] >= distance
                cdf[index] = (np.trapezoid(analytical['f_ov'][mask],
                                           analytical['d_ov_grid'][mask])
                              if np.any(mask) else analytical['over_mass'])
            else:
                distance = np.log(wealth / floor)
                mask = analytical['x_grid'] <= distance
                survived_below = (np.trapezoid(analytical['d_total'][mask],
                                               analytical['x_grid'][mask])
                                  if np.any(mask) else 0)
                cdf[index] = analytical['over_mass'] + analytical['floor'] + survived_below

        color = colors[name]
        ax.plot(wealth_grid, cdf, color=color, lw=2, label=labels[name])
        ax.axvline(floor, color=color, ls=':', lw=0.8, alpha=0.5)
        for quantile in [0.05, 0.5]:
            index = np.searchsorted(cdf, quantile)
            if index < len(wealth_grid):
                ax.plot(wealth_grid[index], quantile, 'o', color=color, markersize=4)
                ax.annotate(f'{quantile:.0%}: {wealth_grid[index]:.0f}',
                            xy=(wealth_grid[index], quantile),
                            xytext=(wealth_grid[index] + 5, quantile + 0.03),
                            fontsize=7, color=color)

    ax.axhline(1.0, color='gray', ls='-', lw=0.5, alpha=0.3)
    ax.set_title('Mandate Terminal Wealth CDF ($T$=10y)', fontsize=13)
    ax.set_xlabel('Terminal wealth $\\Pi_T$', fontsize=11)
    ax.set_ylabel('$P(\\Pi_T \\leq \\Pi)$', fontsize=11)
    ax.legend(fontsize=10, loc='lower right')
    ax.set_xlim(-20, 350)
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    for extension in ['pdf', 'png']:
        fig.savefig(f'/mnt/user-data/outputs/mandate_cdf_overlay.{extension}',
                    bbox_inches='tight', dpi=150)
    print("Saved mandate_cdf_overlay")


def run_local(local: Locals) -> None:
    """Run the selected mandate development figure."""
    if local == Locals.PER_MANDATE:
        print("=" * 60)
        _figure_per_mandate()
    elif local == Locals.DENSITY_OVERLAY:
        print()
        _figure_density_overlay()
    elif local == Locals.CDF_OVERLAY:
        print()
        _figure_cdf_overlay()
    elif local == Locals.ALL_FIGURES:
        print("=" * 60)
        _figure_per_mandate()
        print()
        _figure_density_overlay()
        print()
        _figure_cdf_overlay()
    else:
        raise NotImplementedError(local)
    print("\nAll done.")


if __name__ == "__main__":
    run_local(local=Locals.ALL_FIGURES)
