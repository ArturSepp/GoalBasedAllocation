"""Development runner for the investment opportunity set."""

from enum import Enum

import matplotlib.pyplot as plt

from goal_based_allocation.opportunity_set import (
    PI0,
    AdvisorSpec,
    build_opportunity_set,
)


class Locals(Enum):
    """Runnable opportunity-set development cases."""

    OPPORTUNITY_SET = 1


def run_local(local: Locals) -> None:
    """Build, print, and plot the representative investment opportunity set."""
    if local != Locals.OPPORTUNITY_SET:
        raise NotImplementedError(local)

    spec = AdvisorSpec(omega_0=1.0, c=0.0, q=2/3, q_dd=2.0)
    print(f"Advisor spec: ω₀={spec.omega_0}, c={spec.c:.0%}, q={spec.q:.2f}, q_dd={spec.q_dd}")
    opportunity_set = build_opportunity_set(spec)

    print(f'\n{"w_bd":>5} {"w_eq":>5} {"w_pe":>5} | {"r_impl":>7} '
          f'{"E[Π]":>6} {"Std":>5} {"q5":>5} {"q25":>5} {"q50":>5} '
          f'{"q75":>5} {"q95":>5} {"Surv":>5} {"Cost%":>6} {"L_T":>4}')
    print('=' * 90)
    for point in opportunity_set:
        print(f'{point["w_bd"]:5.0%} {point["w_eq"]:5.0%} {point["w_pe"]:5.0%} | '
              f'{point["r_impl"]:7.2%} {point["E"]:6.1f} {point["Std"]:5.1f} '
              f'{point["q5"]:5.0f} {point["q25"]:5.0f} {point["q50"]:5.0f} '
              f'{point["q75"]:5.0f} {point["q95"]:5.0f} '
              f'{point["S"]:5.1%} {point["floor_cost_pct"]:5.1%} {point["L_T"]:4.0f}')

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    r_impls = [point['r_impl'] * 100 for point in opportunity_set]
    w_bds = [point['w_bd'] * 100 for point in opportunity_set]

    ax = axes[0, 0]
    ax.plot(w_bds, r_impls, 'b-o', lw=2, ms=6)
    ax.set_xlabel('Bond weight (%)', fontsize=12)
    ax.set_ylabel('Implied return $r_{impl}$ (%)', fontsize=12)
    ax.set_title('Implied Return vs Bond Weight', fontsize=13)
    ax.invert_xaxis()

    ax = axes[0, 1]
    ax.plot([point['Std'] for point in opportunity_set], r_impls, 'r-o', lw=2, ms=6)
    ax.set_xlabel('Std[$\\Pi_T$]', fontsize=12)
    ax.set_ylabel('$r_{impl}$ (%)', fontsize=12)
    ax.set_title('Efficient Frontier (MV with Floor)', fontsize=13)

    ax = axes[1, 0]
    selected = [
        opportunity_set[0],
        opportunity_set[len(opportunity_set) // 3],
        opportunity_set[2 * len(opportunity_set) // 3],
        opportunity_set[-1],
    ]
    for point, color in zip(selected, ['C0', 'C1', 'C2', 'C3']):
        label = f"w_bd={point['w_bd']:.0%}, r={point['r_impl']:.1%}"
        ax.plot(point['Pi_cdf'], point['cdf'], color=color, lw=2, label=label)
    ax.set_xlabel('Terminal wealth $\\Pi_T$', fontsize=12)
    ax.set_ylabel('CDF', fontsize=12)
    ax.set_title('Wealth Distribution for Selected Portfolios', fontsize=13)
    ax.legend(fontsize=9)
    ax.set_xlim(50, 300)

    ax = axes[1, 1]
    ax.fill_between(r_impls, [point['q5'] for point in opportunity_set],
                    [point['q95'] for point in opportunity_set],
                    alpha=0.15, color='C0', label='5%–95%')
    ax.fill_between(r_impls, [point['q25'] for point in opportunity_set],
                    [point['q75'] for point in opportunity_set],
                    alpha=0.3, color='C0', label='25%–75%')
    ax.plot(r_impls, [point['q50'] for point in opportunity_set],
            'C0-', lw=2, label='Median')
    ax.plot(r_impls, [point['L_T'] for point in opportunity_set],
            'r--', lw=1.5, label='$L_T$ (floor)')
    ax.axhline(PI0, color='gray', ls=':', alpha=0.5)
    ax.set_xlabel('Implied return $r_{impl}$ (%)', fontsize=12)
    ax.set_ylabel('Terminal wealth $\\Pi_T$', fontsize=12)
    ax.set_title('Quantile Fan Chart', fontsize=13)
    ax.legend(fontsize=10)

    fig.suptitle(f'Investment Opportunity Set (ω₀={spec.omega_0:.0%}, c={spec.c:.0%}, '
                 f'q={spec.q:.2f}, q_dd={spec.q_dd})', fontsize=14)
    fig.tight_layout()
    fig.savefig('/mnt/user-data/outputs/opportunity_set.png', dpi=150)
    print('\nSaved opportunity_set.png')
    plt.close('all')


if __name__ == "__main__":
    run_local(local=Locals.OPPORTUNITY_SET)
