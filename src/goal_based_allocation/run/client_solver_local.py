"""Development runner for client-profile calibration."""

from enum import Enum

from goal_based_allocation.client_solver import ClientProfile, solve_client


class Locals(Enum):
    """Runnable client-solver development cases."""

    CLIENT_PROFILES = 1


def run_local(local: Locals) -> None:
    """Solve and summarize the representative client profiles."""
    if local != Locals.CLIENT_PROFILES:
        raise NotImplementedError(local)

    profiles = [
        ClientProfile("Conservative", c=0.025, g=0.005,
                      x_50=101, x_75=102, q_dd=2.0),
        ClientProfile("Balanced", c=0.02, g=0.015,
                      x_50=106, x_75=109, q_dd=2.0),
        ClientProfile("Growth", c=0.00, g=0.045,
                      x_50=134, x_75=142, q_dd=2.0),
    ]

    results = []
    for profile in profiles:
        print(f"\n{'=' * 75}")
        result = solve_client(profile, n_grid=8, verbose=True)
        results.append(result)
        print()

    valid = [result for result in results if result is not None]
    if not valid:
        print("No valid results.")
        return

    print("\n" + "=" * 75)
    print("SUMMARY")
    print("=" * 75)
    headers = [result['profile'].name for result in valid]
    print(f"{'':>18}" + "".join(f"{header:>16}" for header in headers))
    print("-" * 66)

    rows = [
        ('c', [result['profile'].c for result in valid], '%'),
        ('g', [result['profile'].g for result in valid], '%'),
        ('r_h', [result['r_h'] for result in valid], '%'),
        ('q_dd', [result['profile'].q_dd for result in valid], 'f1'),
        ('Bd/Eq/PE', None, None),
        ('k', [result['k'] for result in valid], 'f2'),
        ('σ_unc', [result['sig_unc'] for result in valid], '%'),
        ('x_25 (=L_T)', [result['x_25_input'] for result in valid], 'f1'),
        ('q25 achieved', [result['q25'] for result in valid], 'f1'),
        ('q50 target', [result['profile'].x_50 for result in valid], 'f0'),
        ('q50 achieved', [result['q50'] for result in valid], 'f1'),
        ('q75 target', [result['profile'].x_75 for result in valid], 'f0'),
        ('q75 achieved', [result['q75'] for result in valid], 'f1'),
        ('q90', [result['q90'] for result in valid], 'f1'),
        ('E[Π_T]', [result['mu'] for result in valid], 'f1'),
        ('Π*_T', [result['Pi_star_T'] for result in valid], 'f0'),
        ('Survival', [result['surv'] for result in valid], '%'),
        ('|ω*_a|', [result['w_a'] for result in valid], 'f3'),
    ]
    for label, values, format_name in rows:
        if values is None:
            weights = [
                f"{result['w_bd']:.0%}/{result['w_eq']:.0%}/{result['w_pe']:.0%}"
                for result in valid
            ]
            print(f"{'Bd/Eq/PE':>18}" + "".join(f"{weight:>16}" for weight in weights))
            continue
        row = f"{label:>18}"
        for value in values:
            if format_name == '%':
                row += f"{value:15.1%}"
            elif format_name == 'f0':
                row += f"{value:16.0f}"
            elif format_name == 'f1':
                row += f"{value:16.1f}"
            elif format_name == 'f2':
                row += f"{value:16.2f}"
            elif format_name == 'f3':
                row += f"{value:16.3f}"
        print(row)


if __name__ == "__main__":
    run_local(local=Locals.CLIENT_PROFILES)
