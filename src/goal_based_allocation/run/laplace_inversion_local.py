"""Development runner for numerical Laplace inversion diagnostics."""

from enum import Enum

import numpy as np
from scipy.special import erfc

from goal_based_allocation.laplace_inversion import (
    laplace_invert_abate_whitt,
    laplace_invert_stehfest,
)


class Locals(Enum):
    """Runnable Laplace-inversion development cases."""

    KNOWN_TRANSFORMS = 1


def _run_known_transforms() -> None:
    """Compare both inversion methods with known transform pairs."""
    print("=" * 70)
    print("TESTING LAPLACE INVERSION METHODS")
    print("=" * 70)

    coefficient = 2.0

    def exponential_transform(p):
        return (1.0 / (p + coefficient)).reshape(-1, 1)

    print("\nTest 1: f(t) = exp(-a*t), a=2")
    for time in [0.5, 1.0, 2.0, 5.0]:
        exact = np.exp(-coefficient * time)
        stehfest = laplace_invert_stehfest(exponential_transform, time)[0]
        abate_whitt = laplace_invert_abate_whitt(exponential_transform, time)[0]
        print(f"  t={time:.1f}: exact={exact:.8f}, Stehfest={stehfest:.8f} "
              f"(err={abs(stehfest - exact):.2e}), AbateWhitt={abate_whitt:.8f} "
              f"(err={abs(abate_whitt - exact):.2e})")

    def weighted_exponential_transform(p):
        return (1.0 / (p + coefficient) ** 2).reshape(-1, 1)

    print(f"\nTest 2: f(t) = t*exp(-a*t), a={coefficient}")
    for time in [0.5, 1.0, 2.0, 5.0]:
        exact = time * np.exp(-coefficient * time)
        stehfest = laplace_invert_stehfest(weighted_exponential_transform, time)[0]
        abate_whitt = laplace_invert_abate_whitt(weighted_exponential_transform, time)[0]
        print(f"  t={time:.1f}: exact={exact:.8f}, Stehfest={stehfest:.8f} "
              f"(err={abs(stehfest - exact):.2e}), AbateWhitt={abate_whitt:.8f} "
              f"(err={abs(abate_whitt - exact):.2e})")

    frequency = 3.0

    def sine_transform(p):
        return (frequency / (p ** 2 + frequency ** 2)).reshape(-1, 1)

    print(f"\nTest 3: f(t) = sin(w*t), w={frequency} (oscillatory)")
    for time in [0.5, 1.0, 2.0, 5.0]:
        exact = np.sin(frequency * time)
        stehfest = laplace_invert_stehfest(sine_transform, time)[0]
        abate_whitt = laplace_invert_abate_whitt(sine_transform, time)[0]
        print(f"  t={time:.1f}: exact={exact:+.8f}, Stehfest={stehfest:+.8f} "
              f"(err={abs(stehfest - exact):.2e}), AbateWhitt={abate_whitt:+.8f} "
              f"(err={abs(abate_whitt - exact):.2e})")

    def constant_transform(p):
        return (1.0 / p).reshape(-1, 1)

    print("\nTest 4: f(t) = 1 (constant)")
    for time in [0.5, 1.0, 2.0, 5.0]:
        exact = 1.0
        stehfest = laplace_invert_stehfest(constant_transform, time)[0]
        abate_whitt = laplace_invert_abate_whitt(constant_transform, time)[0]
        print(f"  t={time:.1f}: exact={exact:.8f}, Stehfest={stehfest:.8f} "
              f"(err={abs(stehfest - exact):.2e}), AbateWhitt={abate_whitt:.8f} "
              f"(err={abs(abate_whitt - exact):.2e})")

    def multi_output_transform(p):
        result = np.zeros((len(p), 3), dtype=complex if np.iscomplexobj(p) else float)
        for index, coefficient_j in enumerate([1.0, 2.0, 3.0]):
            result[:, index] = 1.0 / (p + coefficient_j)
        return result

    print("\nTest 5: Multi-output f(t) = [exp(-t), exp(-2t), exp(-3t)]")
    time = 1.0
    exact = np.array([np.exp(-1), np.exp(-2), np.exp(-3)])
    stehfest = laplace_invert_stehfest(multi_output_transform, time)
    abate_whitt = laplace_invert_abate_whitt(multi_output_transform, time)
    for index in range(3):
        print(f"  f_{index + 1}: exact={exact[index]:.8f}, Stehfest={stehfest[index]:.8f} "
              f"(err={abs(stehfest[index] - exact[index]):.2e}), "
              f"AbateWhitt={abate_whitt[index]:.8f} "
              f"(err={abs(abate_whitt[index] - exact[index]):.2e})")

    def survival_transform(p):
        return (np.exp(-np.sqrt(2 * p)) / p).reshape(-1, 1)

    print("\nTest 6: f(t) = erfc(1/sqrt(2t)) — survival-type function")
    for time in [0.5, 1.0, 2.0, 5.0, 10.0]:
        exact = erfc(1.0 / np.sqrt(2 * time))
        stehfest = laplace_invert_stehfest(survival_transform, time)[0]
        abate_whitt = laplace_invert_abate_whitt(survival_transform, time)[0]
        print(f"  t={time:5.1f}: exact={exact:.8f}, Stehfest={stehfest:.8f} "
              f"(err={abs(stehfest - exact):.2e}), AbateWhitt={abate_whitt:.8f} "
              f"(err={abs(abate_whitt - exact):.2e})")

    print("\n" + "=" * 70)
    print("ALL TESTS COMPLETE")
    print("=" * 70)


def run_local(local: Locals) -> None:
    """Run the selected Laplace-inversion diagnostic."""
    if local == Locals.KNOWN_TRANSFORMS:
        _run_known_transforms()
    else:
        raise NotImplementedError(local)


if __name__ == "__main__":
    run_local(local=Locals.KNOWN_TRANSFORMS)
