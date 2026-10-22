---
myst:
  html_meta:
    description: >-
      Install goal-based-allocation and run the offline quickstart: one balanced mandate with its
      floor-protected terminal wealth distribution and exact buy-and-hold benchmark, and where each
      printed number is derived in the handbook.
---

# Getting started

*Author: [Artur Sepp](https://github.com/ArturSepp) / First recorded: [2026-08-19](https://github.com/ArturSepp/GoalBasedAllocation/commit/d5a04a57ecb27f8686e2d18046c8546f3eca3577)*

Project: [GoalBasedAllocation](https://github.com/ArturSepp/GoalBasedAllocation).
Software citation: [CITATION.cff](https://github.com/ArturSepp/GoalBasedAllocation/blob/main/CITATION.cff).

The quickstart installs the package from PyPI and evaluates one point of the investment opportunity
set, a balanced mandate, offline: its calibrated mean-variance policy, the survival probability of
its wealth floor, its expected terminal wealth and the exact buy-and-hold benchmark.

## Reproducible released workflow

This first-use workflow is verified with **goal-based-allocation 0.5.0** on Python 3.12. The
package requires Python 3.11 or newer. Install core without extras; save and run the
release-pinned script in a working directory outside synchronized folders:

```powershell
python -m pip install "goal-based-allocation==0.5.0"
python -c "from importlib.metadata import version; import goal_based_allocation as p; print(version('goal-based-allocation')); print(p.__file__)"
Invoke-WebRequest -Uri 'https://raw.githubusercontent.com/ArturSepp/GoalBasedAllocation/343c9a12bcd49da4ff90cf0af295c7a7d4ec0d61/examples/getting_started/quickstart.py' -OutFile quickstart.py
python quickstart.py
```

Installation and saving the script need internet access; the calculation below runs offline.
Measure installation separately from execution; this NumPy/SciPy workflow does not use Numba.
No universal time-to-first-result is promised.

This small CPU task evaluates one balanced mandate and its analytical buy-and-hold benchmark. It uses a 10-year horizon, initial wealth 100 and continuously compounded annual rates; the printed quantities are model outputs. It needs no data, credentials, GUI, pandas or optional extras, and writes no files. NumPy, SciPy and Matplotlib are the core runtime. Keep the rate, wealth, floor and horizon conventions together when adapting inputs.

[Stable documentation](https://goalbasedallocation.readthedocs.io/en/stable/) is the release channel;
[latest](https://goalbasedallocation.readthedocs.io/en/latest/) follows main and may include unreleased
changes. The commands above pin both the 0.5.0 package and its quickstart source. The expected
output below is verified against that release.

## Install

Use the pinned commands above to reproduce the released workflow. For development, follow the
[contributor setup](https://github.com/ArturSepp/GoalBasedAllocation/blob/main/CONTRIBUTING.md)
and install the matching checkout outside synchronized folders. Development versions can differ
from the released example:

```bash
python -m pip install .
```

The core runtime requires Python 3.11 or newer, NumPy, SciPy, and Matplotlib. Once installed, the
released workflow runs offline without credentials, market data, a display, or a source checkout.

## Run one balanced mandate

The authoritative script computes one balanced mandate and its exact buy-and-hold benchmark. It
writes no files; execution time depends on the platform and numerical dependencies.

```{literalinclude} ../examples/getting_started/quickstart.py
:language: python
:caption: examples/getting_started/quickstart.py
```

[View the 0.5.0 script on GitHub](https://github.com/ArturSepp/GoalBasedAllocation/blob/343c9a12bcd49da4ff90cf0af295c7a7d4ec0d61/examples/getting_started/quickstart.py).

Verified output for 0.5.0 (minor platform differences affect only trailing digits);
`tests/test_quickstart.py` checks these values:

```text
GoalBasedAllocation quickstart
horizon=10y, initial_wealth=100, rates=continuous annual
mandate weights: bonds=35.0%, equity=43.3%, private_equity=21.7%
MV-optimal: expected_wealth=139.509, std=45.990, survival=79.071%
floor_atom=13.100%, jump_overshoot=7.830%
buy-and-hold: expected_wealth=150.637, std=74.848, implied_return=4.097%
floor_protection_cost=7.387% of terminal value
```

The floor-protection cost is the relative difference between the total values of the floor-protected
mandate and the buy-and-hold benchmark. It is not a fee or a guaranteed realised cost.

## Where the numbers come from

| Printed value | Derivation |
|---|---|
| Mandate weights | The bond-weight curve of [mandate aggregation](mandate_aggregation.md) |
| `expected_wealth`, `std`, `survival` | The [terminal wealth distribution](terminal_wealth_distribution.md) of the calibrated policy |
| `floor_atom`, `jump_overshoot` | The three components of stopped wealth, from the [Laplace framework](laplace_barrier_framework.md) |
| Buy-and-hold values | The exact [buy-and-hold moments](buy_and_hold_moments.md) |
| `floor_protection_cost` | The total values of the [opportunity set](investment_opportunity_set.md) |

The [opportunity-set chapter](investment_opportunity_set.md#worked-example) reproduces every printed
number from the public functions and checks it against an independent computation.

## First parameters to change

- `w_bd` controls the bond share. The remaining risky share is split by `AdvisorSpec.q`.
- `AdvisorSpec.omega_0` is the initial risky allocation target.
- `AdvisorSpec.c` is the continuous annual consumption rate.
- `AdvisorSpec.q_dd` scales the drawdown/floor tolerance.

The package-level opportunity-set workflow uses a 10-year horizon, initial wealth 100, and a 2%
annual continuously compounded rate. These are model inputs, not forecasts or investment advice.

Next: [the investment opportunity set and investor selection](investment_opportunity_set.md), or start
the handbook with [the regime-switching jump-diffusion](regime_switching_model.md).
