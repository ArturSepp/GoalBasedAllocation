---
myst:
  html_meta:
    description: >-
      Install goal-based-allocation and run the offline quickstart: one balanced mandate with its
      floor-protected terminal wealth distribution and exact buy-and-hold benchmark, and where each
      printed number is derived in the handbook.
---

# Getting started

*Author: [Artur Sepp](https://github.com/ArturSepp)*

Project: [GoalBasedAllocation](https://github.com/ArturSepp/GoalBasedAllocation).
Software citation: [CITATION.cff](https://github.com/ArturSepp/GoalBasedAllocation/blob/main/CITATION.cff).

The quickstart installs the package from PyPI and evaluates one point of the investment opportunity
set, a balanced mandate, offline: its calibrated mean-variance policy, the survival probability of
its wealth floor, its expected terminal wealth and the exact buy-and-hold benchmark.

## Install

```bash
python -m pip install goal-based-allocation
```

The core runtime requires Python 3.10 or newer, NumPy, SciPy, and Matplotlib. The first-success
workflow is offline: it needs no credentials, market data, display, or source checkout.

## Run one balanced mandate

The authoritative script computes one balanced mandate and its exact buy-and-hold benchmark. It
normally completes in under 15 seconds and writes no files.

```{literalinclude} ../examples/getting_started/quickstart.py
:language: python
:caption: examples/getting_started/quickstart.py
```

[View the script on GitHub](https://github.com/ArturSepp/GoalBasedAllocation/blob/main/examples/getting_started/quickstart.py).

Expected output (minor platform differences affect only trailing digits); `tests/test_quickstart.py`
checks these values:

```text
GoalBasedAllocation quickstart
horizon=10y, initial_wealth=100, rates=continuous annual
mandate weights: bonds=35.0%, equity=43.3%, private_equity=21.7%
MV-optimal: expected_wealth=139.040, std=46.087, survival=78.703%
floor_atom=13.431%, jump_overshoot=7.866%
buy-and-hold: expected_wealth=150.637, std=74.848, implied_return=4.097%
floor_protection_cost=7.699% of terminal value
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
