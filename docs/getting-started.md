# Getting started

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

Expected output at version 0.3.0 (minor platform differences affect only trailing digits):

```text
GoalBasedAllocation quickstart
horizon=10y, initial_wealth=100, rates=continuous annual
mandate weights: bonds=35.0%, equity=43.3%, private_equity=21.7%
MV-optimal: expected_wealth=139.040, std=46.087, survival=78.703%
floor_atom=13.431%, jump_overshoot=7.866%
buy-and-hold: expected_wealth=150.637, std=74.848, implied_return=4.097%
floor_protection_cost=7.699% of terminal value
```

The floor-protection cost is the relative difference between the terminal-value measures used by
the analytical mandate and buy-and-hold benchmark. It is not a fee or guaranteed realised cost.

## First parameters to change

- `w_bd` controls the bond share. The remaining risky share is split by `AdvisorSpec.q`.
- `AdvisorSpec.omega_0` is the initial risky allocation target.
- `AdvisorSpec.c` is the continuous annual consumption rate.
- `AdvisorSpec.q_dd` scales the drawdown/floor tolerance.

The package-level opportunity-set workflow uses a 10-year horizon, initial wealth 100, and a 2%
annual continuously compounded rate. These are model inputs, not forecasts or investment advice.

Next: [mandates and opportunity sets](user-guide/mandates-opportunity-set.md).
