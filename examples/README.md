# Examples

Minimal, self-contained illustrations of the core objects in the paper. Each
script uses only the public API of `goal_based_allocation`, runs in a few
seconds, and saves a single PNG next to itself.

```bash
python examples/wealth_process_simulation.py
python examples/terminal_wealth_distribution.py
python examples/investment_opportunity_set.py
python examples/regime_switch_smile.py
```

| Script | Paper figure | What it shows |
|---|---|---|
| `wealth_process_simulation.py` | Figure 1 | MV-optimal wealth paths (survived vs stopped), target `Π*(t)`, expected wealth, and the absorbing floor `L_t`, for the balanced mandate. |
| `terminal_wealth_distribution.py` | Figure 2 | Terminal wealth density decomposed into survived density, floor atom, and jump-overshoot density, overlaid with a Monte Carlo histogram. |
| `investment_opportunity_set.py` | Figure 4 | The one-parameter opportunity set: efficient frontier (implied return vs volatility) and the terminal wealth quantile fan vs implied return. |
| `regime_switch_smile.py` | — | Regime-switching implied-volatility smile from the Laplace vanilla pricer, plus Fourier and Monte Carlo reference pricers and validation checks (`UnitTests`). |

The balanced-mandate examples fix the target return directly
(`target_return=0.04`) rather than calibrating `ω*(0)=1`, so the survival and
floor-atom values are illustrative and differ slightly from the calibrated
figures in the paper. The generated PNGs are reproducible from the scripts.
