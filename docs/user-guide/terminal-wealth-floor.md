# Absorbing floor and terminal-wealth distribution

## Probability decomposition

At horizon $T$, total probability is split into:

1. survived wealth above the floor;
2. an atom at the floor from diffusion paths that hit the absorbing barrier; and
3. wealth below the floor from crash jumps that overshoot the barrier.

The three masses should reconcile to one within numerical integration tolerance.

## Public functions

| Function | Output |
|---|---|
| `compute_survival` | probability of remaining above the absorbing barrier |
| `compute_density` | regime-conditional bounded transition densities |
| `compute_tilted_survival` | exponentially tilted survival transform used in moments |
| `compute_overshoot_density` | density of jump distance beyond the barrier |

For a terminal gap process, integrate `compute_density` on the positive gap grid and
`compute_overshoot_density` on the overshoot grid. The floor atom is the residual
$1-S-O$ after survival mass $S$ and overshoot mass $O$ are computed. Use a grid that is wide and
fine enough for the desired tolerance; do not clip a negative residual silently without checking
integration error.

## Units and interpretation

- Density grids are in the log-gap or overshoot coordinate expected by the corresponding API, not
  directly in wealth units.
- Wealth mapping requires the target wealth and buffer from the same Riccati solution.
- A positive overshoot mass is expected in a jump model and is economically different from the
  diffusion floor atom.

## Runnable source

[`examples/terminal_wealth_distribution.py`](https://github.com/ArturSepp/GoalBasedAllocation/blob/main/examples/terminal_wealth_distribution.py)
constructs the full decomposition and overlays an independent Monte Carlo histogram. It saves one
PNG. For regression tolerances, see [validation](../validation.md).

## Failure modes

- Root-finding failure when Laplace characteristic roots are not separated cleanly.
- An inversion/grid range that truncates material tail mass.
- Mixing an asset specification with a gap process produced by another Riccati solution.
- Confusing the exponential jump rate with its reciprocal mean.

API: [Laplace and regime-switching modules](../api/index.md).
