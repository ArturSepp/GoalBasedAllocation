# MV-optimal policy and glide path

## Question answered

Given an effective risky asset, horizon, target return, hurdle rate, consumption rate, and wealth
floor, solve the pre-commitment mean-variance policy and inspect its target-wealth trajectory and
allocation intensity.

The policy has the form

$$
\omega^*(t)=|\omega_a^*|\left(\frac{\Pi^*(t)}{\Pi_t}-1\right).
$$

`find_ell` solves for the Lagrange parameter and Riccati state. `gap_process_asset` maps the
resulting policy to the log-cushion process used by the Laplace survival/density functions.

## Inputs

| Input | Meaning | Convention |
|---|---|---|
| effective asset | aggregated regime drifts, volatilities, intensities, and jumps | built with `build_effective_asset` or a paper specification |
| horizon | terminal time | years |
| target return | target used to solve the pre-commitment problem | continuous annual rate |
| `r` | hurdle/risk-free rate | continuous annual rate |
| `c` | consumption rate | continuous annual rate |

The lower-level solver returns `ell` and a Riccati result. `derived_at_tau` exposes target wealth,
allocation coefficient, and related quantities at a backward-time coordinate. Keep the paper's
time-direction convention explicit when plotting a path.

## Interpretation

- A larger funding gap $\Pi^*/\Pi-1$ produces a larger risky allocation.
- As realised wealth approaches the target trajectory, the policy de-risks endogenously.
- The policy is a model result, not a guarantee that the wealth floor cannot be crossed by a jump.
- `gap_process_asset` is the correct bridge to `compute_survival` and terminal-density analytics.

## Runnable source

See
[`examples/wealth_process_simulation.py`](https://github.com/ArturSepp/GoalBasedAllocation/blob/main/examples/wealth_process_simulation.py)
for policy paths, target wealth, expected wealth, and the absorbing floor. The example is
illustrative and writes one PNG; the [quickstart](../getting-started.md) is the output-free install
check.

## Common mistakes

- Passing a simple rather than continuously compounded target/rate.
- Reading `tau` as calendar time without reversing the Riccati grid.
- Comparing an effective-asset policy with a benchmark built from different regime/jump inputs.
- Treating Monte Carlo estimates as policy inputs rather than independent validation.

API: [Riccati solver and model modules](../api/index.md).
