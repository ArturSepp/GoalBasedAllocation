# Validation and numerical evidence

Analytical and semi-analytical calculations are the implementation. Independent Monte Carlo and
alternative transforms are validators.

## Fast source suite

```bash
pytest -m "not slow" -q
```

This covers probability bounds/monotonicity, Riccati initial conditions, exact buy-and-hold
moments, option-pricer properties, Fourier agreement, metadata, paths, and quickstart behavior.

## Full suite

```bash
pytest -q
```

The `slow` tests add seeded Monte Carlo option-price cross-checks. The option test accepts a
four-standard-error envelope rather than forcing a deterministic price match.

## Paper validator

From a development install at repository root:

```bash
python papers/goal_based_allocation_2026/generate_paper_figures.py \
  --test --outdir temporary_paper_output/
```

The current CLI runs nine assertions and then generates the ten figures. Always use a temporary
output directory for verification. The assertions cover density normalization, barrier-density
and analytical-survival consistency, horizon monotonicity, asset comparisons, Riccati initial
conditions, a 100K-path Monte Carlo survival comparison, and Table 1 inputs.

## Numerical conventions

- Do not change Laplace inversion contours, quadrature nodes, or ODE tolerances to satisfy a test.
- Do not silently regenerate expected values or paper figures.
- When a plausible change can run but be numerically wrong, require an independent method: Monte
  Carlo for wealth-floor analytics and Fourier/Monte Carlo for option pricing.
- Record exact values and environment for migration/release gates.

See the repository's ignored `agents/` reports for dated local migration evidence; they are
operational records, not public package documentation.
