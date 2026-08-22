# Contributing to GoalBasedAllocation

Thanks for your interest in `goal-based-allocation`. The package implements the analytical model
behind a research paper, so explicit conventions and independently verified numerical results
take priority over convenience.

## Scope

In scope:

- Bug fixes in survival, density, moment, allocation, or option-pricing analytics
- Numerical robustness improvements backed by an independent Monte Carlo or analytical check
- Compatibility work for supported Python, NumPy, SciPy, and Matplotlib releases
- Documentation, deterministic examples, packaging, and tests

Open an issue before writing code that changes the model specification, public signatures,
dependency floors, runtime dependencies, Laplace contours, quadrature nodes, ODE tolerances, or
paper-facing results. Do not replace analytical routines with Monte Carlo implementations, add a
pandas/DataFrame layer, edit `papers/`, or submit generated figures.

## Reporting a bug

Use the bug-report template and include the package version, Python version, operating system, a
minimal self-contained reproducer, and the full traceback or incorrect output. For a numerical
problem, state the regime, rate, annualisation, wealth, floor, horizon, and model parameters, then
name the independent analytical identity or Monte Carlo check used for comparison.

## Development setup

```bash
git clone https://github.com/ArturSepp/GoalBasedAllocation.git
cd GoalBasedAllocation
uv sync --locked --group test
uv run --no-sync pytest -q
uv run --locked --only-group lint ruff check .
```

Build the documentation with the same warning gates used in CI:

```bash
uv sync --locked --extra docs
uv run --no-sync python -m sphinx -E -W --keep-going -b html docs docs/_build/html
uv run --no-sync python -m sphinx -E -W -b linkcheck docs docs/_build/linkcheck
```

`--locked` intentionally fails when `pyproject.toml` and `uv.lock` disagree. Dependency groups are
contributor environments, not package extras: use `test` for pytest, `lint` for Black and Ruff,
and retain the `docs` extra for Read the Docs.

## Pull requests

- Keep one focused topic per pull request.
- Add a regression test that fails before a behavioral fix and passes afterwards.
- Preserve rate, annualisation, wealth, floor, regime, and jump conventions explicitly.
- Verify analytical changes against a separately implemented Monte Carlo or analytical check.
- Do not adjust numerical tolerances or expected values merely to make a test pass.
- Keep runtime dependencies to NumPy, SciPy, and Matplotlib.
- Do not edit `papers/` or commit generated figures, private data, local paths, or environments.
- Run the relevant test, lint, documentation, and artifact checks before submitting.
- Do not bump package or citation versions; releases are handled separately.

## Conduct and licence

Be civil and assume good faith. By contributing, you agree that your contribution is licensed
under the project's MIT licence.
