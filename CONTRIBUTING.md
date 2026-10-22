# Contributing to GoalBasedAllocation

*Author: [Artur Sepp](https://github.com/ArturSepp) / First recorded: [2026-08-22](https://github.com/ArturSepp/GoalBasedAllocation/commit/a3c6dfabe2b80893e32c0e2ca16a979abac1bf1f)*

Source: [GoalBasedAllocation](https://github.com/ArturSepp/GoalBasedAllocation).
Software citation: [CITATION.cff](https://github.com/ArturSepp/GoalBasedAllocation/blob/main/CITATION.cff).

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
pandas/DataFrame layer or submit unapproved generated figures. Paper changes follow
`papers/AGENTS.md` and must preserve the published numerical contract.

## Reporting a bug

Use the bug-report template and include the package version, Python version, operating system, a
minimal self-contained reproducer, and the full traceback or incorrect output. For a numerical
problem, state the regime, rate, annualisation, wealth, floor, horizon, and model parameters, then
name the independent analytical identity or Monte Carlo check used for comparison.

## Starter tasks

These are bounded proposals, not assigned issues or confirmed defects. Recheck source and open
work before agreeing one task. Use the existing [feature form](https://github.com/ArturSepp/GoalBasedAllocation/issues/new?template=feature_request.yml),
Area **Documentation or deterministic example**. Include the task ID, owner, exact paths, reproduction, acceptance and focused
check under Proposed behavior; retain required boundary checkboxes. Use the bug form only for
observed incorrect behaviour. Keep numerical code, public signatures, dependencies, fixtures and
canonical examples unchanged for these documentation tasks.

### GBA-1: clarify released versus source output

Owner: GoalBasedAllocation. Edit only `docs/getting-started.md` to clarify one observed release/source ambiguity. Acceptance: identify the matched snapshot, 10-year horizon, initial wealth 100 and continuous annual rate units. Preserve every computed value and the canonical script.

```console
python examples/getting_started/quickstart.py
python -m pytest tests/test_quickstart.py -q
```

### GBA-2: clarify model conventions

Owner: GoalBasedAllocation. Edit only `docs/conventions.md` to clarify one rate, wealth-unit or floor-state ambiguity. Acceptance: a reader can distinguish a model input from a realised guarantee without changing formulas, contours, quadrature nodes or ODE tolerances.

```console
python -m pytest tests/test_framework.py -q
```

### GBA-3: check release navigation

Owner: GoalBasedAllocation. Edit only `docs/getting-started.md` to clarify one stable/main or API-reference route. Acceptance: retain the pinned canonical example and existing page anchors, without changing method or manuscript results.

```console
python -m pytest tests/test_docs.py -q
```

Run the package's existing strict HTML command after a documentation edit and inspect the
changed passage. On the maintainer's OneDrive checkout, dot-source the shared
`Enter-AgentRepo.ps1` first and use `C:\Python\GoalBasedAllocation312\Scripts\python.exe`.
Keep exports, caches, builds and run output on C. The `uv` setup below is for ordinary contributor
checkouts outside OneDrive; never create `.venv` in the synchronized checkout.

Follow the [shared review guidance](https://github.com/ArturSepp/ArturSepp/blob/main/docs/contributing_to_stack.md#review-attribution-and-automation-assisted-changes).
Record the revision, actual versions, exact checks and skips, rendered review and material
automation assistance. Preserve MIT licensing and attribution. Automated checks do not
establish independent reader success. Review time and acceptance are not promised.

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
- Follow `papers/AGENTS.md` for paper changes. Do not commit unapproved generated
  figures, private data, machine-specific paths, or environments.
- Run the relevant test, lint, documentation, and artifact checks before submitting.
- Do not bump package or citation versions; releases are handled separately.

## Conduct and licence

Be civil and assume good faith. By contributing, you agree that your contribution is licensed
under the project's MIT licence.
