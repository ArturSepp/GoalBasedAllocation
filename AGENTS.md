## Python environment (mandatory)

- Never create, use, or install packages into a Python virtual environment anywhere under `C:\Users\artur\OneDrive`.
- Keep this repository's environment outside OneDrive at `C:\Python\GoalBasedAllocation312`.
- Use `C:\Python\GoalBasedAllocation312\Scripts\python.exe` for Python, tests, linters, and package installation.
- If it is missing, create it with `py -3.12 -m venv C:\Python\GoalBasedAllocation312`.
- Never run plain `uv sync` or plain `uv run` from this checkout: uv otherwise creates `<repo>\.venv` even when uv was launched through a Python executable under `C:\Python`.
- If a uv project operation is required, first set `UV_PROJECT_ENVIRONMENT=C:\Python\GoalBasedAllocation312`; for pip-style operations prefer `uv pip ... --python C:\Python\GoalBasedAllocation312\Scripts\python.exe`.
- If any OneDrive-local environment already exists, do not use it; report it for removal.
- Run standard portfolio tasks through
  `& "$env:USERPROFILE\OneDrive\analytics\my_github\ArturSepp\scripts\repo_governance\Invoke-Repo.ps1" -Task verify`.
  Use `-Task check` or `-Task test` for a narrower run. The launcher selects this repository's
  external interpreter and routes generated state to C:.

# AGENTS.md

Guidance for AI coding agents working in the **GoalBasedAllocation** repository.

## Project overview

`goal-based-allocation` implements dynamic mean-variance portfolio allocation under
regime-switching jump-diffusions with an absorbing wealth floor, solved analytically:
survival probabilities and conditional moments via Laplace transforms, the MV-optimal
policy via a Riccati ODE system, terminal wealth density decomposition, expected glide
paths, investment opportunity sets, and vanilla option pricing under the same model.
Monte Carlo exists only to validate the analytics.

It is the companion code to Sepp (2026), *Dynamic Mean-Variance Portfolio Allocation
under Regime-Switching Jump-Diffusions with Absorbing Barriers and Distribution
Matching* (SSRN 6534579). Distribution name `goal-based-allocation`; import name
`goal_based_allocation`. Licensed MIT (`LICENSE`).

## Ecosystem position

This package is one of ten public Python libraries maintained at
[github.com/ArturSepp](https://github.com/ArturSepp). Check the owning package before
adding a capability or copying code between repositories.

| Package | Repository | Purpose |
|---|---|---|
| `qis` | QuantInvestStrats | performance analytics, backtesting, and factsheet reporting |
| `optimalportfolios` | OptimalPortfolios | portfolio construction and rolling backtesting |
| `factorlasso` | FactorLasso | sparse factor-model estimation |
| `bbg-fetch` | BloombergFetch | Bloomberg data in pandas DataFrames |
| `stochvolmodels` | StochVolModels | stochastic-volatility pricing and calibration |
| `trendfollowing` | TrendFollowingSystems | closed-form trend-following analytics |
| `privateassets` | PrivateAssets | multi-factor PME for private assets |
| `goal-based-allocation` | GoalBasedAllocation | goal-based allocation under regime-switching jump-diffusions |
| `vanilla-option-pricers` | VanillaOptionPricers | Numba-vectorised BSM and Bachelier pricing |
| `option-chain-analytics` | OptionChainAnalytics | point-in-time option-chain data and queries |

Core dependency edges: `optimalportfolios` consumes `qis` and `factorlasso`;
`trendfollowing` and `privateassets` consume `qis`; `stochvolmodels` consumes
`vanilla-option-pricers`; `option-chain-analytics` consumes `qis` and
`vanilla-option-pricers`. The remaining packages have no core stack dependencies.

Optional edges: PrivateAssets' `factors` extra adds `factorlasso`; StochVolModels'
`research` extra adds `qis` and `option-chain-analytics`; OCA's `bloomberg` and `all`
extras add `bbg-fetch`. Core imports must work without optional dependencies.
OCA never imports StochVolModels or the private SigmaStrats consumer. Exact
maintainer-tool exceptions are recorded in `.github/stack-policy.json`; they do
not authorise adding those dependencies to core or importing them at package root.

## Repository layout

```
src/goal_based_allocation/
  laplace_inversion.py     Laplace transform inversion machinery
  riccati_solver.py        Riccati ODE system for the MV-optimal policy
  client_solver.py         mandate-level solver
  mandate_utils.py         portfolio mandate construction
  opportunity_set.py       investment opportunity set construction
  vanilla_option_pricer.py option pricing under the regime-switching model
  variance_swap.py         variance swap analytics
  regime_switch_paper.py   paper-facing entry points
  run/                     source-only component runners (*_local.py; no __init__.py)
tests/     top-level test modules (test_*.py)
papers/    paper replication and research projects
examples/  runnable examples
```

## Commands

```bash
uv sync --locked --group test
uv run --no-sync pytest -q                    # as CI runs it
uv run --no-sync pytest -m "not slow"         # skip Monte Carlo cross-checks
uv run --no-sync pytest tests/test_framework.py -v
uv run --no-sync python examples/getting_started/quickstart.py
uv run --locked --only-group lint ruff check .
uv run --no-sync python -m sphinx -E -W --keep-going -b html docs docs/_build/html
uv build
uv run --no-project python scripts/check_dist_contents.py dist/*
```

The `slow` pytest marker is declared for slower tests such as Monte Carlo cross-checks.
Runtime dependencies are numpy, scipy and matplotlib only. Documentation dependencies live in
the `docs` extra. Supported Python is >= 3.10; CI runs 3.10 - 3.14 under Linux and Python 3.12
under Windows and macOS, then tests built wheel/sdist artifacts outside the checkout.

## Conventions

- Test files are named `test_*.py` and live in the top-level `tests/` directory.
- Component development runners live in `src/goal_based_allocation/run/<subject>_local.py`,
  expose `Locals` and `run_local(local=...)`, and have no `__init__.py`. Never import them from
  production modules or public `__init__.py`; wheel and sdist builds exclude the entire directory.
- Enum-dispatched root examples and paper analyses use the same `Locals` and
  `run_local(local=...)` names, with one explicit selection in the `__main__` guard.
- Mark slow tests with `@pytest.mark.slow` rather than deleting or skipping them.
- Dataclasses carry model parameters and results throughout the package — extend the
  existing containers rather than passing loose tuples.
- The package is numpy/scipy only: there is no pandas dependency and no DataFrame layer.
  Keep it that way.
- Analytical results are cross-checked against the Monte Carlo simulator; a new
  analytical routine should come with the corresponding validation test.

## Constraints — do not do these

- Do not replace analytical computations with Monte Carlo. The point of this package is
  that the quantities are available in closed or semi-closed form; Monte Carlo is the
  validator, not the implementation.
- Do not add pandas, or any dependency beyond numpy, scipy and matplotlib.
- Do not change the model specification (two regimes, exponential jumps at transitions,
  single effective asset after mandate aggregation) — it is the published model.
- Do not adjust Laplace inversion contours, quadrature nodes, or ODE solver tolerances
  to make a test pass; investigate the discrepancy instead.
- Do not commit generated figures.

<!-- ===== SHARED AGENT CORE (standalone variant) — begin =====
     Generated from SHARED_AGENT_CORE.md in the maintainer's project knowledge. Do not hand-edit
     between these markers — propose the change to the maintainer instead. Variants: builder
     (qis) / consumer / standalone. Last synced 2026-09-13, agent core v1.6 -->

## Domain invariants

- Conventions are stated, never implied: rate conventions, annualisation, units of wealth and
  return. One convention per concept across the stack — if this package and a sibling disagree,
  that is a bug to report, not a difference to accommodate.

## Dependency surface

This package is standalone: it imports nothing from the stack, and its runtime surface — numpy,
scipy, matplotlib — is a design constraint, not a preference. Ask before adding any dependency.

**Never invent a symbol.** If a function, class, or keyword argument is not in the export
surface of this package or of a dependency, it does not exist. Check in one line —
`python -c "import goal_based_allocation as g; print([n for n in dir(g) if not n.startswith('_')])"`
— and say a symbol is missing rather than producing code that calls it.

## Verification loop

- Plan → patch → verify. Name the verification command and its result when proposing a patch.
- A second pass is mandatory where a plausible patch can be numerically wrong and still run
  clean: Laplace inversion, the Riccati system, densities and moments. The Monte Carlo
  simulator is the reference computed a different way — verify against it and say so.
- Prove a new test fails before trusting that it passes: reintroduce the defect, watch it fail,
  restore.

## Escalation and scope

- Stop and propose before proceeding when a change would exceed roughly five files, alter a
  public signature, or touch a numerical path.
- Never change numerical results, random seeds, or computed values unless the change is the
  request.
- A public-signature change carries a `CHANGELOG.md` entry and a version bump in the same
  change. Removing a keyword argument from a function taking `**kwargs` is a silent break — the
  caller's keyword is swallowed and nothing raises. Treat it as breaking.
- Do not refactor beyond the requested scope. Propose the wider change; do not perform it.

## Concurrent sessions

More than one agent or session may work on this checkout at the same time, so a file can change
between your read of it and your write.

- Re-read a file from disk immediately before editing it. Never write a file from an earlier
  read: a whole-file write from a stale copy silently reverts another session's work.
- Prefer minimal anchored edits over whole-file replacement. If the on-disk content is not what
  you expected, stop and reconcile your change onto the current content rather than overwrite.

## Agent-generated artifacts

All agent-generated roadmaps, execution plans, audits, reports, handoffs, and other working
outputs live under the repository-root `agents/` directory, which is local and ignored by Git.
Never create `ROADMAP_*.md`, `Claude outputs/`, `Codex outputs/`, or similar agent-output
artifacts at the repository root. Name feature roadmaps `agents/ROADMAP_<feature>.md`. An
execution request names the file and stage. A stage is complete when its stated verification
command passes; its out-of-scope list is binding.

<!-- ===== SHARED AGENT CORE — end ===== -->

## Replication contract

`papers/` reproduces the figures of Sepp (2026) and contains repository-only research.
Any change to the Laplace inversion, the Riccati solver, or the mandate aggregation requires
re-running those scripts and confirming the figures and reported values are unchanged. The main
paper's `--test` mode also generates figures; use a temporary `--outdir` for verification.

## Release checklist

A release touches four version locations. All four must agree:

1. `version` in `pyproject.toml`
2. `__version__` in `src/goal_based_allocation/__init__.py`
3. `version` and `date-released` in `CITATION.cff`
4. the software BibTeX entry in `README.md` (if it pins a version)

For an authorized publication: commit, tag that exact main-reachable commit as
`v<version>`, then build, verify and publish its artifacts. Frequent PyPI updates are
supported. A GitHub Release page is optional and created only when requested; it is not
required for a local build, pip installation or routine package publication. Development
versions on main may be ahead of PyPI. Do not publish or bump a version for unrelated work.
