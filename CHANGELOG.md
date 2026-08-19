# Changelog

All notable changes to this project are documented in this file. The format is
based on [Keep a Changelog](https://keepachangelog.com/), and the project
follows [Semantic Versioning](https://semver.org/).

## [Unreleased]

## [0.3.0] - 2026-08-19

### Changed
- Moved the installable package from `goal_based_allocation/` to
  `src/goal_based_allocation/` without changing the import name or numerical code.
- Renamed the repository-only research root from `paper_code/` to `papers/`. Existing
  GitHub source and image URLs containing `paper_code/` must use the new path.
- Added an output-free installed-wheel quickstart, artifact-provenance CI, and a gated
  Sphinx/MyST documentation site with task guides and a checked public API catalogue.
- Adopted SPDX license metadata and raised the build-backend floor to setuptools 77;
  runtime dependencies are unchanged.
- Published task-oriented Sphinx documentation on GitHub Pages and added its canonical
  project URL to package metadata.

## [0.2.0] - 2026-07-15

### Added
- `vanilla_option_pricer` module: Laplace-transform European vanilla option
  pricing under the regime-switching jump-diffusion. Exposes `RiskNeutralParams`,
  `price_vanilla`, `implied_vol`, `OptionType`, and `Regime` from the package top
  level. All strikes are priced jointly by a single Abate-Whitt inversion in
  maturity, reusing the degree-6 characteristic roots of the wealth-floor
  machinery.
- `examples/` folder with minimal, self-contained illustrations:
  `wealth_process_simulation.py` (Figure 1), `terminal_wealth_distribution.py`
  (Figure 2), `investment_opportunity_set.py` (Figure 4), and
  `regime_switch_smile.py` (implied-volatility smile with Carr-Madan Fourier and
  Monte Carlo reference pricers).
- `tests/` suite (pytest): option-pricer properties (put-call parity,
  Black-Scholes limit, monotonicity in strike, scalar/array consistency, input
  validation), Fourier and Monte Carlo cross-checks, and core-framework smoke
  tests.
- GitHub Actions CI: runs the test suite on Python 3.10, 3.11, and 3.12.

### Changed
- Repository reorganised: the core library remains in `goal_based_allocation/`;
  the 2026 paper (LaTeX source, compiled PDF, figures, and
  `generate_paper_figures.py`) is now self-contained under
  `papers/goal_based_allocation_2026/`.
- `generate_paper_figures.py` now defaults its output directory to the paper's
  own `figures/` folder, independent of the working directory.
- The `regime_switch_smile` example draws its parameters from the paper's equity
  regime process instead of a standalone base case.
- README updated for the new layout, the option pricer, and the examples.
- Paper title standardised to "Dynamic Mean-Variance Portfolio Allocation under
  Regime-Switching Jump-Diffusions with Absorbing Barriers and Distribution
  Matching" across code docstrings, README, and the citation.

### Fixed
- Reconciled the package version, which was out of sync between `__init__.py`
  (0.1.0) and `pyproject.toml` (0.1.1).

## [0.1.1]

Initial packaged release of the analytical framework: Riccati ODE solver for the
MV-optimal policy, Laplace-domain density / survival / overshoot computations,
buy-and-hold moments via matrix exponential, and the investment opportunity-set
framework.
