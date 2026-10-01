# Papers and research projects

*Author: [Artur Sepp](https://github.com/ArturSepp)*

Project: [GoalBasedAllocation](https://github.com/ArturSepp/GoalBasedAllocation).
Software citation: [CITATION.cff](https://github.com/ArturSepp/GoalBasedAllocation/blob/main/CITATION.cff).

The `papers/` directory is available in a GitHub clone and is intentionally excluded from PyPI
wheel and source distributions.

## Goal-based allocation paper

`papers/goal_based_allocation_2026/` contains the LaTeX source, compiled PDF and tracked figures in
`paper/`, with the figure-generation/integration script in `replication/` for Sepp (2026), *Dynamic Mean-Variance Portfolio
Allocation under Regime-Switching Jump-Diffusions with Absorbing Barriers and Distribution
Matching* ([SSRN 6534579](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=6534579)).

Use the temporary-output command in [validation](validation.md). The `--test` mode also generates
figures; it is not a read-only test switch.

## KOSPI volatility study

`papers/kospi_volatility_fit_jun2026/` is a separate repository-only calibration study using the
shipped regime-switching option pricer. It has additional research dependencies (including
pandas) and stored Bloomberg-derived option-chain snapshots. Those dependencies and data are not
part of the core package contract or a general data service.

See the study's
[`README.md`](https://github.com/ArturSepp/GoalBasedAllocation/blob/main/papers/kospi_volatility_fit_jun2026/README.md)
for its methods, provenance, limitations, and commands.

## Output policy

Do not commit newly generated calibration output or documentation builds. Existing tracked paper
figures are replication artifacts; path-only or packaging changes must preserve their bytes.

The six-section policy is documented in the repository
[paper contract](https://github.com/ArturSepp/GoalBasedAllocation/blob/main/papers/AGENTS.md).
Drafts, private material and agent records are always ignored. New output uses
the external local runtime. The KOSPI scripts and four unchanged static CSVs now
live under `replication/`, while its displayed figures live in `paper/figures/`.
