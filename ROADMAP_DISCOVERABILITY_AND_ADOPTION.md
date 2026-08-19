# GoalBasedAllocation discoverability and adoption roadmap

Version 1.0, 2026-08-19

Source: adapted from the unified OSS discoverability and adoption roadmap at
`C:\Users\artur\OneDrive\analytics\my_github\.agents\ROADMAP_OSS_DISCOVERABILITY_AND_ADOPTION.md`.
The path supplied in the adaptation request,
`C:\Users\artur\OneDrive\analytics\my_github\CLAUDE\ROADMAP_OSS_DISCOVERABILITY_AND_ADOPTION.md`,
does not exist as of 2026-08-19; the portfolio master above is the source used by the existing
TrendFollowingSystems and BloombergFetch adaptations.

Status: local implementation completed through U5 on 2026-08-19. U0, U1, M1, M2, D1, and the
local portions of U2, U3a, U4a, U4b, U5, and U6 pass. Gate B used the roadmap's default defer
decision, so U7 is skipped. Gate A, deployed U3b evidence, U8 release, and U9 measurement remain
pending and require external settings, public deployment, or explicit release approval.

## Outcome

Make `goal-based-allocation` easy to discover, install, evaluate, reproduce, and cite as the
analytical Python implementation of dynamic mean-variance allocation under regime-switching
jump-diffusions with absorbing wealth floors. A qualified new user should be able to:

1. install a released wheel and obtain a deterministic survival/allocation result in less than
   one minute without data, credentials, or a source checkout;
2. understand the model's rate, wealth, jump, regime, floor, and annualisation conventions before
   interpreting a result;
3. solve and compare MV-optimal mandates, terminal-wealth distributions, and buy-and-hold
   benchmarks through documented public APIs;
4. distinguish the core allocation workflow from the secondary vanilla-option and research
   replication workflows; and
5. find the paper, validation evidence, API reference, examples, changelog, and citation from one
   canonical documentation root.

The objective is qualified discovery followed by successful first use, not raw traffic, stars,
or a broader claim that this specialised model is a general portfolio-management system.

## Binding package decisions

These decisions are mandatory for this adaptation.

1. **Migrate the installable package to a src layout.** Move `goal_based_allocation/`
   mechanically to `src/goal_based_allocation/`. Keep `tests/`, `examples/`, and the renamed
   `papers/` directory at the repository root. Preserve the import token
   `goal_based_allocation`, public signatures, numerical outputs, and source bytes.
2. **Rename `paper_code/` to `papers/`.** Use `git mv paper_code papers` in a separate
   migration commit. The old root directory must be absent afterward. Update every repository
   reference, command, image path, layout description, and agent instruction. Preserve the two
   paper/research projects and all tracked data, PDFs, LaTeX, scripts, and figures byte-for-byte
   except for path text that must change.
3. **Keep examples and papers repository-only.** `examples/` is the single public source-example
   location. `papers/` is the single replication/research location. Neither belongs inside the
   installable package, and neither is copied into `docs/`. The PyPI wheel and sdist must state
   and enforce their narrower installed-artifact contract.
4. **Publish a wheel as well as an sdist at the next approved release.** PyPI 0.2.0 exposes only
   a 54.3 kB source distribution. The candidate release must build, inspect, install, and publish
   both artifacts; editable-install success is not release evidence.
5. **Preserve scientific and numerical results.** No model specification, Laplace contour,
   quadrature node, ODE tolerance, random seed, public call signature, stored market data,
   paper value, or figure is changed by this roadmap. Analytical calculations remain the
   implementation and Monte Carlo remains an independent validator.
6. **Keep the runtime dependency surface unchanged.** The core remains NumPy, SciPy, and
   Matplotlib only. Documentation tools may be approved as a `docs` extra; paper-only tools such
   as pandas remain paper-only and do not enter runtime dependencies.
7. **Do not release implicitly.** Layout, README, metadata, documentation, or packaging changes
   reach PyPI only through U8 after explicit maintainer approval.

If a mandatory migration changes a numerical result, import path, public signature, research data
byte, or tracked figure byte, stop and report the mismatch. Do not update expected values or
regenerate figures to make the migration appear successful.

## Package adaptation profile

| Field | Package-specific answer |
|---|---|
| Distribution name | `goal-based-allocation` |
| Import name | `goal_based_allocation` |
| Current public release | 0.2.0, released on PyPI 2026-07-15 |
| One-sentence role | Analytical dynamic mean-variance allocation and terminal-wealth risk under regime-switching jump-diffusions in Python for quantitative researchers and wealth-management model developers |
| Primary users | Quantitative researchers, wealth-management/asset-allocation model developers, and reviewers reproducing Sepp (2026) |
| Priority task 1 | Solve the Riccati system for an MV-optimal allocation policy and interpret its endogenous de-risking glide path |
| Priority task 2 | Compute survival, floor-hit/overshoot risk, moments, and the full terminal-wealth distribution against an absorbing wealth floor |
| Priority task 3 | Construct and compare mandates and investment opportunity sets, including exact buy-and-hold benchmarks |
| Secondary task | Price and validate European vanilla options under the same two-regime jump-diffusion |
| Differentiating workflow | Riccati policy solution and Laplace-domain terminal-wealth decomposition with Monte Carlo cross-validation under one published regime-switching model |
| Canonical repository | `https://github.com/ArturSepp/GoalBasedAllocation` |
| Canonical documentation now | GitHub README and PyPI-rendered release README |
| Canonical documentation target | `https://artursepp.github.io/GoalBasedAllocation/`, subject to the documentation and external-settings gates |
| Package index | `https://pypi.org/project/goal-based-allocation/` |
| Documentation system | None now; proposed Sphinx/MyST static site on GitHub Pages |
| First-success archetype | Offline numerical library |
| First-success contract | Released wheel, no network/data/credentials, deterministic public-API result in less than one minute |
| Release authority | Artur Sepp; PyPI and GitHub credentials required |
| Existing analytics | GitHub/PyPI/Pepy badges and public counters; Search Console and documentation analytics unknown |
| Scientific boundary | The SSRN paper, publication status, claims, and numerical results are separate; this roadmap may link and cite them but cannot alter or overstate them |
| Research-code boundary | `papers/goal_based_allocation_2026/` and `papers/kospi_volatility_fit_jun2026/` are checkout-only research projects, not installed APIs or core first-success requirements |
| Proceed/defer | Proceed: the public capability is mature, the package is lightweight, the companion paper creates qualified demand, and the packaging/documentation gaps are concrete |

## Canonical identity

Proposed public sentence:

> `goal-based-allocation` — analytical dynamic mean-variance allocation and terminal-wealth risk
> under regime-switching jump-diffusions in Python for quantitative researchers and
> wealth-management model developers.

Proposed boundary sentence:

> It models two regimes, exponential jumps at regime transitions, an absorbing wealth floor, and
> multi-asset mandates aggregated to one effective risky asset; it is not a discrete constrained
> multi-asset optimiser, a trading engine, or a production portfolio-management system.

Use `optimalportfolios` as the linked sibling for discrete rolling multi-asset optimisation,
constraints, transaction costs, and backtesting. Keep option pricing visible as a secondary
capability that reuses the same model, not as a competing primary identity.

## Current evidence that sets the priorities

These are dated planning observations from 2026-08-19, not a substitute for the fixed U1
baseline:

- the installable package is at repository root and setuptools discovery has no explicit `src`
  boundary, so checkout imports can mask an incomplete artifact;
- CI installs editable and runs `pytest -q` only; it does not build a distribution, install a
  wheel outside the checkout, or assert import provenance;
- the current local suite passes on Python 3.12: 46 tests in 2.07 seconds, including the marked
  Monte Carlo checks; the README survival smoke values are 0.966021, 0.922337, 0.804665, and
  0.686081 for 1, 2, 5, and 10 years under the paper equity specification;
- PyPI 0.2.0 exposes an sdist but no wheel, while the public installation path leads with
  `pip install goal-based-allocation`;
- the GitHub README is already strong domain documentation, but there is no independent
  documentation root, API site, sitemap, canonical-page policy, or task-level navigation;
- the PyPI-rendered 0.2.0 README is older than the current GitHub README, so its positioning and
  repository layout remain stale until an approved release;
- `paper_code` appears in the root README, AGENTS.md, CHANGELOG.md, and the KOSPI study README;
  public image and command links will need an explicit migration note when it becomes `papers`;
- tests insert `examples/` into `sys.path` to import a reference pricer, which is legitimate for
  independent validation but must not make examples look like installed modules;
- all four current version surfaces—`pyproject.toml`, `goal_based_allocation.__version__`,
  `CITATION.cff`, and README BibTeX—say 0.2.0, although AGENTS.md currently lists only three in
  its release checklist;
- GitHub showed 10 stars, 2 forks, 1 watcher, no GitHub Releases, and 15 commits; these are
  context, not optimisation targets; and
- the repository has useful paper figures and plotting examples, but it lacks one output-free,
  wheel-tested first-success script.

These findings justify structural hardening before expanding public content. They do not justify
changes to the mathematical API.

## Artifacts

Public, candidate-for-commit execution contract:

```text
ROADMAP_DISCOVERABILITY_AND_ADOPTION.md
```

Local operational records under the untracked and ignored `agents/` directory:

```text
agents/ADOPTION_PROFILE.md
agents/DISCOVERABILITY_BASELINE.md
agents/M1_SRC_LAYOUT_REPORT.md
agents/M2_PAPERS_RENAME_REPORT.md
agents/DISCOVERABILITY_AUDIT.md
agents/DISCOVERABILITY_90_DAY_REPORT.md
```

U0 must add `/agents/` to `.gitignore` before creating local reports. Credentials, Search Console
exports, browser state, raw analytics, downloaded competitor material, and release tokens never
enter the repository. User-facing documentation belongs under committed `docs/`.

## Global execution rules

- Execute and verify one stage at a time; use one focused commit per mandatory migration.
- Re-read every target file immediately before editing and reconcile concurrent changes.
- A stage is complete only when its stated commands and independent numerical checks pass.
- Record command, environment, exact result, and commit in the status log and local stage report.
- Use released public symbols in user documentation and verify every named symbol exists.
- State wealth units, rates, compounding, annualisation, return convention, jump-parameter
  convention, regime labels, horizon, and floor convention explicitly.
- A source example is authoritative. Documentation includes it mechanically or tests it; code is
  not copied by hand into multiple sources.
- Keep `examples/` and `papers/` at repository root and out of the installed package.
- Build and test wheel and sdist. Run installed-artifact checks from a temporary directory outside
  the checkout; an editable installation is never sufficient.
- Keep documentation dependencies in a separately approved `docs` extra and never add paper-only
  dependencies to the package runtime.
- Do not commit generated docs HTML, newly generated figures, notebook output, calibration output,
  downloaded data, or analytics exports.
- Preserve existing tracked figures and research data byte-for-byte during path migrations.
- Search observations are dated spot checks, not rank measurements.
- Comparisons use current primary sources and include at least one use case that favours each
  genuine alternative.
- A version bump, GitHub Release, documentation publication, or PyPI upload requires explicit
  maintainer approval at the relevant gate.

## Execution order

The two mandatory migrations precede identity and content work so later pages and links are built
on stable paths.

| Order | Stage | Deliverable | Main gate |
|---:|---|---|---|
| 1 | U0 | Adaptation profile and proceed decision | Role, boundary, and maintenance case are explicit |
| 2 | U1 | Dated discovery/conversion baseline | Pre-change evidence is fixed |
| 3 | M1 | Mandatory src-layout migration | Imports and tests no longer depend on root package placement |
| 4 | M2 | Mandatory `paper_code` to `papers` rename | Research assets and commands survive the path migration |
| 5 | U2 | Canonical identity and repository trust metadata | Public surfaces describe one package |
| 6 | Decision D1 | Documentation stack and dependency approval | No docs dependency is added implicitly |
| 7 | U3a | Local documentation foundation | Static docs build warning-free |
| 8 | U6 | Root first-success workflow | Clean wheel yields deterministic success |
| 9 | U4a | Core allocation task documentation | The three priority jobs are executable and interpretable |
| 10 | U4b | Conventions, validation, options, and paper guides | Secondary workflows have clear boundaries |
| 11 | U5 | Neutral comparison/choice guide | Qualified users can decide fit |
| 12 | Gate A | GitHub About, Pages, Search Console, and deployment | Canonical pages are public and indexable |
| 13 | U3b | Deployed technical discoverability audit | Public technical evidence passes |
| 14 | Gate B | Hosted-notebook decision | Default recommendation: defer |
| 15 | U7 | Optional thin notebook | Only if Gate B approves |
| 16 | U8 | Release, deploy, and trust alignment | Wheel, sdist, docs, tag, and metadata match |
| 17 | U9 | 30/60/90-day measurement | Further investment follows evidence |

---

## U0 — Triage and adapt

**Deliverable:** `agents/ADOPTION_PROFILE.md` and this roadmap.

Record the profile above, current release, dependency position, users, three priority tasks,
offline first-success contract, scientific boundary, research-code boundary, documentation
maturity, and maintenance decision. Add `/agents/` to `.gitignore` before writing the report.

**Acceptance:** the profile contains distinct positioning, concrete tasks, a feasible
first-success path, and a proceed/defer decision; the operational report is ignored by Git.

**Verification:**

```powershell
$text = Get-Content agents/ADOPTION_PROFILE.md -Raw
$required = 'Distribution name','Primary user','Priority task','First-success','Proceed'
$required | ForEach-Object {
    if ($text -notmatch [regex]::Escape($_)) { throw "missing $_" }
}
git check-ignore agents/ADOPTION_PROFILE.md
```

Manually compare the report with `pyproject.toml`, README.md, AGENTS.md, package exports, examples,
papers, PyPI, and GitHub About.

**Out of scope:** implementation, release, publication claims, or rebranding the package
portfolio.

## U1 — Establish the baseline

**Deliverable:** `agents/DISCOVERABILITY_BASELINE.md`.

Fix one dated snapshot before implementation:

- PyPI version, release date, artifact names/sizes/types, project links, description, and rendered
  README;
- GitHub canonical URL, About text/link, stars, forks, watchers, tags, releases, default branch,
  and visible activity;
- current documentation topology and the path from discovery to first successful result;
- README/PyPI differences, broken local links, stale paths, and identity contradictions;
- wheel/sdist contents, src-layout state, example state, paper path, and installed-artifact test
  coverage;
- Search Console, package downloads, docs referrals, citations, issues, and external references,
  marking unavailable credentialed values `unknown`; and
- exact treatment of branded and fixed non-branded queries.

Fixed non-branded queries:

1. `goal based portfolio allocation Python package`
2. `portfolio wealth floor survival probability Python`
3. `dynamic mean variance regime switching Python`

Branded means a query containing `goal-based-allocation`, `goal_based_allocation`, or
`GoalBasedAllocation`, case-insensitive.

Capture the current source-only PyPI artifact explicitly and run the current released sdist in a
clean environment so U8 can compare like for like.

**Acceptance:** every value has a date and source family; unavailable credentialed data remains
unknown; the report contains `Indexing`, `Queries`, `Conversion path`, `Artifacts`, `Adoption
signals`, and `Limitations`.

**Verification:**

```powershell
$required = 'Indexing','Queries','Conversion path','Artifacts','Adoption signals','Limitations'
$text = Get-Content agents/DISCOVERABILITY_BASELINE.md -Raw
$required | ForEach-Object {
    if ($text -notmatch [regex]::Escape($_)) { throw "missing $_" }
}
```

**Out of scope:** changing public settings, repository paths, or documentation while measuring
the baseline.

## M1 — Mandatory migration to src layout

**Deliverable:** one mechanical layout commit and `agents/M1_SRC_LAYOUT_REPORT.md`.

Target layout:

```text
src/
    goal_based_allocation/
        __init__.py
        client_solver.py
        laplace_inversion.py
        mandate_utils.py
        opportunity_set.py
        regime_switch_paper.py
        riccati_solver.py
        vanilla_option_pricer.py
        variance_swap.py
tests/                         remains at repository root
examples/                      remains at repository root
papers/                        remains at repository root after M2
```

Implementation requirements:

1. Record SHA-256 hashes for every file under `goal_based_allocation/`, then use
   `git mv goal_based_allocation src/goal_based_allocation`.
2. Configure setuptools explicitly:

   ```toml
   [tool.setuptools]
   package-dir = {"" = "src"}

   [tool.setuptools.packages.find]
   where = ["src"]
   include = ["goal_based_allocation*"]
   ```

3. Preserve `import goal_based_allocation`, all top-level exports, `__version__`, module imports,
   function signatures, and numerical behavior.
4. Add a structural test that rejects a root `goal_based_allocation/` directory and an import
   provenance test that proves the tested module comes from the installed candidate artifact.
5. Keep the deliberate `examples/` path insertion used by the independent reference pricers,
   but assert that `examples` is not installed as a package and cannot shadow the core import.
6. Update only path/layout references required by the move. Identity and explanatory content
   changes wait for U2/U4.
7. Update CI with a distribution job that builds wheel and sdist, installs each outside the
   checkout, changes to a temporary working directory, imports the package, runs the first-success
   smoke, and executes `pip check`.
8. Add a small artifact-content check that asserts exactly one `goal_based_allocation` package is
   present and `tests/`, `examples/`, `paper_code/` (before M2), `papers/` (after M2), and other
   repository-only files are absent from the wheel. Record the intended sdist contract separately.
9. Keep supported Python and dependency floors unchanged.

Failure-first evidence:

- add the root-package rejection test, temporarily recreate an empty root
  `goal_based_allocation/`, observe failure, remove it, and record the result;
- run the installed-import provenance check before candidate installation and observe failure or
  a checkout path, then run it against the wheel from a temporary directory and observe a
  `site-packages` path; and
- show that the current PyPI release has no wheel, then show that the candidate build produces
  one without publishing it.

Numerical preservation evidence:

- compare the pre/post source hashes and require every mechanically moved module to match;
- run the full 46-test suite, including the Fourier and Monte Carlo reference cross-checks;
- record the pre/post survival vector for horizons 1, 2, 5, and 10;
- record pre/post Riccati initial conditions and buy-and-hold moment smoke values; and
- run the paper integration checks after M2 establishes the final path.

**Acceptance:**

- only `src/goal_based_allocation` supplies the installed package;
- root `tests/`, `examples/`, and `papers/` remain outside it;
- all moved source hashes, public imports, signatures, and numerical values are unchanged;
- the full suite and independent analytical/Monte Carlo checks pass;
- wheel and sdist build successfully and the wheel contains exactly one package copy; and
- both artifacts install and import from a working directory outside the checkout.

**Verification:**

```powershell
python -m pip install -e ".[dev]"
pytest -q
ruff check --select E9,F63,F7,F82,F811 src tests examples
python -m build
python scripts/check_dist_contents.py dist/*
```

Then create clean environments outside the checkout, install the wheel and sdist separately, and
run:

```python
from pathlib import Path
import goal_based_allocation as gba

package_file = Path(gba.__file__).resolve()
assert "GoalBasedAllocation" not in str(package_file)
assert gba.__version__ == "0.2.0"  # replace only at an explicitly approved release stage

eq = gba.create_paper_assets()["equity"]
values = [gba.compute_survival(t, eq.x0, eq) for t in (1.0, 2.0, 5.0, 10.0)]
assert all(values[i] >= values[i + 1] for i in range(3))
```

Use the supported Python 3.10–3.12 CI matrix. Do not turn this mechanical migration into a style
refactor.

**Out of scope:** paper-directory rename, documentation expansion, formula changes, test
relocation, import renaming, dependency changes, version bump, or release.

## M2 — Mandatory rename from `paper_code` to `papers`

**Deliverable:** one path-migration commit and `agents/M2_PAPERS_RENAME_REPORT.md`.

Target layout:

```text
papers/
    goal_based_allocation_2026/
        goal_based_allocation_2026.tex
        goal_based_allocation_2026.pdf
        generate_paper_figures.py
        figures/
    kospi_volatility_fit_jun2026/
        README.md
        data/
        figures/
        run_analysis.py
        regime_switch_calibration.py
        term_structure.py
        vol_surface_utils.py
```

Implementation requirements:

1. Record relative-path, size, and SHA-256 manifests for every file below `paper_code/`.
2. Use `git mv paper_code papers`; do not copy/recreate files or leave a compatibility directory.
3. Update all `paper_code` references. The pre-migration scan identifies four containing files:
   README.md, AGENTS.md, CHANGELOG.md, and the KOSPI study README. Scan again at execution because
   concurrent work may add references.
4. Update root README structure, reproduction commands, and all selected-figure image paths to
   `papers/...`; update repository instructions and the replication contract to name `papers/`.
5. Add an `[Unreleased]` changelog migration note because old GitHub source/image links are not
   redirected by a directory rename.
6. State in README/docs that `papers/` requires a GitHub clone and is intentionally excluded from
   PyPI artifacts. Do not imply that installing the sdist provides the paper checkout.
7. Preserve every research script, CSV, LaTeX file, PDF, and tracked figure byte-for-byte except
   the KOSPI README path text that must change.
8. Keep paper-only dependencies and Bloomberg-derived research data outside the package runtime.
9. Add a repository test that rejects `paper_code/`, verifies the expected `papers/` entry points,
   and checks every repository-local Markdown image/link affected by the move.
10. Do not regenerate paper or example figures as part of this path-only stage.

Failure-first evidence:

- add the repository-path test while `paper_code/` still exists and observe failure;
- run the README local-link check before updating paths and observe the old references fail after
  the move; then update them and pass; and
- verify the artifact-content check rejects any accidental `papers/` inclusion in the wheel.

**Acceptance:**

- `papers/` is the only research/replication root and `paper_code/` is absent;
- `rg` finds no live `paper_code` references outside the migration history/report;
- before/after research-file manifests match after mapping the old root to the new root;
- README images and documented commands resolve at their new paths;
- the goal-based-allocation paper's integration checks pass with unchanged values;
- all package tests still pass; and
- candidate wheel/sdist contents follow the documented checkout-only paper contract.

**Verification:**

```powershell
if (Test-Path paper_code) { throw 'paper_code still exists' }
$hits = rg -n --hidden --glob '!.git/**' --glob '!agents/**' `
  --glob '!ROADMAP_DISCOVERABILITY_AND_ADOPTION.md' --glob '!CHANGELOG.md' `
  --glob '!tests/test_repository_layout.py' --glob '!scripts/check_dist_contents.py' `
  'paper_code' .
if ($LASTEXITCODE -eq 0) { throw "stale paper_code references:`n$hits" }
pytest -q
python papers/goal_based_allocation_2026/generate_paper_figures.py `
  --test --outdir <temporary-directory>
python -m build
python scripts/check_dist_contents.py dist/*
```

Run the KOSPI study's non-writing import/data smoke in its documented research environment and
compare its stored-data and stored-figure hashes. A path-only move does not authorise recalibration
or figure replacement.

**Out of scope:** changing paper text or results, reorganising either paper project internally,
renaming paper project slugs, adding paper dependencies, numerical changes, version bump, or
release.

## U2 — Establish one canonical package identity and trust surface

**Deliverable:** one small commit aligning repository-owned identity and release metadata.

Apply the canonical and boundary sentences above, with natural shortening where field limits
require it. Align:

- `[project].description`, keywords, classifiers, and project URLs in `pyproject.toml`;
- README title, opening, `When to use it`, installation, package structure, paper paths, and
  ecosystem boundary;
- documentation landing title and short title once U3a exists;
- `CITATION.cff`, README BibTeX, and a new `[Unreleased]` changelog section; and
- AGENTS.md layout, commands, src/papers contracts, and the actual four version surfaces.

Add `Documentation`, `Changelog`, and `Paper` project URLs only when their public targets exist or
will exist in the same approved release. Keep `Repository` and `Issues` canonical. Preserve the
distribution/import tokens and avoid unbounded claims such as “best”, “leading”, or
“production-ready”.

**Acceptance:** every primary surface describes the same specialised allocation workflow, makes
the model boundary clear, and distinguishes installed APIs from repository-only research.

**Verification:** inspect built wheel `METADATA` (`Name`, `Version`, `Summary`, `Requires-Python`,
`Project-URL`), build docs warning-free after U3a, run local-link tests, and verify version/citation
consistency.

**Out of scope:** public GitHub settings, publication-status changes, package rename, public API
changes, version bump, or release.

## Decision D1 — Approve the documentation stack

**Recommended decision:** approve a small Sphinx/MyST documentation-only toolchain and GitHub
Pages. Reuse the portfolio's established configuration where appropriate, but do not copy package
content or numerical utilities.

Approval covers:

- a `docs` optional extra containing Sphinx and MyST; choose any additional theme explicitly;
- `sphinx-sitemap` only if the selected static host does not provide a correct native sitemap;
- a `docs/` source tree and warning-free build gate;
- a GitHub Actions Pages deployment workflow; and
- the proposed canonical root `https://artursepp.github.io/GoalBasedAllocation/`.

**Gate evidence:** a short maintainer decision in the roadmap status log naming the approved host,
toolchain, dependency policy, and canonical URL.

If not approved, stop U3–U5 and retain the README as the canonical documentation surface; adapt U6
to be README-tested and record the reduced indexing contract.

## U3a — Build the local documentation foundation

**Deliverable:** a committed `docs/` source tree and docs CI check.

Minimum structure:

```text
docs/
    conf.py
    index.md
    getting-started.md
    conventions.md
    model-boundaries.md
    user-guide/
    validation.md
    papers.md
    api/
```

Requirements:

- install the candidate package before autodoc; never put the repository root on `sys.path` to
  bypass packaging;
- expose one landing page, first-success page, conventions page, validation page, paper page, and
  public API index;
- link to root examples and papers rather than copying their source;
- use `literalinclude` or a tested extraction for authoritative source examples;
- set canonical HTTPS URLs, repository/PyPI/issues/changelog links, and meaningful titles;
- keep generated HTML in `docs/_build/` and gitignore it; and
- configure nitpicky/warning handling so broken references fail CI without creating a broad
  docstring-rewrite project.

**Acceptance:** a clean environment builds HTML warning-free, navigation reaches every page, and
the API reference imports the installed src-layout package.

**Verification:**

```powershell
python -m pip install -e ".[docs]"
python -m sphinx -W --keep-going -b html docs docs/_build/html
```

Also run an internal-link check and assert no generated build output is tracked.

**Out of scope:** full docstring coverage, custom domain, content waves, release, or generated
marketing figures.

## U6 — Create one source of truth for first success

**Deliverable:** `examples/getting_started/quickstart.py` plus a getting-started page that includes
or mechanically checks it.

The script must:

1. use only top-level released public symbols;
2. create the paper asset or one balanced effective mandate without external data;
3. solve one MV-optimal policy or compute its survival result;
4. compare one analytical quantity with an exact buy-and-hold benchmark;
5. print compact deterministic values with conventions and no plots/files; and
6. finish in less than one minute on a normal laptop.

Document the first parameters to change: horizon, target return, rate, consumption/floor growth,
and mandate weights. State that paper-calibrated inputs are illustrative and not investment
advice.

Failure-first evidence: run the source script against the current released artifact in a clean
temporary environment and record any missing-symbol or drift failure before accepting the
candidate. The final CI check copies only the authoritative script to a temporary directory,
installs the built wheel, and runs it with the checkout absent from `sys.path`.

**Acceptance:** a new user can reproduce the promised result from the released wheel and docs;
source and docs cannot silently drift; no network, credential, input data, display, or output file
is required.

**Verification:** run the script from wheel and sdist installations on the supported CI matrix,
assert deterministic output within documented numerical tolerances, and verify import provenance.

**Out of scope:** notebook dependencies, plotting, long Monte Carlo, paper figure generation, or a
second quickstart implementation.

## U4a — Publish the three core task guides

**Deliverable:** three focused user-guide pages:

1. **MV-optimal policy and glide path** — `find_ell`, `gap_process_asset`, policy inputs, Riccati
   outputs, target wealth, allocation intensity, and interpretation;
2. **Absorbing floor and terminal-wealth distribution** — `compute_survival`, `compute_density`,
   `compute_tilted_survival`, `compute_overshoot_density`, floor atom, overshoot mass, and moment
   reconciliation; and
3. **Mandates and opportunity sets** — `build_effective_asset`, `portfolio_sigma_unc`,
   `portfolio_eta_quadrature`, `AdvisorSpec`, `compute_opportunity_point`,
   `build_opportunity_set`, and `bh_moments_rsjd`.

Each page states the practitioner/research problem, model scope, exact inputs/outputs, units and
conventions, a minimal public-API example linked to authoritative source, expected result,
interpretation, failure modes, numerical cost, and important non-goals. Explain which values are
paper specifications versus user assumptions.

**Acceptance:** all three priority tasks have executable entry points; every symbol exists; every
reported value is reproduced; and numerical claims have an independent Monte Carlo or exact
matrix-exponential check where applicable.

**Verification:** warning-free docs build, internal link check, public-name coverage check, and
execution of every snippet/source example under its claimed environment.

**Out of scope:** new calibration algorithms, constraints, more regimes, transaction costs,
path-dependent payoffs, or copying `optimalportfolios` functionality.

## U4b — Publish conventions, validation, options, and paper guides

**Deliverable:** supporting pages that prevent misuse of the core guides:

- conventions for rates, wealth units, return/volatility annualisation, regime numbering,
  transition intensities, and mean/rate jump parameters;
- analytical-versus-Monte-Carlo validation, tolerances, seeds, and why Monte Carlo is not the
  implementation;
- vanilla option pricing with `RiskNeutralParams`, `Regime`, `OptionType`, `price_vanilla`, and
  `implied_vol`, clearly labelled as a secondary workflow; and
- paper reproduction with the new `papers/` paths, checkout-only dependencies, expected runtime,
  tracked-output policy, and the distinction between the main allocation paper and the KOSPI
  calibration study.

Link the KOSPI study's `bbg-fetch` provenance without making `bbg-fetch` or pandas runtime
dependencies. Do not present stored market data as a general bundled data service.

**Acceptance:** conventions are stated once and linked everywhere; reference calculations pass;
paper commands resolve from a clean clone; and no docs page implies that `papers/` ships in the
wheel/sdist.

**Verification:** warning-free docs build, local/deployed link checks, full pytest suite, paper
integration test, and current-source checks for every named public symbol.

**Out of scope:** new option payoffs, new market-data fetching, KOSPI recalibration, paper edits,
or generated figures.

## U5 — Publish a neutral comparison and choice guide

**Deliverable:** one dated page comparing workflows, not popularity.

Recommended alternatives to verify against current primary sources at execution:

- `optimalportfolios` for discrete rolling multi-asset construction, constraints, transaction
  costs, and backtesting;
- PyPortfolioOpt for accessible static portfolio optimisation workflows;
- Riskfolio-Lib for broad risk-measure and portfolio-model coverage; and
- CVXPortfolio or skfolio for their documented multi-period or estimator-oriented workflow.

Compare model assumptions, continuous/dynamic versus discrete optimisation, wealth-floor/barrier
handling, regime-switching jumps, terminal-distribution analytics, constraints/costs, backtesting,
data requirements, and intended user. Include at least one use case favouring each alternative.

**Acceptance:** every nontrivial claim resolves to a current official source; versions and access
dates are recorded; unknowns are explicit; no package is presented as universally superior.

**Verification:** warning-free docs build, internal link check, and manual primary-source citation
audit.

**Out of scope:** performance benchmarks designed to favour this package or adding alternatives as
dependencies.

## Maintainer gate A — Publish and align external identity/indexing

Credentialed actions after U3a/U4/U5 are locally complete:

1. Enable the approved GitHub Pages workflow and canonical documentation URL.
2. Update GitHub About to the canonical sentence and set the website to the docs root; keep the
   SSRN paper linked from README/docs and add PyPI where GitHub permits.
3. Verify the documentation property in Google Search Console using a persistent public method.
4. Submit or confirm the canonical sitemap and inspect landing, getting-started, core task, API,
   and paper pages.
5. Record normal crawl latency as pending rather than as a technical failure.
6. Export or summarise the query baseline needed by U9 without committing private exports.

**Gate evidence:** a redacted local summary containing approval, property type, verification date,
sitemap status, GitHub settings, deployment URL, and priority-page statuses.

## U3b — Audit deployed technical discoverability

**Deliverable:** `agents/DISCOVERABILITY_AUDIT.md` and only the minimal remediation commit if the
public deployment exposes a defect.

Check the deployed root, getting-started page, three core task pages, API index, paper page,
robots policy, and sitemap:

- expected HTTP status and no redirect loops;
- one canonical HTTPS URL per page;
- no accidental `noindex`, `nosnippet`, or robots exclusion;
- server-rendered titles, descriptions, headings, and primary navigation;
- internal navigation from the landing page to every priority page;
- repository, PyPI, issues, changelog, paper, and docs links in both directions; and
- valid sitemap entries with no build paths or duplicate aliases.

**Acceptance:** priority pages are reachable, indexable, canonical, and inspectable in Search
Console.

**Verification:** warning-free local docs build followed by scripted deployed HTTP/canonical/
robots/sitemap checks after Pages finishes deploying.

**Out of scope:** paid SEO tools, custom-domain migration, AI crawler files, or content expansion.

## Maintainer gate B — Decide on a hosted notebook

**Default recommendation: defer.** The wheel-first quickstart is deterministic, lightweight, and
more maintainable than a second interface. Approve a notebook only if user evidence shows that a
hosted numerical trial materially reduces adoption friction.

Approval means one thin, output-free Colab entry point using the released package and the U6
workflow. It does not mean adding Jupyter to package dependencies or reproducing paper figures.

## U7 — Optional thin Colab entry point

**Deliverable:** only if Gate B approves: one output-free notebook, an Open in Colab link, and a
mechanical drift check against U6.

The notebook installs the released version, prints it, executes the same first-success workflow,
states runtime/conventions, and links to versioned docs and the authoritative source. It does not
install an unpublished checkout or embed large results.

**Acceptance:** a clean hosted runtime completes `Run all` and the drift check passes on supported
CI systems.

**Out of scope:** notebook galleries, Binder, long simulations, Jupyter extras, or stored output.

## U8 — Release, deploy, and align trust surfaces

**Deliverable:** an explicitly approved release because the src-layout build, wheel publication,
PyPI README, and metadata cannot reach users otherwise.

Before release:

- choose the next version explicitly; do not infer it from this roadmap;
- align `pyproject.toml`, `src/goal_based_allocation/__init__.py`, `CITATION.cff`, README BibTeX,
  and CHANGELOG;
- run the full supported Python matrix and independent numerical cross-checks;
- run the main paper integration tests from `papers/` without regenerating committed figures;
- build wheel and sdist, inspect contents/metadata, run `twine check`, install each artifact in a
  clean environment outside the checkout, run U6, and execute `pip check`;
- confirm the wheel contains one src-layout package and no repository-only papers/examples/tests;
- confirm the sdist contents match the documented contract;
- confirm docs/examples describe the version and final paths being published; and
- obtain explicit maintainer approval before upload/tag/release.

After release:

- verify PyPI exposes both wheel and sdist and renders the current README/project links;
- verify the Git tag and GitHub Release match the package version;
- verify GitHub README images and every `papers/` command/path;
- verify Pages deployed the intended commit and repeat U3b checks; and
- record commit, tag, package URL, release URL, artifact hashes, docs deployment, and results.

**Acceptance:** all public surfaces expose the same version, identity, paths, citation, and docs;
both artifacts install successfully; the released wheel produces the U6 result; and no scientific
or numerical result changed.

**Verification:**

```powershell
pytest -q
python papers/goal_based_allocation_2026/generate_paper_figures.py `
  --test --outdir <temporary-directory>
python -m sphinx -W --keep-going -b html docs docs/_build/html
python -m build
python -m twine check dist/*
python scripts/check_dist_contents.py dist/*
```

Then perform clean wheel/sdist installs, U6 execution, direct PyPI/GitHub/Pages inspection, and the
repository release checklist. A successful upload alone is not completion.

**Out of scope:** unrelated features, new algorithms/dependencies, changing paper results,
recalibration, or publication without explicit approval.

## U9 — Measure at approximately 30, 60, and 90 days

**Deliverable:** `agents/DISCOVERABILITY_90_DAY_REPORT.md`, updated at each checkpoint.

Fix definitions before the first checkpoint:

- Search Console uses the latest 28 complete days ending at least two days before observation;
- branded means the three exact package/repository tokens defined in U1, case-insensitive;
- non-branded queries are the three fixed U1 task queries;
- priority pages are fixed at U1/Gate A;
- package downloads use one trailing-period source and are not treated as unique users;
- GitHub stars, forks, dependents, watchers, issues, and citations are secondary proxies;
- first-success referrals and task-page entrances are used where privacy-respecting analytics
  exist; and
- missing, delayed, or suppressed values remain missing.

Compare index coverage, query impressions/clicks/CTR, entry pages, docs-to-PyPI/GitHub paths,
downloads, repository signals, citations, qualified issues, and exact-name search treatment. State
that a new Search Console property has no historical pre-registration baseline and do not infer
causality from one metric.

At 90 days recommend one action: deepen a performing task page, repair a demonstrated conversion
failure, improve external distribution, or stop investing in a channel that produced no qualified
use.

**Acceptance:** every checkpoint uses fixed definitions, missing data is explicit, and the final
recommendation follows from multiple observations.

**Scheduling:** after U8, create task-attached 30/60/90-day follow-ups in the maintainer's Europe/
Zurich timezone. Scheduling sets up U9; it does not complete it.

## Status log

Append one line for every completed, skipped, or blocked stage:

```text
YYYY-MM-DD · stage · branch/commit · PASS|PASS-LOCAL|SKIPPED|BLOCKED · concise verification result
```

Use `PASS-LOCAL` only before a required deployment or credentialed check. Replace it with `PASS`
after public verification rather than leaving stale follow-up text.

Initial entry:

```text
2026-08-19 · roadmap · working tree · DRAFTED · package adaptation written; execution not started
2026-08-19 · U0 · working tree · PASS · profile recorded; agents/ is ignored
2026-08-19 · U1 · working tree · PASS · dated public/artifact/conversion baseline recorded
2026-08-19 · M1 · working tree · PASS · byte-identical src move; 48 tests; wheel/sdist validated
2026-08-19 · M2 · working tree · PASS · papers move preserved assets; 50 tests; 9 paper checks
2026-08-19 · U2 · working tree · PASS-LOCAL · identity, SPDX metadata, paths, and citations align
2026-08-19 · D1 · working tree · PASS · Sphinx/MyST/sitemap and gated GitHub Pages approved
2026-08-19 · U3a · working tree · PASS-LOCAL · 12-page docs site builds warning-free
2026-08-19 · U6 · working tree · PASS-LOCAL · wheel quickstart is deterministic and output-free
2026-08-19 · U4a · working tree · PASS-LOCAL · three core allocation task guides complete
2026-08-19 · U4b · working tree · PASS-LOCAL · conventions, validation, options, papers complete
2026-08-19 · U5 · working tree · PASS-LOCAL · dated primary-source comparison guide complete
2026-08-19 · Gate B · working tree · SKIPPED · hosted notebook deferred by default decision
2026-08-19 · U7 · working tree · SKIPPED · no notebook because Gate B deferred it
```

## Definition of complete

The implementation is complete when U0–U8 selected above have passed, both mandatory migrations
are verified, wheel and sdist are public and tested, the canonical documentation deployment and
credentialed gates are recorded, and U9 checkpoints are scheduled.

The roadmap itself is complete only after the final U9 observation and evidence-based
recommendation. Time-based measurement cannot be completed early. The SSRN paper, journal status,
scientific claims, and any future publication roadmap remain separate.
