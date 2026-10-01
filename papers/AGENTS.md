# Paper workspace contract

Root AGENTS.md controls the external Python environment and numerical invariants.

## Layout

Keep each workspace at `papers/<paper_id>/`, outside the installable package.

| Section | Purpose | Git policy |
|---|---|---|
| `paper/` | Current manuscript, reading PDF, dependencies and `figures/` | Exact approved files only |
| `drafts/<version>/` | Earlier manuscripts with their own figures | Always ignored |
| `presentations/<event>/` | Slides with their own figures | Ignored unless individually approved |
| `private/` | Editor correspondence, referee reports, replies and permissions | Always ignored |
| `replication/` | Research code, static `data/`, automated `tests/` | Code and approved static inputs tracked |
| `agents/` | Paper plans, reports and execution records | Always ignored |

Do not track empty placeholders. Paper-specific records belong in each paper's
`agents/`; this explicitly overrides the shared core's root-only record rule.
Repository-wide records remain in root `agents/`.

## Publication and preservation

- Preserve the existing tracked assets in `goal_based_allocation_2026` and
  `kospi_volatility_fit_jun2026`. The paper index describes their scope.
  All other workspaces are ignored until an explicit publication decision.
- Use exact file exceptions for manuscripts, figures, presentations and static
  inputs. Existing inclusion does not approve new versions, vendor snapshots or
  datasets, or establish redistribution rights. SSRN posting is not GitHub permission.
- Keep restricted new inputs in ignored `replication/data/local/`; retain
  confidential permission records in `private/`. Never force-add private content.
- Preserve manuscript text, PDFs, images, static input bytes and numerical logic
  during layout changes. Record input SHA-256 hashes. Do not refresh numerical
  baselines as part of reorganizing folders.
- Retain local files when untracking. Git ignores do not erase prior history;
  ignored material needs private backup outside Git.
- The KOSPI README is the current study narrative; its existing displayed figures
  live under `paper/figures/`. Do not invent a manuscript or a new permission grant.

## Replication

Keep research code in `replication/`, inputs in `replication/data/`, and paper-specific
automated tests in `replication/tests/test_*.py`. Existing package tests stay in root
`tests/`. Keep the main script's numerical integration routine with its producer.
Core package imports must not depend on papers or their research dependencies.

Before Python or checks on this host, dot-source the shared
`ArturSepp/scripts/repo_governance/Enter-AgentRepo.ps1` with this repository's
`-RepoPath` and the stack `-RepositoriesRoot`. Use the external interpreter named
by root AGENTS.md. Never create an environment under OneDrive.

Run documented scripts by their file paths with the package installed. Their
sibling research imports are intentionally not part of the package API.
`GBA_PAPER_OUTPUT_PATH` optionally selects an absolute external parent directory;
each paper gets its own subdirectory. Otherwise output uses the configured local
runtime. The main CLI's `--outdir` selects an exact external figure directory,
including for `--test`. Output must stay outside the checkout and OneDrive.
For LaTeX, copy approved inputs to an external build directory before compiling.
Promote reviewed results into publication assets explicitly.

## Verification

Run `.github/scripts/check_paper_policy.py --worktree` to preview the working tree
and without options to validate the actual index and indexed ignore rules.
Run `.github/scripts/paper_policy_test.py` and the paper tests before committing.
CI checks publication boundaries and paper output routing. Source archives and
wheels must exclude paper workspaces and agent records; use the existing
`scripts/check_dist_contents.py` on both artifacts.

Re-run root numerical verification when analytical logic changes. Layout-only
work must verify moved hashes, imports, output paths and public links; it does
not imply full numerical regeneration or figure visual approval.
