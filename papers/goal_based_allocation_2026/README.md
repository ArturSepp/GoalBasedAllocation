# Goal-based allocation paper

*Author: [Artur Sepp](https://github.com/ArturSepp)*

Project: [GoalBasedAllocation](https://github.com/ArturSepp/GoalBasedAllocation).
Software citation: [CITATION.cff](https://github.com/ArturSepp/GoalBasedAllocation/blob/main/CITATION.cff).

The current manuscript, PDF and six approved figures are under `paper/`.
The unchanged LaTeX uses `figures/` relative to that directory. Earlier drafts,
slides, private correspondence and working records use the standard local sections.

## Replication

From the repository root with the prescribed environment configured:

```console
python papers/goal_based_allocation_2026/replication/generate_paper_figures.py --figure 10
python papers/goal_based_allocation_2026/replication/generate_paper_figures.py --test
python -m pytest papers/goal_based_allocation_2026/replication/tests -q
```

The generator uses synthetic inputs defined in the source. `--test` runs the
existing integration assertions and then generates all ten figures, so it can
be expensive. `--outdir` now applies to test mode too; it must be an absolute
directory outside the checkout and OneDrive. Without it, output goes to this
paper's directory under the configured external local runtime.

New output does not replace the approved figure bundle. Static data metadata is
in `replication/data/`. Build LaTeX from an external copy of `paper/`; this layout
change does not rebuild or alter the reading PDF.
