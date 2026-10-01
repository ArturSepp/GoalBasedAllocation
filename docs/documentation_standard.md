---
myst:
  html_meta:
    description: >-
      Authoring rules for the goal-based-allocation handbook: chapter structure, the seven-row
      convention card, reserved notation, concise proofs, executed worked examples, the single
      bibliography, manuscript references, figure provenance and verification.
---

# Documentation standard

*Author: [Artur Sepp](https://github.com/ArturSepp)*

This standard applies to human-authored documentation for
[GoalBasedAllocation](https://github.com/ArturSepp/GoalBasedAllocation).
Software citation: [CITATION.cff](https://github.com/ArturSepp/GoalBasedAllocation/blob/main/CITATION.cff).

This is the GoalBasedAllocation supplement to the
[shared OSS documentation standard](https://github.com/ArturSepp/ArturSepp/blob/main/docs/documentation_standard.md).
The shared guide owns the common authoring rules; this page records the package-specific handbook
conventions, their tests, the figure policy and the verification commands. It follows the
[qis supplement](https://github.com/ArturSepp/QuantInvestStrats/blob/main/docs/documentation_standard.md)
and the [optimalportfolios supplement](https://github.com/ArturSepp/OptimalPortfolios/blob/main/docs/documentation_standard.md),
so that the three handbooks read alike.

## Article structure

Use the shared [article structure](https://github.com/ArturSepp/ArturSepp/blob/main/docs/documentation_standard.md#user-content-article-structure),
with the H2 `Implementation in goal_based_allocation`. A methodology chapter has, in order: Overview;
Inputs, notation, and assumptions; Methodology; Worked example; Implementation in
goal_based_allocation; Interpretation and limitations; See also; References. Utility pages, such as the
home page, the quickstart, the conventions, the bibliography and this standard, use the shared shorter
form. `tests/test_docs.py` lists the methodology chapters and enforces the headings, the MyST
description, the byline and the software links.

The eleven chapters form the *goal-based allocation handbook*, in five parts: the model; dynamic
mean-variance allocation; the Laplace transform framework; terminal wealth and investor profiles; and
derivatives under the same model. A new chapter enters the toctree of its part on the
[home page](index.md), the `CHAPTERS` list of `tests/test_documentation_examples.py` and the
methodology list of `tests/test_docs.py`.

## Handbook conventions

- **Convention card.** The first table under `Inputs, notation, and assumptions` has the header
  `| Convention | This article |` and exactly these rows, in order: Time and rates, Regimes, Jump
  sizes, Wealth coordinate, Measure, Numerical method, Package default. The
  [conventions page](conventions.md#the-convention-card) defines each row.
- **Reserved notation.** Symbols reserved on the [conventions page](conventions.md#reserved-notation)
  keep one meaning in every chapter. A chapter declares any other symbol in its notation table and
  never reuses a reserved one. Regimes are 1 and 2 in formulas and 0 and 1 in code.
- **Results and concise proofs.** State a result in a paragraph opening with a bold
  **Definition.**, **Identity.**, **Proposition.**, **Lemma.** or **Theorem.** label, and follow each
  identity or proposition with a short **Proof.** paragraph ending in $\square$. A result taken from
  the manuscript names its number, for example Theorem 5.3; cite a long derivation instead of
  reproducing it.
- **The manuscript.** Statement, equation, table and figure numbers refer to the manuscript of
  Sepp (2026) tracked in `papers/goal_based_allocation_2026/paper/`, dated 7 April 2026. The paper is
  public as an SSRN working paper; cite it with its DOI. Where the implementation or a derivation on
  the page differs from the manuscript, say so neutrally, give the evidence on the page, and keep the
  package's numbers unchanged.
- **Insight and Pitfall callouts.** Write `> **Insight.** ...` or `> **Pitfall.** ...` as an ordinary
  blockquote. `docs/_ext/gba_callouts.py`, adapted from qis, renders them as admonitions in Sphinx;
  other viewers show a quotation. A Pitfall states a behaviour of the code that a reader can get
  wrong, with the number that shows it.
- **Executed worked examples.** Every `python` block of a methodology chapter runs, offline and in
  page order in one namespace, in `tests/test_documentation_examples.py`, and asserts every number the
  chapter quotes against an independent computation: a closed form, a second numerical method, an
  exact simulation or a value of the manuscript at its printed precision. Put the comment
  `<!-- docs-test: skip -->` on the line before a schematic block that cannot run. Chapters whose
  examples calibrate mandates or run simulations of more than a few seconds are marked `slow`.
- **One bibliography.** [The bibliography](bibliography.md) holds every cited work once, in one style.
  A chapter's References section is a numbered list whose items begin with a bibliography entry
  verbatim, optionally followed by a note, and it includes the software citation. `tests/test_docs.py`
  enforces the match.
- **Coverage.** Every public export of `goal_based_allocation` appears exactly once on the
  [API page](api/index.md) and is named, in inline code, by at least one methodology chapter;
  `tests/test_docs.py` enforces both.
- **Spelling.** Prose uses British spelling; Python names keep their published spelling.

## Copyable methodology template

Copy the [shared methodology template](https://github.com/ArturSepp/ArturSepp/blob/main/docs/documentation_standard.md#user-content-copyable-methodology-template),
replace `PACKAGE` with `goal_based_allocation` and `REPOSITORY` with `GoalBasedAllocation`, add the
convention card, and supply the topic. Keep the MyST description front matter. Follow the shared
[authorship and date rules](https://github.com/ArturSepp/ArturSepp/blob/main/docs/documentation_standard.md#user-content-authorship-and-dates);
a new uncommitted page uses the linked Artur Sepp byline without a date.

## Portable mathematics

Follow the shared [portable mathematics rules](https://github.com/ArturSepp/ArturSepp/blob/main/docs/documentation_standard.md#user-content-portable-mathematics).
MyST's `dollarmath` extension is enabled. The rendering faults that the optimalportfolios supplement
records for GitHub apply here too: spell every command with letters, write `\lvert` and `\rvert` for
absolute values, `\lt` and `\gt` for inequalities in inline math, never start a line of a display block
with `+`, `-`, `*`, `>` or `#`, and attach a subscript to a letter rather than to a closing brace, for
example `\omega_a^{[i]}` rather than `\omega^{[i]}_a`.

For example, the target of the mean-variance policy is the same in both regimes:

$$
\Pi^{*}(t) = \frac{\ell}{2}e^{-r_c(T - t)} .
$$

## References and source ownership

Apply the shared [reference and example rules](https://github.com/ArturSepp/ArturSepp/blob/main/docs/documentation_standard.md#user-content-references-and-executable-examples).
The public API is the set of names exported by `src/goal_based_allocation/__init__.py`. Objects the
chapters use from modules, such as `laplace_inversion.laplace_invert_abate_whitt` or
`riccati_solver.solve_riccati`, are labelled as module-level on the page and listed in their own group
of the API page. Internal helpers with a leading underscore are named only in contract details.

The package is standalone: it depends on NumPy, SciPy and Matplotlib only. A chapter may cite related
stack packages for neighbouring workflows, but no example imports them.

## Figures

The repository does not commit newly generated figures without an explicit publication decision. The
handbook therefore draws no figures of its own: its numbers live in the executed examples. A chapter
may display an existing approved figure of the manuscript from
`papers/goal_based_allocation_2026/paper/figures/`, by a relative path, when:

- the caption states the question, the numbers the reader should take away, and the manuscript figure
  number;
- the caption names the producer, `papers/goal_based_allocation_2026/replication/generate_paper_figures.py --figure N`,
  and says when the figure uses an approximation that the package does not, as the buy-and-hold
  curves of Figure 2 do;
- the alt text describes the comparison, and a link opens the full-resolution file.

The figures are paper artefacts; their bytes are preserved by the paper policy in
[`papers/AGENTS.md`](https://github.com/ArturSepp/GoalBasedAllocation/blob/main/papers/AGENTS.md), and the
handbook neither regenerates nor replaces them.

## Verification

The API reference is generated at build time: `docs/conf.py` adds the autodoc, autosummary and napoleon
extensions, autosummary writes one stub page per object into `docs/api/generated/` (git-ignored), and a
docstring hook renders the plain-text formulas of package docstrings as literal blocks so that a strict
build accepts them. After the repository's mandatory environment setup, use its external interpreter:

```console
python -m pytest tests/test_docs.py tests/test_documentation_examples.py
python -m pytest tests/test_documentation_examples.py -m "not slow"
python .github/oss_checks.py docs --output-dir <C-local directory>
```

Build from a C-local export, as the shared guide requires: autosummary writes generated sources beside
the inputs, so redirecting only the HTML output is insufficient for a OneDrive checkout. Without
`--working-tree` the shared profile exports the committed revision and builds the export. The CI
workflow passes `--working-tree`, which builds in place; on a OneDrive checkout that writes
`docs/api/generated/` into the checkout, so for uncommitted work copy the working tree to C-local
storage and run `python -m sphinx -E -W --keep-going -b html docs <output>` in the copy. Inspect rendered equations and the displayed figures after the build; source checks do not
certify rendering. The pages of the old `user-guide/` folder are redirect stubs that keep their
published addresses.

## See also

- [Documentation home](index.md)
- [Notation and conventions](conventions.md)
- [Bibliography](bibliography.md)
- [Validation and numerical evidence](validation.md)
- [Contributor guidance](https://github.com/ArturSepp/GoalBasedAllocation/blob/main/AGENTS.md)

## References

- [GoalBasedAllocation software citation](https://github.com/ArturSepp/GoalBasedAllocation/blob/main/CITATION.cff).
- [Shared OSS documentation standard](https://github.com/ArturSepp/ArturSepp/blob/main/docs/documentation_standard.md).
- [MyST: math and equations](https://myst-parser.readthedocs.io/en/latest/syntax/math.html).
- [GitHub: writing mathematical expressions](https://docs.github.com/en/get-started/writing-on-github/working-with-advanced-formatting/writing-mathematical-expressions).
