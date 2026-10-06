"""Sphinx configuration for the goal-based-allocation documentation.

As in the qis and optimalportfolios documentation, the methodology chapters form one handbook and
the API reference is generated at build time rather than maintained by hand. ``api/index.md`` groups
the public exports of ``goal_based_allocation`` by capability, each group linked to the chapters
that derive its formulas; autosummary writes one stub page per object into ``api/generated/``,
which is git-ignored, so the reference cannot drift from the docstrings.

``tests/test_docs.py`` checks that every public export appears on the API page exactly once and is
named by a handbook chapter, and ``tests/test_documentation_examples.py`` executes the worked
examples of every chapter.
"""

import os
import sys
import tomllib
from pathlib import Path

DOCS_DIR = Path(__file__).resolve().parent
REPOSITORY_ROOT = DOCS_DIR.parent
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))
sys.path.insert(0, str(DOCS_DIR / "_ext"))

project = "goal-based-allocation"
author = "Artur Sepp"
copyright = "2026, Artur Sepp"
release = tomllib.loads(
    (REPOSITORY_ROOT / "pyproject.toml").read_text(encoding="utf-8")
)["project"]["version"]
version = release

extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx_sitemap",
    "gba_callouts",
]

# Dollar math keeps chapter source portable across MyST, GitHub and VS Code.
myst_enable_extensions = ["colon_fence", "deflist", "dollarmath"]
# Resolve ordinary Markdown section links with GitHub-compatible heading fragments.
myst_heading_anchors = 3
myst_html_meta = {
    "google-site-verification": "cddUZk3Gsd1MySw42Rwuq_rMzUDcMNkJWekObx-QS9Y",
}
source_suffix = {".rst": "restructuredtext", ".md": "markdown"}
templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# The package docstrings use the numpydoc layout; signatures stay readable and types render in
# the body. Stub pages are written to api/generated/ on every build.
autosummary_generate = True
autodoc_typehints = "description"
autodoc_member_order = "bysource"
napoleon_numpy_docstring = True
napoleon_google_docstring = True
napoleon_use_ivar = True

# SSRN serves the live paper pages to browsers but rejects automated link-check requests. DOI
# redirects commonly end at publisher sites that answer HTTP 403 to automated clients although the
# DOI is registered; the bibliography records were checked against Crossref instead.
linkcheck_ignore = [
    r"https://papers\.ssrn\.com/.*",
    r"https://(?:www\.)?ssrn\.com/.*",
    r"https://doi\.org/.*",
    r"https://www\.risk\.net/.*",
]

html_theme = "furo"
html_title = "goal-based-allocation - goal-based allocation under regime-switching jump-diffusions"
html_short_title = "GoalBasedAllocation"
html_baseurl = os.environ.get(
    "READTHEDOCS_CANONICAL_URL",
    "https://goalbasedallocation.readthedocs.io/en/latest/",
)
html_extra_path = ["robots.txt", "googleccb1e876a2b4bf72.html"]
html_static_path = ["_static"]
html_css_files = ["gba.css"]
html_theme_options = {
    "source_repository": "https://github.com/ArturSepp/GoalBasedAllocation/",
    "source_branch": "main",
    "source_directory": "docs/",
}

sitemap_url_scheme = "{link}"
# The search page is marked noindex, the general index only lists links to other pages, and the
# user-guide pages redirect to the handbook chapters that replaced them.
sitemap_excludes = ["search.html", "genindex.html", "user-guide/*"]

# The former user-guide addresses stay on the site so that existing links keep working, but they
# only redirect to their chapters, so search engines are asked to follow them and not index them.
NOINDEX_PREFIXES = ("user-guide/",)
ROBOTS_NOINDEX = '<meta name="robots" content="noindex, follow">\n'

# Packages whose versions the footer names, with their usual spelling.
BUILD_PACKAGES = {
    "numpy": "NumPy",
    "scipy": "SciPy",
    "matplotlib": "Matplotlib",
}


def _build_versions() -> str:
    """Name the package versions of this build for the footer of every page.

    Chapters state no hand-written version stamps, which go stale between releases; the build reads
    the versions of its own environment instead. The package itself is named by the source version
    in ``pyproject.toml``, because the pages document the checkout being built.

    Returns:
        One sentence naming the source version and the installed numerical dependencies.
    """
    from importlib.metadata import PackageNotFoundError, version as installed_version

    names = []
    for distribution, label in BUILD_PACKAGES.items():
        try:
            names.append(f"{label} {installed_version(distribution)}")
        except PackageNotFoundError:
            continue
    built_with = ", ".join(names[:-1]) + f" and {names[-1]}" if len(names) > 1 else "".join(names)
    return f"Built from goal-based-allocation {release}" + (f" with {built_with}." if names else ".")


# _templates/page.html adds the build versions to the footer of every page.
html_context = {"build_versions": _build_versions()}


def _indent(line: str) -> int:
    """Return the number of leading spaces of a docstring line."""
    return len(line) - len(line.lstrip())


def _literal_formula_blocks(app, what, name, obj, options, lines) -> None:
    """Render the plain-text formulas of a package docstring as literal blocks.

    Several docstrings write formulas as indented plain-text lines under an introducing sentence,
    for example ``Paper eq (nu_eff):`` followed by ``sigma_eff = |omega*_a| * sigma``. Read as
    reStructuredText, those lines become malformed definition lists and ``|...|`` becomes an
    undefined substitution, which a strict build rejects. This hook turns every such indented run
    in the description part of the docstring, before the first napoleon field, into a literal
    block, so the formula keeps its layout and the source modules stay unchanged.

    Args:
        app: Sphinx application; unused.
        what: Kind of object being documented; unused.
        name: Fully qualified object name.
        obj: The documented object; unused.
        options: Autodoc directive options; unused.
        lines: Docstring lines after napoleon processing, edited in place.
    """
    if not name.startswith("goal_based_allocation"):
        return
    result = []
    literal_indent = None
    for index, line in enumerate(lines):
        if line.startswith(":") or line.startswith(".. "):
            result.extend(lines[index:])
            break
        if literal_indent is not None:
            if not line.strip() or _indent(line) > literal_indent:
                result.append(line)
                continue
            literal_indent = None
            if result[-1].strip():
                result.append("")
        following = lines[index + 1] if index + 1 < len(lines) else ""
        if line.strip() and following.strip() and _indent(following) > _indent(line):
            text = line.rstrip().replace("|", r"\|")
            result.extend([text if text.endswith("::") else text + " ::", ""])
            literal_indent = _indent(line)
            continue
        result.append(line.replace("|", r"\|"))
    lines[:] = result


def _use_root_canonical(app, pagename, templatename, context, doctree) -> None:
    """Use the HTTPS site root, rather than index.html, as the landing canonical."""
    if pagename == "index":
        context["pageurl"] = app.config.html_baseurl


def _keep_redirect_stubs_out_of_the_index(app, pagename, templatename, context, doctree) -> None:
    """Append a ``noindex, follow`` robots tag to the redirect stubs of the former user guide.

    Appending to ``metatags`` keeps the description and refresh tags from the page's front matter.
    """
    if pagename.startswith(NOINDEX_PREFIXES):
        context["metatags"] = (context.get("metatags") or "") + ROBOTS_NOINDEX


def setup(app) -> None:
    """Register documentation build hooks."""
    app.connect("autodoc-process-docstring", _literal_formula_blocks)
    app.connect("html-page-context", _use_root_canonical)
    app.connect("html-page-context", _keep_redirect_stubs_out_of_the_index)
