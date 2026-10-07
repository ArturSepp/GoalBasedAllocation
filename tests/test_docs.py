"""Documentation structure, bibliography and public-name coverage tests.

The handbook conventions are recorded in ``docs/documentation_standard.md``; this module enforces
the mechanical ones: page metadata and attribution, the eight sections and the convention card of a
methodology chapter, the callout labels, the single bibliography, the API page and the coverage of
every public export by a chapter. A small Sphinx build checks the page titles, robots tags and
sitemap that the site configuration produces.
"""

import importlib
import inspect
import re
import runpy
import subprocess
import sys
from html.parser import HTMLParser
from pathlib import Path

import pytest

import goal_based_allocation

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DOCS = REPOSITORY_ROOT / "docs"

METHODOLOGY_CHAPTERS = (
    "regime_switching_model.md",
    "buy_and_hold_moments.md",
    "mv_optimal_policy.md",
    "wealth_floor_gap_process.md",
    "laplace_inversion.md",
    "laplace_barrier_framework.md",
    "terminal_wealth_distribution.md",
    "mandate_aggregation.md",
    "investment_opportunity_set.md",
    "european_options.md",
    "variance_swaps.md",
)
UTILITY_PAGES = (
    "index.md",
    "getting-started.md",
    "conventions.md",
    "model-boundaries.md",
    "validation.md",
    "papers.md",
    "comparison.md",
    "bibliography.md",
    "documentation_standard.md",
    "api/index.md",
)
# published addresses of the former user guide, kept as redirects to their chapters
LEGACY_REDIRECTS = {
    "user-guide/mv-optimal-policy.md": "mv_optimal_policy",
    "user-guide/terminal-wealth-floor.md": "terminal_wealth_distribution",
    "user-guide/mandates-opportunity-set.md": "investment_opportunity_set",
    "user-guide/option-pricing.md": "european_options",
}
SECTIONS = (
    "Overview",
    "Inputs, notation, and assumptions",
    "Methodology",
    "Worked example",
    "Implementation in goal_based_allocation",
    "Interpretation and limitations",
    "See also",
    "References",
)
CARD_ROWS = (
    "Time and rates",
    "Regimes",
    "Jump sizes",
    "Wealth coordinate",
    "Measure",
    "Numerical method",
    "Package default",
)
BYLINE = "*Author: [Artur Sepp](https://github.com/ArturSepp)"
CITATION = ("Software citation: "
            "[CITATION.cff](https://github.com/ArturSepp/GoalBasedAllocation/blob/main/CITATION.cff).")
PENDING_MARK = " [pending publisher check]"
SOFTWARE_ENTRY = (
    "Sepp, A. goal-based-allocation: Semi-analytical dynamic mean-variance allocation and "
    "terminal-wealth risk under regime-switching jump-diffusions. [Software citation metadata]"
    "(https://github.com/ArturSepp/GoalBasedAllocation/blob/main/CITATION.cff)."
)


def read(relative: str) -> str:
    """Return a documentation source."""
    return (DOCS / relative).read_text(encoding="utf-8")


def without_code(text: str) -> str:
    """Remove fenced code blocks, keeping line structure."""
    return re.sub(r"```.*?```", lambda match: "\n" * match.group(0).count("\n"), text, flags=re.DOTALL)


def normalise(text: str) -> str:
    """Collapse whitespace so that line wrapping does not change an entry."""
    return re.sub(r"\s+", " ", text).strip()


def list_items(text: str, marker: str) -> list:
    """Items of the Markdown lists in ``text`` whose bullets match ``marker``, continuations joined."""
    items, current = [], None
    for line in text.splitlines():
        match = re.match(rf"^(?:{marker})\s+(.*)$", line)
        if match:
            if current is not None:
                items.append(normalise(current))
            current = match[1]
        elif current is not None and line.startswith(" ") and line.strip():
            current += " " + line.strip()
        elif current is not None:
            items.append(normalise(current))
            current = None
    if current is not None:
        items.append(normalise(current))
    return items


def bibliography_entries() -> list:
    """The canonical entries, without their pending-verification mark."""
    body = read("bibliography.md").split("\n## ", 1)[1]
    return [item[:-len(PENDING_MARK)] if item.endswith(PENDING_MARK) else item
            for item in list_items(body, "-")]


def public_names() -> list:
    """Names exported by the package root, excluding submodules."""
    return sorted(name for name in dir(goal_based_allocation)
                  if not name.startswith("_")
                  and not inspect.ismodule(getattr(goal_based_allocation, name)))


def test_navigation_targets_exist():
    for relative in METHODOLOGY_CHAPTERS + UTILITY_PAGES + tuple(LEGACY_REDIRECTS):
        assert (DOCS / relative).is_file(), relative


def test_pages_have_description_title_byline_and_citation():
    for relative in METHODOLOGY_CHAPTERS + UTILITY_PAGES:
        text = read(relative)
        assert text.startswith("---\nmyst:\n  html_meta:\n    description: >-\n"), relative
        body = without_code(text.split("\n---\n", 1)[1])
        titles = re.findall(r"^# .+$", body, flags=re.MULTILINE)
        assert len(titles) == 1, relative
        after_title = body.split(titles[0], 1)[1].lstrip("\n")
        assert after_title.startswith(BYLINE), relative
        assert CITATION in body, relative


def test_methodology_chapters_have_the_eight_sections_in_order():
    for chapter in METHODOLOGY_CHAPTERS:
        headings = re.findall(r"^## (.+)$", without_code(read(chapter)), flags=re.MULTILINE)
        assert tuple(headings) == SECTIONS, chapter
        assert re.search(r"^Implemented in \[GoalBasedAllocation\]", read(chapter), flags=re.MULTILINE)


def test_methodology_chapters_open_with_the_convention_card():
    for chapter in METHODOLOGY_CHAPTERS:
        section = read(chapter).split("## Inputs, notation, and assumptions\n", 1)[1]
        table = []
        for line in section.lstrip("\n").splitlines():
            if not line.startswith("|"):
                break
            table.append(line)
        assert table[0] == "| Convention | This article |", chapter
        assert table[1] == "|---|---|", chapter
        rows = tuple(line.split("|")[1].strip() for line in table[2:])
        assert rows == CARD_ROWS, chapter


def test_methodology_chapters_are_in_the_handbook_toctree():
    toctree = "\n".join(re.findall(r"```\{toctree\}(.*?)```", read("index.md"), flags=re.DOTALL))
    for chapter in METHODOLOGY_CHAPTERS:
        assert re.search(rf"^{chapter[:-3]}$", toctree, flags=re.MULTILINE), chapter


def test_callouts_use_the_two_labels():
    for chapter in METHODOLOGY_CHAPTERS:
        for line in without_code(read(chapter)).splitlines():
            if line.startswith("> **") and not line.startswith(">  "):
                assert line.startswith(("> **Insight.**", "> **Pitfall.**")), (chapter, line)


def test_bibliography_has_unique_entries():
    entries = bibliography_entries()
    assert len(entries) == len(set(entries))
    assert SOFTWARE_ENTRY in entries


def test_references_are_verbatim_bibliography_entries():
    entries = bibliography_entries()
    cited = set()
    for chapter in METHODOLOGY_CHAPTERS:
        references = read(chapter).split("\n## References\n", 1)[1]
        items = list_items(references, r"\d+\.")
        assert items, chapter
        for item in items:
            matches = [entry for entry in entries if item.startswith(entry)]
            assert matches, (chapter, item[:80])
            cited.update(matches)
        assert any(item.startswith(SOFTWARE_ENTRY) for item in items), chapter
    assert cited == set(entries), sorted(set(entries) - cited)


def test_api_page_lists_every_export_exactly_once():
    page = read("api/index.md")
    listed = re.findall(r"^   goal_based_allocation\.([A-Za-z_][A-Za-z0-9_.]*)$", page, flags=re.MULTILINE)
    root = [name for name in listed if "." not in name]
    assert sorted(root) == public_names()
    assert len(root) == len(set(root))
    for dotted in (name for name in listed if "." in name):
        module_name, attribute = dotted.rsplit(".", 1)
        module = importlib.import_module(f"goal_based_allocation.{module_name}")
        assert hasattr(module, attribute), dotted


def test_every_export_is_named_by_a_chapter():
    chapters = "\n".join(read(chapter) for chapter in METHODOLOGY_CHAPTERS)
    spans = re.findall(r"`([^`\n]+)`", chapters)
    for name in public_names():
        assert any(re.search(rf"\b{name}\b", span) for span in spans), name


def test_chapter_mathematics_avoids_github_escapes():
    for chapter in METHODOLOGY_CHAPTERS + ("conventions.md",):
        prose = without_code(read(chapter))
        assert "\\," not in prose, chapter
        assert "}_" not in prose, chapter


def test_legacy_user_guide_pages_redirect_to_chapters():
    for relative, target in LEGACY_REDIRECTS.items():
        text = read(relative)
        assert text.startswith("---\norphan: true\n"), relative
        assert f'"http-equiv=refresh": "0; url=../{target}.html"' in text, relative
        assert (DOCS / f"{target}.md").is_file(), target


@pytest.mark.parametrize(
    ("service_url", "canonical_url"),
    [
        # stable and latest serve the same pages, so both name latest as canonical
        (
            "https://goalbasedallocation.readthedocs.io/en/stable/",
            "https://goalbasedallocation.readthedocs.io/en/latest/",
        ),
        (
            "https://goalbasedallocation.readthedocs.io/en/latest/",
            "https://goalbasedallocation.readthedocs.io/en/latest/",
        ),
        (
            "https://goalbasedallocation.readthedocs.io/en/0.4.1/",
            "https://goalbasedallocation.readthedocs.io/en/0.4.1/",
        ),
    ],
)
def test_stable_builds_name_latest_as_canonical(monkeypatch, service_url, canonical_url):
    pytest.importorskip("tomllib", reason="docs/conf.py reads pyproject.toml with tomllib")
    monkeypatch.setenv("READTHEDOCS_CANONICAL_URL", service_url)
    assert runpy.run_path(str(DOCS / "conf.py"))["html_baseurl"] == canonical_url


def test_site_build_shortens_titles_and_keeps_redirect_stubs_out_of_the_index(
    monkeypatch, tmp_path
):
    # Furo would end every title with the full html_title, and the redirect stubs, the noindex
    # search page and the general index would be offered to search engines in the sitemap.
    for module in ("sphinx", "furo", "myst_parser", "sphinx_sitemap"):
        pytest.importorskip(module)
    monkeypatch.delenv("READTHEDOCS_CANONICAL_URL", raising=False)
    source = tmp_path / "source"
    (source / "user-guide").mkdir(parents=True)
    (source / "conf.py").write_text(
        "import runpy\n"
        f"_site = runpy.run_path({str(DOCS / 'conf.py')!r})\n"
        "extensions = ['myst_parser', 'sphinx.ext.autodoc', 'sphinx_sitemap']\n"
        "html_theme = 'furo'\n"
        f"templates_path = [{str(DOCS / '_templates')!r}]\n"
        "for _key in ('project', 'html_title', 'html_baseurl', 'sitemap_url_scheme',\n"
        "             'sitemap_excludes'):\n"
        "    if _key in _site:\n"
        "        globals()[_key] = _site[_key]\n"
        "setup = _site['setup']\n",
        encoding="utf-8",
    )
    front_matter = "---\nmyst:\n  html_meta:\n    description: {}\n---\n\n"
    (source / "index.md").write_text(
        front_matter.format("The handbook.") + "# Home\n\n```{toctree}\nchapter\n```\n",
        encoding="utf-8",
    )
    (source / "chapter.md").write_text(
        front_matter.format("A chapter.") + "# The regime-switching jump-diffusion\n\nText.\n",
        encoding="utf-8",
    )
    (source / "user-guide" / "old.md").write_text(
        "---\norphan: true\nmyst:\n  html_meta:\n    description: A moved page.\n"
        '    "http-equiv=refresh": "0; url=../chapter.html"\n---\n\n# A moved page\n\nMoved.\n',
        encoding="utf-8",
    )
    output = tmp_path / "html"
    result = subprocess.run(
        [sys.executable, "-m", "sphinx", "-W", "-q", "-b", "html", str(source), str(output)],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr

    class Head(HTMLParser):
        """Collect the title text and the robots and description tags of a page head."""

        def __init__(self):
            super().__init__()
            self.robots, self.descriptions = [], []

        def handle_starttag(self, tag, attrs):
            attrs = dict(attrs)
            if tag == "meta" and attrs.get("name") == "robots":
                self.robots.append(attrs["content"])
            if tag == "meta" and attrs.get("name") == "description":
                self.descriptions.append(attrs["content"])

    def head(name):
        """Parse the head of a built page and return its titles and tag collector."""
        html = (output / f"{name}.html").read_text(encoding="utf-8").split("</head>")[0]
        parser = Head()
        parser.feed(html)
        return re.findall(r"<title>(.*?)</title>", html), parser

    project = "goal-based-allocation"
    titles, index = head("index")
    assert titles == ["goal-based-allocation - goal-based allocation under regime-switching "
                      "jump-diffusions"]
    assert index.robots == []
    titles, chapter = head("chapter")
    assert titles == [f"The regime-switching jump-diffusion - {project}"]
    assert chapter.robots == []
    titles, stub = head("user-guide/old")
    assert titles == [f"A moved page - {project}"]
    assert stub.robots == ["noindex, follow"]
    assert stub.descriptions == ["A moved page."]
    sitemap = (output / "sitemap.xml").read_text(encoding="utf-8")
    base = "https://goalbasedallocation.readthedocs.io/en/latest/"
    assert sorted(re.findall(r"<loc>(.*?)</loc>", sitemap)) == [
        f"{base}chapter.html",
        f"{base}index.html",
    ]
