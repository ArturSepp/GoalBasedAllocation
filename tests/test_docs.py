"""Documentation structure and public-name coverage tests."""

import re
from pathlib import Path

import goal_based_allocation

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def test_api_catalogue_names_exist():
    api_page = (REPOSITORY_ROOT / "docs" / "api" / "index.md").read_text(encoding="utf-8")
    names = set(re.findall(r"^\| `([A-Za-z][A-Za-z0-9_]*)` \|", api_page, flags=re.MULTILINE))

    assert len(names) >= 30
    missing = sorted(name for name in names if not hasattr(goal_based_allocation, name))
    assert missing == []


def test_docs_navigation_targets_exist():
    expected = (
        "getting-started.md",
        "conventions.md",
        "model-boundaries.md",
        "validation.md",
        "papers.md",
        "comparison.md",
        "api/index.md",
        "user-guide/mv-optimal-policy.md",
        "user-guide/terminal-wealth-floor.md",
        "user-guide/mandates-opportunity-set.md",
        "user-guide/option-pricing.md",
    )
    docs = REPOSITORY_ROOT / "docs"

    for relative_path in expected:
        assert (docs / relative_path).is_file(), relative_path
