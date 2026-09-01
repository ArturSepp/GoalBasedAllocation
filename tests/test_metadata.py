"""Tests for release and repository metadata consistency."""

import re
from pathlib import Path

import goal_based_allocation

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def test_version_surfaces_are_consistent():
    pyproject = (REPOSITORY_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    match = re.search(r'^version = "([^\"]+)"$', pyproject, flags=re.MULTILINE)
    citation = (REPOSITORY_ROOT / "CITATION.cff").read_text(encoding="utf-8")
    readme = (REPOSITORY_ROOT / "README.md").read_text(encoding="utf-8")

    assert match is not None
    version = match.group(1)
    assert goal_based_allocation.__version__ == version
    assert f"version: {version}" in citation
    assert f"version      = {{{version}}}" in readme


def test_core_metadata_uses_canonical_identity():
    pyproject = (REPOSITORY_ROOT / "pyproject.toml").read_text(encoding="utf-8")

    assert 'name = "goal-based-allocation"' in pyproject
    assert 'license = "MIT"' in pyproject
    assert 'description = "Semi-analytical dynamic mean-variance allocation' in pyproject
    assert 'Repository = "https://github.com/ArturSepp/GoalBasedAllocation"' in pyproject
    paper_url = 'Paper = "https://papers.ssrn.com/sol3/papers.cfm?abstract_id=6534579"'
    assert paper_url in pyproject
