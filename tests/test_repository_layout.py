"""Repository layout and import-boundary tests."""

import re
from pathlib import Path

import goal_based_allocation

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def test_installable_package_uses_src_layout():
    """Reject a root package that can mask candidate-artifact defects."""
    root_package = REPOSITORY_ROOT / "goal_based_allocation"
    src_package = REPOSITORY_ROOT / "src" / "goal_based_allocation"

    assert not root_package.exists()
    assert src_package.is_dir()
    assert Path(goal_based_allocation.__file__).resolve().is_relative_to(src_package.resolve())


def test_examples_remain_outside_installable_package():
    """Examples are repository-only reference implementations."""
    assert (REPOSITORY_ROOT / "examples").is_dir()
    assert not (REPOSITORY_ROOT / "src" / "goal_based_allocation" / "examples").exists()


def test_research_projects_use_papers_root():
    """Keep one conventional, repository-only research root."""
    papers = REPOSITORY_ROOT / "papers"

    assert not (REPOSITORY_ROOT / "paper_code").exists()
    assert (papers / "goal_based_allocation_2026" / "generate_paper_figures.py").is_file()
    assert (papers / "kospi_volatility_fit_jun2026" / "run_analysis.py").is_file()


def test_public_path_references_use_papers_root():
    """Reject stale commands and image paths after the papers rename."""
    paths = (
        REPOSITORY_ROOT / "README.md",
        REPOSITORY_ROOT / "AGENTS.md",
        REPOSITORY_ROOT / "papers" / "kospi_volatility_fit_jun2026" / "README.md",
    )
    for path in paths:
        assert "paper_code" not in path.read_text(encoding="utf-8")


def test_readme_local_images_resolve():
    """Protect the paper figures displayed on GitHub and PyPI."""
    readme = (REPOSITORY_ROOT / "README.md").read_text(encoding="utf-8")
    image_paths = re.findall(r'<img src="([^\"]+)"', readme)

    assert image_paths
    for image_path in image_paths:
        assert (REPOSITORY_ROOT / image_path).is_file(), image_path
