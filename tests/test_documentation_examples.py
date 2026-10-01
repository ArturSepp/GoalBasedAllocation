"""Execute the worked examples of every handbook chapter.

Each methodology chapter under ``docs/`` shows its worked example as fenced ``python`` blocks whose
assertions check every number the chapter quotes against an independent computation. The blocks of
one chapter run in order in one namespace, offline, from a temporary working directory, so a chapter
reads like a script. A block that is a schematic fragment carries the comment
``<!-- docs-test: skip -->`` on the line before its fence. Chapters whose examples run Monte Carlo
or calibrate a mandate are marked ``slow``.
"""

import re
from pathlib import Path

import pytest

DOCS = Path(__file__).resolve().parents[1] / "docs"

FENCE = re.compile(r"(?P<skip><!-- docs-test: skip -->\s*\n)?```python\n(?P<code>.*?)\n```", re.DOTALL)

# chapter basename -> whether its examples are slow
CHAPTERS = {
    "regime_switching_model": False,
    "buy_and_hold_moments": False,
    "mv_optimal_policy": True,
    "wealth_floor_gap_process": True,
    "laplace_inversion": False,
    "laplace_barrier_framework": False,
    "terminal_wealth_distribution": True,
    "mandate_aggregation": True,
    "investment_opportunity_set": True,
    "european_options": False,
    "variance_swaps": False,
}


def chapter_blocks(name: str) -> list:
    """Return the executable python blocks of a chapter, in page order.

    Args:
        name: chapter basename under ``docs/``.

    Returns:
        Source strings of the blocks that are not marked as skipped.
    """
    text = (DOCS / f"{name}.md").read_text(encoding="utf-8")
    return [match["code"] for match in FENCE.finditer(text) if not match["skip"]]


def test_every_methodology_chapter_has_examples():
    for name in CHAPTERS:
        assert chapter_blocks(name), f"{name}.md has no executable python block"


@pytest.mark.parametrize(
    "name",
    [pytest.param(name, marks=pytest.mark.slow) if slow else name
     for name, slow in CHAPTERS.items()],
)
def test_chapter_examples_run(name, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("MPLBACKEND", "Agg")
    namespace = {"__name__": f"docs_{name}"}
    for index, code in enumerate(chapter_blocks(name)):
        exec(compile(code, f"docs/{name}.md[block {index}]", "exec"), namespace)
