"""Regressions for CLI output routing without running expensive simulations."""

import importlib
from pathlib import Path
import sys

import pytest


@pytest.fixture
def producer(monkeypatch):
    """Import the repository-only script using its supported sibling imports."""
    replication = Path(__file__).resolve().parents[1]
    monkeypatch.syspath_prepend(str(replication))
    return importlib.import_module("generate_paper_figures")


def test_test_mode_honors_outdir(producer, monkeypatch, tmp_path):
    """The old --test branch silently ignored its explicit output directory."""
    destination = tmp_path / "verification"
    calls = []
    monkeypatch.setattr(producer, "local_integration_tests",
                        lambda outdir=None: (calls.append(outdir) or 9, 0))
    monkeypatch.setattr(sys, "argv", ["generate", "--test", "--outdir", str(destination)])
    monkeypatch.chdir(tmp_path)
    producer.main()
    assert calls == [destination]
    assert destination.is_dir()
    assert not (tmp_path / "figures").exists()


def test_test_failures_return_nonzero(producer, monkeypatch, tmp_path):
    """A failed numerical assertion must not look like a successful CLI run."""
    monkeypatch.setattr(producer, "local_integration_tests", lambda outdir=None: (8, 1))
    monkeypatch.setattr(sys, "argv", ["generate", "--test", "--outdir", str(tmp_path)])
    with pytest.raises(SystemExit) as failure:
        producer.main()
    assert failure.value.code == 1


def test_default_generation_stays_external(producer, monkeypatch, tmp_path):
    """Normal generation uses a paper-specific external folder."""
    monkeypatch.setenv("GBA_PAPER_OUTPUT_PATH", str(tmp_path))
    calls = []
    monkeypatch.setattr(producer, "_generate_all_figures", calls.append)
    monkeypatch.setattr(sys, "argv", ["generate"])
    producer.main()
    assert calls == [tmp_path / "goal_based_allocation_2026" / "figures"]


@pytest.mark.parametrize("choice", ["relative", "checkout", "onedrive"])
def test_reject_output_in_source_or_onedrive(producer, monkeypatch, tmp_path, choice):
    """Refuse dangerous output destinations before calling the computation."""
    import gba_paths
    value = {"relative": "figures", "checkout": gba_paths.REPOSITORY_DIR / "figures",
             "onedrive": tmp_path / "OneDrive" / "figures"}[choice]
    monkeypatch.setattr(sys, "argv", ["generate", "--outdir", str(value)])
    with pytest.raises(ValueError):
        producer.main()


def test_latex_figure_references_survive_move():
    """The unchanged relative graphicspath must still find every figure."""
    import re
    paper = Path(__file__).resolve().parents[2] / "paper"
    source = (paper / "goal_based_allocation_2026.tex").read_text(encoding="utf-8")
    assert r"\graphicspath{{figures/}}" in source
    figures = re.findall(r"\\includegraphics(?:\[[^]]*\])?\{([^}]+)\}", source)
    assert len(figures) == 6
    assert all((paper / "figures" / name).is_file() for name in figures)
