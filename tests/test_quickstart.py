"""Execution test for the authoritative first-success workflow."""

import runpy
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def test_quickstart_runs_without_outputs(capsys, tmp_path, monkeypatch):
    script = REPOSITORY_ROOT / "examples" / "getting_started" / "quickstart.py"
    monkeypatch.chdir(tmp_path)

    runpy.run_path(str(script), run_name="__main__")
    output = capsys.readouterr().out

    assert "MV-optimal: expected_wealth=139.040" in output
    assert "survival=78.703%" in output
    assert "buy-and-hold: expected_wealth=150.637" in output
    assert "floor_protection_cost=7.699%" in output
    assert list(tmp_path.iterdir()) == []
