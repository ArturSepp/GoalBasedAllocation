"""Protect frozen study inputs and external output without vendor dependencies."""

import ast
import hashlib
import importlib
import json
from pathlib import Path


REPLICATION = Path(__file__).resolve().parents[1]


def test_four_static_snapshots_preserve_hashes():
    """Git-normalized bytes, including provenance headers, must survive the move."""
    data = REPLICATION / "data"
    expected = json.loads((data / "sha256.json").read_text(encoding="utf-8"))
    assert len(expected) == 4
    actual = {p.name: hashlib.sha256(p.read_bytes().replace(b"\r\n", b"\n")).hexdigest()
              for p in data.glob("*.csv")}
    assert actual == expected


def test_data_and_output_locations_are_distinct(monkeypatch, tmp_path):
    """Frozen input resolution must not depend on the caller's current directory."""
    monkeypatch.syspath_prepend(str(REPLICATION))
    paths = importlib.import_module("kospi_paths")
    monkeypatch.setenv("GBA_PAPER_OUTPUT_PATH", str(tmp_path))
    monkeypatch.chdir(tmp_path)
    assert paths.DATA_DIR == REPLICATION / "data"
    assert paths.figure_directory() == tmp_path / "kospi_volatility_fit_jun2026" / "figures"
    assert paths.output_root() != paths.DATA_DIR


def test_relocated_sibling_imports_exist():
    """Every study-local import still resolves beside the moved scripts."""
    siblings = {"vol_surface_utils", "regime_switch_calibration", "term_structure",
                "bvol_interpolation_check", "kospi_paths"}
    for script in REPLICATION.glob("*.py"):
        tree = ast.parse(script.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module in siblings:
                assert (REPLICATION / f"{node.module}.py").is_file()
