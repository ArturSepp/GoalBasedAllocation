"""Validate the public wheel and source-distribution boundaries."""

from __future__ import annotations

import glob
import sys
import tarfile
import zipfile
from pathlib import Path

PROHIBITED_ROOTS = (".idea/", "agents/", "paper_code/", "papers/")
PROHIBITED_PACKAGE_ROOTS = ("goal_based_allocation/run/",)
PROHIBITED_SOURCE_ROOTS = ("src/goal_based_allocation/run/",)


def _expand_paths(arguments: list[str]) -> list[Path]:
    paths: list[Path] = []
    for argument in arguments:
        matches = [Path(match) for match in glob.glob(argument)]
        paths.extend(matches or [Path(argument)])
    return paths


def _check_wheel(path: Path) -> None:
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()

    assert names.count("goal_based_allocation/__init__.py") == 1
    assert any(name.endswith(".dist-info/METADATA") for name in names)
    for name in names:
        assert not name.startswith(
            ("tests/", "examples/", *PROHIBITED_ROOTS, *PROHIBITED_PACKAGE_ROOTS)
        ), name
        assert name.startswith("goal_based_allocation/") or ".dist-info/" in name, name


def _strip_sdist_root(names: list[str]) -> list[str]:
    roots = {name.split("/", 1)[0] for name in names if "/" in name}
    assert len(roots) == 1, roots
    prefix = f"{roots.pop()}/"
    return [name.removeprefix(prefix) for name in names if name != prefix]


def _check_sdist(path: Path) -> None:
    with tarfile.open(path, "r:gz") as archive:
        names = _strip_sdist_root(archive.getnames())

    assert names.count("src/goal_based_allocation/__init__.py") == 1
    for required in ("LICENSE", "README.md", "pyproject.toml"):
        assert required in names, required
    for name in names:
        assert not name.startswith((*PROHIBITED_ROOTS, *PROHIBITED_SOURCE_ROOTS)), name


def main(arguments: list[str]) -> int:
    paths = _expand_paths(arguments)
    if not paths:
        raise SystemExit("pass at least one wheel or sdist path")

    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)
        if path.suffix == ".whl":
            _check_wheel(path)
        elif path.name.endswith(".tar.gz"):
            _check_sdist(path)
        else:
            raise ValueError(f"unsupported distribution artifact: {path}")
        print(f"validated {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
