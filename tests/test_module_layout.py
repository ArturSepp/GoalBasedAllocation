"""Contracts separating pytest modules, source runners, and production code."""

from __future__ import annotations

import ast
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = REPOSITORY_ROOT / "src" / "goal_based_allocation"
TESTS_ROOT = REPOSITORY_ROOT / "tests"
EXPECTED_RUNNERS = {
    "run/client_solver_local.py",
    "run/laplace_inversion_local.py",
    "run/mandate_utils_local.py",
    "run/opportunity_set_local.py",
    "run/regime_switch_paper_local.py",
    "run/riccati_solver_local.py",
    "run/variance_swap_local.py",
}
LEGACY_DISPATCHERS = {
    "LocalTest",
    "LocalTests",
    "UnitTest",
    "UnitTests",
    "local_test",
    "run_local_test",
    "run_unit_test",
    "unit_test",
}


def _tree(path: Path) -> ast.Module:
    """Parse one Python module."""
    return ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))


def _definitions(path: Path) -> set[str]:
    """Return top-level class and function names."""
    return {
        node.name
        for node in _tree(path).body
        if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
    }


def _has_test_candidate(path: Path) -> bool:
    """Return whether a module defines a pytest-collectable test."""
    return any(
        isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name.startswith("test_")
        for node in ast.walk(_tree(path))
    )


def _is_main_guard(node: ast.AST) -> bool:
    """Return whether a node is an executable main guard."""
    return (
        isinstance(node, ast.If)
        and isinstance(node.test, ast.Compare)
        and isinstance(node.test.left, ast.Name)
        and node.test.left.id == "__name__"
        and len(node.test.ops) == 1
        and isinstance(node.test.ops[0], ast.Eq)
        and len(node.test.comparators) == 1
        and isinstance(node.test.comparators[0], ast.Constant)
        and node.test.comparators[0].value == "__main__"
    )


def _has_main_guard(path: Path) -> bool:
    """Return whether a module has a top-level executable main guard."""
    return any(_is_main_guard(node) for node in _tree(path).body)


def _main_calls_run_local_directly(path: Path) -> bool:
    """Return whether the sole main statement is ``run_local(local=Locals.*)``."""
    guards = [node for node in _tree(path).body if _is_main_guard(node)]
    if len(guards) != 1 or len(guards[0].body) != 1:
        return False
    statement = guards[0].body[0]
    if not (
        isinstance(statement, ast.Expr)
        and isinstance(statement.value, ast.Call)
        and isinstance(statement.value.func, ast.Name)
        and statement.value.func.id == "run_local"
    ):
        return False
    return any(
        keyword.arg == "local"
        and isinstance(keyword.value, ast.Attribute)
        and isinstance(keyword.value.value, ast.Name)
        and keyword.value.value.id == "Locals"
        for keyword in statement.value.keywords
    )


def _imports_run(path: Path) -> bool:
    """Return whether production code imports development-only run code."""
    for node in ast.walk(_tree(path)):
        if isinstance(node, ast.Import):
            if any("run" in alias.name.split(".") for alias in node.names):
                return True
        elif isinstance(node, ast.ImportFrom):
            parts = (node.module or "").split(".")
            if "run" in parts:
                return True
    return False


def test_pytest_modules_are_central_and_automated() -> None:
    """Top-level test modules remain pure pytest modules."""
    test_modules = sorted(
        path
        for path in TESTS_ROOT.glob("*.py")
        if path.name not in {"__init__.py", "conftest.py"}
    )
    failures = []
    for path in test_modules:
        if not path.name.startswith("test_"):
            failures.append(f"{path.name}: expected test_*.py")
        if not _has_test_candidate(path):
            failures.append(f"{path.name}: no pytest test candidate")
        if _has_main_guard(path):
            failures.append(f"{path.name}: pytest modules cannot be executable runners")
    assert len(test_modules) >= 9, "the automated suite unexpectedly disappeared"
    assert not failures, failures


def test_development_runner_layout() -> None:
    """Source development runners use the requested no-init run/xxx_local contract."""
    python_modules = sorted(PACKAGE_ROOT.rglob("*.py"))
    run_modules = [
        path for path in python_modules if "run" in path.relative_to(PACKAGE_ROOT).parts
    ]
    runners = [path for path in run_modules if path.name.endswith("_local.py")]
    failures = []
    for path in runners:
        relative = path.relative_to(PACKAGE_ROOT).as_posix()
        definitions = _definitions(path)
        if not {"Locals", "run_local"} <= definitions:
            failures.append(f"{relative}: expected Locals plus run_local")
        if LEGACY_DISPATCHERS & definitions:
            failures.append(f"{relative}: retains legacy dispatcher names")
        if not _main_calls_run_local_directly(path):
            failures.append(f"{relative}: main guard must contain only run_local(local=Locals.*)")
        if _has_test_candidate(path):
            failures.append(f"{relative}: contains pytest tests")

    actual_runners = {path.relative_to(PACKAGE_ROOT).as_posix() for path in runners}
    support_modules = sorted(
        path.relative_to(PACKAGE_ROOT).as_posix()
        for path in run_modules
        if not path.name.endswith("_local.py")
    )
    local_modules_outside_run = sorted(
        path.relative_to(PACKAGE_ROOT).as_posix()
        for path in python_modules
        if path.name.endswith("_local.py") and "run" not in path.relative_to(PACKAGE_ROOT).parts
    )
    assert actual_runners == EXPECTED_RUNNERS
    assert not failures, failures
    assert not support_modules, f"unexpected run support modules: {support_modules}"
    assert not local_modules_outside_run, local_modules_outside_run


def test_production_modules_do_not_own_development_dispatchers() -> None:
    """Production modules stay independent of source-only development runners."""
    failures = []
    for path in sorted(PACKAGE_ROOT.rglob("*.py")):
        if "run" in path.relative_to(PACKAGE_ROOT).parts:
            continue
        relative = path.relative_to(PACKAGE_ROOT).as_posix()
        if LEGACY_DISPATCHERS & _definitions(path):
            failures.append(f"{relative}: owns a legacy development dispatcher")
        if _has_main_guard(path):
            failures.append(f"{relative}: owns an executable development runner")
        if _imports_run(path):
            failures.append(f"{relative}: imports development-only run code")
    assert not failures, failures


def test_development_runners_are_excluded_from_distributions() -> None:
    """Setuptools, the source manifest, and artifact checker exclude source runners."""
    pyproject = (REPOSITORY_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    manifest = (REPOSITORY_ROOT / "MANIFEST.in").read_text(encoding="utf-8")
    checker = (REPOSITORY_ROOT / "scripts" / "check_dist_contents.py").read_text(
        encoding="utf-8"
    )
    assert '"goal_based_allocation.run*"' in pyproject
    assert "prune src/goal_based_allocation/run" in manifest
    assert '"goal_based_allocation/run/"' in checker
