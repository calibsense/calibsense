"""Guards on the two shipping targets: the pip package and the frozen binary.

These are cheap checks for mistakes that are invisible in a normal test run and
only show up once something is distributed.
"""

from __future__ import annotations

import ast
import os
import pathlib
import subprocess
import sys

import pytest

import caltrust

ROOT = pathlib.Path(caltrust.__file__).resolve().parent
REPO = ROOT.parent


def test_entry_point_imports_absolutely():
    """PyInstaller runs `__main__.py` with no parent package.

    A relative import there fails at run time and, worse, leaves PyInstaller's
    dependency graph empty at build time, producing a binary that builds without
    error and cannot start.
    """
    tree = ast.parse((ROOT / "__main__.py").read_text())
    relative = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.level > 0
    ]
    assert not relative, (
        "caltrust/__main__.py must use absolute imports; found "
        f"{[node.module for node in relative]}"
    )


def test_python_dash_m_runs_the_cli():
    result = subprocess.run(
        [sys.executable, "-m", "caltrust", "--version"],
        capture_output=True, text=True, cwd=str(REPO),
    )
    assert result.returncode == 0, result.stderr
    assert caltrust.__version__ in result.stdout


def test_version_is_declared_once():
    """`pyproject.toml` and `caltrust._version` must not drift apart."""
    pyproject = (REPO / "pyproject.toml").read_text()
    assert f'version = "{caltrust.__version__}"' in pyproject


def test_runtime_dependencies_stay_at_three():
    """The binary target is why this list is short; keep it deliberate."""
    import re

    text = (REPO / "pyproject.toml").read_text()
    block = re.search(r"^dependencies = \[(.*?)\]", text, re.S | re.M)
    assert block, "pyproject.toml must declare dependencies"
    names = re.findall(r'"([A-Za-z0-9_.\-]+)', block.group(1))
    assert sorted(names) == ["numpy", "opencv-contrib-python", "pyyaml"]


def test_no_module_imports_a_dependency_we_do_not_declare():
    forbidden = {"scipy", "pandas", "matplotlib", "torch", "click", "typer",
                 "rich", "PIL", "requests"}
    offenders = []
    for path in ROOT.rglob("*.py"):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [alias.name.split(".")[0] for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                names = [node.module.split(".")[0]]
            else:
                continue
            offenders += [
                (str(path.relative_to(ROOT)), name)
                for name in names if name in forbidden
            ]
    assert not offenders, f"undeclared imports: {offenders}"


def test_the_package_ships_no_data_files():
    """No data files means no `sys._MEIPASS` handling is needed when frozen."""
    extras = [
        str(p.relative_to(ROOT))
        for p in ROOT.rglob("*")
        if p.is_file() and p.suffix not in (".py", ".pyc") and "__pycache__" not in str(p)
    ]
    assert not extras, f"unexpected non-Python files in the package: {extras}"


def test_every_module_is_reachable_by_a_static_import():
    """No registry may rely on a filesystem scan, which a frozen build lacks."""
    import importlib

    for path in ROOT.rglob("*.py"):
        if path.name == "__main__.py":
            continue
        relative = path.relative_to(ROOT).with_suffix("")
        parts = [p for p in relative.parts if p != "__init__"]
        importlib.import_module(".".join(["caltrust", *parts]))


@pytest.mark.skipif(
    not (REPO / "dist" / "caltrust").exists(),
    reason="no frozen binary built; run `make binary` first",
)
@pytest.mark.slow
def test_the_frozen_binary_starts_in_a_bare_environment():
    binary = REPO / "dist" / "caltrust"
    result = subprocess.run(
        [str(binary), "formats"],
        capture_output=True, text=True,
        env={"HOME": os.environ.get("HOME", "/tmp"), "PATH": "/usr/bin:/bin"},
    )
    assert result.returncode == 0, result.stderr
    assert "checkerboard" in result.stdout
    assert "pinhole_brown_conrady" in result.stdout


def test_the_diagnostic_registry_is_a_literal_not_a_scan():
    """A frozen build has no filesystem to scan, so the list has to be static."""
    source = (ROOT / "diagnose" / "report.py").read_text()
    tree = ast.parse(source)
    assignments = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.AnnAssign)
        and isinstance(node.target, ast.Name)
        and node.target.id == "DIAGNOSTICS"
    ]
    assert assignments, "DIAGNOSTICS must be defined in diagnose/report.py"
    assert isinstance(assignments[0].value, ast.Tuple)
    assert all(isinstance(item, ast.Name) for item in assignments[0].value.elts)


def test_every_diagnostic_in_the_registry_is_statically_imported():
    from caltrust.diagnose.report import DIAGNOSTICS

    source = (ROOT / "diagnose" / "report.py").read_text()
    for cls in DIAGNOSTICS:
        assert f"import" in source and cls.__name__ in source, cls.__name__
