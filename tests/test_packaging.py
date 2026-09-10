# calibsense - measurement uncertainty for camera calibration.
# Copyright (C) 2026 Abhishek Gola
#
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License, version 3, as published by
# the Free Software Foundation. This program is distributed WITHOUT ANY WARRANTY;
# without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
# PARTICULAR PURPOSE. See the LICENSE file, or <https://www.gnu.org/licenses/>.

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

import calibsense

ROOT = pathlib.Path(calibsense.__file__).resolve().parent
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
        "calibsense/__main__.py must use absolute imports; found "
        f"{[node.module for node in relative]}"
    )


def test_python_dash_m_runs_the_cli():
    result = subprocess.run(
        [sys.executable, "-m", "calibsense", "--version"],
        capture_output=True, text=True, cwd=str(REPO),
    )
    assert result.returncode == 0, result.stderr
    assert calibsense.__version__ in result.stdout


def test_version_is_declared_once():
    """`pyproject.toml` and `calibsense._version` must not drift apart."""
    pyproject = (REPO / "pyproject.toml").read_text()
    assert f'version = "{calibsense.__version__}"' in pyproject


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
        importlib.import_module(".".join(["calibsense", *parts]))


@pytest.mark.skipif(
    not (REPO / "dist" / "calibsense").exists(),
    reason="no frozen binary built; run `make binary` first",
)
@pytest.mark.slow
def test_the_frozen_binary_starts_in_a_bare_environment():
    binary = REPO / "dist" / "calibsense"
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
    from calibsense.diagnose.report import DIAGNOSTICS

    source = (ROOT / "diagnose" / "report.py").read_text()
    for cls in DIAGNOSTICS:
        assert f"import" in source and cls.__name__ in source, cls.__name__


def test_the_licence_file_is_the_real_agpl_text():
    text = (REPO / "LICENSE").read_text()
    assert "GNU AFFERO GENERAL PUBLIC LICENSE" in text.splitlines()[0]
    assert "Version 3, 19 November 2007" in text
    # Section 13 is what distinguishes the AGPL from the GPL; if it is missing,
    # the file is the wrong licence.
    assert "Remote Network Interaction" in text
    assert len(text) > 30_000


def test_pyproject_declares_the_same_licence():
    """PEP 639 form: an SPDX expression, not a table, and no classifier.

    A `license` table and a `License ::` classifier both still build, but
    setuptools deprecates them and a first release should not ship warnings.
    """
    text = (REPO / "pyproject.toml").read_text()
    assert 'license = "AGPL-3.0-only"' in text
    # Under PEP 639 both keys belong to [project]; setuptools warns if
    # `license-files` is left behind in [tool.setuptools].
    project = text.split("[tool.setuptools")[0]
    assert 'license-files = ["LICENSE"]' in project
    # The expression carries the licence now, so the classifier is redundant.
    assert "License :: OSI Approved" not in text


def test_every_source_file_carries_the_spdx_header():
    """A licence nobody can find in the file they are reading is not much use."""
    missing = []
    for root in ("calibsense", "tests", "examples"):
        base = REPO / root
        if not base.exists():
            continue
        for path in base.rglob("*.py"):
            if "__pycache__" in str(path):
                continue
            head = path.read_text()[:800]
            if "SPDX-License-Identifier: AGPL-3.0-only" not in head:
                missing.append(str(path.relative_to(REPO)))
    assert not missing, f"no SPDX header in: {missing}"


def test_the_pyinstaller_spec_carries_it_too():
    head = (REPO / "packaging" / "calibsense.spec").read_text()[:800]
    assert "SPDX-License-Identifier: AGPL-3.0-only" in head


def test_the_header_sits_above_the_docstring_not_instead_of_it():
    """Comments before a docstring are fine; replacing it would break pdoc."""
    import calibsense
    import calibsense.refit.covariance
    import calibsense.report.pdf
    import calibsense.task.tasks

    for module in (calibsense, calibsense.refit.covariance, calibsense.report.pdf,
                   calibsense.task.tasks):
        assert module.__doc__, f"{module.__name__} lost its docstring"
        assert "SPDX" not in module.__doc__, f"{module.__name__} header ate the docstring"


def test_the_version_flag_states_the_licence():
    """The AGPL asks an interactive program to say so."""
    result = subprocess.run(
        [sys.executable, "-m", "calibsense", "--version"],
        capture_output=True, text=True, cwd=str(REPO),
    )
    assert result.returncode == 0
    assert "AGPL-3.0-only" in result.stdout
    assert "NO WARRANTY" in result.stdout
    assert "network access" in result.stdout


def test_no_stale_permissive_licence_claim_remains():
    """The project was Apache-2.0 first; a leftover claim would be a real problem."""
    offenders = []
    for name in ("README.md", "pyproject.toml", "tracker.md", "open-items.md"):
        path = REPO / name
        if not path.exists():
            continue
        for number, line in enumerate(path.read_text().splitlines(), 1):
            if "Apache" in line and "OpenCV" not in line:
                offenders.append(f"{name}:{number}: {line.strip()}")
    assert not offenders, f"stale licence claims: {offenders}"
