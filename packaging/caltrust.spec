# caltrust - metric trust for camera calibration.
# Copyright (C) 2026 Abhishek Gola
#
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License, version 3, as published by
# the Free Software Foundation. This program is distributed WITHOUT ANY WARRANTY;
# without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
# PARTICULAR PURPOSE. See the LICENSE file, or <https://www.gnu.org/licenses/>.

# PyInstaller spec for a single-file caltrust binary.
#
# Build with `make binary`, which runs:
#     python -m PyInstaller --clean --noconfirm packaging/caltrust.spec
#
# caltrust is written to survive freezing. Every registry -- camera models,
# targets, detectors, calibration readers -- is a literal dictionary populated
# by static imports, so nothing here needs a hidden-import list to find them,
# and the package ships no data files, so nothing needs sys._MEIPASS handling.
# The CLI is argparse, so there is no third-party console framework to hook.
#
# The binary is dominated by OpenCV; expect roughly 80-120 MB depending on
# platform. The excludes below drop the GUI and plotting stacks that OpenCV and
# NumPy pull in transitively but caltrust never calls.

import os

BLOCK_CIPHER = None
ROOT = os.path.abspath(os.path.join(SPECPATH, os.pardir))

analysis = Analysis(
    [os.path.join(ROOT, "caltrust", "__main__.py")],
    pathex=[ROOT],
    binaries=[],
    datas=[],
    hiddenimports=[],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        "tkinter",
        "matplotlib",
        "PIL",
        "PyQt5",
        "PyQt6",
        "PySide2",
        "PySide6",
        "IPython",
        "pytest",
        "setuptools",
        "pydoc_data",
        "scipy",
        "pandas",
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=BLOCK_CIPHER,
    noarchive=False,
)

archive = PYZ(analysis.pure, analysis.zipped_data, cipher=BLOCK_CIPHER)

executable = EXE(
    archive,
    analysis.scripts,
    analysis.binaries,
    analysis.zipfiles,
    analysis.datas,
    [],
    name="caltrust",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
