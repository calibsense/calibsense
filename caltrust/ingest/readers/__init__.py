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

"""Readers for calibrations produced by other tools.

The reader list is ordered most-specific first: ROS and Kalibr are both YAML and
both would be plausible to a looser matcher, so each is asked before the
general OpenCV FileStorage reader gets a turn.
"""

from __future__ import annotations

import os
from typing import List, Optional, Sequence, Tuple

from ...core.session import CalibrationRecord
from ...errors import UnsupportedFormatError
from .base import SNIFF_BYTES, CalibrationReader
from .kalibr import KalibrReader
from .native import NativeJsonReader, write_calibration
from .opencv_fs import OpenCVFileStorageReader
from .ros import RosCameraInfoReader


def default_readers() -> List[CalibrationReader]:
    """Every reader, in the order `read_calibration` tries them."""
    return [
        NativeJsonReader(),
        RosCameraInfoReader(),
        KalibrReader(),
        OpenCVFileStorageReader(),
    ]


def reader_names() -> Tuple[Tuple[str, str], ...]:
    """Name and one-line description of every reader, for CLI help."""
    return tuple((r.name, r.description) for r in default_readers())


def _head(path: str) -> str:
    try:
        with open(path, "rb") as handle:
            return handle.read(SNIFF_BYTES).decode("utf-8", errors="ignore")
    except OSError as exc:
        raise UnsupportedFormatError(f"could not read {path}: {exc}") from exc


def read_calibration(
    path: str,
    fmt: Optional[str] = None,
    readers: Optional[Sequence[CalibrationReader]] = None,
) -> CalibrationRecord:
    """Read a calibration file, detecting its format.

    Args:
        path: The file to read.
        fmt: Force one reader by name, skipping detection. Useful when a file
            is recognised by the wrong reader, or carries no usable header.
        readers: Override the reader list, for tests.

    Returns:
        The calibration the file describes.

    Raises:
        UnsupportedFormatError: The file is missing, no reader matched, or the
            named reader does not exist.
    """
    if not os.path.isfile(path):
        raise UnsupportedFormatError(f"no such calibration file: {path}")
    candidates = list(readers) if readers is not None else default_readers()
    if fmt is not None:
        chosen = [r for r in candidates if r.name == fmt]
        if not chosen:
            raise UnsupportedFormatError(
                f"unknown calibration format {fmt!r}; have "
                f"{[r.name for r in candidates]}"
            )
        return chosen[0].read(path)

    head = _head(path)
    for reader in candidates:
        if reader.sniff(path, head):
            return reader.read(path)
    raise UnsupportedFormatError(
        f"no reader recognised {path}; supported formats are "
        f"{[r.name for r in candidates]}. Force one with --calibration-format."
    )


__all__ = [
    "CalibrationReader",
    "KalibrReader",
    "NativeJsonReader",
    "OpenCVFileStorageReader",
    "RosCameraInfoReader",
    "default_readers",
    "read_calibration",
    "reader_names",
    "write_calibration",
]
