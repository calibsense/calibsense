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

"""calibsense's own JSON calibration format.

This is the format to emit when scripting against calibsense, and the one that
carries the least ambiguity: the distortion model is named explicitly, so no
reader has to guess from a coefficient count.
"""

from __future__ import annotations

import json
import os
from typing import Any, ClassVar, Dict

from ...core.session import CalibrationRecord
from ...errors import UnsupportedFormatError
from .base import CalibrationReader

FORMAT = "calibsense.calibration"


class NativeJsonReader(CalibrationReader):
    """Reads a JSON file carrying a serialised `CalibrationRecord`."""

    name: ClassVar[str] = "native"
    description: ClassVar[str] = "calibsense JSON calibration"

    def sniff(self, path: str, head: str) -> bool:
        """Match a JSON file that names the calibsense calibration format."""
        return os.path.splitext(path)[1].lower() == ".json" and FORMAT in head

    def read(self, path: str) -> CalibrationRecord:
        """Parse a calibsense JSON calibration.

        Args:
            path: The file to read.

        Returns:
            The calibration it describes.

        Raises:
            UnsupportedFormatError: The JSON will not parse or is missing a key.
        """
        try:
            with open(path, "r", encoding="utf-8") as handle:
                payload: Dict[str, Any] = json.load(handle)
        except (OSError, json.JSONDecodeError) as exc:
            raise UnsupportedFormatError(f"could not parse {path}: {exc}") from exc
        if payload.get("format") != FORMAT:
            raise UnsupportedFormatError(
                f"{path} declares format {payload.get('format')!r}, expected {FORMAT!r}"
            )
        try:
            record = CalibrationRecord.from_dict(payload["calibration"])
        except KeyError as exc:
            raise UnsupportedFormatError(f"{path} is missing {exc}") from exc
        metadata = dict(record.metadata)
        metadata["path"] = os.path.abspath(path)
        return CalibrationRecord(
            camera=record.camera,
            image_size=record.image_size,
            source=record.source if record.source != "unknown" else f"native:{os.path.basename(path)}",
            reported_rms=record.reported_rms,
            metadata=metadata,
        )


def write_calibration(record: CalibrationRecord, path: str) -> str:
    """Write a calibration in calibsense's own JSON format.

    Args:
        record: The calibration to write.
        path: Destination path.

    Returns:
        The path written.

    Raises:
        UnsupportedFormatError: The file could not be written.
    """
    payload = {"format": FORMAT, "calibration": record.to_dict()}
    try:
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
    except OSError as exc:
        raise UnsupportedFormatError(f"could not write {path}: {exc}") from exc
    return path
