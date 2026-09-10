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

"""The reader contract for existing calibration files."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import ClassVar

from ...core.session import CalibrationRecord

#: Bytes of a file offered to `sniff`. Enough for any header worth matching.
SNIFF_BYTES = 4096

#: Markers unique to OpenCV's own serialiser. `!!opencv-` covers both
#: `opencv-matrix` and the `opencv-nd-matrix` that OpenCV 5 writes for a 1-D
#: array, and `%YAML:` is the colon form OpenCV 4 emits, which is not valid YAML
#: and therefore appears nowhere else.
_OPENCV_MARKERS = ("!!opencv-", "<opencv_storage>", "%YAML:")


def looks_like_opencv_filestorage(head: str) -> bool:
    """Whether a file header is OpenCV's own serialisation format.

    Used in both directions: the OpenCV reader requires it, and the YAML readers
    for ROS and Kalibr require its absence. Without that second check an OpenCV
    file that happens to carry a `distortion_model` key is claimed by the ROS
    reader, which then fails on a tag PyYAML cannot construct.

    Args:
        head: The first `SNIFF_BYTES` bytes of the file, decoded as text.

    Returns:
        `True` when the header carries an OpenCV FileStorage marker.
    """
    return any(marker in head for marker in _OPENCV_MARKERS)


class CalibrationReader(ABC):
    """Reads one third-party calibration format.

    A reader is asked to recognise a file before it is asked to parse it, so
    that an unrecognised file produces "no reader matched" rather than the
    parse error of whichever reader happened to be tried last.
    """

    #: Short name used in `--format` and in provenance strings.
    name: ClassVar[str] = ""
    #: One line, shown by the CLI when listing supported formats.
    description: ClassVar[str] = ""

    @abstractmethod
    def sniff(self, path: str, head: str) -> bool:
        """Decide whether this reader should attempt the file.

        Args:
            path: The file path, for extension checks.
            head: The first `SNIFF_BYTES` bytes decoded as text, errors ignored.

        Returns:
            `True` if this reader recognises the format.
        """

    @abstractmethod
    def read(self, path: str) -> CalibrationRecord:
        """Parse the file.

        Args:
            path: The file to read.

        Returns:
            The calibration it describes.

        Raises:
            UnsupportedFormatError: The file matched but could not be parsed.
        """
