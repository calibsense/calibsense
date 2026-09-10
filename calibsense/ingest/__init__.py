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

"""M1 — ingest.

Two entry points, one output. `session_from_images` runs a detector over a
capture; `session_from_calibration` takes a calibration that already exists
together with the detections behind it.
"""

from __future__ import annotations

from .detections import read_detections, write_detections
from .detectors import DetectorOptions, TargetDetector, detector_for, registered_detectors
from .images import detect_in_images, find_images, read_image
from .pipeline import session_from_calibration, session_from_images
from .readers import read_calibration, reader_names, write_calibration
from .robot import read_robot_poses, write_robot_poses

__all__ = [
    "DetectorOptions",
    "TargetDetector",
    "detect_in_images",
    "detector_for",
    "find_images",
    "read_calibration",
    "read_detections",
    "read_image",
    "read_robot_poses",
    "reader_names",
    "registered_detectors",
    "session_from_calibration",
    "session_from_images",
    "write_calibration",
    "write_detections",
    "write_robot_poses",
]
