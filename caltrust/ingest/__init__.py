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
