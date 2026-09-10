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

"""The two ingest paths, as one function each.

Milestone M1 in full: either detect a target across a folder of images, or take
a calibration someone already produced together with the detections behind it.
Both end at the same `CalibrationSession`, so nothing downstream knows or cares
which route was taken.
"""

from __future__ import annotations

import os
from typing import Any, Mapping, Optional, Sequence, Tuple

from ..core.camera import CameraModel
from ..core.observations import ObservationSet
from ..core.session import CalibrationRecord, CalibrationSession, RobotPoses
from ..core.target import TargetSpec
from ..errors import IngestError, ValidationError
from .detections import read_detections
from .detectors import DetectorOptions
from .images import ProgressCallback, detect_in_images, find_images
from .readers import read_calibration
from .robot import read_robot_poses


def session_from_images(
    images: str,
    target: TargetSpec,
    options: Optional[DetectorOptions] = None,
    calibration: Optional[str] = None,
    calibration_format: Optional[str] = None,
    robot_poses: Optional[str] = None,
    robot_convention: str = "gripper2base",
    robot_units: str = "mm",
    recursive: bool = True,
    progress: Optional[ProgressCallback] = None,
    metadata: Optional[Mapping[str, Any]] = None,
) -> CalibrationSession:
    """Detect a target across a folder of images and build a session.

    Args:
        images: A directory of calibration images, or a single image.
        target: The physical target to look for.
        options: Detector settings.
        calibration: Optional path to the calibration already in use, so the
            audit can compare against what was shipped.
        calibration_format: Force a calibration reader by name.
        robot_poses: Optional path to robot poses for hand-eye.
        robot_convention: `"gripper2base"` or `"base2gripper"`.
        robot_units: Length unit of the robot translations.
        recursive: Descend into subdirectories of `images`.
        progress: Per-image callback, for CLI output.
        metadata: Free-form provenance to attach.

    Returns:
        A session ready for `calibsense.refit`.

    Raises:
        IngestError: No image yielded a detection, or the inputs disagree.
    """
    paths = find_images(images, recursive=recursive)
    observations = detect_in_images(paths, target, options, progress)
    prior = _read_prior(calibration, calibration_format, observations)
    robot = _read_robot(robot_poses, robot_convention, robot_units)
    return CalibrationSession(
        observations=observations,
        prior=prior,
        robot=robot,
        metadata=_provenance(metadata, source="images", root=os.path.abspath(images)),
    )


def session_from_calibration(
    calibration: str,
    detections: str,
    target: Optional[TargetSpec] = None,
    image_size: Optional[Tuple[int, int]] = None,
    calibration_format: Optional[str] = None,
    robot_poses: Optional[str] = None,
    robot_convention: str = "gripper2base",
    robot_units: str = "mm",
    metadata: Optional[Mapping[str, Any]] = None,
) -> CalibrationSession:
    """Build a session from an existing calibration and its detections.

    Args:
        calibration: Path to the calibration file to audit.
        detections: Path to the corner detections behind it.
        target: The target, when the detections file does not name one.
        image_size: Frame size, when neither file names one.
        calibration_format: Force a calibration reader by name.
        robot_poses: Optional path to robot poses for hand-eye.
        robot_convention: `"gripper2base"` or `"base2gripper"`.
        robot_units: Length unit of the robot translations.
        metadata: Free-form provenance to attach.

    Returns:
        A session ready for `calibsense.refit`.

    Raises:
        IngestError: The calibration and the detections describe different
            image sizes, which means they are not from the same capture.
    """
    prior = read_calibration(calibration, calibration_format)
    observations = read_detections(
        detections,
        target=target,
        image_size=image_size,
        fallback_image_size=prior.image_size,
    )
    if observations.image_size != prior.image_size:
        raise IngestError(
            f"the calibration is for {prior.image_size} images but the detections "
            f"are {observations.image_size}; these are not the same capture"
        )
    robot = _read_robot(robot_poses, robot_convention, robot_units)
    return CalibrationSession(
        observations=observations,
        prior=prior,
        robot=robot,
        metadata=_provenance(
            metadata,
            source="calibration",
            calibration=os.path.abspath(calibration),
            detections=os.path.abspath(detections),
        ),
    )


def _read_prior(
    path: Optional[str], fmt: Optional[str], observations: ObservationSet
) -> Optional[CalibrationRecord]:
    if path is None:
        return None
    prior = read_calibration(path, fmt)
    if prior.image_size != observations.image_size:
        raise IngestError(
            f"{path} is a calibration for {prior.image_size} images, but the "
            f"detected images are {observations.image_size}"
        )
    return prior


def _read_robot(
    path: Optional[str], convention: str, units: str
) -> Optional[RobotPoses]:
    return None if path is None else read_robot_poses(path, convention, units)


def _provenance(metadata: Optional[Mapping[str, Any]], **fields: Any) -> dict:
    payload = dict(metadata or {})
    payload.setdefault("ingest", fields)
    return payload
