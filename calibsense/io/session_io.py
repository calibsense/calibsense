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

"""Reading and writing calibration sessions.

Views are stored as one concatenated array pair plus offsets rather than one
array per view, so a thousand-view session is still a handful of members inside
the archive.
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np

from ..core.camera import camera_from_dict
from ..core.observations import (
    DetectionSummary,
    ObservationSet,
    ViewObservations,
)
from ..core.poses import poses_from_array, poses_to_array
from ..core.session import CalibrationRecord, CalibrationSession, RobotPoses
from ..core.target import target_from_dict
from ..errors import SerializationError
from .bundle import check_format, read_bundle, write_bundle

FORMAT = "calibsense.session"
FORMAT_VERSION = 1


def save_session(session: CalibrationSession, path: str) -> str:
    """Write a session to a single file.

    Args:
        session: The session to store.
        path: Destination path; `.npz` is appended when missing.

    Returns:
        The path actually written.
    """
    views = session.observations.views
    point_ids = np.concatenate([v.point_ids for v in views])
    image_points = np.concatenate([v.image_points for v in views])
    offsets = np.cumsum([0] + [v.n_points for v in views])

    manifest: Dict[str, Any] = {
        "format": FORMAT,
        "format_version": FORMAT_VERSION,
        "created": session.created,
        "calibsense_version": session.calibsense_version,
        "metadata": dict(session.metadata),
        "target": session.observations.target.to_dict(),
        "image_size": list(session.observations.image_size),
        "views": [v.to_manifest() for v in views],
        "summary": session.observations.summary.to_dict(),
        "prior": session.prior.to_dict() if session.prior else None,
        "robot": session.robot.to_manifest() if session.robot else None,
    }
    arrays = {
        "point_ids": point_ids.astype(np.int64),
        "image_points": image_points.astype(np.float64),
        "view_offsets": offsets.astype(np.int64),
    }
    if session.robot is not None:
        arrays["robot_poses"] = poses_to_array(session.robot.poses)
    return write_bundle(path, manifest, arrays)


def load_session(path: str) -> CalibrationSession:
    """Read a session written by `save_session`.

    Args:
        path: Path to the bundle.

    Returns:
        The reconstructed session.

    Raises:
        SerializationError: The file is not a session bundle or is inconsistent.
    """
    manifest, arrays = read_bundle(path)
    check_format(manifest, FORMAT, FORMAT_VERSION)
    try:
        offsets = arrays["view_offsets"]
        point_ids = arrays["point_ids"]
        image_points = arrays["image_points"]
        view_meta = manifest["views"]
    except KeyError as exc:
        raise SerializationError(f"session bundle is missing {exc}") from exc

    if len(view_meta) != len(offsets) - 1:
        raise SerializationError(
            f"{len(view_meta)} view records but {len(offsets) - 1} offset spans"
        )
    views = []
    for index, meta in enumerate(view_meta):
        start, stop = int(offsets[index]), int(offsets[index + 1])
        views.append(
            ViewObservations(
                view_id=meta["view_id"],
                point_ids=point_ids[start:stop],
                image_points=image_points[start:stop],
                source=meta.get("source"),
                metadata=meta.get("metadata", {}),
            )
        )
    observations = ObservationSet(
        target=target_from_dict(manifest["target"]),
        image_size=tuple(manifest["image_size"]),
        views=tuple(views),
        summary=DetectionSummary.from_dict(manifest.get("summary", {})),
    )
    prior = (
        CalibrationRecord.from_dict(manifest["prior"]) if manifest.get("prior") else None
    )
    robot = None
    if manifest.get("robot"):
        robot = RobotPoses(
            poses=poses_from_array(arrays["robot_poses"]),
            view_ids=tuple(manifest["robot"]["view_ids"]),
            source=manifest["robot"].get("source", "unknown"),
        )
    return CalibrationSession(
        observations=observations,
        prior=prior,
        robot=robot,
        created=manifest.get("created", ""),
        calibsense_version=manifest.get("calibsense_version", ""),
        metadata=manifest.get("metadata", {}),
    )
