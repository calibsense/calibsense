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

"""Synthetic captures with known truth.

Two jobs. It is the backbone of the test suite, because a claim about
uncertainty can only be checked against a case whose true parameters are known.
It is also what a later milestone needs in order to answer "what would happen if
you added six views at 400 mm and 1200 mm" — that question is a covariance
prediction over a capture that does not exist yet, and predicting it means being
able to generate it.

Everything is deterministic given a seed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np

from .core.camera import CameraModel
from .core.observations import DetectionSummary, ObservationSet, ViewObservations
from .core.poses import Pose, rotvec_to_matrix
from .core.target import TargetSpec
from .errors import ValidationError
from .refit.projection import projector_for


def _rng(seed: Optional[int]) -> np.random.Generator:
    return np.random.default_rng(seed)


def pose_for_view(
    target: TargetSpec,
    distance_mm: float,
    tilt_rad: float = 0.0,
    tilt_axis_rad: float = 0.0,
    roll_rad: float = 0.0,
    offset_mm: Sequence[float] = (0.0, 0.0),
) -> Pose:
    """Place a target in front of the camera at a chosen distance and tilt.

    The board's centre, not its origin corner, is what gets placed, so a tilt
    rotates the board about its own middle and keeps it in frame.

    Args:
        target: The target being placed.
        distance_mm: Distance from the camera to the board centre, along the
            optical axis, in millimetres.
        tilt_rad: Angle between the board normal and the optical axis.
        tilt_axis_rad: Direction of the tilt within the image plane.
        roll_rad: Rotation of the board about the optical axis.
        offset_mm: Lateral `(x, y)` offset of the board centre, in millimetres.

    Returns:
        The board-to-camera pose.

    Raises:
        ValidationError: The distance is not positive.
    """
    if not np.isfinite(distance_mm) or distance_mm <= 0:
        raise ValidationError(f"distance must be positive, got {distance_mm}")
    axis = np.array([np.cos(tilt_axis_rad), np.sin(tilt_axis_rad), 0.0]) * tilt_rad
    rotation = rotvec_to_matrix([0.0, 0.0, roll_rad]) @ rotvec_to_matrix(axis)
    centre = target.object_points().mean(axis=0)
    translation = np.array(
        [float(offset_mm[0]), float(offset_mm[1]), float(distance_mm)]
    ) - rotation @ centre
    return Pose(rotation, translation)


def frontoparallel_poses(
    target: TargetSpec,
    n_views: int,
    distance_mm: float = 800.0,
    lateral_mm: float = 40.0,
    seed: Optional[int] = 0,
) -> List[Pose]:
    """The degenerate capture: every view flat-on at one distance.

    This is the common real-world case and the one that makes reprojection error
    lie. Focal length and distance trade off almost freely, so the fit is
    excellent and the metric scale is not determined.

    Args:
        target: The target to place.
        n_views: How many views to generate.
        distance_mm: The single working distance.
        lateral_mm: How far the board wanders sideways between views.
        seed: Random seed.

    Returns:
        One pose per view.
    """
    rng = _rng(seed)
    return [
        pose_for_view(
            target,
            distance_mm,
            tilt_rad=rng.uniform(0.0, 0.02),
            tilt_axis_rad=rng.uniform(0.0, 2 * np.pi),
            roll_rad=rng.uniform(-0.05, 0.05),
            offset_mm=rng.uniform(-lateral_mm, lateral_mm, 2),
        )
        for _ in range(n_views)
    ]


def diverse_poses(
    target: TargetSpec,
    n_views: int,
    distances_mm: Sequence[float] = (400.0, 800.0, 1200.0),
    max_tilt_rad: float = 0.6,
    lateral_mm: float = 120.0,
    seed: Optional[int] = 0,
) -> List[Pose]:
    """A well-conditioned capture: several distances, real tilt, wide coverage.

    Args:
        target: The target to place.
        n_views: How many views to generate.
        distances_mm: Working distances, cycled across the views.
        max_tilt_rad: Largest board tilt.
        lateral_mm: How far the board wanders sideways between views.
        seed: Random seed.

    Returns:
        One pose per view.
    """
    rng = _rng(seed)
    distances = list(distances_mm)
    if not distances:
        raise ValidationError("need at least one working distance")
    poses = []
    for index in range(n_views):
        distance = distances[index % len(distances)]
        scale = distance / max(distances)
        poses.append(
            pose_for_view(
                target,
                distance,
                tilt_rad=rng.uniform(0.15, max_tilt_rad),
                tilt_axis_rad=rng.uniform(0.0, 2 * np.pi),
                roll_rad=rng.uniform(-np.pi, np.pi),
                offset_mm=rng.uniform(-lateral_mm, lateral_mm, 2) * scale,
            )
        )
    return poses


@dataclass(frozen=True)
class SyntheticCapture:
    """A generated capture together with the truth that produced it.

    Attributes:
        observations: The detections, ready to feed to a refit.
        camera: The true intrinsics.
        poses: The true board-to-camera poses, one per kept view.
        noise_px: Standard deviation of the noise added to each coordinate.
    """

    observations: ObservationSet
    camera: CameraModel
    poses: Tuple[Pose, ...]
    noise_px: float


def synthesise(
    camera: CameraModel,
    target: TargetSpec,
    poses: Sequence[Pose],
    image_size: Tuple[int, int] = (1280, 720),
    noise_px: float = 0.0,
    min_points: int = 8,
    seed: Optional[int] = 0,
    keep_out_of_frame: bool = False,
) -> SyntheticCapture:
    """Project a target through a known camera to produce detections.

    Points that fall outside the frame are dropped, and views left with too few
    points are dropped whole, so the generated capture behaves like a real one
    rather than an idealised full grid every time.

    Args:
        camera: The true intrinsics.
        target: The target to project.
        poses: Board-to-camera poses, one per intended view.
        image_size: Frame size as `(width, height)`.
        noise_px: Standard deviation of independent Gaussian noise added to each
            image coordinate, in pixels.
        min_points: Fewest surviving points for a view to be kept.
        seed: Random seed for the noise.
        keep_out_of_frame: Keep points outside the frame instead of dropping
            them. Useful for isolating geometry from visibility in a test.

    Returns:
        The capture and the truth behind it.

    Raises:
        ValidationError: No view survived, so there is nothing to calibrate.
    """
    rng = _rng(seed)
    projector = projector_for(camera)
    width, height = image_size
    all_ids = np.arange(target.num_points)
    object_points = target.object_points()

    views: List[ViewObservations] = []
    kept_poses: List[Pose] = []
    for index, pose in enumerate(poses):
        in_front = pose.apply(object_points)[:, 2] > 1e-6
        if not np.any(in_front):
            continue
        projected = projector.project(camera, pose, object_points[in_front])
        if noise_px > 0:
            projected = projected + rng.normal(0.0, noise_px, projected.shape)
        ids = all_ids[in_front]
        if not keep_out_of_frame:
            visible = (
                (projected[:, 0] >= 0)
                & (projected[:, 1] >= 0)
                & (projected[:, 0] <= width - 1)
                & (projected[:, 1] <= height - 1)
            )
            projected, ids = projected[visible], ids[visible]
        if ids.size < min_points:
            continue
        views.append(
            ViewObservations(
                view_id=f"synthetic{index:04d}",
                point_ids=ids,
                image_points=projected,
                metadata={"synthetic": True},
            )
        )
        kept_poses.append(pose)

    if not views:
        raise ValidationError(
            f"no view kept at least {min_points} visible points; the target is "
            "out of frame or behind the camera in every pose"
        )
    return SyntheticCapture(
        observations=ObservationSet(
            target=target,
            image_size=(int(width), int(height)),
            views=tuple(views),
            summary=DetectionSummary(
                attempted=len(poses), detector=f"synthetic:{target.kind}"
            ),
        ),
        camera=camera,
        poses=tuple(kept_poses),
        noise_px=float(noise_px),
    )
