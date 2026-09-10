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

"""The measurement tasks.

Each is specified as a *scene* rather than as a set of observations, because the
question an engineer asks before building a cell is "what error will I get at
800 mm", not "what error did I get on this frame". The scene is projected once
through the fitted calibration to produce the pixels a sensor would have seen,
and measured back under every sampled calibration.

Where a task depends on something calibsense did not calibrate — a stereo baseline
supplied by the user, a hand-eye transform from elsewhere — that input is treated
as exact and the task says so, because propagating an uncertainty nobody
measured would be worse than admitting it is missing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, ClassVar, Dict, Optional, Sequence, Tuple

import numpy as np

from ..core.poses import Pose, rotvec_to_matrix
from ..errors import ValidationError
from ..refit.projection import projector_for
from .base import Quantity, Task
from .sampling import ParameterSample


def _positive(value: float, name: str) -> float:
    if not np.isfinite(value) or value <= 0:
        raise ValidationError(f"{name} must be a positive finite length, got {value}")
    return float(value)


@dataclass(frozen=True)
class LengthAtDepth(Task):
    """Measure the distance between two image points at a known depth.

    The headline task, and the simplest one that produces a millimetre. Two
    features sit on a plane perpendicular to the optical axis at `depth_mm`,
    separated by `length_mm`. Their pixels come from the fitted calibration;
    each sampled calibration then lifts those pixels back onto the plane and
    measures the separation.

    To first order the relative error in the length equals the relative error in
    the focal length, which is why this task is so sensitive to exactly the
    parameter a frontoparallel capture fails to determine.

    Attributes:
        depth_mm: Distance from the camera to the plane along the optical axis.
        length_mm: True separation of the two features.
        centre_mm: Where the midpoint sits on the plane, as `(x, y)` in
            millimetres from the optical axis. Off-axis measurements probe the
            distortion, on-axis ones probe the focal length.
        orientation_deg: Direction of the separation within the plane, measured
            from the image x axis.
    """

    depth_mm: float = 800.0
    length_mm: float = 100.0
    centre_mm: Tuple[float, float] = (0.0, 0.0)
    orientation_deg: float = 0.0

    kind: ClassVar[str] = "length_at_depth"
    title: ClassVar[str] = "Length at a known depth"

    def __post_init__(self) -> None:
        object.__setattr__(self, "depth_mm", _positive(self.depth_mm, "depth_mm"))
        object.__setattr__(self, "length_mm", _positive(self.length_mm, "length_mm"))
        object.__setattr__(
            self, "centre_mm", (float(self.centre_mm[0]), float(self.centre_mm[1]))
        )
        object.__setattr__(self, "orientation_deg", float(self.orientation_deg))

    def quantities(self) -> Tuple[Quantity, ...]:
        """The measured separation."""
        return (
            Quantity(
                "length_mm",
                "mm",
                f"a {self.length_mm:g} mm feature measured at {self.depth_mm:g} mm",
            ),
        )

    def scene_points(self) -> np.ndarray:
        """The two feature positions in camera-frame millimetres."""
        angle = np.radians(self.orientation_deg)
        offset = 0.5 * self.length_mm * np.array([np.cos(angle), np.sin(angle)])
        centre = np.asarray(self.centre_mm, dtype=float)
        return np.array([
            [centre[0] - offset[0], centre[1] - offset[1], self.depth_mm],
            [centre[0] + offset[0], centre[1] + offset[1], self.depth_mm],
        ])

    def observe(self, sample: ParameterSample) -> np.ndarray:
        """Project both features through the fitted camera."""
        projector = projector_for(sample.camera)
        return projector.project(sample.camera, Pose.identity(), self.scene_points())

    def measure(self, sample: ParameterSample, image_points: np.ndarray) -> np.ndarray:
        """Lift both pixels onto the plane and measure their separation."""
        projector = projector_for(sample.camera)
        lifted = projector.backproject(sample.camera, image_points, self.depth_mm)
        return np.array([float(np.linalg.norm(lifted[1] - lifted[0]))])

    def describe(self) -> str:
        """Human description including the geometry."""
        return (
            f"Measuring a {self.length_mm:g} mm feature at a working distance of "
            f"{self.depth_mm:g} mm, centred {np.linalg.norm(self.centre_mm):g} mm "
            f"off axis, oriented {self.orientation_deg:g} degrees in the image plane"
        )

    def to_dict(self) -> Dict[str, Any]:
        """Serialise the task's configuration."""
        payload = super().to_dict()
        payload.update(
            depth_mm=self.depth_mm,
            length_mm=self.length_mm,
            centre_mm=list(self.centre_mm),
            orientation_deg=self.orientation_deg,
        )
        return payload


@dataclass(frozen=True)
class PlaneLocation(Task):
    """Locate a plane from a grid of features on it.

    A pose is solved from the observed grid under each sampled calibration, and
    two things are reported: how far the plane is, measured perpendicular to
    itself, and which way it faces. These fail differently — a focal-length
    error moves the plane without tilting it, while a principal-point error
    tilts it without moving it much — so both are worth separating.

    Attributes:
        depth_mm: Distance from the camera to the plane's centre.
        tilt_deg: Angle between the plane's normal and the optical axis.
        extent_mm: Side length of the square grid of features.
        grid: Features across and down.
    """

    depth_mm: float = 800.0
    tilt_deg: float = 25.0
    extent_mm: float = 200.0
    grid: Tuple[int, int] = (5, 5)

    kind: ClassVar[str] = "plane_location"
    title: ClassVar[str] = "Plane location"

    def __post_init__(self) -> None:
        object.__setattr__(self, "depth_mm", _positive(self.depth_mm, "depth_mm"))
        object.__setattr__(self, "extent_mm", _positive(self.extent_mm, "extent_mm"))
        object.__setattr__(self, "tilt_deg", float(self.tilt_deg))
        across, down = (int(v) for v in self.grid)
        if across < 2 or down < 2:
            raise ValidationError(f"grid must be at least 2x2, got {self.grid}")
        object.__setattr__(self, "grid", (across, down))

    def quantities(self) -> Tuple[Quantity, ...]:
        """Perpendicular distance to the plane, and its tilt."""
        return (
            Quantity(
                "distance_mm", "mm",
                f"the perpendicular distance to a plane at {self.depth_mm:g} mm",
            ),
            Quantity(
                "tilt_deg", "deg",
                f"the orientation of a plane at {self.depth_mm:g} mm",
            ),
        )

    def plane_points(self) -> np.ndarray:
        """Feature positions in the plane's own frame, in millimetres."""
        across, down = self.grid
        half = 0.5 * self.extent_mm
        x, y = np.meshgrid(
            np.linspace(-half, half, across), np.linspace(-half, half, down)
        )
        return np.column_stack([x.ravel(), y.ravel(), np.zeros(x.size)])

    def nominal_pose(self) -> Pose:
        """The plane's true pose in the camera frame."""
        rotation = rotvec_to_matrix([np.radians(self.tilt_deg), 0.0, 0.0])
        return Pose(rotation, [0.0, 0.0, self.depth_mm])

    def observe(self, sample: ParameterSample) -> np.ndarray:
        """Project the grid through the fitted camera."""
        projector = projector_for(sample.camera)
        return projector.project(sample.camera, self.nominal_pose(), self.plane_points())

    def measure(self, sample: ParameterSample, image_points: np.ndarray) -> np.ndarray:
        """Solve the plane's pose, then report its distance and tilt."""
        projector = projector_for(sample.camera)
        pose = projector.solve_pose(sample.camera, self.plane_points(), image_points)
        normal = pose.rotation[:, 2]
        # The plane's centre is its frame origin, so the perpendicular distance
        # from the camera to the plane is the projection of the translation onto
        # the normal.
        distance = abs(float(normal @ pose.translation))
        tilt = float(np.degrees(np.arccos(np.clip(abs(normal[2]), 0.0, 1.0))))
        return np.array([distance, tilt])

    def describe(self) -> str:
        """Human description including the geometry."""
        across, down = self.grid
        return (
            f"Locating a plane at {self.depth_mm:g} mm, tilted {self.tilt_deg:g} "
            f"degrees, from a {across}x{down} grid of features spanning "
            f"{self.extent_mm:g} mm"
        )

    def to_dict(self) -> Dict[str, Any]:
        """Serialise the task's configuration."""
        payload = super().to_dict()
        payload.update(
            depth_mm=self.depth_mm,
            tilt_deg=self.tilt_deg,
            extent_mm=self.extent_mm,
            grid=list(self.grid),
        )
        return payload


@dataclass(frozen=True)
class StereoTriangulation(Task):
    """Triangulate a point from two camera positions.

    One camera at two places, or two identical cameras — either way calibsense
    calibrated one set of intrinsics and both views use it, which is the usual
    arrangement and also the one where an intrinsic error does not cancel.

    The baseline is **treated as exact.** It is not something calibsense measured,
    so propagating an invented uncertainty for it would be worse than saying so;
    for a stereo rig the baseline's own uncertainty usually dominates, and it has
    to come from a stereo calibration this milestone does not do.

    Attributes:
        baseline_mm: Separation of the two camera positions along their shared
            x axis.
        depth_mm: Distance to the point along the first camera's optical axis.
        lateral_mm: The point's `(x, y)` offset from the first camera's axis.
    """

    baseline_mm: float = 200.0
    depth_mm: float = 800.0
    lateral_mm: Tuple[float, float] = (0.0, 0.0)

    kind: ClassVar[str] = "stereo_triangulation"
    title: ClassVar[str] = "Stereo triangulation"

    def __post_init__(self) -> None:
        object.__setattr__(self, "baseline_mm", _positive(self.baseline_mm, "baseline_mm"))
        object.__setattr__(self, "depth_mm", _positive(self.depth_mm, "depth_mm"))
        object.__setattr__(
            self, "lateral_mm", (float(self.lateral_mm[0]), float(self.lateral_mm[1]))
        )

    def quantities(self) -> Tuple[Quantity, ...]:
        """Depth error and total position error."""
        return (
            Quantity(
                "depth_mm", "mm",
                f"the triangulated depth of a point at {self.depth_mm:g} mm on a "
                f"{self.baseline_mm:g} mm baseline",
            ),
            Quantity(
                "range_mm", "mm",
                f"the triangulated range to a point at {self.depth_mm:g} mm on a "
                f"{self.baseline_mm:g} mm baseline",
            ),
        )

    def baseline(self) -> Pose:
        """The transform taking first-camera coordinates into the second camera."""
        return Pose(np.eye(3), [-self.baseline_mm, 0.0, 0.0])

    def scene_point(self) -> np.ndarray:
        """The point's position in the first camera's frame, in millimetres."""
        return np.array([self.lateral_mm[0], self.lateral_mm[1], self.depth_mm])

    def observe(self, sample: ParameterSample) -> np.ndarray:
        """Project the point into both cameras through the fitted intrinsics."""
        projector = projector_for(sample.camera)
        point = self.scene_point().reshape(1, 3)
        left = projector.project(sample.camera, Pose.identity(), point)
        right = projector.project(sample.camera, self.baseline(), point)
        return np.vstack([left, right])

    def measure(self, sample: ParameterSample, image_points: np.ndarray) -> np.ndarray:
        """Triangulate, then report the depth and the range."""
        projector = projector_for(sample.camera)
        rays = projector.rays(sample.camera, image_points)
        baseline = self.baseline()
        # Both rays expressed in the first camera's frame, with their origins.
        origins = np.array([[0.0, 0.0, 0.0], baseline.inverse().translation])
        directions = np.array([rays[0], baseline.rotation.T @ rays[1]])
        point = _closest_approach(origins, directions)
        return np.array([float(point[2]), float(np.linalg.norm(point))])

    def describe(self) -> str:
        """Human description including the geometry."""
        return (
            f"Triangulating a point at {self.depth_mm:g} mm from a "
            f"{self.baseline_mm:g} mm baseline, with the baseline treated as "
            "exact because calibsense did not measure it"
        )

    def to_dict(self) -> Dict[str, Any]:
        """Serialise the task's configuration."""
        payload = super().to_dict()
        payload.update(
            baseline_mm=self.baseline_mm,
            depth_mm=self.depth_mm,
            lateral_mm=list(self.lateral_mm),
            baseline_is_exact=True,
        )
        return payload


def _closest_approach(origins: np.ndarray, directions: np.ndarray) -> np.ndarray:
    """The point minimising the squared distance to a set of rays.

    Args:
        origins: An `(m, 3)` array of ray origins.
        directions: An `(m, 3)` array of unit ray directions.

    Returns:
        The `(3,)` least-squares intersection.

    Raises:
        ValidationError: The rays are parallel, so no intersection is determined.
    """
    normal = np.eye(3) - np.einsum("mi,mj->mij", directions, directions)
    matrix = normal.sum(axis=0)
    vector = np.einsum("mij,mj->i", normal, origins)
    eigenvalues = np.linalg.eigvalsh(matrix)
    if eigenvalues[-1] <= 0 or eigenvalues[0] / eigenvalues[-1] < 1e-12:
        raise ValidationError(
            "the rays are parallel to within numerical precision, so no "
            "intersection is determined; widen the baseline"
        )
    return np.linalg.solve(matrix, vector)


@dataclass(frozen=True)
class CameraToBase(Task):
    """Transform a point from the camera frame into the robot base frame.

    The measurement a robot cell actually makes: a feature is seen at some
    depth, and the arm has to be told where it is. Three uncertainties reach the
    answer — the intrinsics that turn a pixel into a ray, the hand-eye transform
    that carries the camera frame onto the arm, and the pixel noise at
    measurement time — and the hand-eye contribution usually dominates.

    The robot's flange pose at measurement time is treated as exact. Robot
    repeatability is a specification of the arm, not something calibsense
    measured, and inventing a number for it would be worse than saying so.

    Attributes:
        hand_eye_result: The solved hand-eye transform, whose covariance is
            sampled alongside the intrinsics.
        depth_mm: Depth of the feature along the camera's optical axis.
        lateral_mm: The feature's `(x, y)` offset from the optical axis.
        flange: The robot's flange-to-base pose at measurement time. Required
            for an eye-in-hand mounting, ignored for eye-to-hand where the
            camera does not move with the arm.
    """

    hand_eye_result: Any = None
    depth_mm: float = 800.0
    lateral_mm: Tuple[float, float] = (0.0, 0.0)
    flange: Optional[Pose] = None

    kind: ClassVar[str] = "camera_to_base"
    title: ClassVar[str] = "Camera to robot base"

    def __post_init__(self) -> None:
        if self.hand_eye_result is None:
            raise ValidationError(
                "this task needs a solved hand-eye transform; run "
                "calibsense.handeye.solve_hand_eye first"
            )
        object.__setattr__(self, "depth_mm", _positive(self.depth_mm, "depth_mm"))
        object.__setattr__(
            self, "lateral_mm", (float(self.lateral_mm[0]), float(self.lateral_mm[1]))
        )
        if self.hand_eye_result.mounting == "eye_in_hand" and self.flange is None:
            raise ValidationError(
                "an eye-in-hand mounting needs the flange pose at measurement "
                "time, because the camera moves with the arm"
            )

    def hand_eye(self):
        """The hand-eye result whose covariance is sampled."""
        return self.hand_eye_result

    def quantities(self) -> Tuple[Quantity, ...]:
        """Per-axis and total position error in the base frame."""
        return (
            Quantity("x_mm", "mm", f"the base-frame x of a feature at {self.depth_mm:g} mm"),
            Quantity("y_mm", "mm", f"the base-frame y of a feature at {self.depth_mm:g} mm"),
            Quantity("z_mm", "mm", f"the base-frame z of a feature at {self.depth_mm:g} mm"),
            Quantity(
                "position_mm", "mm",
                f"the base-frame position of a feature at {self.depth_mm:g} mm",
            ),
        )

    def scene_point(self) -> np.ndarray:
        """The feature's position in the camera frame, in millimetres."""
        return np.array([self.lateral_mm[0], self.lateral_mm[1], self.depth_mm])

    def observe(self, sample: ParameterSample) -> np.ndarray:
        """Project the feature through the fitted camera."""
        projector = projector_for(sample.camera)
        return projector.project(
            sample.camera, Pose.identity(), self.scene_point().reshape(1, 3)
        )

    def measure(self, sample: ParameterSample, image_points: np.ndarray) -> np.ndarray:
        """Backproject the pixel, then carry it into the base frame."""
        projector = projector_for(sample.camera)
        in_camera = projector.backproject(sample.camera, image_points, self.depth_mm)[0]
        transform = sample.hand_eye or self.hand_eye_result.camera
        if self.hand_eye_result.mounting == "eye_in_hand":
            # base <- flange <- camera
            to_base = self.flange.compose(transform)
        else:
            # base <- camera directly; the camera is bolted to the cell
            to_base = transform
        in_base = to_base.apply(in_camera.reshape(1, 3))[0]
        return np.array([
            in_base[0], in_base[1], in_base[2], float(np.linalg.norm(in_base))
        ])

    def describe(self) -> str:
        """Human description including the mounting and what is held exact."""
        mounting = self.hand_eye_result.mounting.replace("_", "-")
        return (
            f"Locating a feature at {self.depth_mm:g} mm in the robot base frame "
            f"through a {mounting} hand-eye transform, with the flange pose "
            "treated as exact because robot repeatability is not something "
            "calibsense measured"
        )

    def to_dict(self) -> Dict[str, Any]:
        """Serialise the task's configuration."""
        payload = super().to_dict()
        payload.update(
            depth_mm=self.depth_mm,
            lateral_mm=list(self.lateral_mm),
            mounting=self.hand_eye_result.mounting,
            flange=None if self.flange is None else self.flange.matrix.tolist(),
            flange_is_exact=True,
        )
        return payload
