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

"""The unit of work: everything calibsense needs to audit one camera.

A session is what M1 produces and every later milestone consumes. It is
deliberately a plain data bundle — detections, optionally the calibration the
engineer already has, optionally the robot poses that go with the views — so
that the expensive part of an audit is done once, saved, and re-analysed
without touching the images again.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np

from .._version import __version__
from ..errors import ValidationError
from .camera import CameraModel, camera_from_dict
from .observations import ObservationSet
from .poses import Pose

#: Hand-eye conventions calibsense accepts on input. Everything is stored as
#: gripper-to-base, which is what `cv2.calibrateHandEye` expects.
HAND_EYE_CONVENTIONS = ("gripper2base", "base2gripper")


@dataclass(frozen=True, eq=False)
class CalibrationRecord:
    """A calibration that already exists, as read from a file.

    Attributes:
        camera: The intrinsic model.
        image_size: Frame size as `(width, height)` in pixels.
        source: Where it came from, for provenance in the report.
        reported_rms: The reprojection RMS the producing tool claimed, if it
            recorded one. This is the number the audit is about to contradict.
        metadata: Anything else the reader recovered.
    """

    camera: CameraModel
    image_size: Tuple[int, int]
    source: str = "unknown"
    reported_rms: Optional[float] = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        width, height = (int(v) for v in self.image_size)
        if width <= 0 or height <= 0:
            raise ValidationError(f"image size must be positive, got {(width, height)}")
        if self.reported_rms is not None:
            rms = float(self.reported_rms)
            if not np.isfinite(rms) or rms < 0:
                raise ValidationError(f"reported_rms must be non-negative, got {rms}")
            object.__setattr__(self, "reported_rms", rms)
        object.__setattr__(self, "image_size", (width, height))
        object.__setattr__(self, "metadata", dict(self.metadata))

    def to_dict(self) -> Dict[str, Any]:
        """Serialise to a plain JSON-compatible dictionary."""
        return {
            "camera": self.camera.to_dict(),
            "image_size": list(self.image_size),
            "source": self.source,
            "reported_rms": self.reported_rms,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "CalibrationRecord":
        """Rebuild from `to_dict` output."""
        return cls(
            camera=camera_from_dict(payload["camera"]),
            image_size=tuple(payload["image_size"]),
            source=payload.get("source", "unknown"),
            reported_rms=payload.get("reported_rms"),
            metadata=payload.get("metadata", {}),
        )


@dataclass(frozen=True, eq=False)
class RobotPoses:
    """Robot poses attached to views, for a later hand-eye calibration.

    Poses are normalised to gripper-to-base on construction, so a caller that
    exported base-to-gripper does not have to remember which way round their
    logger wrote them.

    Attributes:
        poses: One pose per entry of `view_ids`, gripper-to-base.
        view_ids: The views these poses belong to.
        source: Where the poses came from.
    """

    poses: Tuple[Pose, ...]
    view_ids: Tuple[str, ...]
    source: str = "unknown"

    def __post_init__(self) -> None:
        poses = tuple(self.poses)
        view_ids = tuple(str(v) for v in self.view_ids)
        if len(poses) != len(view_ids):
            raise ValidationError(
                f"{len(poses)} robot poses for {len(view_ids)} view ids"
            )
        if not poses:
            raise ValidationError("robot poses cannot be empty")
        if len(set(view_ids)) != len(view_ids):
            raise ValidationError("a view id is repeated in the robot poses")
        object.__setattr__(self, "poses", poses)
        object.__setattr__(self, "view_ids", view_ids)

    @classmethod
    def from_matrices(
        cls,
        matrices: Sequence[np.ndarray],
        view_ids: Sequence[str],
        convention: str = "gripper2base",
        source: str = "unknown",
    ) -> "RobotPoses":
        """Build from 4x4 transforms in either hand-eye convention.

        Args:
            matrices: One `(4, 4)` homogeneous transform per view.
            view_ids: The view each transform belongs to.
            convention: Either `"gripper2base"` or `"base2gripper"`.
            source: Provenance string.

        Returns:
            Poses stored as gripper-to-base.

        Raises:
            ValidationError: The convention is unknown or a matrix is not a
                valid rigid transform.
        """
        if convention not in HAND_EYE_CONVENTIONS:
            raise ValidationError(
                f"unknown hand-eye convention {convention!r}; "
                f"expected one of {list(HAND_EYE_CONVENTIONS)}"
            )
        poses = [Pose.from_matrix(m) for m in matrices]
        if convention == "base2gripper":
            poses = [p.inverse() for p in poses]
        return cls(tuple(poses), tuple(view_ids), source)

    def aligned_with(self, observations: ObservationSet) -> Tuple[Pose, ...]:
        """Poses reordered to match an observation set, one per view.

        Args:
            observations: The views to align to.

        Returns:
            One pose per view, in view order.

        Raises:
            ValidationError: A view has no pose.
        """
        lookup = dict(zip(self.view_ids, self.poses))
        missing = [v for v in observations.view_ids if v not in lookup]
        if missing:
            raise ValidationError(
                f"{len(missing)} view(s) have no robot pose, first is {missing[0]!r}"
            )
        return tuple(lookup[v] for v in observations.view_ids)

    def to_manifest(self) -> Dict[str, Any]:
        """The non-numeric payload, for the JSON side of persistence."""
        return {"view_ids": list(self.view_ids), "source": self.source}


@dataclass(frozen=True, eq=False)
class CalibrationSession:
    """Everything calibsense needs to audit one camera.

    Attributes:
        observations: The detected target points.
        prior: The calibration the engineer already has, if any.
        robot: Robot poses for hand-eye, if any.
        created: ISO-8601 UTC timestamp of ingest.
        calibsense_version: Version that produced the session.
        metadata: Free-form provenance.
    """

    observations: ObservationSet
    prior: Optional[CalibrationRecord] = None
    robot: Optional[RobotPoses] = None
    created: str = ""
    calibsense_version: str = __version__
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.created:
            object.__setattr__(
                self, "created", datetime.now(timezone.utc).isoformat(timespec="seconds")
            )
        if self.prior is not None and self.prior.image_size != self.observations.image_size:
            raise ValidationError(
                f"prior calibration is for {self.prior.image_size} images but the "
                f"detections are {self.observations.image_size}"
            )
        if self.robot is not None:
            # Fail here rather than at hand-eye time, when the images are gone.
            self.robot.aligned_with(self.observations)
        object.__setattr__(self, "metadata", dict(self.metadata))

    @property
    def target(self):
        """The calibration target, a shortcut through the observation set."""
        return self.observations.target

    @property
    def image_size(self) -> Tuple[int, int]:
        """Frame size as `(width, height)`, a shortcut through the observations."""
        return self.observations.image_size

    @property
    def has_hand_eye(self) -> bool:
        """Whether robot poses are attached."""
        return self.robot is not None

    def summary_lines(self) -> Tuple[str, ...]:
        """A short human summary, used by the CLI and by reports."""
        observations = self.observations
        points = observations.points_per_view()
        lines = [
            f"target       {observations.target.describe()}",
            f"image size   {observations.image_size[0]} x {observations.image_size[1]}",
            f"views        {observations.n_views}",
            f"points       {observations.total_points} total, "
            f"{points.min()}-{points.max()} per view "
            f"(median {int(np.median(points))})",
        ]
        summary = observations.summary
        if summary.attempted:
            lines.append(
                f"detection    {summary.succeeded}/{summary.attempted} images "
                f"({summary.success_rate:.0%}) via {summary.detector}"
            )
        if self.prior is not None:
            rms = self.prior.reported_rms
            reported = f", reported RMS {rms:.4f} px" if rms is not None else ""
            lines.append(f"prior        {self.prior.source}{reported}")
        if self.robot is not None:
            lines.append(f"robot poses  {len(self.robot.poses)} from {self.robot.source}")
        return tuple(lines)
