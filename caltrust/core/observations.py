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

"""Detected target points, per view.

Every view carries its own point ids rather than a fixed-length grid. That is
mandatory for ChArUco, where partial detections are the normal case, and it
costs nothing for a checkerboard, where the ids happen to be `0..n-1` every
time. Keeping one representation means the refit code never branches on target
type.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple

import numpy as np

from ..errors import ValidationError
from .target import TargetSpec


@dataclass(frozen=True, eq=False)
class ViewObservations:
    """Points detected in one image.

    Attributes:
        view_id: Stable identifier, unique within an `ObservationSet`. Usually
            the image filename stem.
        point_ids: Target point ids, one per detected point.
        image_points: Pixel coordinates, shape `(n, 2)`, in the same order.
        source: Where the view came from, typically an image path.
        metadata: Detector-specific extras, kept for reporting only.
    """

    view_id: str
    point_ids: np.ndarray
    image_points: np.ndarray
    source: Optional[str] = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        ids = np.asarray(self.point_ids, dtype=np.int64).reshape(-1)
        points = np.asarray(self.image_points, dtype=float).reshape(-1, 2)
        if ids.size != points.shape[0]:
            raise ValidationError(
                f"view {self.view_id!r}: {ids.size} point ids for "
                f"{points.shape[0]} image points"
            )
        if ids.size == 0:
            raise ValidationError(f"view {self.view_id!r} has no points")
        if np.unique(ids).size != ids.size:
            raise ValidationError(f"view {self.view_id!r} repeats a point id")
        if ids.min() < 0:
            raise ValidationError(f"view {self.view_id!r} has a negative point id")
        if not np.all(np.isfinite(points)):
            raise ValidationError(f"view {self.view_id!r} has non-finite image points")
        object.__setattr__(self, "point_ids", ids)
        object.__setattr__(self, "image_points", points)
        object.__setattr__(self, "view_id", str(self.view_id))
        object.__setattr__(self, "metadata", dict(self.metadata))

    @property
    def n_points(self) -> int:
        """Number of detected points in this view."""
        return int(self.point_ids.size)

    def object_points(self, target: TargetSpec) -> np.ndarray:
        """Board-frame coordinates of this view's points, in millimetres.

        Args:
            target: The target these points were detected on.

        Returns:
            An `(n, 3)` array aligned with `image_points`.
        """
        return target.object_points(self.point_ids)

    def centroid(self) -> np.ndarray:
        """Mean image position of the detected points, in pixels."""
        return self.image_points.mean(axis=0)

    def area_fraction(self, image_size: Tuple[int, int]) -> float:
        """Fraction of the frame spanned by the detection's bounding box.

        A small value means corner localisation noise dominates, which is one of
        the causes a calibration audit has to be able to name.

        Args:
            image_size: The `(width, height)` of the frame in pixels.

        Returns:
            Bounding-box area divided by frame area, clipped to `[0, 1]`.
        """
        width, height = image_size
        span = self.image_points.max(axis=0) - self.image_points.min(axis=0)
        return float(np.clip((span[0] * span[1]) / (width * height), 0.0, 1.0))

    def to_arrays(self) -> Dict[str, np.ndarray]:
        """The numeric payload, for array-based persistence."""
        return {"point_ids": self.point_ids, "image_points": self.image_points}

    def to_manifest(self) -> Dict[str, Any]:
        """The non-numeric payload, for the JSON side of persistence."""
        return {
            "view_id": self.view_id,
            "source": self.source,
            "metadata": dict(self.metadata),
            "n_points": self.n_points,
        }


@dataclass(frozen=True)
class DetectionFailure:
    """One image that produced no usable detection.

    Attributes:
        source: The image path or identifier.
        reason: Why it was rejected, in words fit for a report.
    """

    source: str
    reason: str

    def to_dict(self) -> Dict[str, str]:
        """Serialise to a plain dictionary."""
        return {"source": self.source, "reason": self.reason}


@dataclass(frozen=True)
class DetectionSummary:
    """How ingest went, counted honestly.

    Attributes:
        attempted: Images or records offered to the detector.
        failures: Every rejection, with its reason.
        detector: Name of the detector that ran, if one did.
    """

    attempted: int = 0
    failures: Tuple[DetectionFailure, ...] = ()
    detector: Optional[str] = None

    @property
    def succeeded(self) -> int:
        """Images that yielded a usable view."""
        return self.attempted - len(self.failures)

    @property
    def success_rate(self) -> float:
        """Fraction of attempts that produced a view; zero when none were made."""
        return self.succeeded / self.attempted if self.attempted else 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Serialise to a plain dictionary."""
        return {
            "attempted": self.attempted,
            "detector": self.detector,
            "failures": [f.to_dict() for f in self.failures],
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "DetectionSummary":
        """Rebuild from `to_dict` output."""
        return cls(
            attempted=int(payload.get("attempted", 0)),
            failures=tuple(
                DetectionFailure(f["source"], f["reason"])
                for f in payload.get("failures", ())
            ),
            detector=payload.get("detector"),
        )


@dataclass(frozen=True, eq=False)
class ObservationSet:
    """Every view detected on one target with one camera.

    Attributes:
        target: The physical target the points belong to.
        image_size: Frame size as `(width, height)` in pixels.
        views: The detected views, in ingest order.
        summary: Detection accounting, including what failed.
    """

    target: TargetSpec
    image_size: Tuple[int, int]
    views: Tuple[ViewObservations, ...]
    summary: DetectionSummary = field(default_factory=DetectionSummary)

    def __post_init__(self) -> None:
        views = tuple(self.views)
        if not views:
            raise ValidationError("an observation set needs at least one view")
        seen = set()
        for view in views:
            if view.view_id in seen:
                raise ValidationError(f"duplicate view id {view.view_id!r}")
            seen.add(view.view_id)
            # Raises if any id is outside the target, which catches a mismatched
            # target spec at ingest instead of as a strange residual later.
            self.target.object_points(view.point_ids)
        width, height = (int(v) for v in self.image_size)
        if width <= 0 or height <= 0:
            raise ValidationError(f"image size must be positive, got {(width, height)}")
        object.__setattr__(self, "views", views)
        object.__setattr__(self, "image_size", (width, height))

    @property
    def n_views(self) -> int:
        """Number of views."""
        return len(self.views)

    @property
    def total_points(self) -> int:
        """Total detected points across every view."""
        return int(sum(v.n_points for v in self.views))

    @property
    def view_ids(self) -> Tuple[str, ...]:
        """View identifiers, in order."""
        return tuple(v.view_id for v in self.views)

    def index_of(self, view_id: str) -> int:
        """Position of a view by id.

        Args:
            view_id: The identifier to look up.

        Returns:
            The zero-based index of that view.

        Raises:
            ValidationError: No view carries that id.
        """
        try:
            return self.view_ids.index(view_id)
        except ValueError:
            raise ValidationError(f"no view with id {view_id!r}") from None

    def select(self, indices: Iterable[int]) -> "ObservationSet":
        """A new set holding only the given views, order preserved.

        This is how held-out folds are formed in later milestones, so it keeps
        the target and image size but resets nothing else.

        Args:
            indices: View positions to keep.

        Returns:
            A new `ObservationSet`.
        """
        chosen = [int(i) for i in indices]
        out_of_range = [i for i in chosen if not 0 <= i < self.n_views]
        if out_of_range:
            raise ValidationError(f"view index out of range: {out_of_range}")
        return ObservationSet(
            self.target,
            self.image_size,
            tuple(self.views[i] for i in chosen),
            self.summary,
        )

    def filter_min_points(self, minimum: int) -> "ObservationSet":
        """Drop views with too few points to constrain a pose.

        Args:
            minimum: Fewest points a view may keep.

        Returns:
            A new `ObservationSet` without the thin views.

        Raises:
            ValidationError: Every view would be dropped.
        """
        keep = [i for i, v in enumerate(self.views) if v.n_points >= minimum]
        if not keep:
            raise ValidationError(
                f"no view has at least {minimum} points; "
                f"largest is {max(v.n_points for v in self.views)}"
            )
        return self.select(keep)

    def points_per_view(self) -> np.ndarray:
        """Detected point count for each view."""
        return np.array([v.n_points for v in self.views], dtype=int)

    def out_of_frame(self) -> int:
        """Count of detected points lying outside the frame.

        A non-zero count is not fatal — sub-pixel refinement can push a corner a
        fraction of a pixel past the edge — but a large one means the declared
        image size is wrong.
        """
        width, height = self.image_size
        count = 0
        for view in self.views:
            x, y = view.image_points[:, 0], view.image_points[:, 1]
            count += int(np.count_nonzero((x < 0) | (y < 0) | (x > width) | (y > height)))
        return count
