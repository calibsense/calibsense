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

"""Calibration target geometry.

A target spec answers one question: given a point id reported by a detector,
where is that point on the physical board? Everything metric downstream — the
covariance, the millimetres at a working distance — rests on this being right,
so the object points are generated here analytically rather than trusted to a
detector's incidental output, and checked against OpenCV in the test suite.

Ids are always the detector's own index convention for the pattern, so a partial
detection (routine for ChArUco, possible for nothing else) round-trips without
the caller tracking a separate mapping.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, ClassVar, Dict, Mapping, Optional, Sequence, Tuple, Type

import numpy as np

from .. import units as units_mod
from ..errors import ValidationError

_TARGETS: Dict[str, Type["TargetSpec"]] = {}


def register_target(cls: Type["TargetSpec"]) -> Type["TargetSpec"]:
    """Register a target type so `target_from_dict` can find it by `kind`.

    Args:
        cls: The target class to register.

    Returns:
        `cls`, unchanged.
    """
    _TARGETS[cls.kind] = cls
    return cls


class TargetSpec(ABC):
    """A physical calibration target.

    Concrete targets are frozen dataclasses that declare `units` as their final
    field. The base stays a plain ABC so that subclass constructors keep their
    own fields in positional order, which a shared base dataclass field would
    silently shift.

    Attributes:
        units: The unit the board was measured in. Object points are always
            returned in millimetres regardless; this records what the engineer
            typed so a report can echo it back.
    """

    kind: ClassVar[str] = ""
    units: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "units", units_mod.normalise(self.units))

    @property
    @abstractmethod
    def num_points(self) -> int:
        """Total number of detectable points on a fully visible target."""

    @abstractmethod
    def _all_object_points_mm(self) -> np.ndarray:
        """All object points in millimetres, indexed by point id."""

    @abstractmethod
    def describe(self) -> str:
        """A one-line human description, for reports and CLI output."""

    def object_points(self, point_ids: Optional[Sequence[int]] = None) -> np.ndarray:
        """Object points in millimetres for the requested ids.

        Args:
            point_ids: Point ids to look up. `None` returns every point in id
                order.

        Returns:
            An `(n, 3)` array of board-frame coordinates in millimetres. The
            board is planar, so the third column is zero.

        Raises:
            ValidationError: An id is out of range or repeated.
        """
        points = self._all_object_points_mm()
        if point_ids is None:
            return points.copy()
        ids = np.asarray(point_ids, dtype=np.int64).reshape(-1)
        if ids.size and (ids.min() < 0 or ids.max() >= points.shape[0]):
            raise ValidationError(
                f"point id out of range for {self.kind}: got "
                f"[{ids.min()}, {ids.max()}], target has {points.shape[0]} points"
            )
        if np.unique(ids).size != ids.size:
            raise ValidationError("duplicate point ids in a single view")
        return points[ids]

    @property
    def extent_mm(self) -> Tuple[float, float]:
        """Width and height of the point pattern in millimetres."""
        points = self._all_object_points_mm()
        spans = points.max(axis=0) - points.min(axis=0)
        return float(spans[0]), float(spans[1])

    @property
    def diagonal_mm(self) -> float:
        """Diagonal of the point pattern in millimetres."""
        width, height = self.extent_mm
        return float(np.hypot(width, height))

    def to_dict(self) -> Dict[str, Any]:
        """Serialise to a plain JSON-compatible dictionary."""
        payload = {"kind": self.kind, "units": self.units}
        payload.update(self._fields())
        return payload

    @abstractmethod
    def _fields(self) -> Dict[str, Any]:
        """Type-specific fields for serialisation."""


def _grid_dimension(value: int, name: str, minimum: int = 2) -> int:
    if int(value) != value or value < minimum:
        raise ValidationError(f"{name} must be an integer >= {minimum}, got {value}")
    return int(value)


def _positive_length(value: float, name: str) -> float:
    if not np.isfinite(value) or value <= 0:
        raise ValidationError(f"{name} must be a positive finite length, got {value}")
    return float(value)


@register_target
@dataclass(frozen=True)
class Checkerboard(TargetSpec):
    """A plain checkerboard, described by its *inner* corner counts.

    Attributes:
        columns: Inner corners across, matching `patternSize[0]` in OpenCV.
        rows: Inner corners down, matching `patternSize[1]`.
        square_size: Side of one square, in `units`.
    """

    columns: int = 9
    rows: int = 6
    square_size: float = 25.0
    units: str = units_mod.CANONICAL

    kind: ClassVar[str] = "checkerboard"

    def __post_init__(self) -> None:
        super().__post_init__()
        object.__setattr__(self, "columns", _grid_dimension(self.columns, "columns"))
        object.__setattr__(self, "rows", _grid_dimension(self.rows, "rows"))
        object.__setattr__(
            self, "square_size", _positive_length(self.square_size, "square_size")
        )

    @property
    def num_points(self) -> int:
        """Inner corner count, `columns * rows`."""
        return self.columns * self.rows

    @property
    def pattern_size(self) -> Tuple[int, int]:
        """The `(columns, rows)` pair OpenCV's detector expects."""
        return (self.columns, self.rows)

    def _all_object_points_mm(self) -> np.ndarray:
        step = units_mod.to_mm(self.square_size, self.units)
        col, row = np.meshgrid(np.arange(self.columns), np.arange(self.rows))
        return np.stack(
            [col.ravel() * step, row.ravel() * step, np.zeros(self.num_points)], axis=1
        )

    def describe(self) -> str:
        """Human-readable summary, for example `9x6 checkerboard, 25.0 mm squares`."""
        return (
            f"{self.columns}x{self.rows} checkerboard, "
            f"{self.square_size:g} {self.units} squares"
        )

    def _fields(self) -> Dict[str, Any]:
        return {
            "columns": self.columns,
            "rows": self.rows,
            "square_size": self.square_size,
        }


@register_target
@dataclass(frozen=True)
class CircleGrid(TargetSpec):
    """A symmetric or asymmetric grid of circles.

    Attributes:
        columns: Circles per row, matching `patternSize[0]` in OpenCV.
        rows: Number of rows, matching `patternSize[1]`.
        spacing: Centre-to-centre spacing, in `units`. For the asymmetric
            pattern this is the vertical row pitch, and OpenCV's convention puts
            successive rows offset by one `spacing` horizontally.
        asymmetric: Whether the staggered asymmetric layout is used.
    """

    columns: int = 4
    rows: int = 11
    spacing: float = 20.0
    asymmetric: bool = True
    units: str = units_mod.CANONICAL

    kind: ClassVar[str] = "circle_grid"

    def __post_init__(self) -> None:
        super().__post_init__()
        object.__setattr__(self, "columns", _grid_dimension(self.columns, "columns"))
        object.__setattr__(self, "rows", _grid_dimension(self.rows, "rows"))
        object.__setattr__(self, "spacing", _positive_length(self.spacing, "spacing"))
        object.__setattr__(self, "asymmetric", bool(self.asymmetric))

    @property
    def num_points(self) -> int:
        """Circle count, `columns * rows`."""
        return self.columns * self.rows

    @property
    def pattern_size(self) -> Tuple[int, int]:
        """The `(columns, rows)` pair OpenCV's detector expects."""
        return (self.columns, self.rows)

    def _all_object_points_mm(self) -> np.ndarray:
        step = units_mod.to_mm(self.spacing, self.units)
        col, row = np.meshgrid(np.arange(self.columns), np.arange(self.rows))
        col = col.ravel().astype(float)
        row = row.ravel().astype(float)
        x = (2.0 * col + row % 2.0) * step if self.asymmetric else col * step
        return np.stack([x, row * step, np.zeros(self.num_points)], axis=1)

    def describe(self) -> str:
        """Human-readable summary naming the layout and pitch."""
        layout = "asymmetric" if self.asymmetric else "symmetric"
        return (
            f"{self.columns}x{self.rows} {layout} circle grid, "
            f"{self.spacing:g} {self.units} spacing"
        )

    def _fields(self) -> Dict[str, Any]:
        return {
            "columns": self.columns,
            "rows": self.rows,
            "spacing": self.spacing,
            "asymmetric": self.asymmetric,
        }


@register_target
@dataclass(frozen=True)
class CharucoBoard(TargetSpec):
    """A ChArUco board: a checkerboard with ArUco markers in the white squares.

    Point ids are chessboard corner ids, not marker ids, and run row-major over
    the `(squares_x - 1) x (squares_y - 1)` interior corners. Partial detections
    are the normal case for this target, which is exactly why it is worth using.

    Attributes:
        squares_x: Number of squares across the board.
        squares_y: Number of squares down the board.
        square_size: Side of one chessboard square, in `units`.
        marker_size: Side of one ArUco marker, in `units`. Must be smaller than
            `square_size`.
        dictionary: Name of the ArUco dictionary, such as `"DICT_5X5_1000"`.
        legacy_pattern: Use OpenCV's pre-4.6 marker layout. Boards printed from
            an older OpenCV need this set, and a wrong choice produces a
            confident, wrong calibration rather than a detection failure.
    """

    squares_x: int = 8
    squares_y: int = 11
    square_size: float = 20.0
    marker_size: float = 15.0
    dictionary: str = "DICT_5X5_1000"
    legacy_pattern: bool = False
    units: str = units_mod.CANONICAL

    kind: ClassVar[str] = "charuco"

    def __post_init__(self) -> None:
        super().__post_init__()
        object.__setattr__(self, "squares_x", _grid_dimension(self.squares_x, "squares_x"))
        object.__setattr__(self, "squares_y", _grid_dimension(self.squares_y, "squares_y"))
        object.__setattr__(
            self, "square_size", _positive_length(self.square_size, "square_size")
        )
        object.__setattr__(
            self, "marker_size", _positive_length(self.marker_size, "marker_size")
        )
        if self.marker_size >= self.square_size:
            raise ValidationError(
                f"marker_size ({self.marker_size}) must be smaller than "
                f"square_size ({self.square_size})"
            )
        object.__setattr__(self, "dictionary", str(self.dictionary))
        object.__setattr__(self, "legacy_pattern", bool(self.legacy_pattern))

    @property
    def num_points(self) -> int:
        """Interior chessboard corner count."""
        return (self.squares_x - 1) * (self.squares_y - 1)

    @property
    def corner_grid(self) -> Tuple[int, int]:
        """Interior corners as `(across, down)`."""
        return (self.squares_x - 1, self.squares_y - 1)

    def _all_object_points_mm(self) -> np.ndarray:
        step = units_mod.to_mm(self.square_size, self.units)
        across, down = self.corner_grid
        col, row = np.meshgrid(np.arange(across), np.arange(down))
        return np.stack(
            [
                (col.ravel() + 1) * step,
                (row.ravel() + 1) * step,
                np.zeros(self.num_points),
            ],
            axis=1,
        )

    def describe(self) -> str:
        """Human-readable summary naming the square grid and dictionary."""
        return (
            f"{self.squares_x}x{self.squares_y} ChArUco, "
            f"{self.square_size:g}/{self.marker_size:g} {self.units} "
            f"square/marker, {self.dictionary}"
        )

    def _fields(self) -> Dict[str, Any]:
        return {
            "squares_x": self.squares_x,
            "squares_y": self.squares_y,
            "square_size": self.square_size,
            "marker_size": self.marker_size,
            "dictionary": self.dictionary,
            "legacy_pattern": self.legacy_pattern,
        }


def target_from_dict(payload: Mapping[str, Any]) -> TargetSpec:
    """Rebuild a target spec from its serialised form.

    Args:
        payload: A mapping as produced by `TargetSpec.to_dict`.

    Returns:
        The reconstructed target.

    Raises:
        ValidationError: `kind` is missing, unknown, or a field is unexpected.
    """
    fields = dict(payload)
    kind = fields.pop("kind", None)
    if kind not in _TARGETS:
        raise ValidationError(
            f"unknown target kind {kind!r}; expected one of {sorted(_TARGETS)}"
        )
    cls = _TARGETS[kind]
    accepted = set(cls.__dataclass_fields__)
    unknown = set(fields) - accepted
    if unknown:
        raise ValidationError(
            f"unexpected field(s) {sorted(unknown)} for target kind {kind!r}"
        )
    return cls(**fields)


def registered_targets() -> Tuple[str, ...]:
    """Names of every registered target kind."""
    return tuple(sorted(_TARGETS))
