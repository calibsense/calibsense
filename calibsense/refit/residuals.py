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

"""Residual statistics, per view and per corner.

A single RMS over a whole calibration hides everything worth knowing. The same
0.25 px can be 0.25 px everywhere, or 0.1 px over most views and 1.2 px in three
of them, or 0.1 px in the centre rising to 0.9 px at the edge. Those are three
different problems with three different fixes, and telling them apart needs the
distribution rather than its summary.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np

from ..core.observations import ObservationSet
from ..errors import ValidationError

#: Robust z-score above which a view is called an outlier. 3.5 on a
#: median-absolute-deviation scale is the conventional cut.
OUTLIER_Z = 3.5

#: Scale factor making the median absolute deviation a consistent estimator of
#: the standard deviation for normally distributed data.
MAD_TO_SIGMA = 1.4826


@dataclass(frozen=True)
class ViewResiduals:
    """Residual summary for one view.

    Attributes:
        view_id: The view's identifier.
        n_points: Points contributing.
        rms: Root-mean-square reprojection error in pixels, per point.
        bias: Mean signed residual as `(x, y)`, in pixels. A view with a real
            bias is not fitting; it is being fitted around.
        median: Median point error magnitude.
        p95: 95th percentile point error magnitude.
        maximum: Largest single point error.
        robust_z: How far this view's RMS sits from the median view, in robust
            standard deviations.
    """

    view_id: str
    n_points: int
    rms: float
    bias: Tuple[float, float]
    median: float
    p95: float
    maximum: float
    robust_z: float = 0.0

    @property
    def is_outlier(self) -> bool:
        """Whether this view's error stands out from the rest of the capture."""
        return self.robust_z > OUTLIER_Z


@dataclass(frozen=True)
class RadialProfile:
    """Residuals binned by distance from the principal point.

    A distortion model that does not match the lens does not fail uniformly. It
    fails at the edge, where the unmodelled terms are largest, and it shows up
    as a systematic signed radial residual that grows with radius. That signature
    is invisible in an RMS and obvious here.

    Attributes:
        edges: Bin edges in pixels, length `n_bins + 1`.
        counts: Points per bin.
        radial_mean: Mean signed residual along the outward radial direction.
        radial_rms: RMS of the radial component.
        tangential_rms: RMS of the component perpendicular to the radius.
    """

    edges: np.ndarray
    counts: np.ndarray
    radial_mean: np.ndarray
    radial_rms: np.ndarray
    tangential_rms: np.ndarray

    @property
    def n_bins(self) -> int:
        """Number of radial bins."""
        return int(self.counts.size)


@dataclass(frozen=True)
class ResidualStatistics:
    """Everything the standard calibration call throws away about residuals.

    Attributes:
        per_view: One summary per view, in view order.
        residuals: Concatenated per-corner residuals, shape `(total_points, 2)`.
        view_offsets: Start index of each view within `residuals`, length
            `n_views + 1`.
        rms: Overall root-mean-square error in pixels, per point. This is the
            number `cv2.calibrateCamera` returns.
        bias: Overall mean signed residual as `(x, y)`.
        radial: Residuals binned by image radius.
    """

    per_view: Tuple[ViewResiduals, ...]
    residuals: np.ndarray
    view_offsets: np.ndarray
    rms: float
    bias: Tuple[float, float]
    radial: RadialProfile

    @property
    def n_views(self) -> int:
        """Number of views."""
        return len(self.per_view)

    @property
    def total_points(self) -> int:
        """Total points across every view."""
        return int(self.residuals.shape[0])

    def view_residuals(self, index: int) -> np.ndarray:
        """The `(n, 2)` residual array for one view.

        Args:
            index: View position.

        Returns:
            That view's residuals in pixels.
        """
        if not 0 <= index < self.n_views:
            raise ValidationError(f"view index {index} out of range")
        start, stop = int(self.view_offsets[index]), int(self.view_offsets[index + 1])
        return self.residuals[start:stop]

    def outlier_views(self) -> Tuple[ViewResiduals, ...]:
        """Views whose error stands out from the rest, worst first."""
        return tuple(
            sorted(
                (v for v in self.per_view if v.is_outlier),
                key=lambda v: -v.robust_z,
            )
        )

    def worst_views(self, count: int = 5) -> Tuple[ViewResiduals, ...]:
        """The highest-RMS views, worst first.

        Args:
            count: How many to return.

        Returns:
            Up to `count` view summaries.
        """
        return tuple(sorted(self.per_view, key=lambda v: -v.rms)[:count])


def _summarise_view(view_id: str, residual: np.ndarray) -> ViewResiduals:
    magnitude = np.linalg.norm(residual, axis=1)
    return ViewResiduals(
        view_id=view_id,
        n_points=int(residual.shape[0]),
        rms=float(np.sqrt(np.mean(magnitude ** 2))),
        bias=(float(residual[:, 0].mean()), float(residual[:, 1].mean())),
        median=float(np.median(magnitude)),
        p95=float(np.percentile(magnitude, 95)),
        maximum=float(magnitude.max()),
    )


def _robust_z(values: np.ndarray) -> np.ndarray:
    median = float(np.median(values))
    deviation = float(np.median(np.abs(values - median))) * MAD_TO_SIGMA
    if deviation <= 0:
        # Every view agrees; either the fit is perfect or there is one view.
        return np.zeros_like(values)
    return (values - median) / deviation


def radial_profile(
    image_points: np.ndarray,
    residuals: np.ndarray,
    principal_point: Tuple[float, float],
    n_bins: int = 10,
) -> RadialProfile:
    """Bin residuals by distance from the principal point.

    Args:
        image_points: Observed pixel positions, shape `(n, 2)`.
        residuals: Matching residuals, shape `(n, 2)`.
        principal_point: The `(cx, cy)` the radii are measured from.
        n_bins: Number of equal-width radial bins.

    Returns:
        The binned profile. Bins holding no points come back as zero rather
        than `nan`, with a zero count to say so.

    Raises:
        ValidationError: The arrays disagree, or `n_bins` is not positive.
    """
    points = np.asarray(image_points, dtype=float).reshape(-1, 2)
    residual = np.asarray(residuals, dtype=float).reshape(-1, 2)
    if points.shape != residual.shape:
        raise ValidationError(
            f"{points.shape[0]} image points against {residual.shape[0]} residuals"
        )
    if n_bins < 1:
        raise ValidationError(f"n_bins must be positive, got {n_bins}")

    offset = points - np.asarray(principal_point, dtype=float).reshape(1, 2)
    radius = np.linalg.norm(offset, axis=1)
    largest = float(radius.max()) if radius.size else 1.0
    edges = np.linspace(0.0, max(largest, 1e-9), n_bins + 1)

    safe = np.where(radius[:, None] > 0, radius[:, None], 1.0)
    outward = offset / safe
    perpendicular = np.stack([-outward[:, 1], outward[:, 0]], axis=1)
    radial_component = np.sum(residual * outward, axis=1)
    tangential_component = np.sum(residual * perpendicular, axis=1)

    index = np.clip(np.digitize(radius, edges[1:-1]), 0, n_bins - 1)
    counts = np.bincount(index, minlength=n_bins)
    nonzero = np.maximum(counts, 1)
    radial_mean = np.bincount(index, radial_component, minlength=n_bins) / nonzero
    radial_rms = np.sqrt(
        np.bincount(index, radial_component ** 2, minlength=n_bins) / nonzero
    )
    tangential_rms = np.sqrt(
        np.bincount(index, tangential_component ** 2, minlength=n_bins) / nonzero
    )
    empty = counts == 0
    for array in (radial_mean, radial_rms, tangential_rms):
        array[empty] = 0.0
    return RadialProfile(edges, counts, radial_mean, radial_rms, tangential_rms)


def summarise(
    observations: ObservationSet,
    residuals: Sequence[np.ndarray],
    principal_point: Tuple[float, float],
    n_bins: int = 10,
) -> ResidualStatistics:
    """Turn per-view residual arrays into the full residual report.

    Args:
        observations: The detections the residuals belong to.
        residuals: One `(n_i, 2)` array per view, in view order.
        principal_point: The `(cx, cy)` used for radial binning.
        n_bins: Number of radial bins.

    Returns:
        Per-view summaries, per-corner residuals and the radial profile.

    Raises:
        ValidationError: The residual arrays do not match the views.
    """
    if len(residuals) != observations.n_views:
        raise ValidationError(
            f"{len(residuals)} residual arrays for {observations.n_views} views"
        )
    summaries: List[ViewResiduals] = []
    for view, residual in zip(observations.views, residuals):
        array = np.asarray(residual, dtype=float).reshape(-1, 2)
        if array.shape[0] != view.n_points:
            raise ValidationError(
                f"view {view.view_id!r}: {array.shape[0]} residuals for "
                f"{view.n_points} points"
            )
        summaries.append(_summarise_view(view.view_id, array))

    zeds = _robust_z(np.array([s.rms for s in summaries]))
    summaries = [
        ViewResiduals(
            s.view_id, s.n_points, s.rms, s.bias, s.median, s.p95, s.maximum,
            float(z),
        )
        for s, z in zip(summaries, zeds)
    ]

    stacked = np.concatenate([np.asarray(r, dtype=float).reshape(-1, 2) for r in residuals])
    offsets = np.cumsum([0] + [v.n_points for v in observations.views]).astype(np.int64)
    all_points = np.concatenate([v.image_points for v in observations.views])
    magnitude = np.linalg.norm(stacked, axis=1)
    return ResidualStatistics(
        per_view=tuple(summaries),
        residuals=stacked,
        view_offsets=offsets,
        rms=float(np.sqrt(np.mean(magnitude ** 2))),
        bias=(float(stacked[:, 0].mean()), float(stacked[:, 1].mean())),
        radial=radial_profile(all_points, stacked, principal_point, n_bins),
    )
