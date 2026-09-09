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

"""Diagnostics on where in the frame the target was seen, and how big it was.

Distortion is a function of image radius, so a calibration constrains it only
over the radii it actually observed. Corners that never leave the middle of the
frame leave the periphery to extrapolation, which is exactly where the model
will be used and exactly where it is worst.
"""

from __future__ import annotations

from typing import ClassVar, Tuple

import numpy as np

from .base import Diagnostic, DiagnosticContext, Finding, Severity

#: Default occupancy grid, in cells across and down.
GRID = (8, 6)

COVERAGE_CRITICAL = 0.35
COVERAGE_WARNING = 0.65
PERIPHERAL_WARNING = 0.35
REACH_CRITICAL = 0.5
REACH_WARNING = 0.75

#: Median corner spacing below which localisation noise dominates, in pixels.
SPACING_CRITICAL_PX = 10.0
SPACING_WARNING_PX = 20.0


def occupancy(
    points: np.ndarray, image_size: Tuple[int, int], grid: Tuple[int, int] = GRID
) -> np.ndarray:
    """Count detected corners per cell of an image-space grid.

    Args:
        points: Pooled corner positions, shape `(n, 2)`.
        image_size: Frame size as `(width, height)`.
        grid: Cells across and down.

    Returns:
        A `(down, across)` integer array of counts.
    """
    width, height = image_size
    across, down = grid
    column = np.clip((points[:, 0] / width * across).astype(int), 0, across - 1)
    row = np.clip((points[:, 1] / height * down).astype(int), 0, down - 1)
    counts = np.zeros((down, across), dtype=int)
    np.add.at(counts, (row, column), 1)
    return counts


def border_mask(grid: Tuple[int, int] = GRID) -> np.ndarray:
    """A boolean mask selecting the outermost ring of an occupancy grid.

    Args:
        grid: Cells across and down.

    Returns:
        A `(down, across)` boolean array, `True` on the border ring.
    """
    across, down = grid
    mask = np.zeros((down, across), dtype=bool)
    mask[0, :] = mask[-1, :] = True
    mask[:, 0] = mask[:, -1] = True
    return mask


def nearest_neighbour_spacing(points: np.ndarray) -> float:
    """Median distance from each point to its closest neighbour, in pixels.

    A target-agnostic proxy for corner spacing: it needs no grid topology, so it
    works identically for a full checkerboard and a partial ChArUco detection.

    Args:
        points: Corner positions for one view, shape `(n, 2)`.

    Returns:
        The median nearest-neighbour distance, or zero for fewer than two points.
    """
    if points.shape[0] < 2:
        return 0.0
    distances = np.linalg.norm(points[:, None, :] - points[None, :, :], axis=2)
    np.fill_diagonal(distances, np.inf)
    return float(np.median(distances.min(axis=1)))


class ImageCoverage(Diagnostic):
    """Did the corners reach the parts of the frame that matter?

    Three numbers, because they fail independently: how much of the frame was
    visited at all, whether the outer ring was visited, and how far out in radius
    the corners ever got. The last is the one that decides whether the distortion
    coefficients are interpolating or extrapolating.
    """

    cause: ClassVar[str] = "image_coverage"
    title: ClassVar[str] = "Image coverage"

    def run(self, context: DiagnosticContext) -> Finding:
        """Measure grid occupancy, peripheral occupancy and radial reach."""
        points = context.all_image_points
        width, height = context.image_size
        counts = occupancy(points, context.image_size)
        border = border_mask()
        coverage = float((counts > 0).mean())
        peripheral = float((counts[border] > 0).mean())

        camera = context.fit.camera
        centre = np.array([camera.cx, camera.cy])
        radii = np.linalg.norm(points - centre, axis=1)
        corners = np.array([[0.0, 0.0], [width, 0.0], [0.0, height], [width, height]])
        furthest_possible = float(np.linalg.norm(corners - centre, axis=1).max())
        reach = float(radii.max() / furthest_possible) if furthest_possible > 0 else 0.0

        metrics = dict(
            coverage=coverage,
            peripheral_coverage=peripheral,
            radial_reach=reach,
            max_radius_px=float(radii.max()),
            frame_corner_radius_px=furthest_possible,
            grid=list(GRID),
            empty_cells=int((counts == 0).sum()),
        )
        shared = (
            f"corners occupy {coverage:.0%} of an {GRID[0]}x{GRID[1]} grid, "
            f"{peripheral:.0%} of the border ring, and reach {reach:.0%} of the "
            "way to the frame corner"
        )
        if coverage < COVERAGE_CRITICAL or reach < REACH_CRITICAL:
            return self._finding(
                Severity.CRITICAL,
                shared
                + "; the distortion coefficients are extrapolating over most of "
                "the frame",
                "Move the board into the corners and edges of the image, not just "
                "the middle. Distortion grows with radius, so the coefficients are "
                "only measured over the radii you actually visit.",
                **metrics,
            )
        if coverage < COVERAGE_WARNING or peripheral < PERIPHERAL_WARNING or reach < REACH_WARNING:
            return self._finding(
                Severity.WARNING,
                shared + "; the periphery is thin",
                "Add views with the board against the frame edges and corners. A "
                "ChArUco target helps here, because it still yields labelled "
                "points when half the board is out of frame.",
                **metrics,
            )
        return self._finding(Severity.OK, shared, **metrics)


class TargetScale(Diagnostic):
    """Was the target big enough in frame for corner noise not to dominate?

    Corner localisation error is roughly constant in pixels, so the useful
    quantity is not the error but its ratio to the corner spacing. At 10 px
    spacing a 0.2 px localisation error is 2% of the baseline between adjacent
    points, and the pose and the distortion terms both suffer for it.
    """

    cause: ClassVar[str] = "target_scale"
    title: ClassVar[str] = "Target scale"

    def run(self, context: DiagnosticContext) -> Finding:
        """Measure the median corner spacing and the board's share of the frame."""
        spacings = np.array(
            [nearest_neighbour_spacing(v.image_points) for v in context.observations.views]
        )
        areas = np.array(
            [v.area_fraction(context.image_size) for v in context.observations.views]
        )
        median_spacing = float(np.median(spacings))
        sigma = context.fit.covariance.sigma
        relative = sigma / median_spacing if median_spacing > 0 else float("inf")
        metrics = dict(
            median_spacing_px=median_spacing,
            min_spacing_px=float(spacings.min()),
            max_spacing_px=float(spacings.max()),
            median_area_fraction=float(np.median(areas)),
            residual_sigma_px=sigma,
            noise_over_spacing=relative,
        )
        shared = (
            f"median corner spacing is {median_spacing:.1f} px "
            f"({spacings.min():.1f} to {spacings.max():.1f} across views), and the "
            f"board fills {np.median(areas):.0%} of the frame; residual sigma is "
            f"{relative:.1%} of the spacing"
        )
        if median_spacing < SPACING_CRITICAL_PX:
            return self._finding(
                Severity.CRITICAL,
                shared + ", so corner localisation noise dominates the geometry",
                "Fill more of the frame, or use a target with fewer, larger "
                "squares. A board this small in frame cannot support a precise "
                "calibration however many views you take.",
                **metrics,
            )
        if median_spacing < SPACING_WARNING_PX:
            return self._finding(
                Severity.WARNING,
                shared,
                "Bring the board closer, or print a larger one. Corner spacing "
                "below 20 px leaves little margin over localisation noise.",
                **metrics,
            )
        return self._finding(Severity.OK, shared, **metrics)
