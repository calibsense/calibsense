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

"""Diagnostics on the pose set: diversity, depth spread, frontoparallel dominance.

These three are the ones that decide whether a focal length is knowable at all,
and the thresholds below are not guesses. They come from measuring synthetic
rigs against known truth, which is what `examples/degeneracy_demo.py`
reproduces: at 5 degrees of tilt the focal length carries a standard deviation of
28 px, at 20 degrees 3.9 px, at 40 degrees 2.2 px, while the reprojection RMS
sits at 0.275 px throughout.
"""

from __future__ import annotations

from typing import ClassVar

import numpy as np

from .base import Diagnostic, DiagnosticContext, Finding, Severity

#: Tilt below which a view counts as frontoparallel, in degrees.
FRONTOPARALLEL_DEGREES = 15.0

#: Normal spread below which pose diversity is critical, then merely poor.
DIVERSITY_CRITICAL_DEGREES = 5.0
DIVERSITY_WARNING_DEGREES = 20.0

#: Fraction of frontoparallel views above which the capture is dominated by them.
FRONTOPARALLEL_CRITICAL_FRACTION = 0.8
FRONTOPARALLEL_WARNING_FRACTION = 0.5

#: Max-over-min working distance below which depth variation is critical, then thin.
DEPTH_CRITICAL_RATIO = 1.2
DEPTH_WARNING_RATIO = 1.5


def normal_spread_degrees(normals: np.ndarray) -> float:
    """Largest angle between any two board normals, in degrees.

    Folded by absolute value, because a plane's normal has no meaningful sign
    and a board rotated by 180 degrees in its own plane is the same board.

    Args:
        normals: Unit normals, shape `(v, 3)`.

    Returns:
        The maximum pairwise angle, or zero for fewer than two views.
    """
    if normals.shape[0] < 2:
        return 0.0
    cosines = np.clip(np.abs(normals @ normals.T), 0.0, 1.0)
    return float(np.degrees(np.arccos(cosines.min())))


def orientation_tensor(normals: np.ndarray) -> np.ndarray:
    """Eigenvalues of the normal scatter matrix, largest first.

    The tensor `(1/v) * sum(n n')` has trace one and is invariant to each
    normal's sign, which makes it the right summary for directions that are
    identified with their opposites. All normals equal gives `(1, 0, 0)`;
    normals confined to a plane gives a near-zero third eigenvalue; normals
    spread over the sphere approaches `(1/3, 1/3, 1/3)`.

    Args:
        normals: Unit normals, shape `(v, 3)`.

    Returns:
        Three eigenvalues in descending order.
    """
    tensor = (normals.T @ normals) / normals.shape[0]
    return np.sort(np.linalg.eigvalsh(tensor))[::-1]


class PoseDiversity(Diagnostic):
    """Are the board orientations varied enough to determine a focal length?

    A frontoparallel planar target gives one homography per view, and because
    every view carries its own free translation, rescaling all focal lengths and
    all depths together reproduces the images exactly. Perspective foreshortening
    inside a single view is what breaks that, so what matters is how far the
    board normals spread, not how many views there are.
    """

    cause: ClassVar[str] = "pose_diversity"
    title: ClassVar[str] = "Pose diversity"

    def run(self, context: DiagnosticContext) -> Finding:
        """Measure the spread of board normals on the sphere."""
        normals = context.board_normals
        tilts = context.tilt_degrees
        spread = normal_spread_degrees(normals)
        eigenvalues = orientation_tensor(normals)
        metrics = dict(
            normal_spread_deg=spread,
            tilt_min_deg=float(tilts.min()),
            tilt_median_deg=float(np.median(tilts)),
            tilt_max_deg=float(tilts.max()),
            concentration=float(eigenvalues[0]),
            planarity=float(eigenvalues[2]),
            n_views=context.n_views,
        )
        shared = (
            f"board normals span {spread:.1f} degrees, tilts run "
            f"{tilts.min():.1f} to {tilts.max():.1f} degrees over "
            f"{context.n_views} views"
        )
        if spread < DIVERSITY_CRITICAL_DEGREES:
            return self._finding(
                Severity.CRITICAL,
                f"{shared}; every view sees the board at effectively the same "
                "orientation, so the focal length is not determined by this capture",
                "Tilt the board. Aim for at least 30 degrees of tilt in several "
                "different directions; that single change matters more than any "
                "other, and more views at the current orientation will not help.",
                **metrics,
            )
        if spread < DIVERSITY_WARNING_DEGREES:
            return self._finding(
                Severity.WARNING,
                f"{shared}; that is enough to determine the focal length but not "
                "enough to determine it well",
                "Add views at 30 to 45 degrees of tilt, in tilt directions you do "
                "not already have. Going from 5 to 20 degrees of tilt cut the "
                "focal-length uncertainty roughly sevenfold on a reference rig.",
                **metrics,
            )
        if eigenvalues[2] < 1e-3:
            return self._finding(
                Severity.NOTE,
                f"{shared}, but the normals lie almost in a plane, so the board is "
                "being tilted about one axis only",
                "Tilt about the other axis too, so the normals cover a patch of "
                "the sphere rather than an arc of it.",
                **metrics,
            )
        return self._finding(Severity.OK, shared, **metrics)


class DepthVariation(Diagnostic):
    """Do the views span a range of working distances?

    Secondary to tilt, and genuinely secondary: on a reference rig, adding two
    further working distances to an already-tilted capture roughly halved the
    focal-length uncertainty, while adding them to a flat capture changed
    nothing at all, because each new view arrives with its own free translation.
    """

    cause: ClassVar[str] = "depth_variation"
    title: ClassVar[str] = "Depth variation"

    def run(self, context: DiagnosticContext) -> Finding:
        """Measure the ratio of the furthest to the nearest board."""
        distances = context.board_distances_mm
        nearest, furthest = float(distances.min()), float(distances.max())
        ratio = furthest / nearest if nearest > 0 else float("inf")
        correlation = self._focal_correlation(context)
        # Depths come from the fitted poses, so on a rank-deficient fit their
        # absolute values are wrong by exactly the factor the focal length is
        # wrong by -- the two rescale together, which is the degeneracy. The
        # *ratio* survives that rescale, so the finding stands either way; only
        # the millimetre figures have to be withheld.
        absolute = context.fit.conditioning.identifiable
        metrics = dict(
            depth_ratio=ratio,
            nearest_mm=nearest,
            furthest_mm=furthest,
            median_mm=float(np.median(distances)),
            focal_distance_correlation=correlation,
            distances_are_absolute=absolute,
        )
        if absolute:
            shared = (
                f"working distances run {nearest:.0f} to {furthest:.0f} mm, "
                f"a ratio of {ratio:.2f}x"
            )
        else:
            shared = (
                f"working distances span a ratio of {ratio:.2f}x; their absolute "
                "values are not reported because this capture does not determine "
                "the focal length, and depth scales with it"
            )
        # The correlation is only worth quoting when the fit is determined. On a
        # rank-deficient system the unconstrained direction is cut, so it carries
        # no variance to correlate and the number reads near zero on exactly the
        # captures where the confounding is total.
        if correlation is None:
            reading = (
                "; the focal-distance correlation is not meaningful here, because "
                "the fit is rank deficient and the confounded direction carries no "
                "variance to correlate"
            )
        else:
            reading = (
                f"; focal length and working distance are correlated at "
                f"{correlation:+.2f}"
            )

        near_advice = (
            f"{nearest * 0.6:.0f} mm and {furthest * 1.5:.0f} mm"
            if absolute
            else "0.6x and 1.5x your current working distance"
        )
        if ratio < DEPTH_CRITICAL_RATIO:
            return self._finding(
                Severity.CRITICAL,
                shared + reading + ", so scale and focal length are hard to separate",
                f"Capture at a second and third distance, around {near_advice}. Do "
                "this after fixing any tilt problem, not instead of it.",
                **metrics,
            )
        if ratio < DEPTH_WARNING_RATIO:
            return self._finding(
                Severity.WARNING,
                shared + reading,
                f"Widen the range. Views around {near_advice} would decorrelate "
                "focal length from scale.",
                **metrics,
            )
        return self._finding(Severity.OK, shared + reading, **metrics)

    @staticmethod
    def _focal_correlation(context: DiagnosticContext):
        """Mean correlation between `fx` and each view's depth, or `None`."""
        covariance = context.fit.covariance
        if "fx" not in covariance.intrinsic_names:
            return None
        if not context.fit.conditioning.identifiable:
            return None
        return float(covariance.correlation_with_poses("fx")[:, 5].mean())


class FrontoparallelDominance(Diagnostic):
    """How much of the capture is flat-on to the camera?

    Distinct from pose diversity: this measures absolute tilt rather than
    variety. A capture where every board sits within a few degrees of the image
    plane is the single most common way a calibration ends up confidently wrong,
    and it is also the case where distortion is least constrained, because a flat
    board at a fixed distance covers a narrow band of image radii.
    """

    cause: ClassVar[str] = "frontoparallel_dominance"
    title: ClassVar[str] = "Frontoparallel dominance"

    def run(self, context: DiagnosticContext) -> Finding:
        """Measure the fraction of views within `FRONTOPARALLEL_DEGREES` of flat."""
        tilts = context.tilt_degrees
        flat = tilts < FRONTOPARALLEL_DEGREES
        fraction = float(flat.mean())
        undetermined = context.weak_parameters & {"fx", "fy"}
        metrics = dict(
            frontoparallel_fraction=fraction,
            frontoparallel_views=int(flat.sum()),
            n_views=context.n_views,
            threshold_deg=FRONTOPARALLEL_DEGREES,
        )
        shared = (
            f"{int(flat.sum())} of {context.n_views} views ({fraction:.0%}) are "
            f"within {FRONTOPARALLEL_DEGREES:.0f} degrees of the image plane"
        )
        action = (
            "Tilt the board on most of the capture, not a few frames of it. "
            "Frames within 15 degrees of flat contribute almost nothing towards "
            "separating focal length from distance."
        )
        if undetermined and fraction >= FRONTOPARALLEL_WARNING_FRACTION:
            # The causal chain is complete here, so say so rather than leaving
            # the reader to connect two separate findings.
            return self._finding(
                Severity.CRITICAL,
                shared
                + f", and {' and '.join(sorted(undetermined))} came out undetermined; "
                "this is the cause of that",
                action,
                **metrics,
            )
        if fraction >= FRONTOPARALLEL_CRITICAL_FRACTION:
            return self._finding(
                Severity.CRITICAL,
                shared + ", which leaves the focal length resting on distortion alone",
                action,
                **metrics,
            )
        if fraction >= FRONTOPARALLEL_WARNING_FRACTION:
            return self._finding(Severity.WARNING, shared, action, **metrics)
        return self._finding(Severity.OK, shared, **metrics)
