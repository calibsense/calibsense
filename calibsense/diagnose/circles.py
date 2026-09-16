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

"""The one error a circle grid has that no other target does.

A projected circle's centroid is not the projection of its centre: perspective
enlarges the near half of the disc, so the centroid slides away from the centre
by an amount set by how large the circle looks. Every detected point is offset,
in a direction fixed by the tilt rather than by noise, so it does not average
down over points or over views the way corner-localisation error does.

The size of it is a closed form rather than a guess, because a perspective map
sends a circle to an ellipse and an ellipse's centroid is its centre.
"""

from __future__ import annotations

from typing import ClassVar

import numpy as np

from .. import units as units_mod
from ..core.target import CircleGrid
from .base import Diagnostic, DiagnosticContext, Finding, Severity

#: Coefficient in `bias ≈ COEFFICIENT * f * (r / d)^2`, taken at the tilt that
#: maximises it. Computed from the conic at 45 degrees rather than fitted.
BIAS_COEFFICIENT = 0.49

#: Fractions of the residual sigma at which the bias is worth saying something
#: about. It is systematic, so it does not average down and a tenth of sigma is
#: already a different thing from a tenth of sigma's worth of noise.
BIAS_WARNING = 0.5
BIAS_NOTE = 0.1


class CircleGridBias(Diagnostic):
    """Is the centroid bias large enough to matter for this rig?"""

    cause: ClassVar[str] = "circle_centroid_bias"
    title: ClassVar[str] = "Circle centroid bias"

    def run(self, context: DiagnosticContext) -> Finding:
        """Compare the predicted bias against the residual scale."""
        target = context.observations.target
        if not isinstance(target, CircleGrid):
            return self._finding(
                Severity.OK,
                "not a circle grid, so the centroid bias does not arise",
                applies=False,
            )
        if not target.diameter:
            return self._finding(
                Severity.NOTE,
                "this circle grid does not record its circle diameter, so the "
                "centroid bias cannot be evaluated; it is the one error this "
                "target has that a checkerboard does not",
                "Re-state the target with the printed circle diameter, as "
                "`circles:COLSxROWS:SPACING:LAYOUT:DIAMETER`. Nothing needs it "
                "to calibrate and only this needs it to judge the result.",
                applies=True,
                measured=False,
            )

        radius_mm = 0.5 * units_mod.to_mm(target.diameter, target.units)
        distances = context.board_distances_mm
        focal = float(context.fit.camera.fx)
        # Worst over the capture, taken at the nearest view: the bias goes as
        # the square of the angular radius, so the closest board dominates.
        nearest = float(np.min(distances))
        bias_px = BIAS_COEFFICIENT * focal * (radius_mm / nearest) ** 2
        sigma = float(context.fit.covariance.sigma)
        share = bias_px / sigma if sigma > 0 else float("inf")
        metrics = dict(
            applies=True,
            measured=True,
            bias_px=bias_px,
            sigma_px=sigma,
            share_of_sigma=share,
            nearest_distance_mm=nearest,
            circle_radius_mm=radius_mm,
        )
        shared = (
            f"a {2 * radius_mm:g} mm circle at {nearest:.0f} mm biases its own "
            f"centroid by about {bias_px:.3f} px, against a residual sigma of "
            f"{sigma:.3f} px"
        )
        action = (
            "This is systematic, not noise: it points the same way in every "
            "view at a given tilt, so it does not average down and no number of "
            "extra views removes it. Move the target further away — the bias "
            "goes as the square of the angular radius, so doubling the distance "
            "quarters it — or use smaller circles, or switch to a checkerboard "
            "or ChArUco target, which do not have this error at all."
        )
        if share > BIAS_WARNING:
            return self._finding(
                Severity.WARNING,
                f"{shared}, which is {share:.0%} of it", action, **metrics,
            )
        if share > BIAS_NOTE:
            return self._finding(
                Severity.NOTE, f"{shared}, or {share:.0%} of it", action, **metrics,
            )
        return self._finding(
            Severity.OK,
            f"{shared}, which is {share:.0%} of it and not worth acting on",
            **metrics,
        )
