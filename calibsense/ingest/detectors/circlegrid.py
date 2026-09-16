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

"""Circle grid centre detection."""

from __future__ import annotations

from typing import ClassVar, Optional

import cv2
import numpy as np

from ...core.observations import ViewObservations
from ...errors import DetectionError
from .base import TargetDetector
from .opencv_support import to_gray


class CircleGridDetector(TargetDetector):
    """Finds the circle centres of a symmetric or asymmetric grid.

    Centres come from blob fitting rather than corner refinement, so no extra
    sub-pixel pass is applied.

    The perspective bias of a projected circle's centroid is real, systematic
    and uncorrected here. A perspective map sends a circle to an ellipse, whose
    centroid is its centre, and that centre is *not* the projection of the
    circle's centre — so every detected point is offset, in a direction set by
    the tilt rather than by noise, and it does not average down over points or
    views.

    Computed exactly from the conic rather than estimated, for a circle of
    radius `r` seen at distance `d` through a lens of focal length `f`:

        bias ≈ 0.49 · f · (r / d)² · g(tilt),  g peaking at 1 near 45 degrees

    Two things follow that the usual description of this effect gets wrong.
    It does **not** simply grow with obliquity: it rises to a maximum around 45
    degrees and falls away again, reaching the same 0.0223 px at 60 degrees as
    at 30. And it is **quadratic in the angular radius**, so it is governed by
    how big the circles look rather than by how oblique they are — halving the
    working distance costs four times as much as going from 10 to 40 degrees of
    tilt.

    Worked, for a 6 mm circle through a 700 px lens at 40 degrees:

        700 mm   0.025 px      a tenth of a typical 0.2 px noise sigma
        500 mm   0.050 px
        300 mm   0.138 px      most of a noise sigma, and systematic

    So it is negligible for a small target at a normal working distance and
    matters for large circles seen close. calibsense cannot tell which case a
    user is in, because `CircleGrid` records the centre-to-centre spacing and
    not the circle diameter — correcting the bias, or even warning about it,
    needs that number first.
    """

    kind: ClassVar[str] = "circle_grid"

    def detect(
        self, image: np.ndarray, view_id: str, source: Optional[str] = None
    ) -> ViewObservations:
        """Find the circle grid in one image.

        Args:
            image: A grayscale or colour image.
            view_id: Identifier to attach to the resulting view.
            source: Where the image came from.

        Returns:
            The detected circle centres in the grid's canonical order.

        Raises:
            DetectionError: The grid was not found.
        """
        gray = to_gray(image)
        pattern = self.target.pattern_size
        layout = (
            cv2.CALIB_CB_ASYMMETRIC_GRID
            if self.target.asymmetric
            else cv2.CALIB_CB_SYMMETRIC_GRID
        )
        attempts = [layout] if self.options.fast else [layout, layout | cv2.CALIB_CB_CLUSTERING]
        for flags in attempts:
            found, centres = cv2.findCirclesGrid(gray, pattern, flags=flags)
            if found:
                points = centres.reshape(-1, 2).astype(np.float64)
                self._check_enough(points.shape[0], view_id)
                return ViewObservations(
                    view_id=view_id,
                    point_ids=np.arange(points.shape[0], dtype=np.int64),
                    image_points=points,
                    source=source,
                    metadata={
                        "detector": "circle_grid",
                        "clustering": bool(flags & cv2.CALIB_CB_CLUSTERING),
                    },
                )
        raise DetectionError(
            f"{view_id}: no {pattern[0]}x{pattern[1]} circle grid found"
        )
