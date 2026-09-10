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
    sub-pixel pass is applied. The perspective bias of a projected circle's
    centroid is a real and often overlooked error source for this target, and it
    is reported as a diagnostic in a later milestone rather than corrected here.
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
