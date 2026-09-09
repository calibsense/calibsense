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

"""Checkerboard corner detection."""

from __future__ import annotations

from typing import ClassVar, Optional

import cv2
import numpy as np

from ...core.observations import ViewObservations
from ...errors import DetectionError
from .base import TargetDetector
from .opencv_support import refine_corners, to_gray


class CheckerboardDetector(TargetDetector):
    """Finds every inner corner of a plain checkerboard.

    Detection is all-or-nothing: OpenCV reports the full grid or nothing, so a
    kept view always carries `columns * rows` points with ids `0..n-1` in
    row-major order.

    The board's 180-degree ambiguity is not resolved here. For intrinsics it is
    harmless, because each view's pose is free and absorbs the flip. For
    hand-eye it is not, which is a reason to prefer a ChArUco target when robot
    poses are involved.
    """

    kind: ClassVar[str] = "checkerboard"

    def detect(
        self, image: np.ndarray, view_id: str, source: Optional[str] = None
    ) -> ViewObservations:
        """Find the checkerboard in one image.

        Args:
            image: A grayscale or colour image.
            view_id: Identifier to attach to the resulting view.
            source: Where the image came from.

        Returns:
            The detected corners, one per inner corner of the board.

        Raises:
            DetectionError: The full grid was not found.
        """
        gray = to_gray(image)
        pattern = self.target.pattern_size
        corners, method = self._detect_sb(gray, pattern)
        if corners is None:
            corners, method = self._detect_classic(gray, pattern)
        if corners is None:
            raise DetectionError(
                f"{view_id}: no {pattern[0]}x{pattern[1]} checkerboard found"
            )
        points = corners.reshape(-1, 2).astype(np.float64)
        self._check_enough(points.shape[0], view_id)
        return ViewObservations(
            view_id=view_id,
            point_ids=np.arange(points.shape[0], dtype=np.int64),
            image_points=points,
            source=source,
            metadata={"detector": "checkerboard", "method": method},
        )

    def _detect_sb(self, gray, pattern):
        # findChessboardCornersSB is both more robust and already sub-pixel, so
        # it is tried first and its result is never passed through cornerSubPix.
        flags = cv2.CALIB_CB_NORMALIZE_IMAGE
        if not self.options.fast:
            flags |= cv2.CALIB_CB_EXHAUSTIVE | cv2.CALIB_CB_ACCURACY
        try:
            found, corners = cv2.findChessboardCornersSB(gray, pattern, flags=flags)
        except cv2.error:
            return None, ""
        return (corners, "findChessboardCornersSB") if found else (None, "")

    def _detect_classic(self, gray, pattern):
        flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
        if self.options.fast:
            flags |= cv2.CALIB_CB_FAST_CHECK
        found, corners = cv2.findChessboardCorners(gray, pattern, flags=flags)
        if not found:
            return None, ""
        if self.options.refine:
            corners = refine_corners(gray, corners, self.options.refine_window)
        return corners, "findChessboardCorners+cornerSubPix"
