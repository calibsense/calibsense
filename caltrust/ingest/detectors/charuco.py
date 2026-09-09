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

"""ChArUco corner detection."""

from __future__ import annotations

from typing import ClassVar, Optional

import cv2
import numpy as np

from ... import units as units_mod
from ...core.observations import ViewObservations
from ...core.target import TargetSpec
from ...errors import DetectionError, ValidationError
from .base import DetectorOptions, TargetDetector
from .opencv_support import aruco_dictionary, to_gray


class CharucoDetector(TargetDetector):
    """Finds the chessboard corners of a ChArUco board.

    Partial detections are normal and welcome: markers identify each corner, so
    a view showing half the board still contributes unambiguous, correctly
    labelled points. That is what makes ChArUco the right target when the board
    must reach the edge of the frame, where distortion is strongest.
    """

    kind: ClassVar[str] = "charuco"

    def __init__(self, target: TargetSpec, options: Optional[DetectorOptions] = None):
        super().__init__(target, options)
        if not hasattr(cv2.aruco, "CharucoDetector"):
            raise ValidationError(
                "this OpenCV build predates cv2.aruco.CharucoDetector; "
                "caltrust needs OpenCV 4.7 or newer for ChArUco targets"
            )
        square_mm = units_mod.to_mm(target.square_size, target.units)
        marker_mm = units_mod.to_mm(target.marker_size, target.units)
        board = cv2.aruco.CharucoBoard(
            (target.squares_x, target.squares_y),
            square_mm,
            marker_mm,
            aruco_dictionary(target.dictionary),
        )
        board.setLegacyPattern(bool(target.legacy_pattern))
        self.board = board
        self._detector = cv2.aruco.CharucoDetector(board)

    def detect(
        self, image: np.ndarray, view_id: str, source: Optional[str] = None
    ) -> ViewObservations:
        """Find ChArUco chessboard corners in one image.

        Args:
            image: A grayscale or colour image.
            view_id: Identifier to attach to the resulting view.
            source: Where the image came from.

        Returns:
            The detected corners with their board corner ids.

        Raises:
            DetectionError: No markers resolved, or too few corners survived.
        """
        gray = to_gray(image)
        corners, ids, _, marker_ids = self._detector.detectBoard(gray)
        if corners is None or ids is None or len(ids) == 0:
            raise DetectionError(f"{view_id}: no ChArUco corners resolved")
        points = np.asarray(corners, dtype=np.float64).reshape(-1, 2)
        point_ids = np.asarray(ids, dtype=np.int64).reshape(-1)
        self._check_enough(points.shape[0], view_id)
        return ViewObservations(
            view_id=view_id,
            point_ids=point_ids,
            image_points=points,
            source=source,
            metadata={
                "detector": "charuco",
                "markers": 0 if marker_ids is None else int(len(marker_ids)),
                "coverage": round(points.shape[0] / self.target.num_points, 4),
            },
        )
