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
                "calibsense needs OpenCV 4.7 or newer for ChArUco targets"
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

    def _no_corners_reason(self, markers_found: int) -> str:
        """Why no chessboard corners came back, when that can be narrowed down.

        Markers read but no corners is the exact signature of the wrong
        `legacy_pattern`: OpenCV finds the ArUco squares either way and then
        fails to pair them with chessboard corners, because the two layouts
        place the markers differently. Measured on an 8x10 board, the wrong
        setting returns 40 markers and 0 corners where the right one returns 40
        and 63.

        Worth saying only when the flag can matter at all. The two layouts are
        **identical** whenever the board has an odd number of squares in y —
        including the 8x11 default — so on those boards this cannot be the
        cause and pointing at it would send the reader somewhere useless.

        Args:
            markers_found: ArUco markers the detector did resolve.

        Returns:
            The reason text, without the view id.
        """
        if markers_found == 0:
            return "no ChArUco corners resolved, and no ArUco markers either"
        if self.target.squares_y % 2 == 1:
            return (
                f"{markers_found} ArUco markers were read but no chessboard "
                "corners resolved"
            )
        return (
            f"{markers_found} ArUco markers were read but no chessboard corners "
            f"resolved, which is what the wrong legacy_pattern looks like: this "
            f"board is {self.target.squares_x}x{self.target.squares_y} and the "
            "two marker layouts differ whenever the square count in y is even. "
            f"Try legacy_pattern={not self.target.legacy_pattern}"
        )

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
            found = 0 if marker_ids is None else int(len(marker_ids))
            raise DetectionError(f"{view_id}: {self._no_corners_reason(found)}")
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
