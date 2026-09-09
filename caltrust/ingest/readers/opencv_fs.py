"""OpenCV FileStorage calibrations, as written by the OpenCV calibration samples."""

from __future__ import annotations

import os
from typing import Any, ClassVar, Optional

import cv2
import numpy as np

from ...core.camera import FisheyeKannalaBrandt, PinholeBrownConrady
from ...core.session import CalibrationRecord
from ...errors import UnsupportedFormatError
from .base import CalibrationReader, looks_like_opencv_filestorage

_MATRIX_KEYS = ("camera_matrix", "cameraMatrix", "K", "intrinsic", "M1")
_DISTORTION_KEYS = (
    "distortion_coefficients", "distCoeffs", "D", "dist_coeffs", "distortion",
)
_RMS_KEYS = ("avg_reprojection_error", "rms", "reprojection_error", "avg_error")


class OpenCVFileStorageReader(CalibrationReader):
    """Reads `.yml`, `.yaml` and `.xml` files written by `cv2.FileStorage`.

    These files record `K` and `D` but not which distortion model produced them.
    A four-coefficient vector is therefore genuinely ambiguous between
    Brown-Conrady and Kannala-Brandt. Rather than guess silently, the record is
    returned as pinhole and flagged `model_ambiguous` so the caller can insist
    on an explicit choice.
    """

    name: ClassVar[str] = "opencv"
    description: ClassVar[str] = "OpenCV FileStorage (.yml / .yaml / .xml)"

    def sniff(self, path: str, head: str) -> bool:
        """Match OpenCV's XML root tag, its YAML directive, or a matrix tag."""
        suffix = os.path.splitext(path)[1].lower()
        if suffix == ".xml":
            return "<opencv_storage>" in head
        if suffix not in (".yml", ".yaml"):
            return False
        return looks_like_opencv_filestorage(head)

    def read(self, path: str) -> CalibrationRecord:
        """Parse an OpenCV FileStorage calibration.

        Args:
            path: The file to read.

        Returns:
            The calibration, with `model_ambiguous` set in its metadata when the
            distortion model could not be determined from the file.

        Raises:
            UnsupportedFormatError: The file will not open, or lacks `K` or `D`.
        """
        storage = cv2.FileStorage(path, cv2.FILE_STORAGE_READ)
        if not storage.isOpened():
            raise UnsupportedFormatError(f"{path} did not open as OpenCV FileStorage")
        try:
            matrix = self._first_matrix(storage, _MATRIX_KEYS)
            distortion = self._first_matrix(storage, _DISTORTION_KEYS)
            width = self._first_scalar(storage, ("image_width", "image_Width", "width"))
            height = self._first_scalar(storage, ("image_height", "image_Height", "height"))
            rms = self._first_scalar(storage, _RMS_KEYS)
            declared = self._first_string(
                storage, ("distortion_model", "camera_model", "model")
            )
        finally:
            storage.release()

        if matrix is None or matrix.shape != (3, 3):
            raise UnsupportedFormatError(
                f"{path} has no 3x3 camera matrix under any of {list(_MATRIX_KEYS)}"
            )
        if distortion is None:
            raise UnsupportedFormatError(
                f"{path} has no distortion vector under any of {list(_DISTORTION_KEYS)}"
            )
        coefficients = np.asarray(distortion, dtype=float).reshape(-1)
        fisheye = declared is not None and (
            "fisheye" in declared.lower() or "equidistant" in declared.lower()
        )
        ambiguous = declared is None and coefficients.size == 4

        if fisheye:
            camera: Any = FisheyeKannalaBrandt(
                matrix[0, 0], matrix[1, 1], matrix[0, 2], matrix[1, 2],
                coefficients[:4], alpha=float(matrix[0, 1] / matrix[0, 0]),
            )
        else:
            camera = PinholeBrownConrady(
                matrix[0, 0], matrix[1, 1], matrix[0, 2], matrix[1, 2], coefficients
            )
        if width is None or height is None:
            raise UnsupportedFormatError(
                f"{path} does not record image_width and image_height, so its "
                "principal point cannot be checked against the frame"
            )
        return CalibrationRecord(
            camera=camera,
            image_size=(int(width), int(height)),
            source=f"opencv:{os.path.basename(path)}",
            reported_rms=rms,
            metadata={
                "declared_model": declared,
                "model_ambiguous": ambiguous,
                "path": os.path.abspath(path),
            },
        )

    @staticmethod
    def _first_matrix(storage, keys) -> Optional[np.ndarray]:
        for key in keys:
            node = storage.getNode(key)
            if node.empty():
                continue
            value = node.mat()
            if value is not None:
                return np.asarray(value, dtype=float)
        return None

    @staticmethod
    def _first_scalar(storage, keys) -> Optional[float]:
        for key in keys:
            node = storage.getNode(key)
            if not node.empty() and (node.isReal() or node.isInt()):
                return float(node.real())
        return None

    @staticmethod
    def _first_string(storage, keys) -> Optional[str]:
        for key in keys:
            node = storage.getNode(key)
            if not node.empty() and node.isString():
                return str(node.string())
        return None
