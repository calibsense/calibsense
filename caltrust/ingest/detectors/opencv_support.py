"""Small OpenCV helpers the detectors share."""

from __future__ import annotations

from typing import Tuple

import cv2
import numpy as np

from ...errors import DetectionError, ValidationError

#: `cornerSubPix` stopping criteria: 40 iterations or 0.001 px movement.
SUBPIX_CRITERIA = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 1e-3)


def to_gray(image: np.ndarray) -> np.ndarray:
    """Return an 8-bit single-channel view of an image.

    Args:
        image: A 2-D grayscale or 3-channel colour image, any integer or float
            dtype.

    Returns:
        An 8-bit single-channel array.

    Raises:
        DetectionError: The array is not a shape OpenCV can treat as an image.
    """
    array = np.asarray(image)
    if array.ndim == 3:
        if array.shape[2] == 4:
            array = cv2.cvtColor(array, cv2.COLOR_BGRA2GRAY)
        elif array.shape[2] == 3:
            array = cv2.cvtColor(array, cv2.COLOR_BGR2GRAY)
        elif array.shape[2] == 1:
            array = array[:, :, 0]
        else:
            raise DetectionError(f"unsupported channel count: {array.shape[2]}")
    elif array.ndim != 2:
        raise DetectionError(f"expected a 2-D or 3-D image, got shape {array.shape}")
    if array.dtype != np.uint8:
        finite = array[np.isfinite(array)] if array.dtype.kind == "f" else array
        if finite.size == 0:
            raise DetectionError("image has no finite pixels")
        peak = float(finite.max())
        scale = 255.0 / peak if peak > 0 else 1.0
        array = np.clip(np.nan_to_num(array) * scale, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(array)


def image_size_of(image: np.ndarray) -> Tuple[int, int]:
    """The `(width, height)` of an image array.

    Args:
        image: Any 2-D or 3-D image array.

    Returns:
        Width then height, matching OpenCV's `Size` order rather than NumPy's.
    """
    array = np.asarray(image)
    if array.ndim < 2:
        raise ValidationError(f"expected an image array, got shape {array.shape}")
    return int(array.shape[1]), int(array.shape[0])


def aruco_dictionary(name: str):
    """Look up a predefined ArUco dictionary by name.

    Args:
        name: A dictionary name such as `"DICT_5X5_1000"`, case-insensitive.

    Returns:
        The OpenCV dictionary object.

    Raises:
        ValidationError: OpenCV was built without `aruco`, or the name is not a
            predefined dictionary.
    """
    if not hasattr(cv2, "aruco"):
        raise ValidationError(
            "this OpenCV build has no aruco module; install opencv-contrib-python"
        )
    key = name.strip().upper()
    if not key.startswith("DICT_") or not hasattr(cv2.aruco, key):
        available = sorted(d for d in dir(cv2.aruco) if d.startswith("DICT_"))
        raise ValidationError(
            f"unknown ArUco dictionary {name!r}; expected one of {available}"
        )
    return cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, key))


def refine_corners(
    gray: np.ndarray, corners: np.ndarray, window: int
) -> np.ndarray:
    """Sub-pixel refine corner locations, shrinking the window near an edge.

    OpenCV 4 returns detected corners as `(n, 1, 2)` and OpenCV 5 as `(n, 2)`,
    so both are accepted and the caller gets its own shape back.

    Args:
        gray: The 8-bit single-channel image the corners came from.
        corners: Corner estimates, `(n, 2)` or `(n, 1, 2)`.
        window: Requested half-width of the search window, in pixels.

    Returns:
        The refined corners, in the shape they arrived in. Corners hugging the
        image border are returned unrefined, because no window fits around them.
    """
    original = np.asarray(corners)
    points = original.astype(np.float32).reshape(-1, 1, 2)
    height, width = gray.shape[:2]
    x, y = points[:, 0, 0], points[:, 0, 1]
    margin = int(np.floor(min(
        x.min(), y.min(), width - 1 - x.max(), height - 1 - y.max()
    )))
    if margin < 2:
        return original
    half = int(np.clip(min(window, margin - 1), 1, window))
    refined = cv2.cornerSubPix(
        gray, points, (half, half), (-1, -1), SUBPIX_CRITERIA
    )
    return refined.reshape(original.shape)
