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

"""Turning a folder of calibration images into detected views."""

from __future__ import annotations

import os
from typing import Callable, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from ..core.observations import (
    DetectionFailure,
    DetectionSummary,
    ObservationSet,
    ViewObservations,
)
from ..core.target import TargetSpec
from ..errors import DetectionError, IngestError
from .detectors import DetectorOptions, detector_for
from .detectors.opencv_support import image_size_of

#: Extensions treated as calibration images, matched case-insensitively.
IMAGE_SUFFIXES = (
    ".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".pgm", ".ppm", ".webp",
)

#: Called with `(index, total, view_id, error_or_none)` after each image.
ProgressCallback = Callable[[int, int, str, Optional[str]], None]


def find_images(root: str, recursive: bool = True) -> List[str]:
    """List calibration images under a directory, in sorted order.

    Sorting is by path, which is what makes a session reproducible: the same
    folder always produces the same view order and therefore the same view ids.

    Args:
        root: A directory, or a single image file.
        recursive: Descend into subdirectories.

    Returns:
        Absolute paths of every image found.

    Raises:
        IngestError: The path does not exist, or holds no images.
    """
    if not os.path.exists(root):
        raise IngestError(f"no such path: {root}")
    if os.path.isfile(root):
        return [os.path.abspath(root)]
    found: List[str] = []
    for directory, subdirectories, filenames in os.walk(root):
        subdirectories.sort()
        for name in sorted(filenames):
            if name.lower().endswith(IMAGE_SUFFIXES):
                found.append(os.path.join(directory, name))
        if not recursive:
            break
    if not found:
        raise IngestError(
            f"no images under {root}; looked for {', '.join(IMAGE_SUFFIXES)}"
        )
    return [os.path.abspath(p) for p in found]


def read_image(path: str) -> np.ndarray:
    """Read one image from disk.

    Decoding goes through `numpy.fromfile` rather than `cv2.imread` so that
    non-ASCII paths work identically on every platform.

    Args:
        path: Path to an image file.

    Returns:
        The decoded image.

    Raises:
        IngestError: The file could not be read or decoded.
    """
    try:
        raw = np.fromfile(path, dtype=np.uint8)
    except OSError as exc:
        raise IngestError(f"could not read {path}: {exc}") from exc
    if raw.size == 0:
        raise IngestError(f"{path} is empty")
    image = cv2.imdecode(raw, cv2.IMREAD_UNCHANGED)
    if image is None:
        raise IngestError(f"{path} is not an image OpenCV can decode")
    return image


def view_id_for(path: str, taken: Sequence[str]) -> str:
    """Derive a unique view id from an image path.

    Args:
        path: The image path.
        taken: View ids already in use.

    Returns:
        The filename stem, suffixed with a counter only if it collides — which
        happens when the same stem appears in two subdirectories.
    """
    stem = os.path.splitext(os.path.basename(path))[0]
    if stem not in taken:
        return stem
    index = 2
    while f"{stem}#{index}" in taken:
        index += 1
    return f"{stem}#{index}"


def detect_in_images(
    paths: Iterable[str],
    target: TargetSpec,
    options: Optional[DetectorOptions] = None,
    progress: Optional[ProgressCallback] = None,
) -> ObservationSet:
    """Detect a target across a set of images.

    Failures are recorded rather than raised, because a calibration capture
    routinely contains images the detector cannot use, and the *rate* of those
    failures is itself a diagnostic worth reporting.

    Args:
        paths: Image paths, in the order they should become views.
        target: The target to look for.
        options: Detector settings.
        progress: Optional per-image callback, for CLI output.

    Returns:
        Every successful view, with a summary of what failed.

    Raises:
        IngestError: No image yielded a detection, or the images are not all the
            same size.
    """
    paths = list(paths)
    detector = detector_for(target, options)
    views: List[ViewObservations] = []
    failures: List[DetectionFailure] = []
    image_size: Optional[Tuple[int, int]] = None

    for index, path in enumerate(paths):
        view_id = view_id_for(path, [v.view_id for v in views])
        reason: Optional[str] = None
        try:
            image = read_image(path)
        except IngestError as exc:
            # An unreadable file is one bad frame among many, not a reason to
            # throw away a capture session.
            reason = str(exc)
        else:
            size = image_size_of(image)
            if image_size is None:
                image_size = size
            elif size != image_size:
                # A size change means two different cameras or two different
                # capture settings, and no honest single calibration exists.
                raise IngestError(
                    f"{path} is {size[0]}x{size[1]} but earlier images are "
                    f"{image_size[0]}x{image_size[1]}; one calibration cannot "
                    "mix image sizes"
                )
            try:
                views.append(detector.detect(image, view_id, source=path))
            except DetectionError as exc:
                reason = str(exc)
        if reason is not None:
            failures.append(DetectionFailure(path, reason))
        if progress is not None:
            progress(index + 1, len(paths), view_id, reason)

    if not views:
        raise IngestError(
            f"no image yielded a detection of {target.describe()} "
            f"({len(failures)} attempted)"
        )
    return ObservationSet(
        target=target,
        image_size=image_size,
        views=tuple(views),
        summary=DetectionSummary(
            attempted=len(paths), failures=tuple(failures), detector=target.kind
        ),
    )
