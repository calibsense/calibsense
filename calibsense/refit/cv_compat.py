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

"""Resolving OpenCV calibration flags across versions.

OpenCV 4 keeps the fisheye flags in `cv2.fisheye` with their own bit numbering,
and defines same-named constants in `cv2` with *different* values for the
pinhole path. OpenCV 5 merged them into one namespace on the pinhole numbering
and moved the fisheye-only flags to high bits.

Reading `cv2.CALIB_FIX_K1` and handing it to `cv2.fisheye.calibrate` therefore
silently fixes the wrong coefficient on OpenCV 4 and the right one on OpenCV 5.
Every flag is resolved by name through this module so that mistake cannot be
made, and the test suite checks the resolved values against the fisheye
namespace wherever it exists.
"""

from __future__ import annotations

from typing import Dict

import cv2

from ..errors import RefitError

_cache: Dict[str, int] = {}


def fisheye_flag(name: str) -> int:
    """Resolve a fisheye calibration flag by name.

    Args:
        name: A flag name such as `"CALIB_FIX_SKEW"`, without a namespace.

    Returns:
        The flag value appropriate to the installed OpenCV, preferring
        `cv2.fisheye` when it defines the name.

    Raises:
        RefitError: Neither namespace defines the flag.
    """
    key = f"fisheye.{name}"
    if key in _cache:
        return _cache[key]
    namespace = getattr(cv2, "fisheye", None)
    value = getattr(namespace, name, None) if namespace is not None else None
    if value is None:
        value = getattr(cv2, name, None)
    if value is None:
        raise RefitError(
            f"this OpenCV build ({cv2.__version__}) defines no fisheye flag "
            f"{name!r} in either cv2.fisheye or cv2"
        )
    _cache[key] = int(value)
    return _cache[key]


def pinhole_flag(name: str) -> int:
    """Resolve a pinhole calibration flag by name.

    Args:
        name: A flag name such as `"CALIB_RATIONAL_MODEL"`.

    Returns:
        The flag value.

    Raises:
        RefitError: The installed OpenCV does not define the flag.
    """
    if name in _cache:
        return _cache[name]
    value = getattr(cv2, name, None)
    if value is None:
        raise RefitError(
            f"this OpenCV build ({cv2.__version__}) defines no flag {name!r}; "
            "calibsense needs OpenCV 4.8 or newer"
        )
    _cache[name] = int(value)
    return _cache[name]


def set_single_threaded(enabled: bool = True) -> int:
    """Make OpenCV's arithmetic bit-reproducible, at the cost of speed.

    `cv2.calibrateCamera` reduces across threads, and floating-point addition is
    not associative, so the same inputs give answers that differ in the last few
    bits from run to run. Measured on a ten-thread machine, eight identical
    refits produced eight different focal lengths spanning 1.1e-12 px — harmless
    for any conclusion, and enough to make a byte-comparison of two reports
    fail. Pinning OpenCV to one thread removes it entirely.

    Args:
        enabled: Pin to one thread, or restore OpenCV's own default.

    Returns:
        The thread count in effect before the call, so a caller can restore it.
    """
    previous = cv2.getNumThreads()
    cv2.setNumThreads(1 if enabled else 0)
    return previous
