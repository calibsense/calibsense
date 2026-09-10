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

OpenCV 4 and OpenCV 5 disagree about where the fisheye flags live *and* about
their values, so a name resolved from the wrong namespace fixes the wrong
coefficient. This is the guard against that.
"""

from __future__ import annotations

import cv2
import pytest

from calibsense.errors import RefitError
from calibsense.refit.cv_compat import fisheye_flag, pinhole_flag

FISHEYE_NAMES = (
    "CALIB_USE_INTRINSIC_GUESS",
    "CALIB_RECOMPUTE_EXTRINSIC",
    "CALIB_CHECK_COND",
    "CALIB_FIX_SKEW",
    "CALIB_FIX_K1",
    "CALIB_FIX_K2",
    "CALIB_FIX_K3",
    "CALIB_FIX_K4",
    "CALIB_FIX_PRINCIPAL_POINT",
    "CALIB_FIX_FOCAL_LENGTH",
)

PINHOLE_NAMES = (
    "CALIB_USE_INTRINSIC_GUESS",
    "CALIB_FIX_ASPECT_RATIO",
    "CALIB_FIX_PRINCIPAL_POINT",
    "CALIB_FIX_FOCAL_LENGTH",
    "CALIB_ZERO_TANGENT_DIST",
    "CALIB_RATIONAL_MODEL",
    "CALIB_THIN_PRISM_MODEL",
    "CALIB_TILTED_MODEL",
    "CALIB_FIX_K1",
    "CALIB_FIX_K6",
)


@pytest.mark.parametrize("name", FISHEYE_NAMES)
def test_every_fisheye_flag_resolves(name):
    assert fisheye_flag(name) > 0


@pytest.mark.parametrize("name", PINHOLE_NAMES)
def test_every_pinhole_flag_resolves(name):
    assert pinhole_flag(name) > 0


@pytest.mark.parametrize("name", FISHEYE_NAMES)
def test_a_fisheye_flag_prefers_the_fisheye_namespace(name):
    """On OpenCV 4 the two namespaces disagree, and cv2.fisheye has to win."""
    namespace = getattr(cv2, "fisheye", None)
    expected = getattr(namespace, name, None)
    if expected is None:
        pytest.skip(f"this OpenCV has no cv2.fisheye.{name}")
    assert fisheye_flag(name) == expected


def test_fisheye_flags_are_distinct_bits():
    values = [fisheye_flag(n) for n in FISHEYE_NAMES]
    assert len(set(values)) == len(values)


def test_pinhole_flags_are_distinct_bits():
    values = [pinhole_flag(n) for n in PINHOLE_NAMES]
    assert len(set(values)) == len(values)


def test_an_unknown_flag_names_the_opencv_version():
    with pytest.raises(RefitError, match=cv2.__version__):
        pinhole_flag("CALIB_MAKE_IT_PERFECT")
    with pytest.raises(RefitError, match=cv2.__version__):
        fisheye_flag("CALIB_MAKE_IT_PERFECT")


def test_lookups_are_cached_and_stable():
    assert pinhole_flag("CALIB_RATIONAL_MODEL") == pinhole_flag("CALIB_RATIONAL_MODEL")
