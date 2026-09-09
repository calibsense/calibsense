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

"""Shared fixtures.

Everything is built from the synthetic generator rather than checked-in data, so
the suite has no fixture files to drift out of date and every test can state the
truth it expects to recover.
"""

from __future__ import annotations

import numpy as np
import pytest

from caltrust.core.camera import FisheyeKannalaBrandt, PinholeBrownConrady
from caltrust.core.observations import ObservationSet, ViewObservations
from caltrust.core.session import CalibrationRecord, CalibrationSession
from caltrust.core.target import CharucoBoard, Checkerboard, CircleGrid
from caltrust.synthetic import diverse_poses, frontoparallel_poses, synthesise

IMAGE_SIZE = (1280, 720)
NOISE_PX = 0.2


@pytest.fixture
def pinhole():
    """A pinhole camera with five distortion coefficients."""
    return PinholeBrownConrady(900.0, 905.0, 639.5, 359.5, [-0.21, 0.06, 0.001, -0.002, 0.01])


@pytest.fixture
def fisheye():
    """A fisheye camera with four Kannala-Brandt coefficients."""
    return FisheyeKannalaBrandt(430.0, 432.0, 639.5, 359.5, [-0.05, 0.01, -0.002, 0.0004])


@pytest.fixture
def checkerboard():
    """A 9x6 checkerboard with 25 mm squares."""
    return Checkerboard(9, 6, 25.0)


@pytest.fixture
def charuco():
    """An 8x11 ChArUco board with 20 mm squares and 15 mm markers."""
    return CharucoBoard(8, 11, 20.0, 15.0)


@pytest.fixture
def circle_grid():
    """A 4x11 asymmetric circle grid with 20 mm spacing."""
    return CircleGrid(4, 11, 20.0, True)


@pytest.fixture
def good_capture(pinhole, checkerboard):
    """A well-conditioned pinhole capture: real tilt, three working distances."""
    return synthesise(
        pinhole, checkerboard, diverse_poses(checkerboard, 16, seed=4),
        IMAGE_SIZE, noise_px=NOISE_PX, seed=17,
    )


@pytest.fixture
def degenerate_capture(pinhole, checkerboard):
    """A frontoparallel capture, where the focal length is not identifiable."""
    return synthesise(
        pinhole, checkerboard, frontoparallel_poses(checkerboard, 16, 800.0, seed=4),
        IMAGE_SIZE, noise_px=NOISE_PX, seed=17,
    )


@pytest.fixture
def fisheye_capture(fisheye, charuco):
    """A well-conditioned fisheye capture on a ChArUco board."""
    return synthesise(
        fisheye, charuco,
        diverse_poses(charuco, 18, distances_mm=(300.0, 500.0, 800.0),
                      max_tilt_rad=0.8, seed=6),
        IMAGE_SIZE, noise_px=NOISE_PX, seed=17,
    )


@pytest.fixture
def good_session(good_capture):
    """A session over `good_capture`, with no existing calibration."""
    return CalibrationSession(observations=good_capture.observations)


@pytest.fixture
def session_with_prior(good_capture, pinhole):
    """A session whose existing calibration is deliberately a little wrong."""
    wrong = PinholeBrownConrady(870.0, 875.0, 630.0, 350.0, [-0.19, 0.04, 0.0, 0.0, 0.0])
    return CalibrationSession(
        observations=good_capture.observations,
        prior=CalibrationRecord(wrong, IMAGE_SIZE, "shipped.yml", reported_rms=0.2412),
    )


@pytest.fixture
def tiny_observations(checkerboard):
    """Three views of made-up points, for validation tests that need no geometry."""
    rng = np.random.default_rng(0)
    views = tuple(
        ViewObservations(
            f"v{i}", np.arange(checkerboard.num_points),
            rng.uniform(50, 600, (checkerboard.num_points, 2)),
        )
        for i in range(3)
    )
    return ObservationSet(checkerboard, IMAGE_SIZE, views)
