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

"""The renderer is test infrastructure, so it needs its own tests.

If the renderer is wrong, every detector test is wrong in the same direction and
none of them notice. The check that matters is that the projection model the
renderer paints with agrees with the projection model caltrust computes: render
the board, detect it, and confirm the corners land where `projectPoints` says.
"""

from __future__ import annotations

import numpy as np
import pytest

from caltrust.core.target import CharucoBoard, Checkerboard, CircleGrid
from caltrust.refit.projection import projector_for
from caltrust.synthetic import pose_for_view

from .rendering import BOARD_MARGIN_MM, board_texture, pattern_extent, render_view

pytestmark = pytest.mark.slow


def test_checkerboard_extent_runs_half_a_square_past_the_outer_corners():
    target = Checkerboard(9, 6, 25.0)
    low, high = pattern_extent(target)
    assert low.tolist() == pytest.approx([-25.0 - BOARD_MARGIN_MM] * 2)
    assert high[0] == pytest.approx(9 * 25.0 + BOARD_MARGIN_MM)
    assert high[1] == pytest.approx(6 * 25.0 + BOARD_MARGIN_MM)


def test_charuco_extent_covers_the_whole_board_not_just_its_corners():
    target = CharucoBoard(8, 11, 20.0, 15.0)
    low, high = pattern_extent(target)
    assert low.tolist() == pytest.approx([-BOARD_MARGIN_MM] * 2)
    assert high[0] == pytest.approx(8 * 20.0 + BOARD_MARGIN_MM)
    assert high[1] == pytest.approx(11 * 20.0 + BOARD_MARGIN_MM)


def test_circle_grid_extent_is_the_point_bounding_box():
    target = CircleGrid(4, 11, 20.0, True)
    low, high = pattern_extent(target)
    points = target.object_points()[:, :2]
    assert low.tolist() == pytest.approx((points.min(axis=0) - BOARD_MARGIN_MM).tolist())


@pytest.mark.parametrize(
    "target",
    [Checkerboard(9, 6, 25.0), CircleGrid(4, 11, 20.0, True), CharucoBoard(8, 11, 20.0, 15.0)],
    ids=["checkerboard", "circle_grid", "charuco"],
)
def test_texture_is_bimodal_and_the_right_shape(target):
    texture, rectangle_mm, rectangle_texture = board_texture(target)
    assert texture.dtype == np.uint8
    assert texture.ndim == 2
    assert texture.min() < 60 and texture.max() > 200
    assert rectangle_mm.shape == (4, 2)
    assert rectangle_texture.min() == pytest.approx(0.0)


def test_an_unsupported_target_has_no_texture_renderer():
    """A subclass of a known board still renders; a genuinely new one must not."""
    from dataclasses import dataclass

    from caltrust.core.target import TargetSpec

    @dataclass(frozen=True)
    class Dartboard(TargetSpec):
        rings: int = 3
        units: str = "mm"
        kind = "dartboard"

        @property
        def num_points(self):
            return self.rings

        def _all_object_points_mm(self):
            return np.column_stack([np.arange(self.rings) * 10.0,
                                    np.zeros(self.rings), np.zeros(self.rings)])

        def describe(self):
            return "a dartboard"

        def _fields(self):
            return {"rings": self.rings}

    with pytest.raises(TypeError, match="no texture renderer"):
        board_texture(Dartboard())


def test_rendered_corners_agree_with_the_projection_model(pinhole):
    """The renderer's geometry and caltrust's projector must be the same model."""
    from caltrust.ingest.detectors import detector_for

    target = CircleGrid(4, 11, 20.0, True)
    pose = pose_for_view(target, 650.0, tilt_rad=0.3, tilt_axis_rad=0.9, roll_rad=0.2)
    image = render_view(pinhole, target, pose, (1280, 720))
    detected = detector_for(target).detect(image, "v0")
    truth = projector_for(pinhole).project(
        pinhole, pose, target.object_points(detected.point_ids)
    )
    error = np.linalg.norm(detected.image_points - truth, axis=1)
    assert error.mean() < 0.3


def test_blur_and_noise_change_the_image_without_breaking_detection(pinhole):
    from caltrust.ingest.detectors import detector_for

    target = Checkerboard(9, 6, 25.0)
    pose = pose_for_view(target, 700.0, tilt_rad=0.25)
    clean = render_view(pinhole, target, pose, (1280, 720))
    rough = render_view(pinhole, target, pose, (1280, 720), blur=1.2, noise=4.0)
    assert not np.array_equal(clean, rough)
    assert detector_for(target).detect(rough, "v0").n_points == target.num_points


def test_the_background_level_is_honoured(pinhole):
    target = Checkerboard(9, 6, 25.0)
    pose = pose_for_view(target, 2500.0)
    image = render_view(pinhole, target, pose, (1280, 720), background=17)
    assert image[0, 0] == 17


def test_rendering_is_deterministic(pinhole):
    target = Checkerboard(9, 6, 25.0)
    pose = pose_for_view(target, 700.0, tilt_rad=0.25)
    first = render_view(pinhole, target, pose, (1280, 720), noise=3.0, seed=5)
    second = render_view(pinhole, target, pose, (1280, 720), noise=3.0, seed=5)
    assert np.array_equal(first, second)


def test_fisheye_rendering_bends_straight_lines(fisheye, pinhole):
    """A distortion the renderer applies must actually show up in the pixels."""
    target = Checkerboard(9, 6, 25.0)
    pose = pose_for_view(target, 400.0)
    curved = render_view(fisheye, target, pose, (1280, 720))
    straight = render_view(
        type(pinhole)(fisheye.fx, fisheye.fy, fisheye.cx, fisheye.cy, np.zeros(5)),
        target, pose, (1280, 720),
    )
    assert np.abs(curved.astype(int) - straight.astype(int)).mean() > 1.0


def test_the_renderers_own_bias_dominates_a_fit_but_not_a_static_capture(pinhole, tmp_path):
    """Quantifies open item 18, and demonstrates what the noise floor is for.

    The renderer interpolates twice, which leaves a deterministic sub-pixel
    error. In a multi-pose capture that error varies per view and lands in the
    residuals looking exactly like noise, so the fit's `sigma` reports it as
    such — at zero sensor noise the fit still claims a sigma. In a static
    capture the same error is identical in every frame, so it cancels out of the
    scatter and the measured floor sees only the sensor.

    If this ever fails because the fit's sigma tracks the sensor noise, the
    renderer's bias has been fixed and item 18 can be closed.
    """
    from caltrust import instrument, measure_noise_floor
    from caltrust.ingest import detect_in_images, find_images, session_from_images
    from caltrust.synthetic import diverse_poses

    from .rendering import write_static_capture, write_views

    target = Checkerboard(9, 6, 25.0)
    poses = diverse_poses(
        target, 14, distances_mm=(450.0, 700.0, 1000.0),
        max_tilt_rad=0.55, lateral_mm=100.0, seed=3,
    )

    def fit_sigma(noise):
        directory = tmp_path / f"cal{noise}"
        write_views(str(directory), pinhole, target, poses, (1280, 720),
                    noise=noise, blur=1.0)
        return instrument(session_from_images(str(directory), target)).covariance.sigma

    def floor_sigma(noise):
        directory = tmp_path / f"static{noise}"
        write_static_capture(
            str(directory), pinhole, target,
            pose_for_view(target, 700.0, tilt_rad=0.25), 40, (1280, 720),
            noise=noise, blur=1.0, seed=500,
        )
        observations = detect_in_images(find_images(str(directory)), target)
        return measure_noise_floor(observations).irreducible_sigma_px

    # A fit reports a sigma even with no sensor noise at all: that is the
    # renderer, not the sensor.
    assert fit_sigma(0.0) > 0.05
    # And it does not grow with sensor noise, because the bias dominates.
    assert fit_sigma(8.0) < fit_sigma(0.0) * 1.2
    # The static capture does track the sensor, roughly in proportion.
    quiet, loud = floor_sigma(2.0), floor_sigma(8.0)
    assert loud > quiet * 1.5
    assert loud < fit_sigma(0.0)
