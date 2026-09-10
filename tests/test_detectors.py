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

"""Detectors, run against rendered images.

A detector can only be tested against pixels. These render a board through a
known camera and check that OpenCV finds the pattern on its own, and that the
points it returns land where the projection says they should.

Checkerboard ordering carries a 180-degree ambiguity that OpenCV resolves by its
own convention, so the localisation check for that target compares point sets
rather than assuming the ids line up. For intrinsics the ambiguity is harmless,
because each view's pose is free and absorbs the flip.
"""

from __future__ import annotations

import numpy as np
import pytest

from calibsense.core.target import CharucoBoard, Checkerboard, CircleGrid
from calibsense.errors import DetectionError, ValidationError
from calibsense.ingest.detectors import (
    CharucoDetector,
    CheckerboardDetector,
    DetectorOptions,
    detector_for,
    registered_detectors,
)
from calibsense.ingest.detectors.opencv_support import (
    aruco_dictionary,
    image_size_of,
    to_gray,
)
from calibsense.refit.projection import projector_for
from calibsense.synthetic import pose_for_view

from .rendering import render_view

pytestmark = pytest.mark.slow


def localisation_error(detected, camera, target, pose, ordered):
    truth = projector_for(camera).project(
        camera, pose, target.object_points(detected.point_ids)
    )
    if ordered:
        return np.linalg.norm(detected.image_points - truth, axis=1)
    distances = np.linalg.norm(
        detected.image_points[:, None, :] - truth[None, :, :], axis=2
    )
    assert len({int(i) for i in distances.argmin(axis=1)}) == len(detected.point_ids)
    return distances.min(axis=1)


@pytest.mark.parametrize(
    "target,ordered",
    [
        (Checkerboard(9, 6, 25.0), False),
        (CircleGrid(4, 11, 20.0, True), True),
        (CharucoBoard(8, 11, 20.0, 15.0), True),
    ],
    ids=["checkerboard", "circle_grid", "charuco"],
)
def test_detects_a_full_board_to_sub_pixel_accuracy(pinhole, target, ordered):
    pose = pose_for_view(target, 700.0, tilt_rad=0.35, tilt_axis_rad=0.7, roll_rad=0.1)
    image = render_view(pinhole, target, pose, (1280, 720), noise=1.5)
    detected = detector_for(target).detect(image, "v0")
    assert detected.n_points == target.num_points
    error = localisation_error(detected, pinhole, target, pose, ordered)
    assert error.mean() < 0.6
    assert error.max() < 1.5


@pytest.mark.parametrize(
    "target,ordered",
    [
        (Checkerboard(9, 6, 25.0), False),
        (CircleGrid(4, 11, 20.0, True), True),
        (CharucoBoard(8, 11, 20.0, 15.0), True),
    ],
    ids=["checkerboard", "circle_grid", "charuco"],
)
def test_detects_through_a_fisheye_lens(fisheye, target, ordered):
    pose = pose_for_view(target, 420.0, tilt_rad=0.3, tilt_axis_rad=1.2, roll_rad=-0.2)
    image = render_view(fisheye, target, pose, (1280, 720), noise=1.5)
    detected = detector_for(target).detect(image, "v0")
    assert detected.n_points >= target.num_points // 2
    error = localisation_error(detected, fisheye, target, pose, ordered)
    assert error.mean() < 1.0


def test_checkerboard_ids_run_over_the_whole_grid(pinhole):
    target = Checkerboard(9, 6, 25.0)
    pose = pose_for_view(target, 700.0, tilt_rad=0.2)
    detected = detector_for(target).detect(
        render_view(pinhole, target, pose, (1280, 720)), "v0"
    )
    assert np.array_equal(detected.point_ids, np.arange(target.num_points))
    assert detected.metadata["detector"] == "checkerboard"


def test_charuco_handles_a_partly_visible_board(pinhole):
    """Partial detection is the reason to choose this target."""
    target = CharucoBoard(8, 11, 20.0, 15.0)
    pose = pose_for_view(target, 300.0, tilt_rad=0.25, offset_mm=(60.0, 30.0))
    detected = detector_for(target).detect(
        render_view(pinhole, target, pose, (1280, 720), noise=1.0), "v0"
    )
    assert 0 < detected.n_points < target.num_points
    assert detected.point_ids.max() < target.num_points
    assert 0.0 < detected.metadata["coverage"] < 1.0
    assert detected.metadata["markers"] > 0
    error = localisation_error(detected, pinhole, target, pose, ordered=True)
    assert error.mean() < 1.0


def test_a_blank_image_yields_no_detection(pinhole):
    target = Checkerboard(9, 6, 25.0)
    blank = np.full((720, 1280), 200, np.uint8)
    with pytest.raises(DetectionError, match="no 9x6 checkerboard found"):
        detector_for(target).detect(blank, "blank")


def test_a_blank_image_yields_no_circle_grid():
    target = CircleGrid(4, 11, 20.0, True)
    blank = np.full((720, 1280), 200, np.uint8)
    with pytest.raises(DetectionError, match="no 4x11 circle grid found"):
        detector_for(target).detect(blank, "blank")


def test_a_blank_image_yields_no_charuco():
    target = CharucoBoard(8, 11, 20.0, 15.0)
    blank = np.full((720, 1280), 200, np.uint8)
    with pytest.raises(DetectionError, match="no ChArUco corners resolved"):
        detector_for(target).detect(blank, "blank")


def test_min_points_rejects_a_thin_charuco_detection(pinhole):
    target = CharucoBoard(8, 11, 20.0, 15.0)
    pose = pose_for_view(target, 260.0, tilt_rad=0.2, offset_mm=(110.0, 70.0))
    image = render_view(pinhole, target, pose, (1280, 720), noise=1.0)
    loose = detector_for(target, DetectorOptions(min_points=4)).detect(image, "v0")
    strict = detector_for(target, DetectorOptions(min_points=loose.n_points + 1))
    with pytest.raises(DetectionError, match="below the minimum"):
        strict.detect(image, "v0")


def test_colour_and_float_images_are_accepted(pinhole):
    target = Checkerboard(9, 6, 25.0)
    pose = pose_for_view(target, 700.0, tilt_rad=0.2)
    gray = render_view(pinhole, target, pose, (1280, 720))
    colour = np.repeat(gray[:, :, None], 3, axis=2)
    detector = detector_for(target)
    assert detector.detect(colour, "colour").n_points == target.num_points
    as_float = gray.astype(np.float32) / 255.0
    assert detector.detect(as_float, "float").n_points == target.num_points


def test_fast_mode_still_detects_a_clean_board(pinhole):
    target = Checkerboard(9, 6, 25.0)
    pose = pose_for_view(target, 700.0, tilt_rad=0.2)
    image = render_view(pinhole, target, pose, (1280, 720))
    detected = detector_for(target, DetectorOptions(fast=True)).detect(image, "v0")
    assert detected.n_points == target.num_points


def test_detector_kind_must_match_the_target():
    with pytest.raises(ValidationError, match="handles 'checkerboard' targets"):
        CheckerboardDetector(CircleGrid(4, 11, 20.0))


def test_no_detector_for_an_unknown_target_kind():
    class Odd(Checkerboard):
        kind = "dartboard"

    with pytest.raises(ValidationError, match="no detector for target kind"):
        detector_for(Odd(9, 6, 25.0))


def test_all_three_targets_have_a_detector():
    assert registered_detectors() == ("charuco", "checkerboard", "circle_grid")


@pytest.mark.parametrize(
    "kwargs,message",
    [({"refine_window": 0}, "refine_window"), ({"min_points": 3}, "min_points")],
)
def test_invalid_detector_options_are_rejected(kwargs, message):
    with pytest.raises(ValidationError, match=message):
        DetectorOptions(**kwargs)


def test_charuco_rejects_an_unknown_dictionary():
    with pytest.raises(ValidationError, match="unknown ArUco dictionary"):
        CharucoDetector(CharucoBoard(8, 11, 20.0, 15.0, dictionary="DICT_9X9_1"))


def test_charuco_board_object_points_match_the_target_spec():
    target = CharucoBoard(8, 11, 20.0, 15.0)
    detector = CharucoDetector(target)
    assert np.allclose(
        detector.board.getChessboardCorners(), target.object_points()
    )


def test_aruco_dictionary_lookup_is_case_insensitive():
    assert aruco_dictionary("dict_5x5_1000") is not None


def test_to_gray_handles_every_channel_count():
    assert to_gray(np.zeros((4, 4), np.uint8)).shape == (4, 4)
    assert to_gray(np.zeros((4, 4, 1), np.uint8)).shape == (4, 4)
    assert to_gray(np.zeros((4, 4, 3), np.uint8)).shape == (4, 4)
    assert to_gray(np.zeros((4, 4, 4), np.uint8)).shape == (4, 4)


def test_to_gray_rejects_odd_shapes():
    with pytest.raises(DetectionError, match="unsupported channel count"):
        to_gray(np.zeros((4, 4, 7), np.uint8))
    with pytest.raises(DetectionError, match="2-D or 3-D"):
        to_gray(np.zeros(4, np.uint8))


def test_to_gray_rejects_an_all_nan_float_image():
    with pytest.raises(DetectionError, match="no finite pixels"):
        to_gray(np.full((4, 4), np.nan))


def test_to_gray_scales_a_float_image_into_range():
    scaled = to_gray(np.linspace(0.0, 4.0, 16).reshape(4, 4))
    assert scaled.dtype == np.uint8 and scaled.max() == 255


def test_image_size_is_width_then_height():
    assert image_size_of(np.zeros((480, 640, 3), np.uint8)) == (640, 480)
    with pytest.raises(ValidationError, match="image array"):
        image_size_of(np.zeros(5))


def test_the_classic_detector_path_also_works(pinhole, monkeypatch):
    """findChessboardCornersSB is tried first; the fallback has to work too."""
    import cv2

    target = Checkerboard(9, 6, 25.0)
    pose = pose_for_view(target, 700.0, tilt_rad=0.3, roll_rad=0.15)
    image = render_view(pinhole, target, pose, (1280, 720), noise=1.0)

    monkeypatch.setattr(cv2, "findChessboardCornersSB", lambda *a, **k: (False, None))
    detected = detector_for(target).detect(image, "v0")
    assert detected.n_points == target.num_points
    assert detected.metadata["method"] == "findChessboardCorners+cornerSubPix"
    error = localisation_error(detected, pinhole, target, pose, ordered=False)
    assert error.mean() < 0.6


def test_the_classic_path_can_skip_refinement(pinhole, monkeypatch):
    import cv2

    target = Checkerboard(9, 6, 25.0)
    pose = pose_for_view(target, 700.0, tilt_rad=0.3)
    image = render_view(pinhole, target, pose, (1280, 720))
    monkeypatch.setattr(cv2, "findChessboardCornersSB", lambda *a, **k: (False, None))
    detector = detector_for(target, DetectorOptions(refine=False))
    assert detector.detect(image, "v0").n_points == target.num_points


def test_an_opencv_error_in_the_sb_detector_falls_through(pinhole, monkeypatch):
    import cv2

    target = Checkerboard(9, 6, 25.0)
    pose = pose_for_view(target, 700.0, tilt_rad=0.3)
    image = render_view(pinhole, target, pose, (1280, 720))

    def explode(*args, **kwargs):
        raise cv2.error("SB is unhappy")

    monkeypatch.setattr(cv2, "findChessboardCornersSB", explode)
    assert detector_for(target).detect(image, "v0").n_points == target.num_points


def test_refine_corners_shrinks_its_window_near_an_edge():
    from calibsense.ingest.detectors.opencv_support import refine_corners

    gray = np.zeros((40, 40), np.uint8)
    gray[20:, 20:] = 255
    # A corner one pixel from the border leaves no room for a window at all.
    hugging = np.array([[[1.0, 1.0]]], dtype=np.float32)
    assert np.allclose(refine_corners(gray, hugging, 5), hugging)
    # One with room gets refined towards the intensity corner.
    inside = np.array([[[19.0, 19.0]]], dtype=np.float32)
    refined = refine_corners(gray, inside, 5)
    assert refined.shape == inside.shape
