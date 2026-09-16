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

import cv2
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


# --------------------------------------------------------------------------
# the ChArUco legacy marker layout
# --------------------------------------------------------------------------

def _charuco_image(squares_x, squares_y, legacy, pixels=100, margin=40):
    """A rendered ChArUco board under one of the two marker layouts."""
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_1000)
    board = cv2.aruco.CharucoBoard((squares_x, squares_y), 20.0, 15.0, dictionary)
    board.setLegacyPattern(legacy)
    return board.generateImage(
        (squares_x * pixels, squares_y * pixels), marginSize=margin
    )


@pytest.mark.parametrize("squares_y", [3, 5, 7, 9, 11])
def test_the_legacy_layout_is_a_no_op_when_the_y_count_is_odd(squares_y):
    """Including the 8x11 default, where the flag cannot cause anything.

    open-items treated this as a live hazard on every board. The two layouts
    are the same image unless the square count in y is even.
    """
    modern = _charuco_image(8, squares_y, False)
    legacy = _charuco_image(8, squares_y, True)
    assert np.array_equal(modern, legacy)


@pytest.mark.parametrize("squares_y", [4, 6, 8, 10])
def test_the_legacy_layout_differs_when_the_y_count_is_even(squares_y):
    modern = _charuco_image(8, squares_y, False)
    legacy = _charuco_image(8, squares_y, True)
    assert not np.array_equal(modern, legacy)


def test_the_wrong_legacy_setting_fails_detection_rather_than_calibrating_wrongly():
    """open-items had this backwards, and the difference matters.

    It recorded that a wrong setting produces a confident, wrong calibration
    rather than a detection failure. It produces the failure: the ArUco markers
    are read either way and no chessboard corners come back at all, so every
    image is rejected and nothing reaches the fit.
    """
    image = _charuco_image(8, 10, legacy=False)
    wrong = CharucoDetector(
        CharucoBoard(8, 10, 20.0, 15.0, "DICT_5X5_1000", legacy_pattern=True)
    )
    with pytest.raises(DetectionError) as raised:
        wrong.detect(image, "v0")
    message = str(raised.value)
    assert "ArUco markers were read" in message
    assert "legacy_pattern" in message

    right = CharucoDetector(
        CharucoBoard(8, 10, 20.0, 15.0, "DICT_5X5_1000", legacy_pattern=False)
    )
    assert right.detect(image, "v0").n_points > 40


def test_the_legacy_hint_is_withheld_where_the_flag_cannot_be_the_cause():
    """On an odd y count the layouts agree, so blaming the flag misdirects."""
    detector = CharucoDetector(
        CharucoBoard(8, 11, 20.0, 15.0, "DICT_5X5_1000", legacy_pattern=True)
    )
    reason = detector._no_corners_reason(40)
    assert "ArUco markers were read" in reason
    assert "legacy_pattern" not in reason

    assert "no ArUco markers either" in detector._no_corners_reason(0)


# --------------------------------------------------------------------------
# the circle grid's centroid bias
# --------------------------------------------------------------------------

def _ellipse_centre(conic):
    """Centre of a conic given as a 3x3 symmetric matrix."""
    a, b, d = conic[0, 0], conic[0, 1], conic[0, 2]
    c, e = conic[1, 1], conic[1, 2]
    return np.linalg.solve(np.array([[a, b], [b, c]]), -np.array([d, e]))


def _centroid_bias_px(focal, radius_mm, depth_mm, tilt_deg):
    """The gap between a circle's projected centre and its projected centroid.

    Exact rather than rendered: a perspective map sends a circle to an ellipse
    and an ellipse's centroid is its centre, so both sides are closed forms and
    neither the renderer's own bias nor the blob detector is in the way.
    """
    angle = np.radians(tilt_deg)
    rotation = np.array([
        [1.0, 0.0, 0.0],
        [0.0, np.cos(angle), -np.sin(angle)],
        [0.0, np.sin(angle), np.cos(angle)],
    ])
    homography = (
        np.diag([focal, focal, 1.0])
        @ np.column_stack([rotation[:, 0], rotation[:, 1], [0.0, 0.0, depth_mm]])
    )
    inverse = np.linalg.inv(homography)
    projected = inverse.T @ np.diag([1.0, 1.0, -radius_mm ** 2]) @ inverse
    centre = homography @ np.array([0.0, 0.0, 1.0])
    return float(np.linalg.norm(_ellipse_centre(projected) - centre[:2] / centre[2]))


def test_the_centroid_bias_peaks_near_45_degrees_rather_than_growing():
    """The usual description of this effect, and open-items', has it as rising
    with obliquity. It rises to a maximum near 45 degrees and falls away."""
    by_tilt = {
        tilt: _centroid_bias_px(700.0, 6.0, 700.0, tilt)
        for tilt in (0, 10, 20, 30, 40, 50, 60, 70)
    }
    assert by_tilt[0] == pytest.approx(0.0, abs=1e-9)
    peak = max(by_tilt, key=by_tilt.get)
    assert 35 <= peak <= 55, by_tilt
    assert by_tilt[60] < by_tilt[40]
    assert by_tilt[70] < by_tilt[30]


def test_the_centroid_bias_is_quadratic_in_the_angular_radius():
    """Which is why it is governed by circle size and distance, not by tilt.

    Doubling the radius or halving the distance each cost four times as much,
    where the whole span of tilt from 10 to 45 degrees costs about three.
    """
    base = _centroid_bias_px(700.0, 6.0, 700.0, 40.0)
    assert _centroid_bias_px(700.0, 12.0, 700.0, 40.0) == pytest.approx(
        4.0 * base, rel=0.05
    )
    assert _centroid_bias_px(700.0, 6.0, 350.0, 40.0) == pytest.approx(
        4.0 * base, rel=0.05
    )


def test_the_centroid_bias_is_small_at_a_normal_working_distance():
    """Negligible for a small target far away, material for big circles close."""
    far = _centroid_bias_px(700.0, 6.0, 700.0, 40.0)
    near = _centroid_bias_px(700.0, 6.0, 300.0, 40.0)
    assert far < 0.05
    assert near > 0.1


def test_the_circle_grid_spec_can_carry_what_the_bias_needs():
    """The diameter is optional and unrecorded by default.

    Nothing needs it to detect or to calibrate, and only judging the result
    needs it, so requiring it would make every existing circle-grid session
    invalid to state a caveat. Absent, `CircleGridBias` says it cannot check
    rather than staying silent — silence would read as the problem being absent
    rather than the input.
    """
    stated = CircleGrid(4, 11, 20.0, True, diameter=6.0)
    assert stated.diameter == 6.0
    assert "6 mm circles" in stated.describe()

    unstated = CircleGrid(4, 11, 20.0, True)
    assert unstated.diameter == 0.0
    assert "circles" not in unstated.describe().split("spacing")[-1]

    with pytest.raises(ValidationError, match="would touch"):
        CircleGrid(4, 11, 20.0, True, diameter=25.0)
