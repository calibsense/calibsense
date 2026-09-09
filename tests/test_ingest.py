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

"""Image ingest and the two pipeline entry points."""

from __future__ import annotations

import os

import cv2
import numpy as np
import pytest

from caltrust.core.target import Checkerboard
from caltrust.errors import IngestError, UnsupportedFormatError
from caltrust.ingest import (
    DetectorOptions,
    detect_in_images,
    find_images,
    read_image,
    session_from_calibration,
    session_from_images,
    write_detections,
)
from caltrust.ingest.images import IMAGE_SUFFIXES, view_id_for
from caltrust.synthetic import diverse_poses

from .rendering import write_views

pytestmark = pytest.mark.slow

TARGET = Checkerboard(9, 6, 25.0)


@pytest.fixture
def capture_directory(tmp_path, pinhole):
    """A rendered capture, plus one blank image and one corrupt file."""
    directory = tmp_path / "images"
    poses = diverse_poses(
        TARGET, 10, distances_mm=(450.0, 700.0, 1000.0),
        max_tilt_rad=0.55, lateral_mm=100.0, seed=9,
    )
    write_views(str(directory), pinhole, TARGET, poses, (1280, 720), noise=1.5, blur=0.6)
    cv2.imwrite(str(directory / "blank_zz.png"), np.full((720, 1280), 200, np.uint8))
    (directory / "broken_zz.png").write_bytes(b"not an image")
    return directory


@pytest.fixture
def opencv_calibration(tmp_path):
    """A deliberately wrong calibration, in OpenCV FileStorage form."""
    path = tmp_path / "shipped.yml"
    storage = cv2.FileStorage(str(path), cv2.FILE_STORAGE_WRITE)
    storage.write("image_width", 1280)
    storage.write("image_height", 720)
    storage.write("camera_matrix", np.array([[870.0, 0.0, 630.0], [0.0, 875.0, 350.0],
                                             [0.0, 0.0, 1.0]]))
    storage.write("distortion_coefficients", np.array([-0.19, 0.04, 0.0, 0.0, 0.0]))
    storage.write("avg_reprojection_error", 0.2412)
    storage.release()
    return path


def test_find_images_is_sorted_and_absolute(capture_directory):
    paths = find_images(str(capture_directory))
    assert paths == sorted(paths)
    assert all(os.path.isabs(p) for p in paths)
    assert len(paths) == 12


def test_find_images_accepts_a_single_file(capture_directory):
    one = find_images(str(capture_directory))[0]
    assert find_images(one) == [one]


def test_find_images_descends_by_default(tmp_path, pinhole):
    nested = tmp_path / "top" / "inner"
    write_views(str(nested), pinhole, TARGET, diverse_poses(TARGET, 2, seed=1))
    assert len(find_images(str(tmp_path / "top"))) == 2
    with pytest.raises(IngestError, match="no images under"):
        find_images(str(tmp_path / "top"), recursive=False)


def test_find_images_reports_a_missing_path(tmp_path):
    with pytest.raises(IngestError, match="no such path"):
        find_images(str(tmp_path / "absent"))


def test_find_images_reports_an_empty_directory(tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(IngestError, match="no images under"):
        find_images(str(empty))


def test_every_listed_suffix_is_lowercase_with_a_dot():
    assert all(s.startswith(".") and s.islower() for s in IMAGE_SUFFIXES)


def test_read_image_rejects_an_empty_and_a_corrupt_file(tmp_path):
    empty = tmp_path / "empty.png"
    empty.write_bytes(b"")
    with pytest.raises(IngestError, match="is empty"):
        read_image(str(empty))
    corrupt = tmp_path / "corrupt.png"
    corrupt.write_bytes(b"nope")
    with pytest.raises(IngestError, match="not an image"):
        read_image(str(corrupt))


def test_view_ids_come_from_the_filename_and_disambiguate():
    assert view_id_for("/a/b/view007.png", []) == "view007"
    assert view_id_for("/a/b/view007.png", ["view007"]) == "view007#2"
    assert view_id_for("/a/b/view007.png", ["view007", "view007#2"]) == "view007#3"


def test_detection_records_failures_without_stopping(capture_directory):
    observations = detect_in_images(find_images(str(capture_directory)), TARGET)
    assert observations.n_views == 10
    assert observations.summary.attempted == 12
    assert len(observations.summary.failures) == 2
    reasons = " ".join(f.reason for f in observations.summary.failures)
    assert "no 9x6 checkerboard found" in reasons
    assert "not an image" in reasons
    assert observations.summary.success_rate == pytest.approx(10 / 12)


def test_detection_reports_progress(capture_directory):
    seen = []
    detect_in_images(
        find_images(str(capture_directory)), TARGET,
        progress=lambda index, total, view_id, error: seen.append((index, total, error)),
    )
    assert len(seen) == 12
    assert seen[-1][0] == 12 and seen[-1][1] == 12
    assert sum(1 for _, _, error in seen if error is not None) == 2


def test_mixed_image_sizes_abort_the_ingest(capture_directory, pinhole):
    odd = capture_directory / "zz_small.png"
    cv2.imwrite(str(odd), np.full((360, 640), 200, np.uint8))
    with pytest.raises(IngestError, match="cannot\nmix image sizes|mix image sizes"):
        detect_in_images(find_images(str(capture_directory)), TARGET)


def test_a_capture_with_no_detections_is_an_error(tmp_path):
    directory = tmp_path / "blanks"
    directory.mkdir()
    for index in range(3):
        cv2.imwrite(str(directory / f"b{index}.png"), np.full((720, 1280), 200, np.uint8))
    with pytest.raises(IngestError, match="no image yielded a detection"):
        detect_in_images(find_images(str(directory)), TARGET)


def test_session_from_images_recovers_truth(capture_directory, pinhole):
    from caltrust.refit import instrument

    session = session_from_images(str(capture_directory), TARGET)
    assert session.observations.n_views == 10
    assert session.prior is None
    assert session.metadata["ingest"]["source"] == "images"
    fit = instrument(session)
    assert abs(fit.camera.fx - pinhole.fx) < 4.0 * fit.covariance.intrinsic_std()[0]


def test_session_from_images_attaches_the_existing_calibration(
    capture_directory, opencv_calibration
):
    session = session_from_images(
        str(capture_directory), TARGET, calibration=str(opencv_calibration)
    )
    assert session.prior is not None
    assert session.prior.reported_rms == pytest.approx(0.2412)
    assert session.prior.camera.fx == pytest.approx(870.0)


def test_a_calibration_for_a_different_image_size_is_refused(capture_directory, tmp_path):
    path = tmp_path / "wrong.yml"
    storage = cv2.FileStorage(str(path), cv2.FILE_STORAGE_WRITE)
    storage.write("image_width", 640)
    storage.write("image_height", 480)
    storage.write("camera_matrix", np.eye(3) * 400.0)
    storage.write("distortion_coefficients", np.zeros(5))
    storage.release()
    with pytest.raises(IngestError, match="calibration for"):
        session_from_images(str(capture_directory), TARGET, calibration=str(path))


def test_detector_options_flow_through(capture_directory):
    session = session_from_images(
        str(capture_directory), TARGET, options=DetectorOptions(min_points=8, fast=True)
    )
    assert session.observations.n_views >= 8


def test_session_from_calibration_round_trips_through_detections(
    capture_directory, opencv_calibration, tmp_path
):
    first = session_from_images(str(capture_directory), TARGET)
    detections = write_detections(first.observations, str(tmp_path / "d.json"))
    session = session_from_calibration(str(opencv_calibration), detections)
    assert session.observations.n_views == first.observations.n_views
    assert session.prior.camera.fx == pytest.approx(870.0)
    assert session.metadata["ingest"]["source"] == "calibration"


def test_session_from_calibration_needs_matching_image_sizes(
    capture_directory, tmp_path
):
    first = session_from_images(str(capture_directory), TARGET)
    detections = write_detections(first.observations, str(tmp_path / "d.json"))
    path = tmp_path / "small.yml"
    storage = cv2.FileStorage(str(path), cv2.FILE_STORAGE_WRITE)
    storage.write("image_width", 640)
    storage.write("image_height", 480)
    storage.write("camera_matrix", np.eye(3) * 400.0)
    storage.write("distortion_coefficients", np.zeros(5))
    storage.release()
    with pytest.raises(IngestError, match="not the same capture"):
        session_from_calibration(str(path), detections)


def test_session_from_images_attaches_robot_poses(capture_directory, tmp_path):
    import json

    from caltrust.core.poses import Pose

    first = session_from_images(str(capture_directory), TARGET)
    poses = [
        {"view_id": view_id,
         "matrix": Pose.from_rvec_tvec([0.05 * i, 0.0, 0.0], [i, 0.0, 400.0]).matrix.tolist()}
        for i, view_id in enumerate(first.observations.view_ids)
    ]
    path = tmp_path / "r.json"
    path.write_text(json.dumps({"poses": poses}))
    session = session_from_images(str(capture_directory), TARGET, robot_poses=str(path))
    assert session.has_hand_eye
    assert len(session.robot.aligned_with(session.observations)) == session.observations.n_views


def test_a_missing_robot_pose_file_is_reported(capture_directory, tmp_path):
    with pytest.raises(UnsupportedFormatError, match="no such robot pose file"):
        session_from_images(
            str(capture_directory), TARGET, robot_poses=str(tmp_path / "absent.json")
        )
