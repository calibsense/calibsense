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

"""Reading calibrations produced by other tools."""

from __future__ import annotations

import json

import cv2
import numpy as np
import pytest

from caltrust.core.camera import FisheyeKannalaBrandt, PinholeBrownConrady
from caltrust.core.session import CalibrationRecord
from caltrust.errors import UnsupportedFormatError
from caltrust.ingest.readers import (
    KalibrReader,
    NativeJsonReader,
    default_readers,
    read_calibration,
    reader_names,
    write_calibration,
)

MATRIX = np.array([[900.0, 0.0, 639.5], [0.0, 905.0, 359.5], [0.0, 0.0, 1.0]])
DISTORTION = np.array([-0.21, 0.06, 0.001, -0.002, 0.01])

ROS_YAML = """image_width: 1280
image_height: 720
camera_name: front
camera_matrix: {rows: 3, cols: 3, data: [900.0, 0.0, 639.5, 0.0, 905.0, 359.5, 0.0, 0.0, 1.0]}
distortion_model: plumb_bob
distortion_coefficients: {rows: 1, cols: 5, data: [-0.21, 0.06, 0.001, -0.002, 0.01]}
"""

KALIBR_YAML = """cam0:
  camera_model: pinhole
  intrinsics: [900.0, 905.0, 639.5, 359.5]
  distortion_model: equidistant
  distortion_coeffs: [-0.02, 0.003, -0.0001, 0.00002]
  resolution: [1280, 720]
cam1:
  camera_model: pinhole
  intrinsics: [800.0, 802.0, 639.5, 359.5]
  distortion_model: radtan
  distortion_coeffs: [-0.2, 0.05, 0.001, -0.001]
  resolution: [1280, 720]
"""


def write_opencv(path, extra=None, matrix=MATRIX, distortion=DISTORTION, size=(1280, 720)):
    storage = cv2.FileStorage(str(path), cv2.FILE_STORAGE_WRITE)
    if size is not None:
        storage.write("image_width", size[0])
        storage.write("image_height", size[1])
    if matrix is not None:
        storage.write("camera_matrix", matrix)
    if distortion is not None:
        storage.write("distortion_coefficients", distortion)
    for key, value in (extra or {}).items():
        storage.write(key, value)
    storage.release()
    return str(path)


def test_opencv_filestorage(tmp_path):
    path = write_opencv(tmp_path / "cv.yml", {"avg_reprojection_error": 0.2412})
    record = read_calibration(path)
    assert isinstance(record.camera, PinholeBrownConrady)
    assert record.camera.fx == pytest.approx(900.0)
    assert record.image_size == (1280, 720)
    assert record.reported_rms == pytest.approx(0.2412)
    assert record.source.startswith("opencv:")


def test_opencv_xml_is_recognised(tmp_path):
    path = write_opencv(tmp_path / "cv.xml", {"avg_reprojection_error": 0.3})
    assert read_calibration(path).camera.fx == pytest.approx(900.0)


def test_four_coefficients_without_a_model_are_flagged_ambiguous(tmp_path):
    path = write_opencv(tmp_path / "cv.yml", distortion=DISTORTION[:4])
    record = read_calibration(path)
    assert record.metadata["model_ambiguous"] is True


def test_five_coefficients_are_not_ambiguous(tmp_path):
    record = read_calibration(write_opencv(tmp_path / "cv.yml"))
    assert record.metadata["model_ambiguous"] is False


def test_a_declared_fisheye_model_is_honoured(tmp_path):
    path = write_opencv(
        tmp_path / "cv.yml",
        {"distortion_model": "fisheye"},
        distortion=np.array([-0.02, 0.003, -1e-4, 2e-5]),
    )
    assert isinstance(read_calibration(path).camera, FisheyeKannalaBrandt)


def test_missing_image_size_is_an_error(tmp_path):
    path = write_opencv(tmp_path / "cv.yml", size=None)
    with pytest.raises(UnsupportedFormatError, match="image_width"):
        read_calibration(path)


def test_missing_camera_matrix_is_an_error(tmp_path):
    path = write_opencv(tmp_path / "cv.yml", matrix=None)
    with pytest.raises(UnsupportedFormatError, match="camera matrix"):
        read_calibration(path)


def test_missing_distortion_is_an_error(tmp_path):
    path = write_opencv(tmp_path / "cv.yml", distortion=None)
    with pytest.raises(UnsupportedFormatError, match="distortion vector"):
        read_calibration(path)


def test_wrong_shaped_camera_matrix_is_an_error(tmp_path):
    path = write_opencv(tmp_path / "cv.yml", matrix=np.eye(4))
    with pytest.raises(UnsupportedFormatError, match="camera matrix"):
        read_calibration(path)


def test_ros_camera_info(tmp_path):
    path = tmp_path / "ros.yaml"
    path.write_text(ROS_YAML)
    record = read_calibration(str(path))
    assert isinstance(record.camera, PinholeBrownConrady)
    assert record.metadata["camera_name"] == "front"
    assert record.metadata["declared_model"] == "plumb_bob"
    assert record.source.startswith("ros:")


def test_ros_equidistant_becomes_fisheye(tmp_path):
    path = tmp_path / "ros.yaml"
    path.write_text(
        ROS_YAML.replace("plumb_bob", "equidistant").replace(
            "cols: 5, data: [-0.21, 0.06, 0.001, -0.002, 0.01]",
            "cols: 4, data: [-0.02, 0.003, -0.0001, 0.00002]",
        )
    )
    assert isinstance(read_calibration(str(path)).camera, FisheyeKannalaBrandt)


def test_ros_rejects_an_unsupported_distortion_model(tmp_path):
    path = tmp_path / "ros.yaml"
    path.write_text(ROS_YAML.replace("plumb_bob", "kannala_brandt_extended"))
    with pytest.raises(UnsupportedFormatError, match="distortion_model"):
        read_calibration(str(path))


def test_ros_rejects_a_missing_key(tmp_path):
    path = tmp_path / "ros.yaml"
    path.write_text(ROS_YAML.replace("image_height: 720\n", ""))
    with pytest.raises(UnsupportedFormatError, match="image_height"):
        read_calibration(str(path))


def test_ros_equidistant_needs_four_coefficients(tmp_path):
    path = tmp_path / "ros.yaml"
    path.write_text(
        ROS_YAML.replace("plumb_bob", "equidistant").replace(
            "cols: 5, data: [-0.21, 0.06, 0.001, -0.002, 0.01]", "cols: 2, data: [-0.02, 0.003]"
        )
    )
    with pytest.raises(UnsupportedFormatError, match="4 coefficients"):
        read_calibration(str(path))


def test_kalibr_takes_the_first_camera_by_default(tmp_path):
    path = tmp_path / "camchain.yaml"
    path.write_text(KALIBR_YAML)
    record = read_calibration(str(path))
    assert isinstance(record.camera, FisheyeKannalaBrandt)
    assert record.metadata["kalibr_camera"] == "cam0"
    assert record.metadata["cameras_in_chain"] == ["cam0", "cam1"]


def test_kalibr_can_select_a_named_camera(tmp_path):
    path = tmp_path / "camchain.yaml"
    path.write_text(KALIBR_YAML)
    record = KalibrReader(camera="cam1").read(str(path))
    assert isinstance(record.camera, PinholeBrownConrady)
    assert record.camera.fx == pytest.approx(800.0)


def test_kalibr_rejects_a_missing_camera(tmp_path):
    path = tmp_path / "camchain.yaml"
    path.write_text(KALIBR_YAML)
    with pytest.raises(UnsupportedFormatError, match="has no 'cam7'"):
        KalibrReader(camera="cam7").read(str(path))


@pytest.mark.parametrize(
    "edit,message",
    [
        ("camera_model: pinhole", "camera_model"),
        ("intrinsics: [900.0, 905.0, 639.5, 359.5]", "intrinsics"),
        ("resolution: [1280, 720]", "resolution"),
        ("distortion_model: equidistant", "distortion_model"),
    ],
)
def test_kalibr_rejects_broken_entries(tmp_path, edit, message):
    replacements = {
        "camera_model: pinhole": "camera_model: omni",
        "intrinsics: [900.0, 905.0, 639.5, 359.5]": "intrinsics: [900.0, 905.0]",
        "resolution: [1280, 720]": "resolution: [1280]",
        "distortion_model: equidistant": "distortion_model: fov",
    }
    path = tmp_path / "camchain.yaml"
    path.write_text(KALIBR_YAML.replace(edit, replacements[edit], 1))
    with pytest.raises(UnsupportedFormatError, match=message):
        read_calibration(str(path))


def test_kalibr_distortion_model_none_becomes_zeros(tmp_path):
    path = tmp_path / "camchain.yaml"
    path.write_text(
        KALIBR_YAML.replace("distortion_model: equidistant", "distortion_model: none")
    )
    record = read_calibration(str(path))
    assert np.allclose(record.camera.distortion, 0.0)


def test_kalibr_needs_at_least_one_camera(tmp_path):
    path = tmp_path / "camchain.yaml"
    path.write_text("imu0:\n  rate: 200\n")
    with pytest.raises(UnsupportedFormatError):
        KalibrReader().read(str(path))


def test_native_json_round_trip(tmp_path, pinhole):
    record = CalibrationRecord(pinhole, (1280, 720), "orig", 0.21, {"a": 1})
    path = write_calibration(record, str(tmp_path / "c.json"))
    back = read_calibration(path)
    assert back.camera == pinhole
    assert back.reported_rms == pytest.approx(0.21)
    assert back.source == "orig"


def test_native_json_rejects_the_wrong_format(tmp_path):
    path = tmp_path / "c.json"
    path.write_text(json.dumps({"format": "something.else", "calibration": {}}))
    with pytest.raises(UnsupportedFormatError, match="declares format"):
        NativeJsonReader().read(str(path))


def test_native_json_rejects_broken_json(tmp_path):
    path = tmp_path / "c.json"
    path.write_text('{"format": "caltrust.calibration",')
    with pytest.raises(UnsupportedFormatError, match="could not parse"):
        NativeJsonReader().read(str(path))


def test_native_json_rejects_a_missing_calibration_key(tmp_path):
    path = tmp_path / "c.json"
    path.write_text(json.dumps({"format": "caltrust.calibration"}))
    with pytest.raises(UnsupportedFormatError, match="missing"):
        NativeJsonReader().read(str(path))


def test_unrecognised_file_lists_the_supported_formats(tmp_path):
    path = tmp_path / "junk.txt"
    path.write_text("hello")
    with pytest.raises(UnsupportedFormatError, match="no reader recognised"):
        read_calibration(str(path))


def test_missing_file_says_so(tmp_path):
    with pytest.raises(UnsupportedFormatError, match="no such calibration file"):
        read_calibration(str(tmp_path / "absent.yml"))


def test_forcing_a_format_skips_detection(tmp_path):
    path = tmp_path / "no-header.yaml"
    path.write_text(ROS_YAML)
    assert read_calibration(str(path), fmt="ros").camera.fx == pytest.approx(900.0)


def test_forcing_an_unknown_format_is_an_error(tmp_path):
    path = write_opencv(tmp_path / "cv.yml")
    with pytest.raises(UnsupportedFormatError, match="unknown calibration format"):
        read_calibration(path, fmt="matlab")


def test_ros_is_tried_before_opencv(tmp_path):
    """Both readers see YAML; the more specific one has to win."""
    order = [reader.name for reader in default_readers()]
    assert order.index("ros") < order.index("opencv")
    assert order.index("kalibr") < order.index("opencv")


def test_reader_names_are_unique_and_described():
    names = reader_names()
    assert len({n for n, _ in names}) == len(names)
    assert all(description for _, description in names)


def test_an_opencv_file_with_a_distortion_model_key_is_not_claimed_by_ros(tmp_path):
    """Regression: both readers see YAML, and only one can parse OpenCV's tags."""
    path = write_opencv(
        tmp_path / "cv.yml",
        {"distortion_model": "fisheye"},
        distortion=np.array([-0.02, 0.003, -1e-4, 2e-5]),
    )
    record = read_calibration(path)
    assert record.source.startswith("opencv:")
    assert isinstance(record.camera, FisheyeKannalaBrandt)


def test_opencv_yaml_holding_only_a_one_dimensional_array_is_recognised(tmp_path):
    """OpenCV 5 tags a 1-D array `opencv-nd-matrix`, which is not `opencv-matrix`."""
    path = write_opencv(tmp_path / "cv.yml", matrix=None)
    with pytest.raises(UnsupportedFormatError, match="camera matrix"):
        read_calibration(path)


def test_an_opencv_file_with_camN_keys_is_not_claimed_by_kalibr(tmp_path):
    path = write_opencv(tmp_path / "cv.yml", {"cam0": "left", "intrinsics": "x"})
    assert read_calibration(path).source.startswith("opencv:")
