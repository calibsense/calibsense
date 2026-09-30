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

"""Exporting the fitted intrinsics, and reading them back.

The point of every round-trip test here is that calibsense's own readers are the
consumers, so a writer that loses a coefficient or renames a model fails loudly
rather than producing a file that looks plausible.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
import yaml

from calibsense.core.session import CalibrationSession
from calibsense.errors import UnsupportedFormatError
from calibsense.ingest import read_calibration
from calibsense.io import EXPORT_FORMATS, export_calibration, resolve_format
from calibsense.io.export import uncertainty_table
from calibsense.refit import RefitOptions, instrument


@pytest.fixture
def fit(good_session):
    """An instrumented pinhole fit, five distortion coefficients."""
    return instrument(good_session, RefitOptions())


@pytest.fixture
def fisheye_fit(fisheye_capture):
    """An instrumented fisheye fit."""
    return instrument(
        CalibrationSession(observations=fisheye_capture.observations),
        RefitOptions(model="fisheye", distortion_terms=4),
    )


@pytest.fixture
def degenerate_fit(degenerate_capture):
    """A frontoparallel fit, where the standard deviations are lower bounds."""
    return instrument(
        CalibrationSession(observations=degenerate_capture.observations), RefitOptions()
    )


def _written(fit, tmp_path, name, fmt=None):
    return export_calibration(fit, str(tmp_path / name), fmt)


@pytest.mark.parametrize(
    "name,fmt", [("c.yml", None), ("c.xml", None), ("c.yaml", "ros"), ("c.json", None)]
)
def test_every_format_round_trips_the_intrinsics_exactly(fit, tmp_path, name, fmt):
    record = read_calibration(_written(fit, tmp_path, name, fmt))
    assert np.allclose(record.camera.camera_matrix, fit.camera.camera_matrix, rtol=1e-9)
    assert np.allclose(
        np.asarray(record.camera.distortion).reshape(-1),
        np.asarray(fit.camera.distortion).reshape(-1),
        rtol=1e-9, atol=1e-12,
    )
    assert record.image_size == fit.image_size


@pytest.mark.parametrize(
    "name,fmt", [("c.yml", None), ("c.xml", None), ("c.yaml", "ros"), ("c.json", None)]
)
def test_a_fisheye_stays_a_fisheye(fisheye_fit, tmp_path, name, fmt):
    """The model family survives the trip, which is the thing item 9 is about.

    A four-coefficient vector in an OpenCV FileStorage file is ambiguous between
    Brown-Conrady and Kannala-Brandt unless something names the model. These
    writers always name it, so a file calibsense wrote is never ambiguous.
    """
    record = read_calibration(_written(fisheye_fit, tmp_path, name, fmt))
    assert record.camera.kind == "fisheye_kannala_brandt"
    assert not record.metadata.get("model_ambiguous", False)


def test_the_standard_deviations_travel_with_the_values(fit, tmp_path):
    payload = json.loads(open(_written(fit, tmp_path, "c.json")).read())
    deviations = payload["calibration"]["metadata"]["std_dev"]
    names, expected = uncertainty_table(fit)
    # The native writer sorts its keys, so compare membership rather than order.
    assert sorted(deviations) == sorted(names)
    assert [deviations[n] for n in names] == pytest.approx(expected)
    assert all(value > 0 for value in expected)


def test_the_opencv_file_records_the_deviations_alongside_their_names(fit, tmp_path):
    import cv2

    path = _written(fit, tmp_path, "c.yml")
    storage = cv2.FileStorage(path, cv2.FILE_STORAGE_READ)
    try:
        names = storage.getNode("parameter_names").string().split()
        deviations = storage.getNode("parameter_std_dev").mat().reshape(-1)
    finally:
        storage.release()
    expected_names, expected = uncertainty_table(fit)
    assert names == expected_names
    assert deviations == pytest.approx(expected)


def test_a_fixed_parameter_is_absent_rather_than_written_as_zero(good_session, tmp_path):
    """Zero would read as "measured to be exact" when it means "not estimated"."""
    fit = instrument(good_session, RefitOptions(fixed=("k3",)))
    names, deviations = uncertainty_table(fit)
    assert "k3" not in names
    assert all(value > 0 for value in deviations)


def test_ros_names_the_distortion_model_it_can_represent(fit, fisheye_fit, tmp_path):
    pinhole = yaml.safe_load(open(_written(fit, tmp_path, "p.yaml", "ros")))
    assert pinhole["distortion_model"] == "plumb_bob"
    fish = yaml.safe_load(open(_written(fisheye_fit, tmp_path, "f.yaml", "ros")))
    assert fish["distortion_model"] == "equidistant"


def test_ros_calls_a_rational_model_by_its_ros_name(good_session, tmp_path):
    fit = instrument(good_session, RefitOptions(distortion_terms=8))
    payload = yaml.safe_load(open(_written(fit, tmp_path, "r.yaml", "ros")))
    assert payload["distortion_model"] == "rational_polynomial"
    assert payload["distortion_coefficients"]["cols"] == 8


def test_ros_carries_the_uncertainty_in_comments_because_the_message_has_no_field(
    fit, tmp_path
):
    text = open(_written(fit, tmp_path, "c.yaml", "ros")).read()
    header = [line for line in text.splitlines() if line.startswith("#")]
    assert any("fx = " in line and "+/-" in line for line in header)
    # Still parses: the comments must not have broken the YAML.
    assert yaml.safe_load(text)["camera_name"] == "camera"


def test_the_projection_matrix_is_k_with_a_zero_column(fit, tmp_path):
    payload = yaml.safe_load(open(_written(fit, tmp_path, "c.yaml", "ros")))
    projection = np.asarray(payload["projection_matrix"]["data"]).reshape(3, 4)
    assert np.allclose(projection[:, :3], fit.camera.camera_matrix)
    assert np.allclose(projection[:, 3], 0.0)


def test_an_unidentifiable_fit_exports_with_a_warning_attached(degenerate_fit, tmp_path):
    """Exporting false confidence silently is the failure this package exists to catch."""
    assert not degenerate_fit.conditioning.identifiable
    payload = json.loads(open(_written(degenerate_fit, tmp_path, "c.json")).read())
    metadata = payload["calibration"]["metadata"]
    assert metadata["identifiable"] is False
    assert "lower bounds" in metadata["warning"]

    text = open(_written(degenerate_fit, tmp_path, "c.yaml", "ros")).read()
    assert "WARNING" in text


def test_the_provenance_says_which_calibsense_produced_it(fit, tmp_path):
    payload = json.loads(open(_written(fit, tmp_path, "c.json")).read())
    metadata = payload["calibration"]["metadata"]
    assert metadata["produced_by"].startswith("calibsense ")
    assert metadata["views"] == fit.n_views
    assert metadata["rms_px"] == pytest.approx(fit.rms)


@pytest.mark.parametrize(
    "path,expected",
    [("c.json", "native"), ("c.xml", "opencv"), ("c.yml", "opencv"), ("c.yaml", "opencv")],
)
def test_the_extension_picks_the_format(path, expected):
    assert resolve_format(path) == expected


def test_an_explicit_format_beats_the_extension():
    assert resolve_format("c.yaml", "ros") == "ros"


def test_an_unknown_extension_is_refused_rather_than_guessed():
    with pytest.raises(UnsupportedFormatError, match="cannot tell what format"):
        resolve_format("calibration.txt")


def test_an_unknown_format_name_lists_the_ones_that_exist():
    with pytest.raises(UnsupportedFormatError, match="unknown calibration format"):
        resolve_format("c.yml", "kalibr")


def test_every_advertised_format_can_actually_be_written(fit, tmp_path):
    for name in EXPORT_FORMATS:
        assert export_calibration(fit, str(tmp_path / f"{name}.out"), name)


def test_ros_says_so_when_it_cannot_carry_a_skew(fisheye_capture, tmp_path):
    """`camera_info` has no skew term, so losing one has to be visible."""
    import dataclasses

    from calibsense.core.camera import FisheyeKannalaBrandt

    fit = instrument(
        CalibrationSession(observations=fisheye_capture.observations),
        RefitOptions(model="fisheye", distortion_terms=4),
    )
    skewed = dataclasses.replace(fit.camera, alpha=0.01)
    assert isinstance(skewed, FisheyeKannalaBrandt)
    assert skewed.camera_matrix[0, 1] != 0.0

    text = open(
        export_calibration(
            dataclasses.replace(fit, camera=skewed), str(tmp_path / "s.yaml"), "ros"
        )
    ).read()
    assert "skew" in text and "WARNING" in text

    # The lossless formats keep it, which is what the warning points at.
    record = read_calibration(
        export_calibration(
            dataclasses.replace(fit, camera=skewed), str(tmp_path / "s.yml")
        )
    )
    assert record.camera.camera_matrix[0, 1] == pytest.approx(
        skewed.camera_matrix[0, 1]
    )


def test_the_opencv_file_carries_the_warning_too(degenerate_fit, tmp_path):
    """Every format has to say it, not just the ones with room for prose."""
    import cv2

    path = _written(degenerate_fit, tmp_path, "c.yml")
    storage = cv2.FileStorage(path, cv2.FILE_STORAGE_READ)
    try:
        assert int(storage.getNode("identifiable").real()) == 0
        assert "lower bounds" in storage.getNode("warning").string()
    finally:
        storage.release()


@pytest.mark.parametrize("name,fmt", [("c.yml", None), ("c.yaml", "ros")])
def test_an_unwritable_destination_fails_loudly(fit, tmp_path, name, fmt):
    """Silently not writing the calibration would be the worst outcome here."""
    missing = tmp_path / "no-such-directory" / name
    with pytest.raises(UnsupportedFormatError):
        export_calibration(fit, str(missing), fmt)
