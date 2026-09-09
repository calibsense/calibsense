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

"""Reading detections produced elsewhere."""

from __future__ import annotations

import json

import numpy as np
import pytest

from caltrust.core.target import Checkerboard
from caltrust.errors import UnsupportedFormatError, ValidationError
from caltrust.ingest.detections import read_detections, write_detections


def test_json_round_trip(good_capture, tmp_path):
    path = write_detections(good_capture.observations, str(tmp_path / "d.json"))
    back = read_detections(path)
    assert back.target == good_capture.observations.target
    assert back.image_size == good_capture.observations.image_size
    assert back.view_ids == good_capture.observations.view_ids
    for a, b in zip(good_capture.observations.views, back.views):
        assert np.allclose(a.image_points, b.image_points)
        assert np.array_equal(a.point_ids, b.point_ids)


def test_json_without_point_ids_defaults_to_a_full_grid(tmp_path, checkerboard):
    path = tmp_path / "d.json"
    points = np.random.default_rng(0).uniform(0, 600, (checkerboard.num_points, 2))
    path.write_text(json.dumps({
        "views": [{"view_id": "a", "image_points": points.tolist()}],
    }))
    observations = read_detections(str(path), checkerboard, (1280, 720))
    assert np.array_equal(observations.views[0].point_ids, np.arange(checkerboard.num_points))


def test_a_supplied_target_overrides_the_file(good_capture, tmp_path):
    path = write_detections(good_capture.observations, str(tmp_path / "d.json"))
    bigger = Checkerboard(9, 6, 50.0)
    assert read_detections(path, target=bigger).target == bigger


def test_missing_target_is_reported(tmp_path):
    path = tmp_path / "d.json"
    path.write_text(json.dumps({"views": [{"image_points": [[1.0, 2.0]]}]}))
    with pytest.raises(ValidationError, match="does not record the target"):
        read_detections(str(path), image_size=(640, 480))


def test_missing_image_size_is_reported(tmp_path, checkerboard):
    path = tmp_path / "d.json"
    path.write_text(json.dumps({"views": [{"image_points": [[1.0, 2.0]]}]}))
    with pytest.raises(ValidationError, match="does not record the image size"):
        read_detections(str(path), target=checkerboard)


def test_npz_with_a_uniform_grid(tmp_path, checkerboard):
    points = np.random.default_rng(1).uniform(0, 600, (3, checkerboard.num_points, 2))
    path = str(tmp_path / "d.npz")
    np.savez(path, image_points=points, image_size=np.array([1280, 720]))
    observations = read_detections(path, target=checkerboard)
    assert observations.n_views == 3
    assert observations.view_ids == ("view0000", "view0001", "view0002")
    assert np.allclose(observations.views[1].image_points, points[1])


def test_npz_accepts_shared_point_ids(tmp_path, checkerboard):
    ids = np.arange(10)
    points = np.random.default_rng(2).uniform(0, 600, (2, 10, 2))
    path = str(tmp_path / "d.npz")
    np.savez(path, image_points=points, point_ids=ids)
    observations = read_detections(path, checkerboard, (1280, 720))
    assert np.array_equal(observations.views[0].point_ids, ids)


def test_npz_rejects_wrong_shaped_points(tmp_path, checkerboard):
    path = str(tmp_path / "d.npz")
    np.savez(path, image_points=np.zeros((3, 4)))
    with pytest.raises(UnsupportedFormatError, match=r"\(views, points, 2\)"):
        read_detections(path, checkerboard, (1280, 720))


def test_npz_rejects_mismatched_point_ids(tmp_path, checkerboard):
    path = str(tmp_path / "d.npz")
    np.savez(path, image_points=np.zeros((2, 10, 2)), point_ids=np.zeros((2, 4)))
    with pytest.raises(UnsupportedFormatError, match="point_ids is"):
        read_detections(path, checkerboard, (1280, 720))


def test_npz_without_image_points_lists_what_it_found(tmp_path, checkerboard):
    path = str(tmp_path / "d.npz")
    np.savez(path, corners=np.zeros((2, 10, 2)))
    with pytest.raises(UnsupportedFormatError, match="no 'image_points' array"):
        read_detections(path, checkerboard, (1280, 720))


def test_an_unknown_extension_is_refused(tmp_path, checkerboard):
    path = tmp_path / "d.csv"
    path.write_text("1,2\n")
    with pytest.raises(UnsupportedFormatError, match="expected .json or .npz"):
        read_detections(str(path), checkerboard, (1280, 720))


def test_a_missing_file_says_so(tmp_path, checkerboard):
    with pytest.raises(UnsupportedFormatError, match="no such detections file"):
        read_detections(str(tmp_path / "absent.json"), checkerboard)


def test_broken_json_says_so(tmp_path, checkerboard):
    path = tmp_path / "d.json"
    path.write_text('{"views": [')
    with pytest.raises(UnsupportedFormatError, match="could not parse"):
        read_detections(str(path), checkerboard, (1280, 720))


def test_json_without_a_views_list_says_so(tmp_path, checkerboard):
    path = tmp_path / "d.json"
    path.write_text(json.dumps({"corners": []}))
    with pytest.raises(UnsupportedFormatError, match="'views' list"):
        read_detections(str(path), checkerboard, (1280, 720))


def test_a_fallback_image_size_does_not_overwrite_the_file(good_capture, tmp_path):
    """Regression: a fallback must not silently agree with a size the file states."""
    path = write_detections(good_capture.observations, str(tmp_path / "d.json"))
    observations = read_detections(path, fallback_image_size=(640, 480))
    assert observations.image_size == good_capture.observations.image_size


def test_a_fallback_image_size_is_used_when_the_file_has_none(tmp_path, checkerboard):
    path = tmp_path / "d.json"
    path.write_text(json.dumps({"views": [{"image_points": [[1.0, 2.0], [3.0, 4.0],
                                                            [5.0, 6.0], [7.0, 8.0]]}]}))
    observations = read_detections(
        str(path), target=checkerboard, fallback_image_size=(640, 480)
    )
    assert observations.image_size == (640, 480)


def test_an_explicit_image_size_beats_both(good_capture, tmp_path):
    path = write_detections(good_capture.observations, str(tmp_path / "d.json"))
    observations = read_detections(
        path, image_size=(800, 600), fallback_image_size=(640, 480)
    )
    assert observations.image_size == (800, 600)
