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

"""Target geometry, checked against OpenCV where OpenCV has an opinion."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from calibsense.core.target import (
    CharucoBoard,
    Checkerboard,
    CircleGrid,
    registered_targets,
    target_from_dict,
)
from calibsense.errors import ValidationError


def test_checkerboard_points_are_row_major_and_planar():
    target = Checkerboard(4, 3, 10.0)
    points = target.object_points()
    assert points.shape == (12, 3)
    assert np.all(points[:, 2] == 0)
    assert points[0].tolist() == [0.0, 0.0, 0.0]
    assert points[1].tolist() == [10.0, 0.0, 0.0]
    assert points[4].tolist() == [0.0, 10.0, 0.0]
    assert points[5].tolist() == [10.0, 10.0, 0.0]
    assert target.pattern_size == (4, 3)


def test_units_convert_to_millimetres():
    assert Checkerboard(2, 2, 1.0, units="m").object_points()[1, 0] == pytest.approx(1000.0)
    assert Checkerboard(2, 2, 1.0, units="in").object_points()[1, 0] == pytest.approx(25.4)
    assert Checkerboard(2, 2, 10.0, units="cm").square_size == 10.0


def test_asymmetric_circle_grid_matches_the_opencv_sample_layout():
    target = CircleGrid(3, 3, 10.0, asymmetric=True)
    points = target.object_points()
    expected = [
        [(2 * c + r % 2) * 10.0, r * 10.0, 0.0]
        for r in range(3) for c in range(3)
    ]
    assert points.tolist() == expected


def test_symmetric_circle_grid_is_a_plain_lattice():
    points = CircleGrid(3, 2, 5.0, asymmetric=False).object_points()
    assert points[:3, 0].tolist() == [0.0, 5.0, 10.0]
    assert points[3, 1] == pytest.approx(5.0)


@pytest.mark.parametrize("squares", [(5, 7), (8, 11), (3, 3)])
def test_charuco_object_points_match_opencv_exactly(squares):
    target = CharucoBoard(squares[0], squares[1], 20.0, 15.0)
    board = cv2.aruco.CharucoBoard(
        squares, 20.0, 15.0,
        cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_1000),
    )
    assert np.allclose(target.object_points(), board.getChessboardCorners())


def test_charuco_point_count_is_the_interior_corners():
    target = CharucoBoard(8, 11, 20.0, 15.0)
    assert target.num_points == 7 * 10
    assert target.corner_grid == (7, 10)


def test_object_points_selects_by_id():
    target = Checkerboard(4, 3, 10.0)
    assert target.object_points([2, 5]).tolist() == target.object_points()[[2, 5]].tolist()


def test_out_of_range_id_is_rejected():
    with pytest.raises(ValidationError, match="out of range"):
        Checkerboard(4, 3, 10.0).object_points([0, 99])


def test_duplicate_ids_are_rejected():
    with pytest.raises(ValidationError, match="duplicate point ids"):
        Checkerboard(4, 3, 10.0).object_points([1, 1])


def test_empty_id_list_yields_no_points():
    assert Checkerboard(4, 3, 10.0).object_points([]).shape == (0, 3)


def test_object_points_returns_a_copy():
    target = Checkerboard(4, 3, 10.0)
    points = target.object_points()
    points[0, 0] = 999.0
    assert target.object_points()[0, 0] == 0.0


def test_extent_and_diagonal():
    target = Checkerboard(9, 6, 25.0)
    assert target.extent_mm == (200.0, 125.0)
    assert target.diagonal_mm == pytest.approx(np.hypot(200.0, 125.0))


@pytest.mark.parametrize(
    "factory,message",
    [
        (lambda: Checkerboard(1, 6, 25.0), "columns"),
        (lambda: Checkerboard(9, 0, 25.0), "rows"),
        (lambda: Checkerboard(9, 6, 0.0), "square_size"),
        (lambda: Checkerboard(9, 6, -1.0), "square_size"),
        (lambda: CircleGrid(4, 11, np.nan), "spacing"),
        (lambda: CharucoBoard(8, 11, 20.0, 20.0), "marker_size"),
        (lambda: CharucoBoard(8, 11, 20.0, 25.0), "marker_size"),
    ],
)
def test_invalid_targets_are_rejected(factory, message):
    with pytest.raises(ValidationError, match=message):
        factory()


def test_dict_round_trip(checkerboard, charuco, circle_grid):
    for target in (checkerboard, charuco, circle_grid):
        assert target_from_dict(target.to_dict()) == target


def test_target_from_dict_rejects_an_unknown_kind():
    with pytest.raises(ValidationError, match="unknown target kind"):
        target_from_dict({"kind": "dartboard"})


def test_target_from_dict_rejects_an_unexpected_field():
    with pytest.raises(ValidationError, match="unexpected field"):
        target_from_dict({"kind": "checkerboard", "columns": 9, "rows": 6, "diameter": 3})


def test_describe_names_the_geometry(checkerboard, charuco, circle_grid):
    assert "9x6 checkerboard" in checkerboard.describe()
    assert "ChArUco" in charuco.describe() and "DICT_5X5_1000" in charuco.describe()
    assert "asymmetric" in circle_grid.describe()


def test_all_three_targets_are_registered():
    assert registered_targets() == ("charuco", "checkerboard", "circle_grid")
