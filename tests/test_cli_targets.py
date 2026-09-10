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

"""Target shorthand and target files."""

from __future__ import annotations

import json

import pytest

from calibsense.cli.targets import (
    ALIASES,
    SHORTHAND_HELP,
    load_target_file,
    parse_shorthand,
    resolve_target,
)
from calibsense.core.target import CharucoBoard, Checkerboard, CircleGrid
from calibsense.errors import ValidationError


@pytest.mark.parametrize(
    "text,expected",
    [
        ("checkerboard:9x6:25mm", Checkerboard(9, 6, 25.0)),
        ("chessboard:9X6:25", Checkerboard(9, 6, 25.0)),
        ("checker:9*6:2.5cm", Checkerboard(9, 6, 2.5, units="cm")),
        ("circles:4x11:20mm", CircleGrid(4, 11, 20.0, True)),
        ("circles:4x11:20mm:symmetric", CircleGrid(4, 11, 20.0, False)),
        ("circle_grid:4x11:20mm:asymmetric", CircleGrid(4, 11, 20.0, True)),
        ("charuco:8x11:20mm:15mm", CharucoBoard(8, 11, 20.0, 15.0)),
        ("charuco:8x11:20mm:15mm:DICT_4X4_50", CharucoBoard(8, 11, 20.0, 15.0, "DICT_4X4_50")),
        ("charuco:8x11:20mm:15mm:DICT_4X4_50:legacy",
         CharucoBoard(8, 11, 20.0, 15.0, "DICT_4X4_50", True)),
        ("charuco:8x11:20mm:15mm:legacy", CharucoBoard(8, 11, 20.0, 15.0, legacy_pattern=True)),
    ],
)
def test_shorthand_parses(text, expected):
    assert parse_shorthand(text) == expected


def test_lengths_may_carry_a_unit():
    target = parse_shorthand("checkerboard:9x6:1in")
    assert target.units == "in"
    assert target.object_points()[1, 0] == pytest.approx(25.4)


def test_charuco_converts_a_marker_size_given_in_another_unit():
    target = parse_shorthand("charuco:8x11:2cm:15mm")
    assert target.units == "cm"
    assert target.marker_size == pytest.approx(1.5)


def test_whitespace_is_tolerated():
    assert parse_shorthand(" checkerboard : 9 x 6 : 25 mm ") == Checkerboard(9, 6, 25.0)


@pytest.mark.parametrize(
    "text,message",
    [
        ("dartboard:9x6:25mm", "unknown target kind"),
        ("checkerboard:9x6", "takes 2 fields"),
        ("checkerboard:9x6:25mm:extra", "takes 2 fields"),
        ("checkerboard:9-6:25mm", "not a grid size"),
        ("checkerboard:9x6:abc", "not a length"),
        ("checkerboard:9x6:25parsec", "unknown length unit"),
        ("circles:4x11:20mm:diagonal", "symmetric"),
        ("circles:4x11", "takes 2 or 3 fields"),
        ("charuco:8x11:20mm", "at least 3 fields"),
        ("charuco:8x11:20mm:25mm", "marker_size"),
        ("charuco:8x11:20mm:15mm:DICT_4X4_50:DICT_5X5_50", "dictionaries"),
    ],
)
def test_bad_shorthand_explains_itself(text, message):
    with pytest.raises(ValidationError, match=message):
        parse_shorthand(text)


def test_every_alias_maps_to_a_real_kind():
    from calibsense.core.target import registered_targets

    assert set(ALIASES.values()) <= set(registered_targets())


def test_help_text_mentions_every_kind():
    for kind in ("checkerboard", "charuco", "circles"):
        assert kind in SHORTHAND_HELP


def test_yaml_target_file(tmp_path):
    path = tmp_path / "t.yaml"
    path.write_text("kind: checkerboard\ncolumns: 9\nrows: 6\nsquare_size: 25.0\nunits: mm\n")
    assert load_target_file(str(path)) == Checkerboard(9, 6, 25.0)


def test_json_target_file(tmp_path):
    path = tmp_path / "t.json"
    path.write_text(json.dumps({"kind": "charuco", "squares_x": 8, "squares_y": 11,
                                "square_size": 20.0, "marker_size": 15.0}))
    assert load_target_file(str(path)) == CharucoBoard(8, 11, 20.0, 15.0)


def test_a_nested_target_key_is_accepted(tmp_path):
    path = tmp_path / "t.yaml"
    path.write_text("target:\n  kind: chessboard\n  columns: 4\n  rows: 3\n  square_size: 10\n")
    assert load_target_file(str(path)) == Checkerboard(4, 3, 10.0)


def test_a_file_may_use_an_alias_for_its_kind(tmp_path):
    path = tmp_path / "t.yaml"
    path.write_text("kind: circles\ncolumns: 4\nrows: 11\nspacing: 20\n")
    assert load_target_file(str(path)) == CircleGrid(4, 11, 20.0, True)


def test_broken_yaml_is_reported(tmp_path):
    path = tmp_path / "t.yaml"
    path.write_text("kind: [unclosed\n")
    with pytest.raises(ValidationError, match="could not parse target file"):
        load_target_file(str(path))


def test_a_non_mapping_target_file_is_reported(tmp_path):
    path = tmp_path / "t.yaml"
    path.write_text("- checkerboard\n")
    with pytest.raises(ValidationError, match="must hold a mapping"):
        load_target_file(str(path))


def test_resolve_prefers_an_existing_file(tmp_path):
    path = tmp_path / "checkerboard:9x6:25mm"
    path.write_text("kind: checkerboard\ncolumns: 4\nrows: 3\nsquare_size: 10\n")
    assert resolve_target(str(path)) == Checkerboard(4, 3, 10.0)


def test_resolve_falls_back_to_shorthand():
    assert resolve_target("checkerboard:9x6:25mm") == Checkerboard(9, 6, 25.0)
