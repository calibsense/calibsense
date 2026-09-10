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

"""Task shorthand on the command line."""

from __future__ import annotations

import pytest

from calibsense.cli.tasks import ALIASES, SHORTHAND_HELP, parse_task, parse_tasks
from calibsense.errors import ValidationError
from calibsense.task.tasks import LengthAtDepth, PlaneLocation, StereoTriangulation


@pytest.mark.parametrize(
    "text,expected",
    [
        ("length:800mm:100mm", LengthAtDepth(800.0, 100.0)),
        ("distance:0.8m:5cm", LengthAtDepth(800.0, 50.0)),
        ("length:800:100", LengthAtDepth(800.0, 100.0)),
        ("plane:800mm", PlaneLocation(800.0, 25.0)),
        ("plane:800mm:35deg", PlaneLocation(800.0, 35.0)),
        ("stereo:800mm:200mm", StereoTriangulation(200.0, 800.0)),
        ("triangulate:1.2m:150mm", StereoTriangulation(150.0, 1200.0)),
    ],
)
def test_shorthand_parses(text, expected):
    assert parse_task(text) == expected


def test_radians_are_converted():
    import numpy as np

    task = parse_task("plane:800mm:0.5rad")
    assert task.tilt_deg == pytest.approx(np.degrees(0.5))


def test_whitespace_is_tolerated():
    assert parse_task(" length : 800 mm : 100 mm ") == LengthAtDepth(800.0, 100.0)


@pytest.mark.parametrize(
    "text,message",
    [
        ("nope:800mm", "unknown task kind"),
        ("length:800mm", "takes 2 fields"),
        ("length:800mm:100mm:extra", "takes 2 fields"),
        ("length:800mm:abc", "not a length"),
        ("length:800parsec:1mm", "unknown length unit"),
        ("plane:800mm:25:35", "takes 1 or 2 fields"),
        ("plane:800mm:xdeg", "not an angle"),
        ("stereo:800mm", "takes 2 fields"),
        ("base:800mm:extra", "takes 1 field"),
        ("base:800mm", "needs a hand-eye solve"),
    ],
)
def test_bad_shorthand_explains_itself(text, message):
    with pytest.raises(ValidationError, match=message):
        parse_task(text)


def test_every_alias_maps_to_a_real_kind():
    kinds = {"length", "plane", "stereo", "base"}
    assert set(ALIASES.values()) == kinds


def test_help_text_mentions_every_kind():
    for kind in ("length", "plane", "stereo", "base"):
        assert kind in SHORTHAND_HELP


def test_several_tasks_parse_in_order():
    tasks = parse_tasks(["length:800mm:100mm", "plane:600mm:30deg"])
    assert [t.kind for t in tasks] == ["length_at_depth", "plane_location"]


@pytest.mark.slow
def test_a_base_task_parses_with_a_hand_eye_result():
    from calibsense.handeye import solve_hand_eye
    from calibsense.refit import instrument

    from . import rigs

    session = rigs.hand_eye_session("eye_to_hand")
    fit = instrument(session)
    result = solve_hand_eye(fit, session, "eye_to_hand", monte_carlo=False)
    task = parse_task("base:800mm", hand_eye=result)
    assert task.kind == "camera_to_base"
    assert task.depth_mm == 800.0
