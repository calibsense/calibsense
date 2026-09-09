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

"""Unit conversion."""

from __future__ import annotations

import pytest

from caltrust import units
from caltrust.errors import ValidationError


@pytest.mark.parametrize(
    "value,unit,expected",
    [(1, "mm", 1.0), (1, "cm", 10.0), (1, "m", 1000.0), (1, "in", 25.4),
     (1000, "um", 1.0), (2.5, "CM", 25.0), (3, " Meter ", 3000.0)],
)
def test_to_mm(value, unit, expected):
    assert units.to_mm(value, unit) == pytest.approx(expected)


@pytest.mark.parametrize("unit", ["mm", "cm", "m", "in", "um"])
def test_round_trip(unit):
    assert units.from_mm(units.to_mm(7.25, unit), unit) == pytest.approx(7.25)


def test_canonical_is_a_known_unit():
    assert units.normalise(units.CANONICAL) == units.CANONICAL


@pytest.mark.parametrize("unit", ["furlong", "", "mm2", "feet"])
def test_unknown_unit_names_the_alternatives(unit):
    with pytest.raises(ValidationError, match="unknown length unit"):
        units.normalise(unit)
