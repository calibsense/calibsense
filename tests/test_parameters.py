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

"""Free-parameter masks and ties."""

from __future__ import annotations

import numpy as np
import pytest

from calibsense.core.parameters import ParameterBlock, extrinsic_names
from calibsense.errors import ValidationError

NAMES = ("fx", "fy", "cx", "cy")


def block(free, ties=()):
    return ParameterBlock(NAMES, np.array(free, dtype=bool), ties)


def test_all_free_reduction_is_identity():
    reduction = block([1, 1, 1, 1]).reduction()
    assert np.array_equal(reduction, np.eye(4))


def test_fixed_parameters_lose_their_column():
    b = block([1, 0, 1, 0])
    assert b.free_names() == ("fx", "cx")
    assert b.n_free == 2
    assert np.array_equal(b.reduction(), np.array([[1.0, 0], [0, 0], [0, 1.0], [0, 0]]))


def test_a_tie_shares_its_leader_column_with_a_scale():
    b = block([1, 1, 0, 0], ((1, 0, 1.05),))
    assert b.free_names() == ("fx",)
    assert b.reduction().tolist() == [[1.0], [1.05], [0.0], [0.0]]


def test_tie_maps_a_free_step_into_both_parameters():
    b = block([1, 1, 1, 1], ((1, 0, 2.0),))
    step = b.expand([1.0, 3.0, 4.0])
    assert step.tolist() == [1.0, 2.0, 3.0, 4.0]


def test_expand_rejects_a_wrong_length_step():
    with pytest.raises(ValidationError, match="expected 4 free values"):
        block([1, 1, 1, 1]).expand([1.0])


def test_mask_length_must_match_names():
    with pytest.raises(ValidationError, match="free mask has 3 entries"):
        ParameterBlock(NAMES, np.ones(3, dtype=bool))


def test_a_tie_to_a_fixed_leader_is_rejected():
    with pytest.raises(ValidationError, match="not free"):
        block([0, 1, 1, 1], ((1, 0, 1.0),)).reduction()


def test_self_tie_is_rejected():
    with pytest.raises(ValidationError, match="cannot be tied to itself"):
        block([1, 1, 1, 1], ((0, 0, 1.0),))


def test_double_tie_on_one_follower_is_rejected():
    with pytest.raises(ValidationError, match="tied twice"):
        block([1, 1, 1, 1], ((1, 0, 1.0), (1, 2, 1.0)))


def test_chained_ties_are_rejected():
    with pytest.raises(ValidationError, match="chained ties"):
        block([1, 1, 1, 1], ((1, 0, 1.0), (2, 1, 1.0)))


@pytest.mark.parametrize("scale", [0.0, np.nan, np.inf])
def test_tie_scale_must_be_finite_and_non_zero(scale):
    with pytest.raises(ValidationError, match="tie scale"):
        block([1, 1, 1, 1], ((1, 0, scale),))


def test_out_of_range_tie_index_is_rejected():
    with pytest.raises(ValidationError, match="out of range"):
        block([1, 1, 1, 1], ((9, 0, 1.0),))


def test_a_fixed_follower_makes_the_tie_inert():
    b = block([1, 0, 1, 1], ((1, 0, 2.0),))
    assert b.free_names() == ("fx", "cx", "cy")
    assert b.reduction()[1].tolist() == [0.0, 0.0, 0.0]


def test_with_all_free_drops_ties_and_the_mask():
    b = block([1, 0, 0, 0], ((1, 0, 2.0),)).with_all_free()
    assert b.n_free == 4 and b.ties == ()


def test_extrinsic_names_are_unique_per_view():
    assert extrinsic_names(0)[0] == "view0.rx"
    assert set(extrinsic_names(1)).isdisjoint(extrinsic_names(2))
    assert len(extrinsic_names(7)) == 6
