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

"""Camera parameter containers."""

from __future__ import annotations

import numpy as np
import pytest

from calibsense.core.camera import (
    FisheyeKannalaBrandt,
    PinholeBrownConrady,
    camera_from_dict,
    registered_models,
)
from calibsense.errors import ValidationError


@pytest.mark.parametrize("size", [4, 5, 8, 12, 14])
def test_pinhole_accepts_every_opencv_distortion_length(size):
    camera = PinholeBrownConrady(900, 905, 640, 360, np.zeros(size))
    assert len(camera.distortion_names()) == size
    assert camera.parameter_names()[:4] == ("fx", "fy", "cx", "cy")
    assert len(camera.parameter_names()) == 4 + size


@pytest.mark.parametrize("size", [0, 1, 3, 6, 7, 15])
def test_pinhole_rejects_other_distortion_lengths(size):
    with pytest.raises(ValidationError, match="distortion coefficients"):
        PinholeBrownConrady(900, 905, 640, 360, np.zeros(size))


def test_distortion_names_follow_opencv_order():
    camera = PinholeBrownConrady(900, 905, 640, 360, np.zeros(14))
    assert camera.distortion_names() == (
        "k1", "k2", "p1", "p2", "k3", "k4", "k5", "k6",
        "s1", "s2", "s3", "s4", "taux", "tauy",
    )


def test_fisheye_takes_exactly_four_coefficients(fisheye):
    assert fisheye.distortion_names() == ("k1", "k2", "k3", "k4")
    with pytest.raises(ValidationError):
        FisheyeKannalaBrandt(430, 432, 640, 360, np.zeros(5))


def test_fisheye_parameter_vector_ends_with_skew():
    camera = FisheyeKannalaBrandt(430, 432, 640, 360, np.arange(4), alpha=0.01)
    assert camera.parameter_names()[-1] == "alpha"
    assert camera.to_vector()[-1] == pytest.approx(0.01)
    assert camera.to_vector().size == 9


def test_camera_matrix_carries_skew_only_for_fisheye(pinhole):
    assert pinhole.camera_matrix[0, 1] == 0.0
    fisheye = FisheyeKannalaBrandt(400, 400, 640, 360, np.zeros(4), alpha=0.02)
    assert fisheye.camera_matrix[0, 1] == pytest.approx(0.02 * 400)


@pytest.mark.parametrize("bad", [{"fx": 0}, {"fy": -1}, {"fx": np.nan}, {"cx": np.inf}])
def test_invalid_intrinsics_are_rejected(bad):
    fields = {"fx": 900.0, "fy": 905.0, "cx": 640.0, "cy": 360.0}
    fields.update(bad)
    with pytest.raises(ValidationError):
        PinholeBrownConrady(distortion=np.zeros(5), **fields)


def test_non_finite_distortion_is_rejected():
    with pytest.raises(ValidationError, match="finite"):
        PinholeBrownConrady(900, 905, 640, 360, [0, np.nan, 0, 0, 0])


def test_non_finite_skew_is_rejected():
    with pytest.raises(ValidationError, match="alpha"):
        FisheyeKannalaBrandt(430, 432, 640, 360, np.zeros(4), alpha=np.nan)


@pytest.mark.parametrize("size", [4, 5, 8, 12, 14])
def test_vector_round_trip_pinhole(size):
    camera = PinholeBrownConrady(900.5, 905.25, 639.5, 359.5, np.linspace(-0.2, 0.2, size))
    assert type(camera).from_vector(camera.to_vector()).allclose(camera)


def test_vector_round_trip_fisheye():
    camera = FisheyeKannalaBrandt(430, 432, 639.5, 359.5, [-0.05, 0.01, -0.002, 4e-4], 0.01)
    assert FisheyeKannalaBrandt.from_vector(camera.to_vector()).allclose(camera)


def test_fisheye_from_vector_defaults_skew_to_zero():
    camera = FisheyeKannalaBrandt.from_vector([430, 432, 640, 360, 0, 0, 0, 0])
    assert camera.alpha == 0.0


@pytest.mark.parametrize("size", [7, 10])
def test_fisheye_from_vector_rejects_other_lengths(size):
    with pytest.raises(ValidationError, match="8 or 9 values"):
        FisheyeKannalaBrandt.from_vector(np.zeros(size))


def test_dict_round_trip_preserves_equality(pinhole, fisheye):
    for camera in (pinhole, fisheye):
        assert camera_from_dict(camera.to_dict()) == camera


def test_camera_from_dict_rejects_an_unknown_kind():
    with pytest.raises(ValidationError, match="unknown camera model"):
        camera_from_dict({"kind": "orthographic", "fx": 1, "fy": 1, "cx": 0, "cy": 0,
                          "distortion": [0, 0, 0, 0]})


def test_models_of_different_kinds_are_never_equal(pinhole):
    other = FisheyeKannalaBrandt(pinhole.fx, pinhole.fy, pinhole.cx, pinhole.cy, np.zeros(4))
    assert pinhole != other
    assert not pinhole.allclose(other)


def test_allclose_is_false_for_different_distortion_lengths():
    a = PinholeBrownConrady(900, 905, 640, 360, np.zeros(5))
    b = PinholeBrownConrady(900, 905, 640, 360, np.zeros(8))
    assert not a.allclose(b)


def test_free_parameters_defaults_to_everything(pinhole):
    assert pinhole.free_parameters().free_names() == pinhole.parameter_names()


def test_free_parameters_honours_fixed_names(pinhole):
    block = pinhole.free_parameters(fixed=("cx", "cy", "k3"))
    assert set(block.free_names()) == set(pinhole.parameter_names()) - {"cx", "cy", "k3"}


def test_tie_aspect_uses_the_current_ratio(pinhole):
    block = pinhole.free_parameters(tie_aspect=True)
    assert block.reduction()[1, 0] == pytest.approx(pinhole.fy / pinhole.fx)
    assert "fy" not in block.free_names()


def test_tie_aspect_needs_both_focal_lengths_free(pinhole):
    with pytest.raises(ValidationError, match="tie_aspect"):
        pinhole.free_parameters(fixed=("fy",), tie_aspect=True)


def test_unknown_fixed_name_is_rejected(pinhole):
    with pytest.raises(ValidationError, match="unknown parameter"):
        pinhole.free_parameters(fixed=("k9",))


def test_repr_is_readable(pinhole):
    text = repr(pinhole)
    assert "PinholeBrownConrady" in text and "fx=900" in text


def test_both_models_are_registered():
    assert registered_models() == ("fisheye_kannala_brandt", "pinhole_brown_conrady")
