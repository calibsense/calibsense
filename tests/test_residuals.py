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

"""Residual statistics, per view and per corner."""

from __future__ import annotations

import numpy as np
import pytest

from caltrust.core.observations import ObservationSet, ViewObservations
from caltrust.errors import ValidationError
from caltrust.refit.residuals import (
    MAD_TO_SIGMA,
    OUTLIER_Z,
    radial_profile,
    summarise,
)


def build(target, per_view_points, image_size=(1280, 720), seed=0):
    rng = np.random.default_rng(seed)
    views = tuple(
        ViewObservations(
            f"v{i}", np.arange(n),
            np.column_stack([rng.uniform(0, image_size[0], n),
                             rng.uniform(0, image_size[1], n)]),
        )
        for i, n in enumerate(per_view_points)
    )
    return ObservationSet(target, image_size, views)


def test_view_summary_matches_a_hand_computation(checkerboard):
    observations = build(checkerboard, [4])
    residual = np.array([[3.0, 4.0], [0.0, 0.0], [-1.0, 0.0], [0.0, 2.0]])
    statistics = summarise(observations, [residual], (640.0, 360.0))
    view = statistics.per_view[0]
    magnitudes = np.array([5.0, 0.0, 1.0, 2.0])
    assert view.n_points == 4
    assert view.rms == pytest.approx(np.sqrt(np.mean(magnitudes ** 2)))
    assert view.maximum == pytest.approx(5.0)
    assert view.median == pytest.approx(1.5)
    assert view.bias[0] == pytest.approx(0.5)
    assert view.bias[1] == pytest.approx(1.5)


def test_overall_rms_and_bias_pool_every_view(checkerboard):
    observations = build(checkerboard, [3, 3])
    residuals = [np.full((3, 2), 1.0), np.full((3, 2), -1.0)]
    statistics = summarise(observations, residuals, (640.0, 360.0))
    assert statistics.rms == pytest.approx(np.sqrt(2.0))
    assert statistics.bias == (pytest.approx(0.0), pytest.approx(0.0))
    assert statistics.total_points == 6
    assert statistics.n_views == 2


def test_view_residuals_are_recoverable_by_index(checkerboard):
    observations = build(checkerboard, [3, 5])
    residuals = [np.full((3, 2), 1.0), np.full((5, 2), 2.0)]
    statistics = summarise(observations, residuals, (640.0, 360.0))
    assert np.allclose(statistics.view_residuals(0), 1.0)
    assert statistics.view_residuals(1).shape == (5, 2)
    assert statistics.view_offsets.tolist() == [0, 3, 8]
    with pytest.raises(ValidationError, match="out of range"):
        statistics.view_residuals(2)


def test_one_bad_view_is_flagged_as_an_outlier(checkerboard):
    counts = [20] * 8
    observations = build(checkerboard, counts)
    rng = np.random.default_rng(1)
    residuals = [rng.normal(0.0, 0.1, (n, 2)) for n in counts]
    residuals[3] = rng.normal(0.0, 3.0, (counts[3], 2))
    statistics = summarise(observations, residuals, (640.0, 360.0))
    outliers = statistics.outlier_views()
    assert [v.view_id for v in outliers] == ["v3"]
    assert outliers[0].robust_z > OUTLIER_Z
    assert statistics.worst_views(1)[0].view_id == "v3"


def test_a_uniform_capture_has_no_outliers(checkerboard):
    counts = [20] * 6
    observations = build(checkerboard, counts)
    residuals = [np.full((n, 2), 0.2) for n in counts]
    statistics = summarise(observations, residuals, (640.0, 360.0))
    assert statistics.outlier_views() == ()
    assert all(v.robust_z == 0.0 for v in statistics.per_view)


def test_a_single_view_is_never_its_own_outlier(checkerboard):
    observations = build(checkerboard, [10])
    residuals = [np.full((10, 2), 5.0)]
    statistics = summarise(observations, residuals, (640.0, 360.0))
    assert statistics.per_view[0].robust_z == 0.0
    assert not statistics.per_view[0].is_outlier


def test_worst_views_are_ordered_and_capped(checkerboard):
    counts = [10] * 5
    observations = build(checkerboard, counts)
    residuals = [np.full((n, 2), 0.1 * (i + 1)) for i, n in enumerate(counts)]
    statistics = summarise(observations, residuals, (640.0, 360.0))
    worst = statistics.worst_views(3)
    assert [v.view_id for v in worst] == ["v4", "v3", "v2"]


def test_mismatched_residual_count_is_rejected(checkerboard):
    observations = build(checkerboard, [3, 3])
    with pytest.raises(ValidationError, match="residual arrays for"):
        summarise(observations, [np.zeros((3, 2))], (640.0, 360.0))


def test_mismatched_point_count_within_a_view_is_rejected(checkerboard):
    observations = build(checkerboard, [3])
    with pytest.raises(ValidationError, match="residuals for"):
        summarise(observations, [np.zeros((5, 2))], (640.0, 360.0))


def test_radial_profile_separates_radial_from_tangential():
    points = np.array([[100.0, 0.0], [0.0, 100.0], [-100.0, 0.0], [0.0, -100.0]])
    points = points + np.array([640.0, 360.0])
    # Each residual points straight outwards with magnitude 2.
    residuals = np.array([[2.0, 0.0], [0.0, 2.0], [-2.0, 0.0], [0.0, -2.0]])
    profile = radial_profile(points, residuals, (640.0, 360.0), n_bins=1)
    assert profile.radial_mean[0] == pytest.approx(2.0)
    assert profile.radial_rms[0] == pytest.approx(2.0)
    assert profile.tangential_rms[0] == pytest.approx(0.0)


def test_radial_profile_detects_a_purely_tangential_residual():
    points = np.array([[100.0, 0.0]]) + np.array([640.0, 360.0])
    residuals = np.array([[0.0, 3.0]])
    profile = radial_profile(points, residuals, (640.0, 360.0), n_bins=1)
    assert profile.radial_mean[0] == pytest.approx(0.0)
    assert profile.tangential_rms[0] == pytest.approx(3.0)


def test_radial_profile_bins_by_distance():
    radii = np.array([10.0, 20.0, 90.0, 100.0])
    points = np.column_stack([640.0 + radii, np.full(4, 360.0)])
    residuals = np.zeros((4, 2))
    profile = radial_profile(points, residuals, (640.0, 360.0), n_bins=2)
    assert profile.n_bins == 2
    assert profile.counts.tolist() == [2, 2]
    assert profile.edges[0] == 0.0 and profile.edges[-1] == pytest.approx(100.0)


def test_an_empty_bin_is_zero_not_nan():
    points = np.array([[740.0, 360.0], [745.0, 360.0]])
    residuals = np.ones((2, 2))
    profile = radial_profile(points, residuals, (640.0, 360.0), n_bins=5)
    assert 0 in profile.counts.tolist()
    assert np.all(np.isfinite(profile.radial_mean))
    empty = profile.counts == 0
    assert np.all(profile.radial_rms[empty] == 0.0)


def test_a_point_at_the_principal_point_does_not_divide_by_zero():
    points = np.array([[640.0, 360.0], [740.0, 360.0]])
    residuals = np.array([[1.0, 1.0], [1.0, 0.0]])
    profile = radial_profile(points, residuals, (640.0, 360.0), n_bins=2)
    assert np.all(np.isfinite(profile.radial_mean))


def test_radial_profile_checks_its_inputs():
    with pytest.raises(ValidationError, match="image points against"):
        radial_profile(np.zeros((3, 2)), np.zeros((4, 2)), (0.0, 0.0))
    with pytest.raises(ValidationError, match="n_bins must be positive"):
        radial_profile(np.zeros((3, 2)), np.zeros((3, 2)), (0.0, 0.0), n_bins=0)


def test_mad_scale_factor_is_the_conventional_one():
    assert MAD_TO_SIGMA == pytest.approx(1.4826)
