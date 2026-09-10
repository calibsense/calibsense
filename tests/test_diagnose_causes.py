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

"""M4 — each diagnostic against the fault it is named for.

The specificity matrix at the top is the important test. A diagnostic that fires
on everything is as useless as one that fires on nothing, so each rig carries a
single deliberate defect and every diagnostic is checked against all of them.
"""

from __future__ import annotations

import numpy as np
import pytest

from calibsense.diagnose import Severity, diagnose
from calibsense.diagnose.coverage import (
    GRID,
    border_mask,
    nearest_neighbour_spacing,
    occupancy,
)
from calibsense.diagnose.geometry import (
    FRONTOPARALLEL_DEGREES,
    normal_spread_degrees,
    orientation_tensor,
)
from calibsense.diagnose.model import (
    MIN_RADIUS_SIGMAS,
    clustered_radial_profile,
    flatness_z,
    radial_trend,
)
from calibsense.diagnose.views import view_influences

from . import rigs

pytestmark = pytest.mark.slow


def severity_of(context, cause):
    return diagnose(context.fit, context.observations).by_cause(cause).severity


def metrics_of(context, cause):
    return diagnose(context.fit, context.observations).by_cause(cause).metrics


# --------------------------------------------------------------------------
# specificity: the fault each rig carries, and the faults it does not
# --------------------------------------------------------------------------

SPECIFICITY = [
    # rig builder, causes that must fire, causes that must stay quiet
    ("frontoparallel", {"pose_diversity", "frontoparallel_dominance"},
     {"target_scale", "outlier_views"}),
    ("single_depth", {"depth_variation"},
     {"pose_diversity", "frontoparallel_dominance", "outlier_views"}),
    ("centre_only", {"image_coverage"},
     {"pose_diversity", "frontoparallel_dominance", "outlier_views"}),
    ("tiny_board", {"target_scale", "image_coverage"},
     {"pose_diversity", "frontoparallel_dominance"}),
    # The fisheye rig has a short focal length, so its board image is genuinely
    # small; target_scale is a real second finding there, not a false positive.
    ("wrong_model", {"distortion_model"},
     {"pose_diversity", "frontoparallel_dominance", "outlier_views"}),
    ("with_bad_view", {"outlier_views"},
     {"pose_diversity", "frontoparallel_dominance", "depth_variation"}),
]


@pytest.mark.parametrize("rig_name,fires,quiet", SPECIFICITY,
                         ids=[row[0] for row in SPECIFICITY])
def test_each_rig_triggers_its_own_fault_and_not_the_others(rig_name, fires, quiet):
    context = getattr(rigs, rig_name)()
    diagnosis = diagnose(context.fit, context.observations, context.cross_validation)
    for cause in fires:
        finding = diagnosis.by_cause(cause)
        assert finding.severity >= Severity.WARNING, (
            f"{rig_name}: {cause} stayed at {finding.severity.label} — {finding.summary}"
        )
    for cause in quiet:
        finding = diagnosis.by_cause(cause)
        assert finding.severity <= Severity.NOTE, (
            f"{rig_name}: {cause} fired at {finding.severity.label} — {finding.summary}"
        )


def test_a_healthy_rig_has_no_critical_geometry_findings():
    context = rigs.healthy()
    diagnosis = diagnose(context.fit, context.observations)
    for cause in ("pose_diversity", "frontoparallel_dominance", "depth_variation",
                  "target_scale", "distortion_model", "outlier_views"):
        assert diagnosis.by_cause(cause).severity <= Severity.NOTE, (
            f"{cause}: {diagnosis.by_cause(cause).summary}"
        )


# --------------------------------------------------------------------------
# pose diversity
# --------------------------------------------------------------------------

def test_normal_spread_folds_antipodal_directions():
    up = np.array([[0.0, 0.0, 1.0]])
    down = np.array([[0.0, 0.0, -1.0]])
    assert normal_spread_degrees(np.vstack([up, down])) == pytest.approx(0.0, abs=1e-6)


def test_normal_spread_of_orthogonal_normals_is_ninety_degrees():
    normals = np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0]])
    assert normal_spread_degrees(normals) == pytest.approx(90.0)


def test_normal_spread_of_one_view_is_zero():
    assert normal_spread_degrees(np.array([[0.0, 0.0, 1.0]])) == 0.0


def test_orientation_tensor_has_unit_trace_and_is_sign_invariant():
    rng = np.random.default_rng(0)
    normals = rng.normal(size=(20, 3))
    normals /= np.linalg.norm(normals, axis=1, keepdims=True)
    eigenvalues = orientation_tensor(normals)
    assert eigenvalues.sum() == pytest.approx(1.0)
    assert np.all(np.diff(eigenvalues) <= 1e-12)
    flipped = normals * np.where(rng.random((20, 1)) > 0.5, 1.0, -1.0)
    assert np.allclose(orientation_tensor(flipped), eigenvalues)


def test_identical_normals_concentrate_the_tensor():
    normals = np.tile([0.0, 0.0, 1.0], (10, 1))
    assert orientation_tensor(normals)[0] == pytest.approx(1.0)


def test_pose_diversity_names_tilt_as_the_fix():
    finding = diagnose(*_pair(rigs.frontoparallel())).by_cause("pose_diversity")
    assert finding.severity is Severity.CRITICAL
    assert "tilt" in finding.action.lower()
    assert finding.metrics["normal_spread_deg"] < 5.0


def test_pose_diversity_reports_the_tilt_range():
    metrics = metrics_of(rigs.healthy(), "pose_diversity")
    assert metrics["tilt_max_deg"] > metrics["tilt_min_deg"]
    assert 0 <= metrics["tilt_min_deg"] <= 90


# --------------------------------------------------------------------------
# depth variation
# --------------------------------------------------------------------------

def test_depth_ratio_is_measured_from_the_fitted_poses():
    metrics = metrics_of(rigs.healthy(), "depth_variation")
    assert metrics["depth_ratio"] > 1.5
    assert metrics["furthest_mm"] > metrics["nearest_mm"]


def test_single_depth_is_flagged_with_a_ratio_near_one():
    finding = diagnose(*_pair(rigs.single_depth())).by_cause("depth_variation")
    assert finding.severity >= Severity.WARNING
    assert finding.metrics["depth_ratio"] < 1.5


def test_focal_distance_correlation_is_reported_when_identifiable():
    metrics = metrics_of(rigs.single_depth(), "depth_variation")
    assert metrics["focal_distance_correlation"] is not None
    assert abs(metrics["focal_distance_correlation"]) <= 1.0


def test_focal_distance_correlation_is_withheld_when_not_identifiable():
    """A cut direction has no variance to correlate, so the number would mislead."""
    finding = diagnose(*_pair(rigs.frontoparallel())).by_cause("depth_variation")
    assert finding.metrics["focal_distance_correlation"] is None
    assert "not meaningful" in finding.summary


def test_absolute_distances_are_withheld_when_the_focal_length_is_unknown():
    """Depth scales with focal length, so a wrong fx makes the millimetres wrong."""
    finding = diagnose(*_pair(rigs.frontoparallel())).by_cause("depth_variation")
    assert finding.metrics["distances_are_absolute"] is False
    assert "mm" not in finding.action
    assert "your current working distance" in finding.action


def test_absolute_distances_are_quoted_when_the_fit_is_sound():
    finding = diagnose(*_pair(rigs.single_depth())).by_cause("depth_variation")
    assert finding.metrics["distances_are_absolute"] is True
    assert "mm" in finding.action


# --------------------------------------------------------------------------
# frontoparallel dominance
# --------------------------------------------------------------------------

def test_frontoparallel_fraction_counts_views_below_the_threshold():
    context = rigs.frontoparallel()
    metrics = metrics_of(context, "frontoparallel_dominance")
    assert metrics["frontoparallel_fraction"] == pytest.approx(1.0)
    assert metrics["threshold_deg"] == FRONTOPARALLEL_DEGREES
    assert metrics["frontoparallel_views"] == context.n_views


def test_frontoparallel_finding_names_itself_as_the_cause_of_undetermined_focals():
    finding = diagnose(*_pair(rigs.frontoparallel())).by_cause("frontoparallel_dominance")
    assert finding.severity is Severity.CRITICAL
    assert "undetermined" in finding.summary
    assert "this is the cause of that" in finding.summary


def test_a_tilted_capture_is_not_frontoparallel_dominated():
    metrics = metrics_of(rigs.healthy(), "frontoparallel_dominance")
    assert metrics["frontoparallel_fraction"] < 0.5


# --------------------------------------------------------------------------
# image coverage
# --------------------------------------------------------------------------

def test_occupancy_counts_land_in_the_right_cells():
    points = np.array([[10.0, 10.0], [11.0, 11.0], [1270.0, 710.0]])
    counts = occupancy(points, (1280, 720), (8, 6))
    assert counts.shape == (6, 8)
    assert counts[0, 0] == 2
    assert counts[-1, -1] == 1
    assert counts.sum() == 3


def test_occupancy_clamps_points_on_the_far_edge():
    counts = occupancy(np.array([[1280.0, 720.0]]), (1280, 720), (4, 4))
    assert counts[-1, -1] == 1


def test_border_mask_selects_the_outer_ring_only():
    mask = border_mask((4, 3))
    assert mask.shape == (3, 4)
    assert mask.sum() == 4 * 3 - 2  # a 4x3 grid has one interior cell pair
    assert not mask[1, 1]


def test_nearest_neighbour_spacing_on_a_known_lattice():
    grid = np.stack(np.meshgrid(np.arange(5) * 20.0, np.arange(4) * 20.0), axis=-1)
    assert nearest_neighbour_spacing(grid.reshape(-1, 2)) == pytest.approx(20.0)


def test_nearest_neighbour_spacing_of_one_point_is_zero():
    assert nearest_neighbour_spacing(np.array([[1.0, 2.0]])) == 0.0


def test_centre_only_capture_reports_low_reach_and_no_border():
    finding = diagnose(*_pair(rigs.centre_only())).by_cause("image_coverage")
    assert finding.severity is Severity.CRITICAL
    assert finding.metrics["radial_reach"] < 0.5
    assert finding.metrics["peripheral_coverage"] < 0.2


def test_coverage_metrics_are_fractions():
    metrics = metrics_of(rigs.healthy(), "image_coverage")
    for key in ("coverage", "peripheral_coverage", "radial_reach"):
        assert 0.0 <= metrics[key] <= 1.0
    assert metrics["grid"] == list(GRID)


# --------------------------------------------------------------------------
# target scale
# --------------------------------------------------------------------------

def test_a_distant_board_collapses_the_corner_spacing():
    finding = diagnose(*_pair(rigs.tiny_board())).by_cause("target_scale")
    assert finding.severity is Severity.CRITICAL
    assert finding.metrics["median_spacing_px"] < 10.0
    assert finding.metrics["noise_over_spacing"] > 0.01


def test_a_close_board_has_healthy_spacing():
    metrics = metrics_of(rigs.healthy(), "target_scale")
    assert metrics["median_spacing_px"] > 20.0
    assert metrics["min_spacing_px"] <= metrics["median_spacing_px"]


# --------------------------------------------------------------------------
# distortion model adequacy
# --------------------------------------------------------------------------

def test_a_correct_model_leaves_no_radial_structure():
    for builder in (rigs.healthy, rigs.right_model):
        finding = diagnose(*_pair(builder())).by_cause("distortion_model")
        assert finding.severity <= Severity.NOTE, finding.summary
        assert abs(finding.metrics["flatness_z"]) < 3.0


def test_a_fisheye_lens_fitted_as_pinhole_is_caught():
    finding = diagnose(*_pair(rigs.wrong_model())).by_cause("distortion_model")
    assert finding.severity >= Severity.WARNING
    assert finding.metrics["flatness_z"] > 3.0
    assert "fisheye" in finding.action.lower()


def test_the_rms_barely_moves_between_the_right_and_wrong_model():
    """The point of the diagnostic: RMS cannot tell these two apart."""
    wrong, right = rigs.wrong_model(), rigs.right_model()
    assert abs(wrong.fit.rms - right.fit.rms) < 0.02
    assert severity_of(wrong, "distortion_model") > severity_of(right, "distortion_model")


def test_views_are_the_independent_unit_not_corners():
    """Pooling corners would inflate the z-scores, because a view shares a pose."""
    context = rigs.healthy()
    profile = clustered_radial_profile(context)
    clustered = float(np.abs(profile.z_scores()).max())

    # The naive alternative: treat every corner as independent.
    display = context.fit.residuals.radial
    sigma = context.fit.covariance.sigma
    populated = display.counts > 0
    naive = float(
        np.abs(
            display.radial_mean[populated]
            / (sigma / np.sqrt(display.counts[populated]))
        ).max()
    )
    assert naive > clustered, (
        f"clustering did not deflate the z-score: naive {naive:.2f} "
        f"vs clustered {clustered:.2f}"
    )


def test_the_inner_radius_is_excluded_where_the_direction_is_ill_defined():
    context = rigs.healthy()
    profile = clustered_radial_profile(context)
    floor = MIN_RADIUS_SIGMAS * context.fit.covariance.sigma
    assert profile.centres.min() > floor


def test_flatness_z_is_near_zero_for_a_flat_profile():
    from calibsense.diagnose.model import ClusteredProfile

    profile = ClusteredProfile(
        centres=np.linspace(100.0, 400.0, 6),
        means=np.zeros(6),
        errors=np.full(6, 0.01),
        view_counts=np.full(6, 10),
    )
    assert flatness_z(profile) < 0.0


def test_flatness_z_grows_with_structure():
    from calibsense.diagnose.model import ClusteredProfile

    centres = np.linspace(100.0, 400.0, 6)
    errors = np.full(6, 0.01)
    flat = ClusteredProfile(centres, np.zeros(6), errors, np.full(6, 10))
    structured = ClusteredProfile(centres, np.full(6, 0.05), errors, np.full(6, 10))
    assert flatness_z(structured) > flatness_z(flat) + 5.0


def test_flatness_catches_oscillation_that_a_slope_misses():
    """A truncated polynomial leaves alternating signs, which a trend cannot see."""
    from calibsense.diagnose.model import ClusteredProfile

    centres = np.linspace(100.0, 400.0, 6)
    # Symmetric about the middle radius and summing to zero, so the pattern is
    # orthogonal to both a constant and a linear term by construction.
    means = 0.05 * np.array([1.0, -1.0, 0.0, 0.0, -1.0, 1.0])
    profile = ClusteredProfile(centres, means, np.full(6, 0.01), np.full(6, 10))
    slope = radial_trend(profile)
    assert slope is not None
    assert abs(slope[2]) < 1e-6, "the slope should see nothing here"
    assert flatness_z(profile) > 5.0, "the omnibus test should see it"


def test_radial_trend_needs_three_bins():
    from calibsense.diagnose.model import ClusteredProfile

    profile = ClusteredProfile(
        np.array([100.0, 200.0]), np.zeros(2), np.full(2, 0.01), np.full(2, 10)
    )
    assert radial_trend(profile) is None
    assert flatness_z(profile) is not None


# --------------------------------------------------------------------------
# outlier and high-leverage views
# --------------------------------------------------------------------------

def test_information_shares_sum_to_one():
    influences = view_influences(rigs.healthy())
    assert sum(v.information_share for v in influences) == pytest.approx(1.0)
    assert all(v.information_share >= -1e-9 for v in influences)


def test_information_shares_cover_every_view_in_order():
    context = rigs.healthy()
    influences = view_influences(context)
    assert [v.view_id for v in influences] == list(context.observations.view_ids)


def test_a_noisy_view_is_flagged_as_an_outlier():
    context = rigs.with_bad_view()
    finding = diagnose(context.fit, context.observations).by_cause("outlier_views")
    assert finding.severity >= Severity.WARNING
    assert finding.metrics["n_outliers"] >= 1
    flagged = [v for v in finding.metrics["views"] if v["is_outlier"]]
    assert flagged


def test_the_report_says_outliers_are_not_down_weighted():
    """An honest limitation, and it has to reach the reader."""
    context = rigs.with_bad_view()
    finding = diagnose(context.fit, context.observations).by_cause("outlier_views")
    assert "down-weight" in finding.action


def test_leverage_is_reported_against_an_even_share():
    context = rigs.healthy()
    finding = diagnose(context.fit, context.observations).by_cause("outlier_views")
    assert finding.metrics["even_share"] == pytest.approx(1.0 / context.n_views)
    assert finding.metrics["max_information_share"] >= finding.metrics["even_share"]


def test_a_view_influence_serialises():
    payload = view_influences(rigs.healthy())[0].to_dict()
    for key in ("view_id", "information_share", "rms", "robust_z", "dominant"):
        assert key in payload


def _pair(context):
    """Unpack a context into the two positional arguments `diagnose` takes."""
    return context.fit, context.observations


# --------------------------------------------------------------------------
# out-of-sample error, as a finding
# --------------------------------------------------------------------------

def test_without_cross_validation_the_finding_says_so():
    context = rigs.healthy()
    finding = diagnose(context.fit, context.observations).by_cause("out_of_sample_error")
    assert finding.severity is Severity.NOTE
    assert finding.metrics["measured"] is False
    assert "in-sample fit statistic" in finding.summary
    assert "cross-validation" in finding.action


def test_a_generalising_fit_passes():
    from calibsense.core.session import CalibrationSession
    from calibsense.validate import cross_validate

    context = rigs.healthy()
    session = CalibrationSession(observations=context.observations)
    validation = cross_validate(session)
    finding = diagnose(context.fit, context.observations, validation).by_cause(
        "out_of_sample_error"
    )
    assert finding.severity is Severity.OK
    assert finding.metrics["measured"] is True
    assert "honest error estimate" in finding.summary


def test_a_degenerate_capture_makes_the_ratio_uninformative():
    from calibsense.core.session import CalibrationSession
    from calibsense.validate import cross_validate

    context = rigs.frontoparallel()
    session = CalibrationSession(observations=context.observations)
    validation = cross_validate(session)
    finding = diagnose(context.fit, context.observations, validation).by_cause(
        "out_of_sample_error"
    )
    assert finding.severity is Severity.CRITICAL
    assert "uninformative" in finding.summary
    assert finding.metrics["degenerate_folds"]
    # It must point at the signal that does work.
    assert "parameter spread" in finding.action


def test_a_high_ratio_is_reported_at_the_right_severity():
    """The ratio branches, tested directly rather than through a contrived rig.

    Producing a ratio above 2 from synthetic data needs a capture so thin that
    the folds go degenerate first, which exercises a different branch. Replacing
    the ratio on a real result keeps the test about the message.
    """
    import dataclasses

    from calibsense.core.session import CalibrationSession
    from calibsense.diagnose.generalisation import OutOfSampleError
    from calibsense.validate import cross_validate

    context = rigs.healthy()
    validation = cross_validate(CalibrationSession(observations=context.observations))
    assert validation.trustworthy

    def finding_for(out_of_sample_rms):
        altered = dataclasses.replace(
            validation, out_of_sample_rms=out_of_sample_rms
        )
        return OutOfSampleError().run(
            dataclasses.replace(context, cross_validation=altered)
        )

    honest = finding_for(validation.in_sample_rms * 1.05)
    mild = finding_for(validation.in_sample_rms * 1.6)
    bad = finding_for(validation.in_sample_rms * 3.0)

    assert honest.severity is Severity.OK
    assert mild.severity is Severity.WARNING
    assert "overfitting" in mild.summary or "overfitting" in mild.action
    assert bad.severity is Severity.CRITICAL
    assert "understates" in bad.summary
    assert "distortion-terms" in bad.action


def test_the_ratio_rises_with_model_complexity(pinhole, checkerboard):
    """The empirical direction, even where the absolute value stays modest."""
    from calibsense.core.session import CalibrationSession
    from calibsense.refit import RefitOptions
    from calibsense.synthetic import diverse_poses, synthesise
    from calibsense.validate import cross_validate

    capture = synthesise(
        pinhole, checkerboard, diverse_poses(checkerboard, 10, seed=4),
        (1280, 720), noise_px=0.3, seed=21,
    )
    session = CalibrationSession(observations=capture.observations)
    modest = cross_validate(session, RefitOptions(distortion_terms=5))
    greedy = cross_validate(session, RefitOptions(distortion_terms=14))
    assert greedy.ratio > modest.ratio
