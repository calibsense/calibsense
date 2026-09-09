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

"""The refit engine: M2 end to end on synthetic captures with known truth."""

from __future__ import annotations

import numpy as np
import pytest

from caltrust.core.camera import FisheyeKannalaBrandt, PinholeBrownConrady
from caltrust.core.observations import ObservationSet, ViewObservations
from caltrust.core.session import CalibrationRecord, CalibrationSession
from caltrust.errors import RefitError, ValidationError
from caltrust.refit import RefitOptions, instrument
from caltrust.refit.result import OPTIMUM_DECREMENT_TOLERANCE
from caltrust.synthetic import diverse_poses, synthesise


def z_scores(fit, truth):
    values = dict(zip(truth.parameter_names(), truth.to_vector()))
    return {
        name: (value - values[name]) / deviation if deviation > 0 else 0.0
        for name, value, deviation, _ in fit.parameter_table()
    }


def test_refit_recovers_truth_within_its_own_uncertainty(good_session, pinhole):
    fit = instrument(good_session)
    assert fit.refitted
    assert fit.at_optimum
    assert fit.conditioning.identifiable
    scores = z_scores(fit, pinhole)
    # The focal length and principal point are what a metric measurement rests
    # on; three sigma is a generous but non-vacuous bound.
    for name in ("fx", "fy", "cx", "cy"):
        assert abs(scores[name]) < 3.0, f"{name} is {scores[name]:.2f} sigma from truth"


def test_reported_sigma_is_near_the_injected_noise(good_session):
    fit = instrument(good_session)
    assert fit.covariance.sigma == pytest.approx(0.2, rel=0.25)


def test_degrees_of_freedom_account_for_every_parameter(good_session):
    fit = instrument(good_session)
    expected = 2 * fit.residuals.total_points - (
        fit.covariance.n_intrinsic + 6 * fit.n_views
    )
    assert fit.covariance.degrees_of_freedom == expected


def test_a_degenerate_capture_is_reported_as_unidentifiable(degenerate_capture):
    fit = instrument(CalibrationSession(observations=degenerate_capture.observations))
    assert not fit.conditioning.identifiable
    assert fit.rms < 0.4
    assert fit.conditioning.scaled_condition_number > 1e6
    text = "\n".join(fit.summary_lines())
    assert "IDENTIFIABILITY" in text
    participation = dict(zip(fit.covariance.intrinsic_names, fit.conditioning.participation))
    assert participation["fx"] > 0.5


def test_fisheye_refit_recovers_truth(fisheye_capture, fisheye):
    fit = instrument(
        CalibrationSession(observations=fisheye_capture.observations),
        RefitOptions(model="fisheye"),
    )
    assert isinstance(fit.camera, FisheyeKannalaBrandt)
    assert fit.at_optimum
    assert fit.initial_guess == "width/pi"
    scores = z_scores(fit, fisheye)
    for name in ("fx", "fy", "cx", "cy"):
        assert abs(scores[name]) < 3.0, f"{name} is {scores[name]:.2f} sigma from truth"


def test_fisheye_skew_is_never_estimated(fisheye_capture):
    fit = instrument(
        CalibrationSession(observations=fisheye_capture.observations),
        RefitOptions(model="fisheye"),
    )
    assert "alpha" not in fit.covariance.intrinsic_names
    assert fit.camera.alpha == 0.0


def test_model_is_taken_from_the_session_calibration(fisheye_capture, fisheye):
    session = CalibrationSession(
        observations=fisheye_capture.observations,
        prior=CalibrationRecord(fisheye, (1280, 720), "k.yml"),
    )
    fit = instrument(session)
    assert isinstance(fit.camera, FisheyeKannalaBrandt)
    assert fit.initial_guess == "prior"


def test_pinhole_is_the_default_model(good_session):
    assert isinstance(instrument(good_session).camera, PinholeBrownConrady)


@pytest.mark.parametrize("terms,expected", [(4, 5), (5, 5), (8, 8), (12, 12), (14, 14)])
def test_distortion_terms_control_the_vector_length(good_session, terms, expected):
    fit = instrument(good_session, RefitOptions(distortion_terms=terms))
    assert fit.camera.distortion.size == expected


def test_four_terms_means_k3_is_held_at_zero(good_session):
    fit = instrument(good_session, RefitOptions(distortion_terms=4))
    assert fit.camera.distortion[4] == 0.0
    assert "k3" not in fit.covariance.intrinsic_names


def test_fixing_the_principal_point_leaves_it_where_it_started(good_capture, pinhole):
    session = CalibrationSession(
        observations=good_capture.observations,
        prior=CalibrationRecord(pinhole, (1280, 720), "a.yml"),
    )
    fit = instrument(session, RefitOptions(fixed=("cx", "cy")))
    assert fit.camera.cx == pytest.approx(pinhole.cx)
    assert fit.camera.cy == pytest.approx(pinhole.cy)
    assert "cx" not in fit.covariance.intrinsic_names


def test_tie_aspect_makes_the_ratio_hold(good_capture, pinhole):
    session = CalibrationSession(
        observations=good_capture.observations,
        prior=CalibrationRecord(pinhole, (1280, 720), "a.yml"),
    )
    fit = instrument(session, RefitOptions(tie_aspect=True))
    assert fit.camera.fy / fit.camera.fx == pytest.approx(pinhole.fy / pinhole.fx, rel=1e-6)
    assert "fy" not in fit.covariance.intrinsic_names


def test_fixing_one_focal_length_alone_is_refused(good_session):
    with pytest.raises(ValidationError, match="both focal lengths or neither"):
        instrument(good_session, RefitOptions(fixed=("fx",)))


def test_fixing_one_principal_coordinate_alone_is_refused(good_session):
    with pytest.raises(ValidationError, match="both principal point"):
        instrument(good_session, RefitOptions(fixed=("cx",)))


def test_fixing_one_tangential_term_alone_is_refused(good_session):
    with pytest.raises(ValidationError, match="both tangential terms"):
        instrument(good_session, RefitOptions(fixed=("p1",)))


def test_fixing_a_parameter_opencv_cannot_pin_is_refused(good_session):
    with pytest.raises(ValidationError, match="cannot hold"):
        instrument(good_session, RefitOptions(distortion_terms=12, fixed=("s1",)))


def test_zeroing_both_tangential_terms_is_accepted(good_session):
    fit = instrument(good_session, RefitOptions(fixed=("p1", "p2")))
    assert fit.camera.distortion[2] == 0.0 and fit.camera.distortion[3] == 0.0


def test_fixing_k1_is_accepted(good_capture, pinhole):
    session = CalibrationSession(
        observations=good_capture.observations,
        prior=CalibrationRecord(pinhole, (1280, 720), "a.yml"),
    )
    fit = instrument(session, RefitOptions(fixed=("k1",)))
    assert fit.camera.distortion[0] == pytest.approx(pinhole.distortion[0])


def test_no_refit_instruments_the_existing_calibration_in_place(session_with_prior):
    prior = session_with_prior.prior.camera
    fit = instrument(session_with_prior, RefitOptions(refit=False))
    assert not fit.refitted
    assert fit.camera.allclose(prior)
    assert fit.prior_rms == pytest.approx(0.2412)
    # A calibration that is not the optimum of these detections must say so.
    assert not fit.at_optimum
    assert fit.relative_decrement > OPTIMUM_DECREMENT_TOLERANCE
    assert "NOT AT OPTIMUM" in "\n".join(fit.summary_lines())


def test_no_refit_recovers_poses_at_the_given_intrinsics(good_capture, pinhole):
    session = CalibrationSession(
        observations=good_capture.observations,
        prior=CalibrationRecord(pinhole, (1280, 720), "a.yml"),
    )
    fit = instrument(session, RefitOptions(refit=False))
    for solved, truth in zip(fit.poses, good_capture.poses):
        assert np.degrees(solved.angle_to(truth)) < 0.5
        assert np.linalg.norm(solved.translation - truth.translation) < 5.0


def test_no_refit_without_a_calibration_is_refused(good_session):
    with pytest.raises(RefitError, match="session carries none"):
        instrument(good_session, RefitOptions(refit=False))


def test_ignoring_the_prior_still_converges(session_with_prior, pinhole):
    fit = instrument(session_with_prior, RefitOptions(use_prior_as_guess=False))
    assert fit.initial_guess == "image size"
    assert fit.at_optimum
    assert abs(fit.camera.fx - pinhole.fx) < 5.0


def test_a_view_with_too_few_points_is_refused(pinhole, checkerboard):
    projector_points = checkerboard.object_points([0, 1, 2])
    from caltrust.refit.projection import projector_for

    pose = diverse_poses(checkerboard, 1, seed=0)[0]
    thin = ViewObservations(
        "thin", [0, 1, 2],
        projector_for(pinhole).project(pinhole, pose, projector_points),
    )
    capture = synthesise(pinhole, checkerboard, diverse_poses(checkerboard, 6, seed=1))
    observations = ObservationSet(
        checkerboard, (1280, 720), capture.observations.views + (thin,)
    )
    with pytest.raises(RefitError, match="fewer than 4 points"):
        instrument(CalibrationSession(observations=observations))


def test_too_little_data_for_any_uncertainty_is_refused(pinhole, checkerboard):
    from caltrust.refit.projection import projector_for

    pose = diverse_poses(checkerboard, 1, seed=0)[0]
    ids = np.arange(7)
    points = projector_for(pinhole).project(pinhole, pose, checkerboard.object_points(ids))
    observations = ObservationSet(
        checkerboard, (1280, 720), (ViewObservations("v", ids, points),)
    )
    with pytest.raises(RefitError, match="nothing left to estimate"):
        instrument(CalibrationSession(observations=observations))


def test_working_distances_match_the_capture(good_session, good_capture):
    fit = instrument(good_session)
    truth = np.array([p.distance_mm for p in good_capture.poses])
    assert np.allclose(np.sort(fit.working_distances_mm()), np.sort(truth), rtol=0.05)


def test_residual_arrays_line_up_with_the_views(good_session):
    fit = instrument(good_session)
    assert fit.residuals.n_views == fit.n_views
    assert tuple(v.view_id for v in fit.residuals.per_view) == fit.view_ids
    assert fit.residuals.total_points == sum(v.n_points for v in fit.residuals.per_view)


def test_parameter_table_covers_every_free_parameter(good_session):
    fit = instrument(good_session)
    table = fit.parameter_table()
    assert [row[0] for row in table] == list(fit.covariance.intrinsic_names)
    assert all(row[2] >= 0 for row in table)


def test_a_uniform_shift_is_absorbed_by_the_pose(good_capture, checkerboard):
    """Translating a whole view is a change of tx and ty, not a residual."""
    views = list(good_capture.observations.views)
    shifted = views[2]
    views[2] = ViewObservations(
        shifted.view_id, shifted.point_ids, shifted.image_points + 2.0
    )
    observations = ObservationSet(checkerboard, (1280, 720), tuple(views))
    fit = instrument(CalibrationSession(observations=observations))
    assert not fit.residuals.outlier_views()


def test_summary_reports_outlier_views(good_capture, checkerboard):
    """A view with its own much larger noise cannot be fitted away."""
    views = list(good_capture.observations.views)
    noisy = views[2]
    rng = np.random.default_rng(0)
    views[2] = ViewObservations(
        noisy.view_id, noisy.point_ids,
        noisy.image_points + rng.normal(0.0, 3.0, noisy.image_points.shape),
    )
    observations = ObservationSet(checkerboard, (1280, 720), tuple(views))
    fit = instrument(CalibrationSession(observations=observations))
    outliers = fit.residuals.outlier_views()
    assert [v.view_id for v in outliers] == [noisy.view_id]
    assert "outlier views" in "\n".join(fit.summary_lines())


def test_options_round_trip_through_a_dictionary():
    options = RefitOptions(
        model="fisheye", distortion_terms=8, fixed=("cx", "cy"), tie_aspect=True,
        refit=False, use_prior_as_guess=False, rcond=1e-10, radial_bins=6,
        check_condition=True,
    )
    assert RefitOptions.from_dict(options.to_dict()) == options


@pytest.mark.parametrize(
    "kwargs,message",
    [
        ({"model": "omni"}, "model must be"),
        ({"distortion_terms": 6}, "distortion_terms"),
        ({"rcond": 0.0}, "rcond"),
        ({"rcond": 1.0}, "rcond"),
        ({"radial_bins": 0}, "radial_bins"),
    ],
)
def test_invalid_options_are_rejected(kwargs, message):
    with pytest.raises(ValidationError, match=message):
        RefitOptions(**kwargs)


def test_radial_bins_option_is_honoured(good_session):
    fit = instrument(good_session, RefitOptions(radial_bins=4))
    assert fit.residuals.radial.n_bins == 4


def test_fit_records_its_provenance(good_session):
    fit = instrument(good_session)
    assert fit.created.endswith("+00:00")
    assert fit.caltrust_version
    assert fit.image_size == (1280, 720)


def test_over_parameterised_model_reports_weak_directions(good_session):
    """Fitting 14 coefficients to data generated with 5 must say the extras are free."""
    fit = instrument(good_session, RefitOptions(distortion_terms=14))
    assert not fit.conditioning.identifiable
    weak = {name for direction in fit.conditioning.weak_directions
            for name, _ in direction.terms}
    assert weak & {"k4", "k5", "k6", "s1", "s2", "s3", "s4", "taux", "tauy"}


def test_fisheye_can_fix_the_principal_point(fisheye_capture, fisheye):
    session = CalibrationSession(
        observations=fisheye_capture.observations,
        prior=CalibrationRecord(fisheye, (1280, 720), "k.yml"),
    )
    fit = instrument(session, RefitOptions(model="fisheye", fixed=("cx", "cy")))
    assert fit.camera.cx == pytest.approx(fisheye.cx)
    assert "cx" not in fit.covariance.intrinsic_names


def test_fisheye_can_fix_the_focal_lengths(fisheye_capture, fisheye):
    session = CalibrationSession(
        observations=fisheye_capture.observations,
        prior=CalibrationRecord(fisheye, (1280, 720), "k.yml"),
    )
    fit = instrument(session, RefitOptions(model="fisheye", fixed=("fx", "fy")))
    assert fit.camera.fx == pytest.approx(fisheye.fx)
    assert "fx" not in fit.covariance.intrinsic_names


def test_fixing_a_fisheye_coefficient_zeroes_it(fisheye_capture, fisheye):
    """OpenCV's fisheye path zeroes a fixed coefficient rather than holding it."""
    session = CalibrationSession(
        observations=fisheye_capture.observations,
        prior=CalibrationRecord(fisheye, (1280, 720), "k.yml"),
    )
    fit = instrument(session, RefitOptions(model="fisheye", fixed=("k3", "k4")))
    assert fit.camera.distortion[2] == 0.0
    assert fit.camera.distortion[3] == 0.0
    assert "k3" not in fit.covariance.intrinsic_names
    assert "k4" not in fit.covariance.intrinsic_names


def test_fixing_a_pinhole_coefficient_holds_its_value(good_capture, pinhole):
    """The pinhole path, by contrast, keeps the value it was handed."""
    session = CalibrationSession(
        observations=good_capture.observations,
        prior=CalibrationRecord(pinhole, (1280, 720), "a.yml"),
    )
    fit = instrument(session, RefitOptions(fixed=("k2", "k3")))
    assert fit.camera.distortion[1] == pytest.approx(pinhole.distortion[1])
    assert fit.camera.distortion[4] == pytest.approx(pinhole.distortion[4])
    assert "k2" not in fit.covariance.intrinsic_names


def test_fisheye_rejects_fixing_only_one_focal_length(fisheye_capture):
    with pytest.raises(ValidationError, match="both focal lengths or neither"):
        instrument(
            CalibrationSession(observations=fisheye_capture.observations),
            RefitOptions(model="fisheye", fixed=("fy",)),
        )


def test_fisheye_rejects_fixing_only_one_principal_coordinate(fisheye_capture):
    with pytest.raises(ValidationError, match="both principal point"):
        instrument(
            CalibrationSession(observations=fisheye_capture.observations),
            RefitOptions(model="fisheye", fixed=("cy",)),
        )


def test_fisheye_rejects_a_parameter_it_does_not_estimate(fisheye_capture):
    with pytest.raises(ValidationError, match="estimates only"):
        instrument(
            CalibrationSession(observations=fisheye_capture.observations),
            RefitOptions(model="fisheye", fixed=("p1",)),
        )


def test_check_condition_is_passed_through(fisheye_capture):
    """It is a request to OpenCV, so the only contract is that it does not crash."""
    fit = instrument(
        CalibrationSession(observations=fisheye_capture.observations),
        RefitOptions(model="fisheye", check_condition=True),
    )
    assert fit.options.check_condition


def test_fisheye_ladder_exhaustion_reports_every_attempt(monkeypatch, fisheye_capture):
    import cv2

    def always_fail(*args, **kwargs):
        raise cv2.error("InitExtrinsics: fabs(norm_u1) > 0")

    monkeypatch.setattr(cv2.fisheye, "calibrate", always_fail)
    with pytest.raises(RefitError, match="failed from every starting point"):
        instrument(
            CalibrationSession(observations=fisheye_capture.observations),
            RefitOptions(model="fisheye"),
        )


def test_fisheye_ladder_recovers_when_the_first_guess_fails(monkeypatch, fisheye_capture):
    """The ladder exists because OpenCV's own initialiser fails on real captures."""
    import cv2

    real = cv2.fisheye.calibrate
    calls = {"n": 0}

    def fail_once(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            raise cv2.error("InitExtrinsics: fabs(norm_u1) > 0")
        return real(*args, **kwargs)

    monkeypatch.setattr(cv2.fisheye, "calibrate", fail_once)
    fit = instrument(
        CalibrationSession(observations=fisheye_capture.observations),
        RefitOptions(model="fisheye"),
    )
    assert calls["n"] == 2
    assert fit.initial_guess == "width/2.5"
    assert fit.at_optimum


def test_a_pinhole_optimiser_failure_is_reported_cleanly(monkeypatch, good_session):
    import cv2

    def always_fail(*args, **kwargs):
        raise cv2.error("not enough points")

    monkeypatch.setattr(cv2, "calibrateCamera", always_fail)
    with pytest.raises(RefitError, match="cv2.calibrateCamera failed"):
        instrument(good_session)


def test_a_pose_that_cannot_be_solved_names_the_view(monkeypatch, session_with_prior):
    import cv2

    from caltrust.refit.projection import PinholeProjector

    def always_fail(self, *args, **kwargs):
        raise cv2.error("solvePnP exploded")

    monkeypatch.setattr(PinholeProjector, "solve_pose", always_fail)
    with pytest.raises(RefitError, match="could not solve a pose"):
        instrument(session_with_prior, RefitOptions(refit=False))
