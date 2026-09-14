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

"""M6 — hand-eye calibration with a covariance.

The rigs derive the robot poses from the hand-eye constraint, so the truth is
known exactly and every claim here is checked against it. That matters more than
usual: a hand-eye solve that returns the wrong transform of the two returns a
plausible-looking pose with small residuals, and only a known truth catches it.
"""

from __future__ import annotations

import numpy as np
import pytest

from calibsense.errors import ValidationError
from calibsense.handeye import (
    MIN_VIEWS,
    MOUNTINGS,
    HandEyeResult,
    closed_form,
    diagnose_hand_eye,
    relative_motions,
    solve_hand_eye,
)
from calibsense.handeye.solve import residual
from calibsense.diagnose import Severity
from calibsense.refit import instrument

from . import rigs

pytestmark = pytest.mark.slow

TRUE_CAMERA = rigs.TRUE_FLANGE_TO_CAMERA
TRUE_TARGET = rigs.TRUE_BASE_TO_TARGET


def solved(mounting="eye_in_hand", monte_carlo=False, **kwargs):
    session = rigs.hand_eye_session(mounting, **kwargs)
    fit = instrument(session)
    return session, fit, solve_hand_eye(
        fit, session, mounting, monte_carlo=monte_carlo, n_samples=60
    )


@pytest.mark.parametrize("mounting", MOUNTINGS)
def test_the_camera_transform_recovers_the_truth(mounting):
    _, _, result = solved(mounting)
    offset = np.linalg.norm(result.camera.translation - TRUE_CAMERA.translation)
    angle = np.degrees(result.camera.angle_to(TRUE_CAMERA))
    assert offset < 6.0, f"{mounting}: camera translation off by {offset:.2f} mm"
    assert angle < 1.0, f"{mounting}: camera rotation off by {angle:.3f} deg"


@pytest.mark.parametrize("mounting", MOUNTINGS)
def test_the_target_transform_recovers_the_truth(mounting):
    _, _, result = solved(mounting)
    offset = np.linalg.norm(result.target.translation - TRUE_TARGET.translation)
    assert offset < 6.0, f"{mounting}: target translation off by {offset:.2f} mm"
    assert np.degrees(result.target.angle_to(TRUE_TARGET)) < 1.0


def test_the_closed_form_returns_the_middle_term_of_the_constraint():
    """Eliminating the constant leaves the camera for one mounting, the target
    for the other. Assigning the wrong one produces a plausible, wrong answer.
    """
    for mounting, expected, other in (
        ("eye_in_hand", TRUE_CAMERA, TRUE_TARGET),
        ("eye_to_hand", TRUE_TARGET, TRUE_CAMERA),
    ):
        session = rigs.hand_eye_session(mounting)
        fit = instrument(session)
        middle = closed_form(
            mounting, list(session.robot.aligned_with(session.observations)), list(fit.poses)
        )
        assert np.linalg.norm(middle.translation - expected.translation) < 6.0, mounting
        assert np.linalg.norm(middle.translation - other.translation) > 50.0, mounting


def test_the_residual_vanishes_at_the_truth():
    for mounting in MOUNTINGS:
        session = rigs.hand_eye_session(mounting, noise_px=0.0)
        fit = instrument(session)
        robots = list(session.robot.aligned_with(session.observations))
        for robot, board in zip(robots, fit.poses):
            value = residual(mounting, TRUE_CAMERA, TRUE_TARGET, robot, board)
            assert np.abs(value[:3]).max() < 2e-3, mounting
            assert np.abs(value[3:]).max() < 1.0, mounting


def test_refinement_reduces_the_residual():
    session = rigs.hand_eye_session("eye_in_hand")
    fit = instrument(session)
    robots = list(session.robot.aligned_with(session.observations))
    result = solve_hand_eye(fit, session, "eye_in_hand", monte_carlo=False)
    initial = closed_form("eye_in_hand", robots, fit.poses)
    before = np.linalg.norm(initial.translation - TRUE_CAMERA.translation)
    after = np.linalg.norm(result.camera.translation - TRUE_CAMERA.translation)
    assert result.iterations > 0
    assert after <= before + 1e-9


def test_residuals_cover_every_view():
    session, fit, result = solved()
    assert result.residuals.shape == (session.observations.n_views, 6)
    assert result.view_ids == session.observations.view_ids
    assert result.n_views == session.observations.n_views
    assert result.rotation_rms_deg() > 0
    assert result.translation_rms_mm() > 0


def test_a_healthy_pose_set_is_identifiable():
    _, _, result = solved()
    assert result.identifiable
    assert result.spectrum.rank == 12
    assert result.spectrum.scaled_condition_number < 1e4


def test_the_monte_carlo_covariance_is_wider_than_the_residual_one():
    """Measured: the residual method treats the camera poses as exact data, and
    their errors are correlated across views because the intrinsics are shared.
    """
    session = rigs.hand_eye_session("eye_in_hand")
    fit = instrument(session)
    resampled = solve_hand_eye(
        fit, session, "eye_in_hand", monte_carlo=True, n_samples=120
    )
    assert resampled.covariance_method == "monte_carlo"
    assert resampled.monte_carlo_samples == 120
    assert resampled.optimism_factor() > 1.2
    assert "monte carlo" in "\n".join(resampled.summary_lines())


def test_the_monte_carlo_covariance_covers_the_actual_error():
    """The point of the resampling: the interval has to contain the truth."""
    session = rigs.hand_eye_session("eye_in_hand")
    fit = instrument(session)
    result = solve_hand_eye(fit, session, "eye_in_hand", n_samples=200)
    error = np.linalg.norm(result.camera.translation - TRUE_CAMERA.translation)
    predicted = float(np.linalg.norm(result.translation_std_mm()))
    assert error < 3.0 * predicted, (
        f"actual error {error:.2f} mm against a predicted {predicted:.2f} mm"
    )


def test_the_residual_covariance_is_kept_for_comparison():
    session = rigs.hand_eye_session("eye_in_hand")
    fit = instrument(session)
    result = solve_hand_eye(fit, session, "eye_in_hand", n_samples=60)
    assert result.residual_covariance.shape == (12, 12)
    assert not np.allclose(result.covariance, result.residual_covariance)


def test_turning_off_monte_carlo_says_it_is_optimistic():
    _, _, result = solved(monte_carlo=False)
    assert result.covariance_method == "residual"
    assert result.optimism_factor() == pytest.approx(1.0)
    assert "optimistic" in "\n".join(result.summary_lines())


def test_covariance_blocks_and_deviations():
    _, _, result = solved()
    assert result.camera_covariance.shape == (6, 6)
    assert result.target_covariance.shape == (6, 6)
    assert result.rotation_std_deg().shape == (3,)
    assert result.translation_std_mm().shape == (3,)
    assert np.all(result.rotation_std_deg() >= 0)
    assert len(result.PARAMETER_NAMES) == 12


def test_result_serialises():
    _, _, result = solved()
    payload = result.to_dict()
    for key in ("mounting", "camera", "target", "covariance", "identifiable",
                "covariance_method", "optimism_factor", "rotation_std_deg"):
        assert key in payload
    assert len(payload["covariance"]) == 12
    assert payload["camera"]["matrix"] and payload["target"]["matrix"]


def test_relative_motions_pair_robot_and_camera():
    session = rigs.hand_eye_session("eye_in_hand")
    fit = instrument(session)
    robots = list(session.robot.aligned_with(session.observations))
    robot_axes, camera_axes, robot_t, camera_t = relative_motions(
        robots, list(fit.poses), "eye_in_hand"
    )
    assert robot_axes.shape == camera_axes.shape
    assert robot_t.shape == camera_t.shape == robot_axes.shape
    # Conjugate rotations have equal angles.
    assert np.allclose(
        np.linalg.norm(robot_axes, axis=1),
        np.linalg.norm(camera_axes, axis=1),
        atol=1e-2,
    )


def test_relative_motions_refuse_a_pure_translation():
    """Hand-eye is determined by rotation and by nothing else."""
    from calibsense.core.poses import Pose

    robots = [Pose(np.eye(3), [i * 50.0, 0.0, 0.0]) for i in range(6)]
    boards = [Pose(np.eye(3), [0.0, 0.0, 800.0 + i]) for i in range(6)]
    with pytest.raises(ValidationError, match="translations alone"):
        relative_motions(robots, boards, "eye_in_hand")


@pytest.mark.parametrize("mounting", ["sideways", ""])
def test_an_unknown_mounting_is_rejected(mounting):
    session = rigs.hand_eye_session("eye_in_hand")
    fit = instrument(session)
    with pytest.raises(ValidationError, match="unknown mounting"):
        solve_hand_eye(fit, session, mounting)


def test_a_session_without_robot_poses_is_rejected():
    from calibsense.core.session import CalibrationSession

    capture = rigs.healthy()
    session = CalibrationSession(observations=capture.observations)
    with pytest.raises(ValidationError, match="no robot poses"):
        solve_hand_eye(capture.fit, session, "eye_in_hand")


def test_too_few_views_is_rejected():
    session = rigs.hand_eye_session("eye_in_hand", n=MIN_VIEWS - 1)
    fit = instrument(session)
    with pytest.raises(ValidationError, match=f"at least {MIN_VIEWS} views"):
        solve_hand_eye(fit, session, "eye_in_hand")


def test_a_mismatched_fit_and_session_are_rejected():
    session = rigs.hand_eye_session("eye_in_hand")
    other = rigs.hand_eye_session("eye_in_hand", seed=9)
    fit = instrument(other)
    if fit.view_ids == session.observations.view_ids:
        pytest.skip("the two rigs happened to keep the same views")
    with pytest.raises(ValidationError, match="different views"):
        solve_hand_eye(fit, session, "eye_in_hand")


def test_a_bad_covariance_shape_is_rejected():
    _, _, result = solved()
    import dataclasses

    with pytest.raises(ValidationError, match="12x12"):
        dataclasses.replace(result, covariance=np.eye(6))
    with pytest.raises(ValidationError, match="unknown covariance method"):
        dataclasses.replace(result, covariance_method="guesswork")


# --------------------------------------------------------------------------
# pose-set sufficiency
# --------------------------------------------------------------------------

def diagnose_of(session, fit, result):
    return diagnose_hand_eye(
        result, list(session.robot.aligned_with(session.observations)), list(fit.poses)
    )


def test_a_healthy_pose_set_has_no_findings():
    """Nothing the pose set can be blamed for, which is not the same as nothing.

    The board-scale note is always present and is deliberately not a warning:
    it reports a systematic the tool cannot see rather than a defect in the
    capture, and no rearrangement of the poses would remove it.
    """
    session, fit, result = solved()
    diagnosis = diagnose_of(session, fit, result)
    assert not diagnosis.critical and not diagnosis.warnings
    assert len(diagnosis.findings) == 5
    assert [f.cause for f in diagnosis.findings if f.severity is not Severity.OK] == [
        "hand_eye_board_scale"
    ]


def test_rotation_about_one_axis_is_caught():
    """R_X is determined only up to a rotation about the single shared axis."""
    session = rigs.hand_eye_session("eye_in_hand", roll_only=True)
    fit = instrument(session)
    result = solve_hand_eye(fit, session, "eye_in_hand", monte_carlo=False)
    diagnosis = diagnose_of(session, fit, result)
    causes = {f.cause for f in diagnosis.critical}
    assert "hand_eye_rotation_axes" in causes
    assert "hand_eye_conditioning" in causes
    # And the error it hides is large.
    assert np.linalg.norm(result.camera.translation - TRUE_CAMERA.translation) > 20.0


def test_mispaired_robot_poses_are_caught():
    """Conjugate rotations have equal angles, so a shuffle is detectable."""
    session = rigs.hand_eye_session("eye_in_hand", mispair=True)
    fit = instrument(session)
    result = solve_hand_eye(fit, session, "eye_in_hand", monte_carlo=False)
    diagnosis = diagnose_of(session, fit, result)
    finding = diagnosis.by_cause("hand_eye_pairing")
    assert finding.severity is Severity.CRITICAL
    assert finding.metrics["median_disagreement_deg"] > 1.0
    assert np.linalg.norm(result.camera.translation - TRUE_CAMERA.translation) > 20.0


def test_tiny_rotations_are_refused_outright():
    session = rigs.hand_eye_session(
        "eye_in_hand", tilt_degrees=(0.2, 0.5), roll_span_rad=0.02
    )
    fit = instrument(session)
    with pytest.raises(ValidationError, match="translations alone|degrees of rotation"):
        solve_hand_eye(fit, session, "eye_in_hand", monte_carlo=False)


def test_the_pairing_check_passes_on_matched_pairs():
    session, fit, result = solved()
    finding = diagnose_of(session, fit, result).by_cause("hand_eye_pairing")
    assert finding.severity is Severity.OK
    assert finding.metrics["median_disagreement_deg"] < 0.2


def test_every_hand_eye_finding_carries_metrics_and_an_action():
    session = rigs.hand_eye_session("eye_in_hand", roll_only=True)
    fit = instrument(session)
    result = solve_hand_eye(fit, session, "eye_in_hand", monte_carlo=False)
    for finding in diagnose_of(session, fit, result).findings:
        assert finding.metrics
        if finding.severity >= Severity.WARNING:
            assert finding.action


def test_diagnose_rejects_mismatched_pose_lists():
    session, fit, result = solved()
    with pytest.raises(ValidationError, match="robot poses against"):
        diagnose_hand_eye(result, [], list(fit.poses))


# --------------------------------------------------------------------------
# board scale: the error nothing else in the package can see
# --------------------------------------------------------------------------

def _mislabelled(session, relative_error):
    """The same detections, told the board is a different size than it is."""
    from calibsense.core.observations import ObservationSet
    from calibsense.core.session import CalibrationSession
    from calibsense.core.target import Checkerboard

    observations = session.observations
    target = observations.target
    nominal = Checkerboard(
        target.columns, target.rows, target.square_size / (1.0 + relative_error)
    )
    return CalibrationSession(
        observations=ObservationSet(
            nominal, observations.image_size, observations.views
        ),
        robot=session.robot,
    )


@pytest.mark.slow
def test_a_board_scale_error_leaves_the_focal_length_and_the_residual_alone():
    """The reason this needs its own finding: nothing else can detect it.

    Scaling the board and scaling every translation by the same factor produces
    identical images, so the fit absorbs a board scale error entirely into the
    poses. The focal length does not move and the residual does not move, which
    means neither the reprojection error, the cross-validation ratio nor the
    noise-model check can see it.
    """
    from calibsense.refit import RefitOptions, instrument

    session = rigs.hand_eye_session("eye_in_hand")
    clean = instrument(_mislabelled(session, 0.0), RefitOptions())
    scaled = instrument(_mislabelled(session, 0.01), RefitOptions())

    assert scaled.camera.fx == pytest.approx(clean.camera.fx, rel=1e-4)
    assert scaled.rms == pytest.approx(clean.rms, rel=1e-6)
    # It went into the distances instead, at very nearly the full board error.
    ratio = scaled.working_distances_mm().mean() / clean.working_distances_mm().mean()
    assert ratio == pytest.approx(1.0 / 1.01, rel=2e-3)


@pytest.mark.slow
@pytest.mark.parametrize("mounting", ["eye_in_hand", "eye_to_hand"])
def test_the_board_scale_sensitivity_predicts_the_real_shift(mounting):
    """The reported sensitivity has to be the actual displacement, not an omen.

    It is measured with a 0.1% probe, so a test that re-derives it at 0.5% also
    checks the response is linear over the range a printed target can be out by.
    """
    from calibsense.refit import RefitOptions, instrument

    session = rigs.hand_eye_session(mounting)
    clean_session = _mislabelled(session, 0.0)
    clean_fit = instrument(clean_session, RefitOptions())
    clean = solve_hand_eye(
        clean_fit, clean_session, mounting=mounting, monte_carlo=False
    )

    error = 0.005
    bad_session = _mislabelled(session, error)
    bad_fit = instrument(bad_session, RefitOptions())
    bad = solve_hand_eye(bad_fit, bad_session, mounting=mounting, monte_carlo=False)

    actual = float(np.linalg.norm(
        np.asarray(bad.camera.translation) - np.asarray(clean.camera.translation)
    ))
    predicted = clean.board_scale_sensitivity_mm * error
    assert predicted == pytest.approx(actual, rel=0.05)
    # And it is the amplification that makes this worth reporting at all.
    magnitude = np.linalg.norm(np.asarray(clean.camera.translation))
    assert clean.board_scale_sensitivity_mm > 3.0 * magnitude


@pytest.mark.slow
def test_the_board_finding_says_when_the_board_beats_the_random_error():
    from calibsense.handeye.diagnose import diagnose_hand_eye
    from calibsense.refit import RefitOptions, instrument

    session = rigs.hand_eye_session("eye_in_hand")
    fit = instrument(session, RefitOptions())
    result = solve_hand_eye(fit, session, mounting="eye_in_hand", monte_carlo=False)
    robots = list(session.robot.aligned_with(session.observations))
    diagnosis = diagnose_hand_eye(result, robots, list(fit.poses))

    finding = next(f for f in diagnosis.findings if f.cause == "hand_eye_board_scale")
    # A note, not a warning: the ratio grows as the capture improves, so
    # grading on it would make a better calibration complain louder.
    assert finding.severity is Severity.NOTE
    assert finding.metrics["dominant"] is True
    assert finding.metrics["ratio"] > 1.0
    assert finding.metrics["implied_translation_error_mm"] > (
        finding.metrics["reported_translation_sd_mm"]
    )
    assert "certificate" in finding.action


def test_a_sensitivity_that_cannot_be_measured_is_reported_as_absent():
    """Returning zero silently would read as "the board costs nothing"."""
    import dataclasses

    from calibsense.handeye.diagnose import board_scale, HandEyeContext

    session, fit, result = solved()
    robots = list(session.robot.aligned_with(session.observations))
    context = HandEyeContext.build(
        dataclasses.replace(result, board_scale_sensitivity_mm=0.0),
        robots, list(fit.poses),
    )
    finding = board_scale(context)
    assert finding.severity is Severity.NOTE
    assert "could not be measured" in finding.summary
    assert finding.metrics["board_scale_sensitivity_mm"] == 0.0


def test_a_perturbed_solve_that_will_not_converge_gives_no_sensitivity(monkeypatch):
    """The probe is an extra solve, and an extra solve can fail."""
    from calibsense.errors import ValidationError
    from calibsense.handeye import solve as solve_module

    session, fit, result = solved()
    robots = list(session.robot.aligned_with(session.observations))

    def refuse(*args, **kwargs):
        raise ValidationError("no")

    monkeypatch.setattr(solve_module, "_solve_transforms", refuse)
    assert solve_module.board_scale_sensitivity(
        "eye_in_hand", robots, list(fit.poses), result.camera
    ) == 0.0
