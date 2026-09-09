"""Parameter covariance.

Three independent checks, because this is the number the product is built on:

* against `cv2.calibrateCameraExtended`, which reports the marginal standard
  deviations and nothing else,
* against a dense `sigma^2 (J'J)^-1`, which the factored form must reproduce
  exactly wherever the system is full rank,
* against Monte Carlo, which is the only check that the covariance actually
  describes the estimator's spread rather than merely being self-consistent.
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from caltrust.core.camera import PinholeBrownConrady
from caltrust.core.poses import Pose
from caltrust.core.session import CalibrationSession
from caltrust.errors import DegenerateSystemError, ValidationError
from caltrust.refit.covariance import (
    WEAK_DIRECTION_THRESHOLD,
    covariance_from_normal_equations,
)
from caltrust.refit.normal import POSE_DIMENSION, assemble
from caltrust.refit.projection import projector_for
from caltrust.synthetic import diverse_poses, synthesise


def opencv_calibrate(observations):
    """Fit with OpenCV and return the camera, poses and its own std deviations."""
    target = observations.target
    objects = [v.object_points(target).astype(np.float32) for v in observations.views]
    images = [v.image_points.astype(np.float32) for v in observations.views]
    rms, matrix, coefficients, rvecs, tvecs, sd_intrinsic, _, _ = (
        cv2.calibrateCameraExtended(objects, images, observations.image_size, None, None)
    )
    camera = PinholeBrownConrady(
        matrix[0, 0], matrix[1, 1], matrix[0, 2], matrix[1, 2],
        np.asarray(coefficients).reshape(-1),
    )
    poses = [
        Pose.from_rvec_tvec(np.asarray(r).reshape(3), np.asarray(t).reshape(3))
        for r, t in zip(rvecs, tvecs)
    ]
    return camera, poses, np.asarray(sd_intrinsic).reshape(-1), float(rms)


def covariance_at(camera, poses, observations, fixed=(), tie_aspect=False):
    block = camera.free_parameters(fixed, tie_aspect)
    equations, _ = assemble(camera, poses, observations, block)
    return equations, covariance_from_normal_equations(equations)


def test_marginal_std_devs_match_opencv(good_capture):
    observations = good_capture.observations
    camera, poses, opencv_sd, rms = opencv_calibrate(observations)
    equations, covariance = covariance_at(camera, poses, observations)
    # OpenCV requires float32 point arrays, so its own RMS is computed on
    # slightly rounded inputs while ours uses the detections at full precision.
    # The gap is float32 resolution, not a disagreement about the residuals.
    assert equations.rms == pytest.approx(rms, rel=1e-6)
    ours = covariance.intrinsic_std()
    assert np.allclose(ours, opencv_sd[: ours.size], rtol=2e-4)


def test_dense_covariance_matches_the_naive_inverse(good_capture):
    observations = good_capture.observations
    camera, poses, _, _ = opencv_calibrate(observations)
    block = camera.free_parameters()
    equations, covariance = covariance_at(camera, poses, observations)

    reduction = block.reduction()
    n_free = reduction.shape[1]
    rows = 2 * observations.total_points
    jacobian = np.zeros((rows, n_free + POSE_DIMENSION * observations.n_views))
    residual = np.zeros(rows)
    projector = projector_for(camera)
    at = 0
    for index, (view, pose) in enumerate(zip(observations.views, poses)):
        predicted, d_intrinsic, d_pose = projector.project_with_jacobian(
            camera, pose, view.object_points(observations.target)
        )
        span = 2 * view.n_points
        jacobian[at : at + span, :n_free] = d_intrinsic @ reduction
        start = n_free + POSE_DIMENSION * index
        jacobian[at : at + span, start : start + POSE_DIMENSION] = d_pose
        residual[at : at + span] = (predicted - view.image_points).reshape(-1)
        at += span

    normal = jacobian.T @ jacobian
    dof = rows - jacobian.shape[1]
    expected = (float(residual @ residual) / dof) * np.linalg.inv(normal)
    assert covariance.sigma2 == pytest.approx(float(residual @ residual) / dof)
    assert covariance.degrees_of_freedom == dof

    dense = covariance.dense()
    assert dense.shape == expected.shape
    assert np.abs(dense - expected).max() / np.abs(expected).max() < 1e-7


def test_dense_covariance_is_symmetric_and_positive_semi_definite(good_capture):
    camera, poses, _, _ = opencv_calibrate(good_capture.observations)
    _, covariance = covariance_at(camera, poses, good_capture.observations)
    dense = covariance.dense()
    assert np.allclose(dense, dense.T)
    assert np.linalg.eigvalsh(dense).min() > -1e-12 * np.abs(dense).max()


def test_blocks_agree_with_the_dense_form(good_capture):
    camera, poses, _, _ = opencv_calibrate(good_capture.observations)
    _, covariance = covariance_at(camera, poses, good_capture.observations)
    dense = covariance.dense()
    p = covariance.n_intrinsic
    assert np.allclose(dense[:p, :p], covariance.intrinsic)
    for i in range(covariance.n_views):
        start = p + POSE_DIMENSION * i
        stop = start + POSE_DIMENSION
        assert np.allclose(dense[:p, start:stop], covariance.cross_block(i))
        assert np.allclose(dense[start:stop, start:stop], covariance.extrinsic_block(i))
        assert np.allclose(
            dense[start:stop, start:stop], covariance.pose_cross_block(i, i)
        )
    if covariance.n_views > 1:
        start_a = p
        start_b = p + POSE_DIMENSION
        assert np.allclose(
            dense[start_a : start_a + POSE_DIMENSION, start_b : start_b + POSE_DIMENSION],
            covariance.pose_cross_block(0, 1),
        )


def test_different_views_are_correlated_only_through_the_intrinsics(good_capture):
    camera, poses, _, _ = opencv_calibrate(good_capture.observations)
    _, covariance = covariance_at(camera, poses, good_capture.observations)
    cross = covariance.pose_cross_block(0, 1)
    assert np.abs(cross).max() > 0.0
    y0, y1 = covariance._y[0], covariance._y[1]
    expected = covariance.sigma2 * (y0 @ covariance._schur_inverse @ y1.T)
    assert np.allclose(cross, expected)


@pytest.mark.slow
def test_covariance_is_calibrated_against_monte_carlo(pinhole, checkerboard):
    """The predicted spread has to match the spread of repeated refits."""
    poses = diverse_poses(checkerboard, 14, seed=2)
    noise = 0.25
    trials = 250

    def refit(seed):
        capture = synthesise(
            pinhole, checkerboard, poses, (1280, 720), noise_px=noise, seed=seed
        )
        camera, solved, _, _ = opencv_calibrate(capture.observations)
        return camera, solved, capture.observations

    reference_camera, reference_poses, reference_observations = refit(9999)
    _, covariance = covariance_at(
        reference_camera, reference_poses, reference_observations
    )
    predicted = covariance.intrinsic

    samples = np.array([refit(seed)[0].to_vector() for seed in range(trials)])
    empirical = np.cov(samples.T)

    predicted_sd = np.sqrt(np.diag(predicted))
    empirical_sd = np.sqrt(np.diag(empirical))
    ratio = predicted_sd / empirical_sd
    assert np.all(ratio > 0.75), f"predicted too small: {ratio}"
    assert np.all(ratio < 1.35), f"predicted too large: {ratio}"

    # The Mahalanobis distance of the samples from truth should follow chi-square
    # with one degree of freedom per parameter.
    deviation = samples - pinhole.to_vector()
    squared = np.einsum("ni,ij,nj->n", deviation, np.linalg.inv(predicted), deviation)
    expected_mean = covariance.n_intrinsic
    standard_error = np.sqrt(2.0 * expected_mean / trials)
    assert abs(squared.mean() - expected_mean) < 5.0 * standard_error


@pytest.mark.slow
def test_predicted_correlations_match_monte_carlo(pinhole, checkerboard):
    poses = diverse_poses(checkerboard, 14, seed=2)
    trials = 250

    def refit(seed):
        capture = synthesise(
            pinhole, checkerboard, poses, (1280, 720), noise_px=0.25, seed=seed
        )
        return opencv_calibrate(capture.observations), capture.observations

    (camera, solved, _, _), observations = refit(9999)
    _, covariance = covariance_at(camera, solved, observations)
    samples = np.array([refit(seed)[0][0].to_vector() for seed in range(trials)])

    empirical = np.corrcoef(samples.T)
    predicted = covariance.intrinsic_correlation()
    assert np.abs(predicted - empirical).max() < 0.2


def test_degenerate_capture_is_flagged_as_unidentifiable(degenerate_capture):
    observations = degenerate_capture.observations
    camera, poses, _, rms = opencv_calibrate(observations)
    _, covariance = covariance_at(camera, poses, observations)
    assert rms < 0.4, "the degenerate fit must still look good by RMS"
    assert not covariance.is_identifiable()
    weak = covariance.weak_directions()
    assert weak
    involved = {name for name, _ in weak[0].terms[:2]}
    assert involved == {"fx", "fy"}
    participation = dict(zip(covariance.intrinsic_names, covariance.parameter_participation()))
    assert participation["fx"] > 0.5 and participation["fy"] > 0.5
    assert participation["cx"] < 0.2


def test_a_well_posed_capture_is_identifiable(good_capture):
    camera, poses, _, _ = opencv_calibrate(good_capture.observations)
    _, covariance = covariance_at(camera, poses, good_capture.observations)
    assert covariance.is_identifiable()
    assert covariance.weak_directions() == []
    assert np.allclose(covariance.parameter_participation(), 0.0)
    assert covariance.spectrum.scaled_condition_number < 1e6


def test_focal_length_and_distance_are_correlated_in_a_normal_capture(good_capture):
    camera, poses, _, _ = opencv_calibrate(good_capture.observations)
    _, covariance = covariance_at(camera, poses, good_capture.observations)
    correlation = covariance.correlation_with_poses("fx")
    assert correlation.shape == (covariance.n_views, POSE_DIMENSION)
    assert correlation[:, 5].mean() > 0.5
    assert np.all(np.abs(correlation) <= 1.0)


def test_correlation_with_poses_rejects_an_unknown_parameter(good_capture):
    camera, poses, _, _ = opencv_calibrate(good_capture.observations)
    _, covariance = covariance_at(camera, poses, good_capture.observations)
    with pytest.raises(ValidationError, match="not a free intrinsic"):
        covariance.correlation_with_poses("k9")


def test_a_fixed_parameter_is_absent_from_the_covariance(good_capture):
    camera, poses, _, _ = opencv_calibrate(good_capture.observations)
    _, covariance = covariance_at(camera, poses, good_capture.observations, fixed=("cx", "cy"))
    assert "cx" not in covariance.intrinsic_names
    assert covariance.n_intrinsic == len(camera.parameter_names()) - 2
    assert covariance.intrinsic.shape == (covariance.n_intrinsic,) * 2


def test_a_tied_aspect_ratio_removes_fy(good_capture):
    camera, poses, _, _ = opencv_calibrate(good_capture.observations)
    _, covariance = covariance_at(camera, poses, good_capture.observations, tie_aspect=True)
    assert "fy" not in covariance.intrinsic_names
    assert "fx" in covariance.intrinsic_names


def test_no_degrees_of_freedom_is_an_error(pinhole, checkerboard):
    """Four points in one view cannot support both a pose and nine intrinsics."""
    from caltrust.core.observations import ObservationSet, ViewObservations

    projector = projector_for(pinhole)
    pose = diverse_poses(checkerboard, 1, seed=0)[0]
    ids = np.arange(6)
    points = projector.project(pinhole, pose, checkerboard.object_points(ids))
    observations = ObservationSet(
        checkerboard, (1280, 720), (ViewObservations("v", ids, points),)
    )
    equations, _ = assemble(pinhole, [pose], observations, pinhole.free_parameters())
    assert equations.degrees_of_freedom <= 0
    with pytest.raises(DegenerateSystemError, match="no degrees of freedom"):
        covariance_from_normal_equations(equations)


def test_dense_refuses_to_build_an_enormous_matrix(good_capture):
    camera, poses, _, _ = opencv_calibrate(good_capture.observations)
    _, covariance = covariance_at(camera, poses, good_capture.observations)
    with pytest.raises(ValidationError, match="above the limit"):
        covariance.dense(limit=10)


def test_view_index_bounds_are_checked(good_capture):
    camera, poses, _, _ = opencv_calibrate(good_capture.observations)
    _, covariance = covariance_at(camera, poses, good_capture.observations)
    for call in (covariance.extrinsic_block, covariance.cross_block):
        with pytest.raises(ValidationError, match="out of range"):
            call(covariance.n_views)


def test_newton_decrement_is_tiny_at_the_optimum(good_capture):
    camera, poses, _, _ = opencv_calibrate(good_capture.observations)
    equations, covariance = covariance_at(camera, poses, good_capture.observations)
    decrement = covariance.newton_decrement(
        equations.gradient_intrinsic, equations.gradient_pose
    )
    assert decrement / equations.cost < 1e-6


def test_newton_decrement_is_large_away_from_the_optimum(good_capture, pinhole):
    observations = good_capture.observations
    wrong = PinholeBrownConrady(820.0, 830.0, 620.0, 340.0, [-0.1, 0.0, 0.0, 0.0, 0.0])
    projector = projector_for(wrong)
    poses = [
        projector.solve_pose(wrong, v.object_points(observations.target), v.image_points)
        for v in observations.views
    ]
    equations, covariance = covariance_at(wrong, poses, observations)
    decrement = covariance.newton_decrement(
        equations.gradient_intrinsic, equations.gradient_pose
    )
    assert decrement / equations.cost > 1e-3


def test_parameter_names_span_intrinsics_then_poses(good_capture):
    camera, poses, _, _ = opencv_calibrate(good_capture.observations)
    _, covariance = covariance_at(camera, poses, good_capture.observations)
    names = covariance.parameter_names()
    assert len(names) == covariance.n_parameters
    assert names[: covariance.n_intrinsic] == covariance.intrinsic_names
    assert names[-1] == f"view{covariance.n_views - 1}.tz"


def test_sigma_is_the_square_root_of_sigma2(good_capture):
    camera, poses, _, _ = opencv_calibrate(good_capture.observations)
    _, covariance = covariance_at(camera, poses, good_capture.observations)
    assert covariance.sigma == pytest.approx(np.sqrt(covariance.sigma2))


def test_weak_direction_threshold_is_a_sane_default():
    assert 0 < WEAK_DIRECTION_THRESHOLD < 1e-3
