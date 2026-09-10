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

"""Normal equation assembly and the Schur complement."""

from __future__ import annotations

import numpy as np
import pytest

from calibsense.core.camera import PinholeBrownConrady
from calibsense.errors import RefitError, ValidationError
from calibsense.refit.normal import POSE_DIMENSION, assemble
from calibsense.refit.projection import projector_for


def dense_jacobian(camera, poses, observations, block):
    """Build the full Jacobian the naive way, for comparison."""
    reduction = block.reduction()
    n_free = reduction.shape[1]
    n_views = observations.n_views
    rows = 2 * observations.total_points
    jacobian = np.zeros((rows, n_free + POSE_DIMENSION * n_views))
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
    return jacobian, residual


def test_blocks_match_the_dense_jacobian(good_capture):
    camera, observations = good_capture.camera, good_capture.observations
    block = camera.free_parameters()
    equations, per_view = assemble(camera, list(good_capture.poses), observations, block)
    jacobian, residual = dense_jacobian(camera, good_capture.poses, observations, block)

    n_free = block.n_free
    normal = jacobian.T @ jacobian
    assert np.allclose(equations.u, normal[:n_free, :n_free], rtol=1e-9, atol=1e-6)
    for index in range(observations.n_views):
        start = n_free + POSE_DIMENSION * index
        stop = start + POSE_DIMENSION
        assert np.allclose(equations.w[index], normal[:n_free, start:stop], atol=1e-6)
        assert np.allclose(equations.v[index], normal[start:stop, start:stop], atol=1e-6)
    gradient = jacobian.T @ residual
    assert np.allclose(equations.gradient_intrinsic, gradient[:n_free], atol=1e-6)
    assert equations.cost == pytest.approx(float(residual @ residual))
    assert equations.n_residuals == 2 * observations.total_points
    assert sum(r.shape[0] for r in per_view) == observations.total_points


def test_schur_complement_matches_the_dense_elimination(good_capture):
    camera, observations = good_capture.camera, good_capture.observations
    block = camera.free_parameters()
    equations, _ = assemble(camera, list(good_capture.poses), observations, block)
    jacobian, _ = dense_jacobian(camera, good_capture.poses, observations, block)
    normal = jacobian.T @ jacobian
    n_free = block.n_free

    u = normal[:n_free, :n_free]
    w = normal[:n_free, n_free:]
    v = normal[n_free:, n_free:]
    expected = u - w @ np.linalg.inv(v) @ w.T

    schur, _, _, singular = equations.schur_complement()
    assert not singular.any()
    assert np.allclose(schur, expected, rtol=1e-6, atol=1e-6)


def test_y_factor_is_the_pose_block_inverse_times_the_cross_block(good_capture):
    camera = good_capture.camera
    block = camera.free_parameters()
    equations, _ = assemble(
        camera, list(good_capture.poses), good_capture.observations, block
    )
    _, pose_inverse, y, _ = equations.schur_complement()
    for index in range(equations.n_views):
        assert np.allclose(y[index], pose_inverse[index] @ equations.w[index].T)


def test_rms_matches_a_hand_computation(good_capture):
    camera = good_capture.camera
    equations, per_view = assemble(
        camera, list(good_capture.poses), good_capture.observations,
        camera.free_parameters(),
    )
    stacked = np.concatenate(per_view)
    expected = np.sqrt(np.mean(np.sum(stacked ** 2, axis=1)))
    assert equations.rms == pytest.approx(expected)


def test_parameter_and_dof_counts(good_capture):
    camera, observations = good_capture.camera, good_capture.observations
    block = camera.free_parameters()
    equations, _ = assemble(camera, list(good_capture.poses), observations, block)
    assert equations.n_intrinsic == block.n_free
    assert equations.n_parameters == block.n_free + 6 * observations.n_views
    assert equations.degrees_of_freedom == equations.n_residuals - equations.n_parameters
    names = equations.parameter_names()
    assert len(names) == equations.n_parameters
    assert names[: block.n_free] == block.free_names()
    assert names[block.n_free] == "view0.rx"


def test_gradient_is_near_zero_at_the_truth(good_capture):
    """Truth is not the least-squares optimum of a noisy sample, but it is close."""
    camera = good_capture.camera
    equations, _ = assemble(
        camera, list(good_capture.poses), good_capture.observations,
        camera.free_parameters(),
    )
    assert equations.gradient_norm / equations.n_residuals < 1.0


def test_fixing_parameters_shrinks_the_system(good_capture):
    camera = good_capture.camera
    full, _ = assemble(
        camera, list(good_capture.poses), good_capture.observations,
        camera.free_parameters(),
    )
    reduced, _ = assemble(
        camera, list(good_capture.poses), good_capture.observations,
        camera.free_parameters(fixed=("cx", "cy", "k3")),
    )
    assert reduced.n_intrinsic == full.n_intrinsic - 3
    assert reduced.cost == pytest.approx(full.cost)


def test_tied_aspect_ratio_merges_two_columns(good_capture):
    camera = good_capture.camera
    tied, _ = assemble(
        camera, list(good_capture.poses), good_capture.observations,
        camera.free_parameters(tie_aspect=True),
    )
    assert "fy" not in tied.intrinsic_names
    assert tied.n_intrinsic == len(camera.parameter_names()) - 1


def test_weights_scale_the_contribution(good_capture):
    camera, observations = good_capture.camera, good_capture.observations
    block = camera.free_parameters()
    plain, plain_residuals = assemble(
        camera, list(good_capture.poses), observations, block
    )
    quadrupled = [np.full(v.n_points, 4.0) for v in observations.views]
    weighted, weighted_residuals = assemble(
        camera, list(good_capture.poses), observations, block, weights=quadrupled
    )
    assert np.allclose(weighted.u, 4.0 * plain.u)
    assert np.allclose(weighted.v, 4.0 * plain.v)
    assert weighted.cost == pytest.approx(4.0 * plain.cost)
    # Reported residuals stay in pixels regardless of weighting, so that the
    # residual statistics read in units an engineer recognises.
    for plain_view, weighted_view in zip(plain_residuals, weighted_residuals):
        assert np.allclose(plain_view, weighted_view)


def test_zero_weight_removes_a_view(good_capture):
    camera, observations = good_capture.camera, good_capture.observations
    block = camera.free_parameters()
    weights = [np.ones(v.n_points) for v in observations.views]
    weights[0] = np.zeros(observations.views[0].n_points)
    equations, _ = assemble(
        camera, list(good_capture.poses), observations, block, weights=weights
    )
    assert np.allclose(equations.v[0], 0.0)
    _, _, _, singular = equations.schur_complement()
    assert singular[0]


def test_wrong_pose_count_is_rejected(good_capture):
    camera = good_capture.camera
    with pytest.raises(ValidationError, match="poses for"):
        assemble(
            camera, list(good_capture.poses)[:-1], good_capture.observations,
            camera.free_parameters(),
        )


def test_a_parameter_block_for_another_model_is_rejected(good_capture):
    camera = good_capture.camera
    other = PinholeBrownConrady(900, 905, 640, 360, np.zeros(8))
    with pytest.raises(ValidationError, match="parameter block covers"):
        assemble(
            camera, list(good_capture.poses), good_capture.observations,
            other.free_parameters(),
        )


def test_no_free_parameters_is_rejected(good_capture):
    camera = good_capture.camera
    block = camera.free_parameters(fixed=camera.parameter_names())
    with pytest.raises(ValidationError, match="no intrinsic parameter is free"):
        assemble(camera, list(good_capture.poses), good_capture.observations, block)


def test_wrong_weight_length_is_rejected(good_capture):
    camera, observations = good_capture.camera, good_capture.observations
    weights = [np.ones(3) for _ in observations.views]
    with pytest.raises(ValidationError, match="weights for"):
        assemble(
            camera, list(good_capture.poses), observations,
            camera.free_parameters(), weights=weights,
        )


def test_negative_weights_are_rejected(good_capture):
    camera, observations = good_capture.camera, good_capture.observations
    weights = [np.full(v.n_points, -1.0) for v in observations.views]
    with pytest.raises(ValidationError, match="non-negative"):
        assemble(
            camera, list(good_capture.poses), observations,
            camera.free_parameters(), weights=weights,
        )


def test_a_pose_behind_the_camera_is_reported_as_a_refit_error(good_capture):
    from calibsense.core.poses import Pose

    camera = good_capture.camera
    poses = list(good_capture.poses)
    poses[0] = Pose(np.eye(3), [0.0, 0.0, 0.0])

    class Exploding:
        def project_with_jacobian(self, *args, **kwargs):
            raise RuntimeError("singular projection")

    with pytest.raises(RefitError, match="could not be projected"):
        assemble(
            camera, poses, good_capture.observations, camera.free_parameters(),
            projector=Exploding(),
        )
