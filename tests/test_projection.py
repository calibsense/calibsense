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

"""Projection, differentiation and pose solving."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from caltrust.core.camera import FisheyeKannalaBrandt, PinholeBrownConrady
from caltrust.core.poses import Pose
from caltrust.errors import RefitError, ValidationError
from caltrust.refit.projection import (
    FisheyeProjector,
    PinholeProjector,
    numerical_jacobian,
    projector_for,
)

POSES = [
    Pose.from_rvec_tvec([0.2, -0.3, 0.15], [15.0, -25.0, 700.0]),
    Pose.from_rvec_tvec([0.9, 0.4, -0.6], [60.0, 40.0, 350.0]),
    Pose.from_rvec_tvec([0.0, 0.0, 0.0], [0.0, 0.0, 900.0]),
]

CAMERAS = [
    PinholeBrownConrady(900, 905, 639.5, 359.5, [-0.21, 0.06, 0.001, -0.002]),
    PinholeBrownConrady(900, 905, 639.5, 359.5, [-0.21, 0.06, 0.001, -0.002, 0.01]),
    PinholeBrownConrady(900, 905, 639.5, 359.5,
                        [-0.2, 0.05, 1e-3, -1e-3, 0.01, 1e-3, 1e-4, 1e-5]),
    PinholeBrownConrady(900, 905, 639.5, 359.5,
                        np.concatenate([[-0.2, 0.05, 1e-3, -1e-3, 0.01, 1e-3, 1e-4, 1e-5],
                                        [1e-4, 1e-4, 1e-4, 1e-4, 0.01, -0.01]])),
    FisheyeKannalaBrandt(430, 432, 639.5, 359.5, [-0.05, 0.01, -0.002, 4e-4]),
    FisheyeKannalaBrandt(430, 432, 639.5, 359.5, [-0.05, 0.01, -0.002, 4e-4], alpha=0.002),
]


@pytest.fixture
def planar_points():
    rng = np.random.default_rng(3)
    return np.column_stack([
        rng.uniform(-80, 80, 60), rng.uniform(-60, 60, 60), np.zeros(60)
    ])


def ids(cameras):
    return [f"{c.kind[:7]}-{c.distortion.size}-a{c.skew:g}" for c in cameras]


@pytest.mark.parametrize("camera", CAMERAS, ids=ids(CAMERAS))
def test_projector_dispatch_matches_the_model(camera):
    projector = projector_for(camera)
    expected = FisheyeProjector if camera.kind.startswith("fisheye") else PinholeProjector
    assert isinstance(projector, expected)


def test_unknown_model_has_no_projector():
    class Odd(PinholeBrownConrady):
        kind = "orthographic"

    with pytest.raises(ValidationError, match="no projector"):
        projector_for(Odd(900, 900, 640, 360, np.zeros(5)))


@pytest.mark.parametrize("camera", CAMERAS, ids=ids(CAMERAS))
@pytest.mark.parametrize("pose", POSES, ids=["tilted", "close", "frontoparallel"])
def test_analytic_jacobians_match_central_differences(camera, pose, planar_points):
    projector = projector_for(camera)
    _, analytic_intrinsic, analytic_pose = projector.project_with_jacobian(
        camera, pose, planar_points
    )
    numeric_intrinsic, numeric_pose = numerical_jacobian(
        projector, camera, pose, planar_points
    )
    for analytic, numeric, label in (
        (analytic_intrinsic, numeric_intrinsic, "intrinsic"),
        (analytic_pose, numeric_pose, "pose"),
    ):
        scale = max(float(np.abs(numeric).max()), 1e-12)
        error = float(np.abs(analytic - numeric).max()) / scale
        assert error < 1e-6, f"{label} Jacobian off by {error:.2e}"


@pytest.mark.parametrize("camera", CAMERAS, ids=ids(CAMERAS))
def test_jacobian_shapes_follow_the_canonical_parameter_order(camera, planar_points):
    projector = projector_for(camera)
    _, d_intrinsic, d_pose = projector.project_with_jacobian(
        camera, POSES[0], planar_points
    )
    assert d_intrinsic.shape == (2 * len(planar_points), len(camera.parameter_names()))
    assert d_pose.shape == (2 * len(planar_points), 6)


def test_pinhole_jacobian_columns_line_up_with_the_names(planar_points):
    """A step in fx must move x and leave y alone; cy must move only y."""
    camera = CAMERAS[1]
    projector = projector_for(camera)
    _, d_intrinsic, _ = projector.project_with_jacobian(camera, POSES[0], planar_points)
    names = camera.parameter_names()
    d_fx = d_intrinsic[:, names.index("fx")]
    d_cy = d_intrinsic[:, names.index("cy")]
    assert np.allclose(d_fx[1::2], 0.0)
    assert np.allclose(d_cy[0::2], 0.0)
    assert np.allclose(d_cy[1::2], 1.0)


def test_fisheye_skew_column_is_last(planar_points):
    camera = CAMERAS[5]
    _, d_intrinsic, _ = projector_for(camera).project_with_jacobian(
        camera, POSES[0], planar_points
    )
    # d/dalpha of the projection is fx * y_distorted, which is non-zero in x
    # and identically zero in y.
    d_alpha = d_intrinsic[:, camera.parameter_names().index("alpha")]
    assert np.abs(d_alpha[0::2]).max() > 1.0
    assert np.allclose(d_alpha[1::2], 0.0)


@pytest.mark.parametrize("camera", CAMERAS, ids=ids(CAMERAS))
@pytest.mark.parametrize("pose", POSES, ids=["tilted", "close", "frontoparallel"])
def test_pose_recovery_is_exact_without_noise(camera, pose, planar_points):
    projector = projector_for(camera)
    image_points = projector.project(camera, pose, planar_points)
    solved = projector.solve_pose(camera, planar_points, image_points)
    assert np.degrees(solved.angle_to(pose)) < 1e-8
    assert np.linalg.norm(solved.translation - pose.translation) < 1e-6


@pytest.mark.parametrize("camera", CAMERAS, ids=ids(CAMERAS))
def test_pose_recovery_degrades_gracefully_with_noise(camera, planar_points):
    projector = projector_for(camera)
    pose = POSES[0]
    rng = np.random.default_rng(0)
    noisy = projector.project(camera, pose, planar_points) + rng.normal(
        0.0, 0.3, (len(planar_points), 2)
    )
    solved = projector.solve_pose(camera, planar_points, noisy)
    assert np.degrees(solved.angle_to(pose)) < 1.0
    assert np.linalg.norm(solved.translation - pose.translation) < 5.0


def test_refine_pose_does_not_move_an_already_optimal_pose(planar_points):
    camera = CAMERAS[1]
    projector = projector_for(camera)
    pose = POSES[0]
    image_points = projector.project(camera, pose, planar_points)
    refined = projector.refine_pose(camera, pose, planar_points, image_points)
    assert refined.allclose(pose, atol=1e-6)


def test_refine_pose_recovers_from_a_perturbed_start(planar_points):
    camera = CAMERAS[1]
    projector = projector_for(camera)
    pose = POSES[0]
    image_points = projector.project(camera, pose, planar_points)
    start = Pose.from_parameter_vector(
        pose.parameter_vector() + np.array([0.05, -0.05, 0.03, 5.0, -5.0, 20.0])
    )
    refined = projector.refine_pose(camera, start, planar_points, image_points)
    assert np.degrees(refined.angle_to(pose)) < 1e-6


def test_refine_pose_never_makes_the_cost_worse(planar_points):
    camera = CAMERAS[1]
    projector = projector_for(camera)
    pose = POSES[1]
    rng = np.random.default_rng(1)
    observed = projector.project(camera, pose, planar_points) + rng.normal(
        0.0, 2.0, (len(planar_points), 2)
    )
    start = Pose.from_parameter_vector(
        pose.parameter_vector() + np.array([0.2, 0.2, 0.2, 30.0, 30.0, 80.0])
    )
    before = np.sum(projector.residuals(camera, start, planar_points, observed) ** 2)
    refined = projector.refine_pose(camera, start, planar_points, observed)
    after = np.sum(projector.residuals(camera, refined, planar_points, observed) ** 2)
    assert after <= before


def test_too_few_points_for_a_pose_is_refused(planar_points):
    camera = CAMERAS[1]
    projector = projector_for(camera)
    with pytest.raises(RefitError, match="at least 4 points"):
        projector.solve_pose(camera, planar_points[:3], np.zeros((3, 2)))


@pytest.mark.parametrize("bad", [np.zeros((0, 3)), np.zeros((4, 2))])
def test_bad_object_point_shapes_are_rejected(bad):
    camera = CAMERAS[1]
    with pytest.raises(ValidationError):
        projector_for(camera).project(camera, POSES[0], bad)


def test_residuals_are_interleaved_x_then_y(planar_points):
    camera = CAMERAS[1]
    projector = projector_for(camera)
    predicted = projector.project(camera, POSES[0], planar_points)
    observed = predicted + np.column_stack([
        np.full(len(planar_points), 1.0), np.full(len(planar_points), -2.0)
    ])
    residual = projector.residuals(camera, POSES[0], planar_points, observed)
    assert residual.shape == (2 * len(planar_points),)
    assert np.allclose(residual[0::2], -1.0)
    assert np.allclose(residual[1::2], 2.0)


def test_pinhole_projection_matches_opencv_directly(planar_points):
    camera = CAMERAS[1]
    reference, _ = cv2.projectPoints(
        planar_points, POSES[0].rvec, POSES[0].translation,
        camera.camera_matrix, camera.distortion,
    )
    ours = projector_for(camera).project(camera, POSES[0], planar_points)
    assert np.allclose(ours, reference.reshape(-1, 2))


def test_fisheye_projection_matches_opencv_directly(planar_points):
    camera = CAMERAS[4]
    reference, _ = cv2.fisheye.projectPoints(
        planar_points.reshape(-1, 1, 3), POSES[0].rvec.reshape(3, 1),
        POSES[0].translation.reshape(3, 1), camera.camera_matrix, camera.distortion,
        alpha=camera.alpha,
    )
    ours = projector_for(camera).project(camera, POSES[0], planar_points)
    assert np.allclose(ours, reference.reshape(-1, 2))


def test_a_short_pinhole_jacobian_is_reported_not_sliced(monkeypatch, planar_points):
    """The column layout is an OpenCV contract; a change must not pass silently."""
    camera = CAMERAS[1]
    truncated = np.zeros((2 * len(planar_points), 8))
    monkeypatch.setattr(
        cv2, "projectPoints", lambda *a, **k: (np.zeros((len(planar_points), 1, 2)), truncated)
    )
    with pytest.raises(RefitError, match="column Jacobian"):
        projector_for(camera).project_with_jacobian(camera, POSES[0], planar_points)


def test_a_wrong_width_fisheye_jacobian_is_reported(monkeypatch, planar_points):
    camera = CAMERAS[4]
    wrong = np.zeros((2 * len(planar_points), 14))
    monkeypatch.setattr(
        cv2.fisheye, "projectPoints",
        lambda *a, **k: (np.zeros((len(planar_points), 1, 2)), wrong),
    )
    with pytest.raises(RefitError, match="fisheye Jacobian"):
        projector_for(camera).project_with_jacobian(camera, POSES[0], planar_points)


def test_a_pinhole_pnp_failure_is_reported(monkeypatch, planar_points):
    camera = CAMERAS[1]
    monkeypatch.setattr(cv2, "solvePnP", lambda *a, **k: (False, np.zeros(3), np.zeros(3)))
    with pytest.raises(RefitError, match="did not converge"):
        projector_for(camera).solve_pose(camera, planar_points, np.zeros((60, 2)))


def test_a_fisheye_pnp_failure_is_reported(monkeypatch, planar_points):
    camera = CAMERAS[4]
    monkeypatch.setattr(cv2, "solvePnP", lambda *a, **k: (False, np.zeros(3), np.zeros(3)))
    with pytest.raises(RefitError, match="undistorted rays"):
        projector_for(camera).solve_pose(camera, planar_points, np.zeros((60, 2)))


def test_refine_pose_survives_a_singular_hessian(monkeypatch, planar_points):
    """A flat direction must damp rather than raise out of the refinement."""
    camera = CAMERAS[1]
    projector = projector_for(camera)
    pose = POSES[0]
    observed = projector.project(camera, pose, planar_points)
    real_solve = np.linalg.solve
    calls = {"n": 0}

    def sometimes_singular(matrix, vector):
        calls["n"] += 1
        if calls["n"] <= 2:
            raise np.linalg.LinAlgError("singular")
        return real_solve(matrix, vector)

    monkeypatch.setattr(np.linalg, "solve", sometimes_singular)
    refined = projector.refine_pose(camera, pose, planar_points, observed)
    assert refined.allclose(pose, atol=1e-4)
