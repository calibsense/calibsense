"""Rigid transforms, checked against OpenCV's Rodrigues."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from caltrust.core.poses import (
    Pose,
    matrix_to_rotvec,
    poses_from_array,
    poses_to_array,
    quaternion_to_matrix,
    rotvec_to_matrix,
    validate_rotation,
)
from caltrust.errors import ValidationError


def random_rotvecs(count, seed=0, max_angle=np.pi):
    rng = np.random.default_rng(seed)
    axes = rng.normal(size=(count, 3))
    axes /= np.linalg.norm(axes, axis=1, keepdims=True)
    return axes * rng.uniform(0, max_angle, (count, 1))


def test_rotvec_to_matrix_agrees_with_opencv():
    worst = 0.0
    for rotvec in random_rotvecs(500, seed=1):
        reference, _ = cv2.Rodrigues(rotvec)
        worst = max(worst, float(np.abs(rotvec_to_matrix(rotvec) - reference).max()))
    assert worst < 1e-12


def test_matrix_to_rotvec_agrees_with_opencv():
    worst = 0.0
    for rotvec in random_rotvecs(500, seed=2):
        matrix = rotvec_to_matrix(rotvec)
        reference, _ = cv2.Rodrigues(matrix)
        worst = max(worst, float(np.abs(matrix_to_rotvec(matrix) - reference.ravel()).max()))
    assert worst < 1e-8


def test_rotvec_round_trip():
    for rotvec in random_rotvecs(500, seed=3):
        assert np.allclose(matrix_to_rotvec(rotvec_to_matrix(rotvec)), rotvec, atol=1e-8)


@pytest.mark.parametrize(
    "rotvec",
    [
        [0.0, 0.0, 0.0],
        [1e-14, 0.0, 0.0],
        [np.pi, 0.0, 0.0],
        [0.0, np.pi, 0.0],
        [0.0, 0.0, np.pi],
        list(np.array([1, 1, 1]) / np.sqrt(3) * np.pi),
        list(np.array([1, -2, 3]) / np.sqrt(14) * (np.pi - 1e-9)),
    ],
)
def test_singular_rotations_round_trip(rotvec):
    matrix = rotvec_to_matrix(rotvec)
    recovered = matrix_to_rotvec(matrix)
    assert np.allclose(rotvec_to_matrix(recovered), matrix, atol=1e-8)


def test_identity_maps_to_zero():
    assert np.allclose(matrix_to_rotvec(np.eye(3)), np.zeros(3))


def test_reflection_is_not_a_rotation():
    with pytest.raises(ValidationError, match="determinant"):
        validate_rotation(np.diag([1.0, 1.0, -1.0]))


def test_non_orthonormal_is_rejected():
    with pytest.raises(ValidationError, match="orthonormal"):
        validate_rotation(np.eye(3) * 1.1)


@pytest.mark.parametrize("bad", [np.zeros((2, 2)), np.zeros((3, 4))])
def test_wrong_shaped_rotation_is_rejected(bad):
    with pytest.raises(ValidationError, match="3x3"):
        validate_rotation(bad)


def test_non_finite_rotation_is_rejected():
    matrix = np.eye(3)
    matrix[0, 0] = np.nan
    with pytest.raises(ValidationError, match="non-finite"):
        validate_rotation(matrix)


def test_pose_inverse_composes_to_identity():
    pose = Pose.from_rvec_tvec([0.3, -0.4, 0.5], [10.0, -20.0, 700.0])
    assert pose.compose(pose.inverse()).allclose(Pose.identity(), atol=1e-12)
    assert pose.inverse().compose(pose).allclose(Pose.identity(), atol=1e-12)


def test_compose_matches_matrix_multiplication():
    a = Pose.from_rvec_tvec([0.1, 0.2, 0.3], [1.0, 2.0, 3.0])
    b = Pose.from_rvec_tvec([-0.4, 0.1, 0.2], [4.0, 5.0, 6.0])
    assert np.allclose(a.compose(b).matrix, a.matrix @ b.matrix)


def test_apply_matches_the_matrix_form():
    pose = Pose.from_rvec_tvec([0.2, 0.1, -0.3], [5.0, 6.0, 700.0])
    points = np.array([[1.0, 2.0, 3.0], [-4.0, 5.0, -6.0]])
    expected = (pose.matrix @ np.column_stack([points, np.ones(2)]).T).T[:, :3]
    assert np.allclose(pose.apply(points), expected)


def test_apply_rejects_wrong_shaped_points():
    with pytest.raises(ValidationError, match=r"\(n, 3\)"):
        Pose.identity().apply(np.zeros((4, 2)))


def test_matrix_round_trip():
    pose = Pose.from_rvec_tvec([0.3, -0.4, 0.5], [10.0, -20.0, 700.0])
    assert Pose.from_matrix(pose.matrix).allclose(pose, atol=1e-12)


def test_parameter_vector_round_trip():
    pose = Pose.from_rvec_tvec([0.3, -0.4, 0.5], [10.0, -20.0, 700.0])
    assert Pose.from_parameter_vector(pose.parameter_vector()).allclose(pose, atol=1e-9)


def test_parameter_vector_must_be_six_long():
    with pytest.raises(ValidationError, match="6 values"):
        Pose.from_parameter_vector([1, 2, 3])


def test_from_matrix_checks_the_bottom_row():
    matrix = np.eye(4)
    matrix[3, 0] = 0.5
    with pytest.raises(ValidationError, match=r"\[0,0,0,1\]"):
        Pose.from_matrix(matrix)


def test_from_matrix_checks_the_shape():
    with pytest.raises(ValidationError, match="4x4"):
        Pose.from_matrix(np.eye(3))


def test_translation_must_have_three_finite_components():
    with pytest.raises(ValidationError, match="3 components"):
        Pose(np.eye(3), [1.0, 2.0])
    with pytest.raises(ValidationError, match="non-finite"):
        Pose(np.eye(3), [1.0, 2.0, np.nan])


def test_distance_and_angle():
    pose = Pose.from_rvec_tvec([0.0, 0.0, 0.0], [3.0, 4.0, 0.0])
    assert pose.distance_mm == pytest.approx(5.0)
    rotated = Pose.from_rvec_tvec([0.0, 0.0, np.pi / 3], [0.0, 0.0, 0.0])
    assert Pose.identity().angle_to(rotated) == pytest.approx(np.pi / 3)


def test_quaternion_matches_rotvec_for_a_known_rotation():
    angle = 0.7
    quaternion = [np.sin(angle / 2), 0.0, 0.0, np.cos(angle / 2)]
    assert np.allclose(quaternion_to_matrix(quaternion), rotvec_to_matrix([angle, 0, 0]))


def test_quaternion_scalar_first_option():
    angle = 0.7
    scalar_first = [np.cos(angle / 2), 0.0, np.sin(angle / 2), 0.0]
    assert np.allclose(
        quaternion_to_matrix(scalar_first, scalar_first=True),
        rotvec_to_matrix([0, angle, 0]),
    )


def test_quaternion_is_normalised_on_the_way_in():
    quaternion = np.array([0.1, 0.2, 0.3, 0.9]) * 7.0
    assert validate_rotation(quaternion_to_matrix(quaternion)) is not None


@pytest.mark.parametrize("bad", [[0, 0, 0, 0], [1, 2, 3], [1, 2, 3, 4, 5]])
def test_invalid_quaternions_are_rejected(bad):
    with pytest.raises(ValidationError):
        quaternion_to_matrix(bad)


def test_pose_array_round_trip():
    poses = [Pose.from_rvec_tvec([0.1 * i, 0.0, 0.0], [i, 0.0, 500.0]) for i in range(4)]
    recovered = poses_from_array(poses_to_array(poses))
    assert all(a.allclose(b, atol=1e-12) for a, b in zip(poses, recovered))


def test_empty_pose_array_round_trip():
    assert poses_to_array([]).shape == (0, 4, 4)
    assert poses_from_array(np.zeros((0, 4, 4))) == ()


def test_poses_from_array_checks_the_shape():
    with pytest.raises(ValidationError, match=r"\(n, 4, 4\)"):
        poses_from_array(np.ones((3, 3, 3)))


def test_repr_shows_rvec_and_millimetres():
    text = repr(Pose.from_rvec_tvec([0.1, 0.2, 0.3], [10.0, 20.0, 600.0]))
    assert "rvec=" in text and "tvec_mm=" in text
