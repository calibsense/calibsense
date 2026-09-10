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

"""Rigid transforms.

A pose here is always "target points expressed in camera coordinates", the same
convention OpenCV's `rvec`/`tvec` pair uses, and translations are always in
millimetres to match `TargetSpec.object_points`.

Rodrigues conversion is implemented in NumPy rather than delegated to OpenCV so
that `calibsense.core` stays importable without OpenCV; the test suite checks both
directions against `cv2.Rodrigues`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, Tuple

import numpy as np

from ..errors import ValidationError

_SMALL_ANGLE = 1e-8


def rotvec_to_matrix(rotvec: Sequence[float]) -> np.ndarray:
    """Convert a rotation vector to a rotation matrix.

    Args:
        rotvec: Three components whose direction is the axis and whose norm is
            the angle in radians.

    Returns:
        A `(3, 3)` rotation matrix.
    """
    r = np.asarray(rotvec, dtype=float).reshape(3)
    theta = float(np.linalg.norm(r))
    skew = np.array([[0.0, -r[2], r[1]], [r[2], 0.0, -r[0]], [-r[1], r[0], 0.0]])
    if theta < _SMALL_ANGLE:
        # Second-order series; exact enough that a round trip is bit-stable.
        return np.eye(3) + skew + 0.5 * (skew @ skew)
    axis_skew = skew / theta
    return (
        np.eye(3)
        + np.sin(theta) * axis_skew
        + (1.0 - np.cos(theta)) * (axis_skew @ axis_skew)
    )


def matrix_to_rotvec(matrix: np.ndarray) -> np.ndarray:
    """Convert a rotation matrix to a rotation vector.

    Args:
        matrix: A `(3, 3)` rotation matrix.

    Returns:
        The equivalent rotation vector, with angle in `[0, pi]`.

    Raises:
        ValidationError: The matrix is not orthonormal with determinant one.
    """
    R = validate_rotation(matrix)
    cos_theta = np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)
    theta = float(np.arccos(cos_theta))
    antisymmetric = np.array(
        [R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]
    )
    if theta < _SMALL_ANGLE:
        return 0.5 * antisymmetric
    if np.pi - theta < 1e-5:
        # sin(theta) vanishes, so the axis magnitudes come from the symmetric
        # part, which near pi is R ~ 2nn' - I.
        diagonal = np.clip((np.diag(R) + 1.0) / 2.0, 0.0, None)
        axis = np.sqrt(diagonal)
        largest = int(np.argmax(axis))
        for other in range(3):
            if other != largest:
                axis[other] = np.copysign(axis[other], R[largest, other])
        # That fixes the axis only up to a global sign. The antisymmetric part
        # is vanishingly small here but still carries the sign, and at exactly
        # pi it is zero and either sign is the same rotation.
        if axis @ antisymmetric < 0:
            axis = -axis
        return axis / np.linalg.norm(axis) * theta
    return antisymmetric * (theta / (2.0 * np.sin(theta)))


def validate_rotation(matrix: np.ndarray, tolerance: float = 1e-6) -> np.ndarray:
    """Check that a matrix is a proper rotation.

    Args:
        matrix: Candidate `(3, 3)` matrix.
        tolerance: Allowed deviation from orthonormality and from a determinant
            of one.

    Returns:
        The matrix as a float array.

    Raises:
        ValidationError: The matrix is the wrong shape, not orthonormal, or a
            reflection.
    """
    R = np.asarray(matrix, dtype=float)
    if R.shape != (3, 3):
        raise ValidationError(f"rotation must be 3x3, got shape {R.shape}")
    if not np.all(np.isfinite(R)):
        raise ValidationError("rotation contains non-finite entries")
    orthonormality = float(np.abs(R.T @ R - np.eye(3)).max())
    if orthonormality > tolerance:
        raise ValidationError(
            f"rotation is not orthonormal: max |R'R - I| = {orthonormality:.3e}"
        )
    determinant = float(np.linalg.det(R))
    if abs(determinant - 1.0) > tolerance:
        raise ValidationError(
            f"rotation has determinant {determinant:.6f}; a reflection is not a pose"
        )
    return R


@dataclass(frozen=True, eq=False)
class Pose:
    """A rigid transform, rotation then translation.

    A point `p` in the source frame maps to `rotation @ p + translation`.

    Attributes:
        rotation: A `(3, 3)` rotation matrix.
        translation: A `(3,)` translation in millimetres.
    """

    rotation: np.ndarray
    translation: np.ndarray

    def __post_init__(self) -> None:
        object.__setattr__(self, "rotation", validate_rotation(self.rotation))
        translation = np.asarray(self.translation, dtype=float).reshape(-1)
        if translation.size != 3:
            raise ValidationError(
                f"translation must have 3 components, got {translation.size}"
            )
        if not np.all(np.isfinite(translation)):
            raise ValidationError("translation contains non-finite entries")
        object.__setattr__(self, "translation", translation)

    @classmethod
    def from_rvec_tvec(cls, rvec: Sequence[float], tvec: Sequence[float]) -> "Pose":
        """Build a pose from OpenCV's rotation-vector and translation pair.

        Args:
            rvec: Rotation vector, three components.
            tvec: Translation, three components in millimetres.

        Returns:
            The corresponding pose.
        """
        return cls(rotvec_to_matrix(np.asarray(rvec).reshape(3)),
                   np.asarray(tvec, dtype=float).reshape(3))

    @classmethod
    def from_matrix(cls, matrix: np.ndarray) -> "Pose":
        """Build a pose from a 4x4 homogeneous transform.

        Args:
            matrix: A `(4, 4)` array whose last row is `[0, 0, 0, 1]`.

        Returns:
            The corresponding pose.

        Raises:
            ValidationError: The shape or the bottom row is wrong.
        """
        M = np.asarray(matrix, dtype=float)
        if M.shape != (4, 4):
            raise ValidationError(f"homogeneous transform must be 4x4, got {M.shape}")
        if not np.allclose(M[3], [0.0, 0.0, 0.0, 1.0], atol=1e-6):
            raise ValidationError(f"bottom row must be [0,0,0,1], got {M[3].tolist()}")
        return cls(M[:3, :3], M[:3, 3])

    @classmethod
    def identity(cls) -> "Pose":
        """The identity pose."""
        return cls(np.eye(3), np.zeros(3))

    @property
    def rvec(self) -> np.ndarray:
        """The rotation as an OpenCV-style rotation vector."""
        return matrix_to_rotvec(self.rotation)

    @property
    def tvec(self) -> np.ndarray:
        """The translation, an alias that pairs with `rvec`."""
        return self.translation

    @property
    def matrix(self) -> np.ndarray:
        """The `(4, 4)` homogeneous form."""
        M = np.eye(4)
        M[:3, :3] = self.rotation
        M[:3, 3] = self.translation
        return M

    def parameter_vector(self) -> np.ndarray:
        """The six-parameter form `[rx, ry, rz, tx, ty, tz]` used in the fit."""
        return np.concatenate([self.rvec, self.translation])

    @classmethod
    def from_parameter_vector(cls, values: Sequence[float]) -> "Pose":
        """Rebuild from `[rx, ry, rz, tx, ty, tz]`.

        Args:
            values: Six numbers, rotation vector then translation.

        Returns:
            The corresponding pose.
        """
        v = np.asarray(values, dtype=float).reshape(-1)
        if v.size != 6:
            raise ValidationError(f"pose vector must hold 6 values, got {v.size}")
        return cls.from_rvec_tvec(v[:3], v[3:])

    def inverse(self) -> "Pose":
        """The inverse transform."""
        rotation = self.rotation.T
        return Pose(rotation, -rotation @ self.translation)

    def compose(self, other: "Pose") -> "Pose":
        """Apply `other` first, then this pose.

        Args:
            other: The transform applied first.

        Returns:
            The composed transform, equivalent to `self.matrix @ other.matrix`.
        """
        return Pose(
            self.rotation @ other.rotation,
            self.rotation @ other.translation + self.translation,
        )

    def apply(self, points: np.ndarray) -> np.ndarray:
        """Transform points.

        Args:
            points: An `(n, 3)` array of source-frame points in millimetres.

        Returns:
            An `(n, 3)` array in the destination frame.
        """
        p = np.asarray(points, dtype=float)
        if p.ndim != 2 or p.shape[1] != 3:
            raise ValidationError(f"points must be (n, 3), got shape {p.shape}")
        return p @ self.rotation.T + self.translation

    @property
    def distance_mm(self) -> float:
        """Distance from the origin to the transformed frame, in millimetres."""
        return float(np.linalg.norm(self.translation))

    def angle_to(self, other: "Pose") -> float:
        """Rotation angle between two poses, in radians.

        Args:
            other: The pose to compare against.

        Returns:
            The geodesic angle between the two rotations.
        """
        return float(np.linalg.norm(matrix_to_rotvec(self.rotation.T @ other.rotation)))

    def allclose(self, other: "Pose", atol: float = 1e-9) -> bool:
        """Compare two poses entrywise within a tolerance.

        Args:
            other: The pose to compare against.
            atol: Absolute tolerance.

        Returns:
            `True` when both rotation and translation match.
        """
        return bool(
            np.allclose(self.rotation, other.rotation, atol=atol)
            and np.allclose(self.translation, other.translation, atol=atol)
        )

    def __repr__(self) -> str:
        rvec = np.round(self.rvec, 5)
        tvec = np.round(self.translation, 3)
        return f"Pose(rvec={rvec.tolist()}, tvec_mm={tvec.tolist()})"


def poses_to_array(poses: Sequence[Pose]) -> np.ndarray:
    """Stack poses into an `(n, 4, 4)` array for storage.

    Args:
        poses: The poses to stack.

    Returns:
        Their homogeneous matrices.
    """
    if not poses:
        return np.zeros((0, 4, 4))
    return np.stack([p.matrix for p in poses])


def poses_from_array(array: np.ndarray) -> Tuple[Pose, ...]:
    """Rebuild poses from an `(n, 4, 4)` array.

    Args:
        array: Homogeneous matrices, as produced by `poses_to_array`.

    Returns:
        The reconstructed poses.
    """
    stack = np.asarray(array, dtype=float)
    if stack.size == 0:
        return ()
    if stack.ndim != 3 or stack.shape[1:] != (4, 4):
        raise ValidationError(f"expected an (n, 4, 4) array, got shape {stack.shape}")
    return tuple(Pose.from_matrix(m) for m in stack)


def quaternion_to_matrix(quaternion: Sequence[float], scalar_first: bool = False) -> np.ndarray:
    """Convert a unit quaternion to a rotation matrix.

    Args:
        quaternion: Four components. The default order is `(x, y, z, w)`, which
            is what ROS, Eigen's coefficient order and most robot loggers emit.
        scalar_first: Interpret the input as `(w, x, y, z)` instead.

    Returns:
        A `(3, 3)` rotation matrix.

    Raises:
        ValidationError: The quaternion has the wrong length or zero norm.
    """
    q = np.asarray(quaternion, dtype=float).reshape(-1)
    if q.size != 4:
        raise ValidationError(f"quaternion must have 4 components, got {q.size}")
    norm = float(np.linalg.norm(q))
    if not np.isfinite(norm) or norm < 1e-12:
        raise ValidationError(f"quaternion has norm {norm}, which is not a rotation")
    q = q / norm
    w, x, y, z = (q[0], q[1], q[2], q[3]) if scalar_first else (q[3], q[0], q[1], q[2])
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])
