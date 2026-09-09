"""Projection, differentiation and pose solving, one implementation per model.

OpenCV already computes analytic derivatives inside `projectPoints`; it just
throws them away at the calibration level. This module keeps them, and reorders
the columns into caltrust's canonical parameter order so that a covariance block
means the same thing whichever camera model produced it.

The column layouts OpenCV actually returns, both verified against central
differences in the test suite:

* `cv2.projectPoints` — `[rvec(3), tvec(3), fx, fy, cx, cy, distortion(n)]`
* `cv2.fisheye.projectPoints` — `[fx, fy, cx, cy, k1..k4, om(3), T(3), alpha]`
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import ClassVar, Optional, Tuple, Type

import cv2
import numpy as np

from ..core.camera import CameraModel, FisheyeKannalaBrandt, PinholeBrownConrady
from ..core.poses import Pose
from ..errors import RefitError, ValidationError

#: Fewest points that can determine a planar pose.
MIN_POINTS_FOR_POSE = 4


class Projector(ABC):
    """Projects target points into a view and differentiates the result."""

    #: The camera model this projector handles.
    model: ClassVar[Type[CameraModel]] = CameraModel

    @abstractmethod
    def project(
        self, camera: CameraModel, pose: Pose, object_points: np.ndarray
    ) -> np.ndarray:
        """Project board-frame points into the image.

        Args:
            camera: The intrinsic model.
            pose: Board-to-camera transform, translation in millimetres.
            object_points: An `(n, 3)` array of board-frame points in millimetres.

        Returns:
            An `(n, 2)` array of pixel coordinates.
        """

    @abstractmethod
    def project_with_jacobian(
        self, camera: CameraModel, pose: Pose, object_points: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Project points and return both parameter Jacobians.

        Args:
            camera: The intrinsic model.
            pose: Board-to-camera transform.
            object_points: An `(n, 3)` array of board-frame points in millimetres.

        Returns:
            A triple `(points, d_intrinsic, d_pose)`. `points` is `(n, 2)`.
            `d_intrinsic` is `(2n, p)` in the order of
            `camera.parameter_names()`. `d_pose` is `(2n, 6)` in the order
            `[rx, ry, rz, tx, ty, tz]`. Both Jacobians use interleaved rows,
            `x` then `y` for each point.
        """

    @abstractmethod
    def solve_pose(
        self, camera: CameraModel, object_points: np.ndarray, image_points: np.ndarray
    ) -> Pose:
        """Estimate a single view's pose at fixed intrinsics.

        Args:
            camera: The intrinsic model, held fixed.
            object_points: An `(n, 3)` array of board-frame points in millimetres.
            image_points: The matching `(n, 2)` pixel observations.

        Returns:
            The board-to-camera pose.

        Raises:
            RefitError: The pose could not be solved.
        """

    def refine_pose(
        self,
        camera: CameraModel,
        pose: Pose,
        object_points: np.ndarray,
        image_points: np.ndarray,
        max_iterations: int = 30,
        tolerance: float = 1e-12,
    ) -> Pose:
        """Levenberg-Marquardt refinement of one pose at fixed intrinsics.

        Runs against the model's own reprojection objective, so it does not
        inherit the accuracy of any intermediate undistortion step.

        Args:
            camera: The intrinsic model, held fixed.
            pose: Starting pose.
            object_points: An `(n, 3)` array of board-frame points.
            image_points: The matching `(n, 2)` pixel observations.
            max_iterations: Iteration cap.
            tolerance: Stop when the relative cost improvement falls below this.

        Returns:
            The refined pose, or the input pose when no step improved on it.
        """
        points = _as_object_points(object_points)
        observed = np.asarray(image_points, dtype=float).reshape(-1, 2)
        current = pose
        cost = float(np.sum(self.residuals(camera, current, points, observed) ** 2))
        damping = 1e-6
        for _ in range(max_iterations):
            projected, _, d_pose = self.project_with_jacobian(camera, current, points)
            residual = (projected - observed).reshape(-1)
            hessian = d_pose.T @ d_pose
            gradient = d_pose.T @ residual
            improved = False
            for _ in range(8):
                damped = hessian + damping * np.diag(np.maximum(np.diag(hessian), 1e-12))
                try:
                    step = np.linalg.solve(damped, -gradient)
                except np.linalg.LinAlgError:
                    damping *= 10.0
                    continue
                candidate = Pose.from_parameter_vector(
                    current.parameter_vector() + step
                )
                trial = float(
                    np.sum(self.residuals(camera, candidate, points, observed) ** 2)
                )
                if trial < cost:
                    improvement = (cost - trial) / max(cost, 1e-30)
                    current, cost = candidate, trial
                    damping = max(damping * 0.3, 1e-12)
                    improved = improvement > tolerance
                    break
                damping *= 10.0
            if not improved:
                break
        return current

    def residuals(
        self,
        camera: CameraModel,
        pose: Pose,
        object_points: np.ndarray,
        image_points: np.ndarray,
    ) -> np.ndarray:
        """Reprojection residuals for one view, flattened and interleaved.

        Args:
            camera: The intrinsic model.
            pose: Board-to-camera transform.
            object_points: An `(n, 3)` array of board-frame points.
            image_points: The matching `(n, 2)` pixel observations.

        Returns:
            A `(2n,)` array of `projected - observed`, ordered
            `[dx0, dy0, dx1, dy1, ...]`.
        """
        predicted = self.project(camera, pose, object_points)
        return (predicted - np.asarray(image_points, dtype=float)).reshape(-1)


def _as_object_points(object_points: np.ndarray) -> np.ndarray:
    points = np.asarray(object_points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValidationError(f"object points must be (n, 3), got shape {points.shape}")
    if points.shape[0] == 0:
        raise ValidationError("cannot project an empty point set")
    return np.ascontiguousarray(points)


class PinholeProjector(Projector):
    """Brown-Conrady projection through `cv2.projectPoints`."""

    model: ClassVar[Type[CameraModel]] = PinholeBrownConrady

    def project(self, camera, pose, object_points):
        """Project points with the pinhole model. See `Projector.project`."""
        points = _as_object_points(object_points)
        projected, _ = cv2.projectPoints(
            points, pose.rvec, pose.translation,
            camera.camera_matrix, camera.distortion,
        )
        return projected.reshape(-1, 2)

    def project_with_jacobian(self, camera, pose, object_points):
        """Project and differentiate. See `Projector.project_with_jacobian`."""
        points = _as_object_points(object_points)
        projected, jacobian = cv2.projectPoints(
            points, pose.rvec, pose.translation,
            camera.camera_matrix, camera.distortion,
        )
        n_intrinsic = 4 + camera.distortion.size
        expected = 6 + n_intrinsic
        if jacobian.shape[1] < expected:
            raise RefitError(
                f"OpenCV returned a {jacobian.shape[1]}-column Jacobian, expected "
                f"at least {expected}; this OpenCV build is not supported"
            )
        return (
            projected.reshape(-1, 2),
            np.ascontiguousarray(jacobian[:, 6 : 6 + n_intrinsic], dtype=float),
            np.ascontiguousarray(jacobian[:, 0:6], dtype=float),
        )

    def solve_pose(self, camera, object_points, image_points):
        """Solve a pose with `cv2.solvePnP`, refined by Levenberg-Marquardt."""
        points = _as_object_points(object_points)
        observed = np.ascontiguousarray(
            np.asarray(image_points, dtype=np.float64).reshape(-1, 1, 2)
        )
        if points.shape[0] < MIN_POINTS_FOR_POSE:
            raise RefitError(
                f"need at least {MIN_POINTS_FOR_POSE} points for a pose, "
                f"got {points.shape[0]}"
            )
        matrix, distortion = camera.camera_matrix, camera.distortion
        found, rvec, tvec = cv2.solvePnP(
            points.reshape(-1, 1, 3), observed, matrix, distortion,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
        if not found:
            raise RefitError("cv2.solvePnP did not converge")
        initial = Pose.from_rvec_tvec(rvec.reshape(3), tvec.reshape(3))
        return self.refine_pose(camera, initial, points, image_points)


class FisheyeProjector(Projector):
    """Kannala-Brandt projection through `cv2.fisheye.projectPoints`."""

    model: ClassVar[Type[CameraModel]] = FisheyeKannalaBrandt

    #: Canonical order is `[fx, fy, cx, cy, k1..k4, alpha]`; OpenCV puts alpha
    #: last of fifteen and the pose in the middle.
    _INTRINSIC_COLUMNS = np.array([0, 1, 2, 3, 4, 5, 6, 7, 14])
    _POSE_COLUMNS = np.array([8, 9, 10, 11, 12, 13])

    def project(self, camera, pose, object_points):
        """Project points with the fisheye model. See `Projector.project`."""
        points = _as_object_points(object_points).reshape(-1, 1, 3)
        projected, _ = cv2.fisheye.projectPoints(
            points, pose.rvec.reshape(3, 1), pose.translation.reshape(3, 1),
            camera.camera_matrix, camera.distortion, alpha=camera.alpha,
        )
        return projected.reshape(-1, 2)

    def project_with_jacobian(self, camera, pose, object_points):
        """Project and differentiate. See `Projector.project_with_jacobian`."""
        points = _as_object_points(object_points).reshape(-1, 1, 3)
        projected, jacobian = cv2.fisheye.projectPoints(
            points, pose.rvec.reshape(3, 1), pose.translation.reshape(3, 1),
            camera.camera_matrix, camera.distortion, alpha=camera.alpha,
        )
        if jacobian.shape[1] != 15:
            raise RefitError(
                f"OpenCV returned a {jacobian.shape[1]}-column fisheye Jacobian, "
                "expected 15; this OpenCV build is not supported"
            )
        return (
            projected.reshape(-1, 2),
            np.ascontiguousarray(jacobian[:, self._INTRINSIC_COLUMNS], dtype=float),
            np.ascontiguousarray(jacobian[:, self._POSE_COLUMNS], dtype=float),
        )

    def solve_pose(self, camera, object_points, image_points):
        """Solve a pose by undistorting to normalised rays, then `cv2.solvePnP`.

        A fisheye image cannot be handed to `solvePnP` directly, so the points
        are lifted to the normalised plane first and solved against an identity
        intrinsic matrix.
        """
        points = _as_object_points(object_points)
        if points.shape[0] < MIN_POINTS_FOR_POSE:
            raise RefitError(
                f"need at least {MIN_POINTS_FOR_POSE} points for a pose, "
                f"got {points.shape[0]}"
            )
        observed = np.asarray(image_points, dtype=np.float64).reshape(-1, 2)
        # cv2.fisheye.undistortPoints ignores K[0, 1], so the skew is removed
        # here rather than silently biasing the initial pose.
        deskewed = observed.copy()
        deskewed[:, 0] -= camera.alpha * camera.fx * (observed[:, 1] - camera.cy) / camera.fy
        skewless = camera.camera_matrix.copy()
        skewless[0, 1] = 0.0
        normalised = cv2.fisheye.undistortPoints(
            np.ascontiguousarray(deskewed.reshape(-1, 1, 2)), skewless, camera.distortion
        )
        found, rvec, tvec = cv2.solvePnP(
            points.reshape(-1, 1, 3), normalised, np.eye(3), np.zeros(4),
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
        if not found:
            raise RefitError("cv2.solvePnP did not converge on undistorted rays")
        initial = Pose.from_rvec_tvec(rvec.reshape(3), tvec.reshape(3))
        return self.refine_pose(camera, initial, points, observed)


_PROJECTORS = {
    PinholeBrownConrady.kind: PinholeProjector,
    FisheyeKannalaBrandt.kind: FisheyeProjector,
}


def projector_for(camera: CameraModel) -> Projector:
    """Build the projector that handles a camera model.

    Args:
        camera: The model to project with.

    Returns:
        A projector instance.

    Raises:
        ValidationError: No projector handles this model.
    """
    try:
        return _PROJECTORS[camera.kind]()
    except KeyError:
        raise ValidationError(
            f"no projector for camera model {camera.kind!r}; "
            f"have {sorted(_PROJECTORS)}"
        ) from None


def numerical_jacobian(
    projector: Projector,
    camera: CameraModel,
    pose: Pose,
    object_points: np.ndarray,
    step: float = 1e-6,
) -> Tuple[np.ndarray, np.ndarray]:
    """Central-difference Jacobians, for checking the analytic ones.

    This is a test and debugging aid, not part of the fit. It is exported
    because "our derivatives match finite differences" is a claim the package
    should let a sceptical user re-run themselves.

    Args:
        projector: The projector under test.
        camera: The intrinsic model to differentiate around.
        pose: The pose to differentiate around.
        object_points: An `(n, 3)` array of board-frame points.
        step: Relative step size; the absolute step never falls below `step`.

    Returns:
        A pair `(d_intrinsic, d_pose)` with the same shapes and column order as
        `Projector.project_with_jacobian`.
    """
    points = _as_object_points(object_points)
    intrinsic0 = camera.to_vector()
    pose0 = pose.parameter_vector()

    def evaluate(intrinsic_vector, pose_vector):
        model = type(camera).from_vector(intrinsic_vector)
        return projector.project(
            model, Pose.from_parameter_vector(pose_vector), points
        ).reshape(-1)

    def differentiate(base, apply):
        columns = []
        for index in range(base.size):
            delta = max(abs(base[index]), 1.0) * step
            forward, backward = base.copy(), base.copy()
            forward[index] += delta
            backward[index] -= delta
            columns.append((apply(forward) - apply(backward)) / (2.0 * delta))
        return np.stack(columns, axis=1)

    d_intrinsic = differentiate(intrinsic0, lambda v: evaluate(v, pose0))
    d_pose = differentiate(pose0, lambda v: evaluate(intrinsic0, v))
    return d_intrinsic, d_pose
