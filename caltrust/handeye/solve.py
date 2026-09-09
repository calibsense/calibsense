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

"""M6 — hand-eye calibration with a covariance.

OpenCV solves hand-eye and returns a transform with no uncertainty attached,
which for a cell that has to state its measurement accuracy is half an answer.
This module re-solves it as a least-squares problem so the covariance falls out
of the same normal equations that produced the estimate.

The formulation is the one that makes the constraint an equality rather than a
pairwise relation. In eye-in-hand, the target does not move in the cell, so

    Z = A_i X B_i    for every view i

with `A_i` the robot's flange pose, `X` the unknown flange-to-camera transform,
`B_i` the target's pose in the camera from the calibration, and `Z` the unknown
target pose in the cell. Eye-to-hand swaps which transform is constant:
`X = A_i Z B_i^-1`. Either way there are twelve unknowns and six residuals per
view, and the residual vanishes at the solution — which is what lets the
covariance be read off the linearisation.

Rotation and translation residuals carry different units, so both scales are
estimated from the residuals themselves and used as weights. Without that the
solve silently optimises whichever has the larger numerical magnitude.

The closed-form initialisation is implemented here rather than delegated.
`cv2.calibrateHandEye` exists in OpenCV 4 but is absent from OpenCV 5's Python
bindings — the `CALIB_HAND_EYE_*` flags are still exported, the function is not —
so depending on it would make this milestone fail on the newer library. The
method used is the standard one: relative motions satisfy `A_ij X = X B_ij`, the
rotation axes of `A_ij` and `B_ij` are related by `a = R_X b`, so `R_X` comes out
of a Procrustes fit on the axis pairs and `t_X` out of a linear least squares.
The test suite cross-checks it against OpenCV wherever OpenCV still has it.
"""

from __future__ import annotations

from dataclasses import replace
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np

from ..core.poses import Pose, matrix_to_rotvec, rotvec_to_matrix
from ..core.session import CalibrationSession
from ..errors import RefitError, ValidationError
from ..refit.linalg import DEFAULT_RCOND, invert_symmetric
from ..refit.result import InstrumentedFit
from .result import MOUNTINGS, HandEyeResult

#: Fewest views a hand-eye solve will accept. Twelve unknowns against six
#: residuals per view needs three views to be determined at all, and three is
#: never enough to be trustworthy.
MIN_VIEWS = 4

#: Gauss-Newton iteration cap.
MAX_ITERATIONS = 40

#: Relative cost improvement below which refinement stops.
TOLERANCE = 1e-12

#: Floors on the estimated residual scales, so a near-perfect synthetic fit
#: cannot produce an infinite weight.
MIN_ROTATION_SIGMA = 1e-9
MIN_TRANSLATION_SIGMA = 1e-6

#: Relative rotations smaller than this carry no information about the camera
#: rotation, and including them only adds noise to the closed-form fit.
MIN_PAIR_ROTATION_RAD = np.radians(5.0)


def residual(mounting: str, camera: Pose, target: Pose, robot: Pose, board: Pose) -> np.ndarray:
    """The six-component constraint violation for one view.

    Args:
        mounting: `"eye_in_hand"` or `"eye_to_hand"`.
        camera: The camera transform being estimated.
        target: The target transform being estimated.
        robot: The robot's flange-to-base pose for this view.
        board: The target's pose in the camera for this view.

    Returns:
        Rotation vector then translation of the residual transform, which is the
        identity at the solution.
    """
    if mounting == "eye_in_hand":
        # Z^-1 A X B should be the identity.
        deviation = target.inverse().compose(robot).compose(camera).compose(board)
    else:
        # X^-1 A Z B^-1 should be the identity.
        deviation = (
            camera.inverse().compose(robot).compose(target).compose(board.inverse())
        )
    return np.concatenate([matrix_to_rotvec(deviation.rotation), deviation.translation])


def _residuals(
    mounting: str,
    camera: Pose,
    target: Pose,
    robots: Sequence[Pose],
    boards: Sequence[Pose],
) -> np.ndarray:
    return np.stack(
        [
            residual(mounting, camera, target, robot, board)
            for robot, board in zip(robots, boards)
        ]
    )


def _weights(rotation_sigma: float, translation_sigma: float) -> np.ndarray:
    """Per-component reciprocal scales, so the two units are commensurate."""
    return np.array([1.0 / rotation_sigma] * 3 + [1.0 / translation_sigma] * 3)


def _estimate_sigmas(residuals: np.ndarray) -> Tuple[float, float]:
    """Residual scales for rotation and translation, from the residuals.

    Each set of `3v` components pays for six of the twelve parameters, which is
    the even split; it is an approximation, and it only affects the absolute
    scale of the covariance rather than its shape.
    """
    n_views = residuals.shape[0]
    dof = max(3 * n_views - 6, 1)
    rotation = float(np.sqrt(np.sum(residuals[:, :3] ** 2) / dof))
    translation = float(np.sqrt(np.sum(residuals[:, 3:] ** 2) / dof))
    return (
        max(rotation, MIN_ROTATION_SIGMA),
        max(translation, MIN_TRANSLATION_SIGMA),
    )


def _apply(camera: Pose, target: Pose, step: np.ndarray) -> Tuple[Pose, Pose]:
    """Right-multiply both transforms by a local increment.

    A right-multiplied update keeps the increment in each transform's own frame,
    which is the well-behaved chart for a rigid transform and avoids the
    distortion an additive update to a rotation vector suffers at large angles.
    """
    return (
        camera.compose(Pose.from_parameter_vector(step[:6])),
        target.compose(Pose.from_parameter_vector(step[6:])),
    )


def _jacobian(
    mounting: str,
    camera: Pose,
    target: Pose,
    robots: Sequence[Pose],
    boards: Sequence[Pose],
    weights: np.ndarray,
    step: float = 1e-7,
) -> Tuple[np.ndarray, np.ndarray]:
    """Central-difference Jacobian of the weighted residuals.

    Numerical rather than analytic, deliberately. The analytic form needs the
    adjoint of four composed transforms per view and is easy to get subtly
    wrong; the numerical form is twenty-four cheap residual evaluations and the
    test suite checks the refinement it drives against known truth.

    Returns:
        The `(6v, 12)` Jacobian and the flattened weighted residual.
    """
    base = (_residuals(mounting, camera, target, robots, boards) * weights).reshape(-1)
    columns = []
    for index in range(12):
        forward = np.zeros(12)
        forward[index] = step
        plus = _apply(camera, target, forward)
        minus = _apply(camera, target, -forward)
        high = (_residuals(mounting, *plus, robots, boards) * weights).reshape(-1)
        low = (_residuals(mounting, *minus, robots, boards) * weights).reshape(-1)
        columns.append((high - low) / (2.0 * step))
    return np.stack(columns, axis=1), base


def _chordal_mean(poses: Sequence[Pose]) -> Pose:
    """The average of several rigid transforms.

    Translations average directly; rotations use the chordal mean, which is the
    closest rotation matrix to the arithmetic mean and is the standard choice
    for averaging orientations that are already close together.
    """
    translation = np.mean([p.translation for p in poses], axis=0)
    stacked = np.mean([p.rotation for p in poses], axis=0)
    u, _, vt = np.linalg.svd(stacked)
    rotation = u @ vt
    if np.linalg.det(rotation) < 0:
        u[:, -1] *= -1.0
        rotation = u @ vt
    return Pose(rotation, translation)


def relative_motions(
    robots: Sequence[Pose], boards: Sequence[Pose], mounting: str
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Relative robot and camera motions over every pair of views.

    The hand-eye constraint becomes `A_ij X = X B_ij` between any two views, and
    that pairwise form is what both the closed-form solve and the pose-set
    diagnostics read. Pairs whose relative rotation is negligible are dropped,
    because they constrain the camera rotation not at all.

    Args:
        robots: Flange-to-base poses, one per view.
        boards: Target-in-camera poses, one per view.
        mounting: `"eye_in_hand"` or `"eye_to_hand"`.

    Returns:
        Four arrays over the kept pairs: robot rotation vectors, camera rotation
        vectors, robot translations and camera translations.
    """
    robot_rotations, camera_rotations = [], []
    robot_translations, camera_translations = [], []
    count = len(robots)
    for i in range(count):
        for j in range(i + 1, count):
            if mounting == "eye_in_hand":
                # Z = A X B for all i, so A_j^-1 A_i X = X B_j B_i^-1.
                motion = robots[j].inverse().compose(robots[i])
                board_motion = boards[j].compose(boards[i].inverse())
            else:
                # X = A Z B^-1 for all i, so A_i^-1 A_j X = X B_i^-1 B_j is not
                # the form; here the camera transform sits on the left instead.
                motion = robots[i].inverse().compose(robots[j])
                board_motion = boards[i].inverse().compose(boards[j])
            axis = matrix_to_rotvec(motion.rotation)
            if np.linalg.norm(axis) < MIN_PAIR_ROTATION_RAD:
                continue
            robot_rotations.append(axis)
            camera_rotations.append(matrix_to_rotvec(board_motion.rotation))
            robot_translations.append(motion.translation)
            camera_translations.append(board_motion.translation)
    if not robot_rotations:
        raise ValidationError(
            "no pair of views differs by more than "
            f"{np.degrees(MIN_PAIR_ROTATION_RAD):.0f} degrees of rotation; a "
            "hand-eye calibration cannot be determined from translations alone"
        )
    return (
        np.stack(robot_rotations),
        np.stack(camera_rotations),
        np.stack(robot_translations),
        np.stack(camera_translations),
    )


def closed_form(
    mounting: str, robots: Sequence[Pose], boards: Sequence[Pose]
) -> Pose:
    """Closed-form estimate of whichever transform sits in the middle.

    Which transform that is depends on the mounting, and getting it wrong is a
    quiet, plausible-looking mistake rather than a crash. Eliminating the
    constant between two views gives

    * eye-in-hand: `A_j^-1 A_i X = X B_j B_i^-1`, so the middle term is the
      **camera** transform,
    * eye-to-hand: `A_i^-1 A_j Z = Z B_i^-1 B_j`, so the middle term is the
      **target** transform.

    Callers should use `_initial_guess`, which converts either answer into both.

    Args:
        mounting: `"eye_in_hand"` or `"eye_to_hand"`.
        robots: Flange-to-base poses, one per view.
        boards: Target-in-camera poses, one per view.

    Returns:
        The camera transform for eye-in-hand, the target transform for
        eye-to-hand.

    Raises:
        ValidationError: The relative motions do not determine a rotation.
    """
    robot_axes, camera_axes, robot_t, camera_t = relative_motions(
        robots, boards, mounting
    )
    # Procrustes on the axis pairs: minimise sum |a - R b|^2.
    correlation = robot_axes.T @ camera_axes
    u, _, vt = np.linalg.svd(correlation)
    rotation = u @ vt
    if np.linalg.det(rotation) < 0:
        u[:, -1] *= -1.0
        rotation = u @ vt

    # Then (R_A - I) t_X = R_X t_B - t_A, stacked over pairs.
    blocks, targets = [], []
    for index in range(robot_axes.shape[0]):
        motion_rotation = rotvec_to_matrix(robot_axes[index])
        blocks.append(motion_rotation - np.eye(3))
        targets.append(rotation @ camera_t[index] - robot_t[index])
    design = np.vstack(blocks)
    translation, *_ = np.linalg.lstsq(design, np.concatenate(targets), rcond=None)
    return Pose(rotation, translation)


def _initial_guess(
    mounting: str, robots: Sequence[Pose], boards: Sequence[Pose]
) -> Tuple[Pose, Pose]:
    """Both transforms, from whichever one the closed form recovers.

    The closed form returns the term that survives eliminating the constant,
    which is the camera transform for eye-in-hand and the target transform for
    eye-to-hand. The other follows from the constraint, averaged over views.
    """
    middle = closed_form(mounting, robots, boards)
    if mounting == "eye_in_hand":
        # Z = A X B
        camera = middle
        target = _chordal_mean(
            [robot.compose(camera).compose(board) for robot, board in zip(robots, boards)]
        )
    else:
        # X = A Z B^-1
        target = middle
        camera = _chordal_mean(
            [
                robot.compose(target).compose(board.inverse())
                for robot, board in zip(robots, boards)
            ]
        )
    return camera, target


def _refine(
    mounting: str,
    camera: Pose,
    target: Pose,
    robots: Sequence[Pose],
    boards: Sequence[Pose],
    weights: np.ndarray,
    iterations: int,
) -> Tuple[Pose, Pose]:
    """A few Gauss-Newton steps from a starting point known to be close."""
    cost = float(
        np.sum((_residuals(mounting, camera, target, robots, boards) * weights) ** 2)
    )
    for _ in range(iterations):
        jacobian, flat = _jacobian(mounting, camera, target, robots, boards, weights)
        try:
            step = np.linalg.lstsq(jacobian.T @ jacobian, -(jacobian.T @ flat), rcond=None)[0]
        except np.linalg.LinAlgError:
            break
        trial_camera, trial_target = _apply(camera, target, step)
        trial = float(
            np.sum(
                (_residuals(mounting, trial_camera, trial_target, robots, boards) * weights)
                ** 2
            )
        )
        if not np.isfinite(trial) or trial >= cost:
            break
        camera, target, cost = trial_camera, trial_target, trial
    return camera, target


def solve_hand_eye(
    fit: InstrumentedFit,
    session: CalibrationSession,
    mounting: str = "eye_in_hand",
    rcond: float = DEFAULT_RCOND,
    monte_carlo: bool = True,
    n_samples: int = 200,
    seed: Optional[int] = 0,
) -> HandEyeResult:
    """Estimate the hand-eye transform and its covariance.

    Args:
        fit: The instrumented refit, whose per-view poses are the target's pose
            in the camera.
        session: The session the fit came from, for its robot poses.
        mounting: `"eye_in_hand"` when the camera rides the flange,
            `"eye_to_hand"` when it is fixed in the cell.
        rcond: Relative eigenvalue cut when inverting the normal equations.
        monte_carlo: Resample calibrations to get a covariance that accounts for
            the camera poses being estimated rather than measured. On by default
            because the residual-only alternative was measured to understate the
            translation deviation threefold; turn it off for speed when the
            number is not going to be quoted.
        n_samples: Calibrations to resample when `monte_carlo` is set.
        seed: Seed for the resampling.

    Returns:
        The transform, its covariance, and the residuals behind them.

    Raises:
        ValidationError: The mounting is unknown, the session has no robot poses,
            there are too few views, or the robot never rotated enough for the
            problem to be determined.
    """
    if mounting not in MOUNTINGS:
        raise ValidationError(
            f"unknown mounting {mounting!r}; expected one of {list(MOUNTINGS)}"
        )
    if session.robot is None:
        raise ValidationError(
            "this session carries no robot poses; ingest them with --robot-poses"
        )
    if fit.view_ids != session.observations.view_ids:
        raise ValidationError(
            "the fit and the session cover different views, so the robot poses "
            "cannot be aligned to the camera poses"
        )
    robots = list(session.robot.aligned_with(session.observations))
    boards = list(fit.poses)
    if len(robots) < MIN_VIEWS:
        raise ValidationError(
            f"hand-eye needs at least {MIN_VIEWS} views, got {len(robots)}"
        )

    camera, target = _initial_guess(mounting, robots, boards)
    weights = _weights(1.0, 1.0)
    cost = float(
        np.sum((_residuals(mounting, camera, target, robots, boards) * weights) ** 2)
    )
    iterations = 0
    # Two passes: refine under unit weights, re-estimate the residual scales,
    # then refine again under the scales the data actually implies.
    for pass_index in range(2):
        if pass_index == 1:
            weights = _weights(
                *_estimate_sigmas(_residuals(mounting, camera, target, robots, boards))
            )
            cost = float(
                np.sum(
                    (_residuals(mounting, camera, target, robots, boards) * weights) ** 2
                )
            )
        for _ in range(MAX_ITERATIONS):
            jacobian, flat = _jacobian(
                mounting, camera, target, robots, boards, weights
            )
            normal = jacobian.T @ jacobian
            gradient = jacobian.T @ flat
            try:
                step = np.linalg.lstsq(normal, -gradient, rcond=None)[0]
            except np.linalg.LinAlgError:
                break
            trial_camera, trial_target = _apply(camera, target, step)
            trial = float(
                np.sum(
                    (
                        _residuals(mounting, trial_camera, trial_target, robots, boards)
                        * weights
                    )
                    ** 2
                )
            )
            iterations += 1
            if not np.isfinite(trial) or trial >= cost:
                break
            improvement = (cost - trial) / max(cost, 1e-30)
            camera, target, cost = trial_camera, trial_target, trial
            if improvement < TOLERANCE:
                break

    residuals = _residuals(mounting, camera, target, robots, boards)
    rotation_sigma, translation_sigma = _estimate_sigmas(residuals)
    weights = _weights(rotation_sigma, translation_sigma)
    jacobian, _ = _jacobian(mounting, camera, target, robots, boards, weights)
    # Weights already carry 1/sigma, so the normal equations are in units where
    # the residual variance is one and no further scaling is needed.
    spectrum = invert_symmetric(jacobian.T @ jacobian, rcond)
    result = HandEyeResult(
        mounting=mounting,
        camera=camera,
        target=target,
        covariance=spectrum.inverse,
        residual_covariance=spectrum.inverse,
        spectrum=spectrum,
        rotation_sigma_rad=rotation_sigma,
        translation_sigma_mm=translation_sigma,
        n_views=len(robots),
        residuals=residuals,
        view_ids=session.observations.view_ids,
        covariance_method="residual",
        iterations=iterations,
    )
    if not monte_carlo:
        return result
    empirical, _ = resample_covariance(
        result, fit, session, robots, weights, n_samples, seed
    )
    return replace(
        result,
        covariance=empirical,
        covariance_method="monte_carlo",
        monte_carlo_samples=n_samples,
    )


def resample_covariance(
    nominal: HandEyeResult,
    fit: InstrumentedFit,
    session: CalibrationSession,
    robots: Sequence[Pose],
    weights: np.ndarray,
    n_samples: int = 200,
    seed: Optional[int] = 0,
    refine_iterations: int = 3,
) -> Tuple[np.ndarray, np.ndarray]:
    """Hand-eye covariance that accounts for the calibration's own uncertainty.

    The residual-based covariance treats the target-in-camera poses as exact
    data. They are not: they come from the calibration, and — this is the part
    that matters — their errors are **correlated across views**, because every
    view shares the same intrinsics. A focal-length error tilts and scales every
    camera pose coherently, so it does not average down the way independent
    per-view noise would, and a covariance built on the independence assumption
    comes out optimistic. Measured on a synthetic rig with known truth, the
    residual-based translation deviation was 0.39 mm against an actual error of
    1.29 mm; resampling gave 1.22 mm.

    Args:
        nominal: The solved hand-eye result to perturb around.
        fit: The instrumented refit whose covariance is resampled.
        session: The session, for its robot poses.
        robots: Flange poses aligned to the views.
        weights: Residual weights from the nominal solve.
        n_samples: Calibrations to resample.
        seed: Seed, so a report is reproducible.
        refine_iterations: Gauss-Newton steps per sample, from the nominal
            solution. Three is enough because each perturbation is small.

    Returns:
        The `(12, 12)` empirical covariance and the `(n, 12)` deviations behind
        it, ordered as `HandEyeResult.PARAMETER_NAMES`.

    Raises:
        ValidationError: Too few resamples converged for a covariance.
    """
    from ..task.sampling import CovarianceSampler

    sampler = CovarianceSampler(fit, tuple(range(len(fit.poses))), seed)
    deviations: List[np.ndarray] = []
    for sample in sampler.draw(n_samples):
        try:
            camera, target = _refine(
                nominal.mounting, nominal.camera, nominal.target,
                robots, list(sample.poses), weights, refine_iterations,
            )
        except (ValidationError, np.linalg.LinAlgError):
            continue
        deviations.append(
            np.concatenate([
                nominal.camera.inverse().compose(camera).parameter_vector(),
                nominal.target.inverse().compose(target).parameter_vector(),
            ])
        )
    if len(deviations) < max(10, n_samples // 2):
        raise ValidationError(
            f"only {len(deviations)} of {n_samples} hand-eye resamples converged; "
            "the pose set is too weak for a resampled covariance"
        )
    stacked = np.stack(deviations)
    return np.cov(stacked.T), stacked
