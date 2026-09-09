"""Rig builders that each trigger one specific diagnostic.

A diagnostic is only worth anything if it fires on the fault it is named for and
stays quiet on the others. These builders make that testable: each one produces a
capture with a single deliberate defect and known truth.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import numpy as np

from caltrust.core.camera import CameraModel, FisheyeKannalaBrandt, PinholeBrownConrady
from caltrust.core.observations import ObservationSet, ViewObservations
from caltrust.core.poses import Pose
from caltrust.core.session import CalibrationSession, RobotPoses
from caltrust.core.target import Checkerboard
from caltrust.diagnose.base import DiagnosticContext
from caltrust.refit import RefitOptions, instrument
from caltrust.refit.projection import projector_for
from caltrust.synthetic import pose_for_view, synthesise

IMAGE_SIZE = (1280, 720)
NOISE_PX = 0.2
BOARD = Checkerboard(9, 6, 25.0)

#: A well-behaved wide lens: the radial function is monotone over the whole frame.
WIDE_PINHOLE = PinholeBrownConrady(700.0, 703.0, 639.5, 359.5,
                                   [-0.30, 0.11, 8e-4, -1.1e-3, -0.022])
STRONG_FISHEYE = FisheyeKannalaBrandt(330.0, 332.0, 639.5, 359.5,
                                      [-0.12, 0.04, -0.01, 0.002])


def poses(
    n: int = 24,
    distances_mm: Sequence[float] = (400.0, 800.0),
    tilt_degrees: Tuple[float, float] = (20.0, 40.0),
    lateral_mm: float = 240.0,
    seed: int = 2,
    target=BOARD,
) -> List[Pose]:
    """Build a pose set with explicit control of every axis a diagnostic reads.

    Args:
        n: Number of views.
        distances_mm: Working distances, sampled uniformly between the extremes
            when two or more are given.
        tilt_degrees: Range of board tilt.
        lateral_mm: How far the board wanders sideways, which drives coverage.
        seed: Random seed.
        target: The target being placed.

    Returns:
        One pose per view.
    """
    rng = np.random.default_rng(seed)
    low, high = min(distances_mm), max(distances_mm)
    return [
        pose_for_view(
            target,
            rng.uniform(low, high) if high > low else low,
            tilt_rad=np.radians(rng.uniform(*tilt_degrees)),
            tilt_axis_rad=rng.uniform(0.0, 2 * np.pi),
            roll_rad=rng.uniform(-np.pi, np.pi),
            offset_mm=rng.uniform(-lateral_mm, lateral_mm, 2),
        )
        for _ in range(n)
    ]


def context(
    camera: CameraModel = WIDE_PINHOLE,
    pose_set: Optional[Sequence[Pose]] = None,
    options: Optional[RefitOptions] = None,
    noise_px: float = NOISE_PX,
    target=BOARD,
    seed: int = 5,
    cross_validation=None,
) -> DiagnosticContext:
    """Synthesise a capture, fit it, and wrap it for the diagnostics.

    Args:
        camera: The true intrinsics.
        pose_set: Poses, or `None` for a healthy default.
        options: Refit settings.
        noise_px: Corner noise standard deviation.
        target: The target to project.
        seed: Seed for the noise.
        cross_validation: Attach an out-of-sample result.

    Returns:
        A context ready to hand to any diagnostic.
    """
    capture = synthesise(
        camera, target, list(pose_set if pose_set is not None else poses()),
        IMAGE_SIZE, noise_px=noise_px, seed=seed,
    )
    session = CalibrationSession(observations=capture.observations)
    fit = instrument(session, options)
    return DiagnosticContext(fit, capture.observations, cross_validation)


def healthy() -> DiagnosticContext:
    """Real tilt, two working distances, wide lateral coverage."""
    return context()


def frontoparallel() -> DiagnosticContext:
    """Every board flat-on at one distance: the focal length is undetermined."""
    return context(pose_set=poses(tilt_degrees=(0.2, 1.5), distances_mm=(800.0,)))


def single_depth() -> DiagnosticContext:
    """Well tilted, but every view at the same distance."""
    return context(pose_set=poses(distances_mm=(700.0,)))


def centre_only() -> DiagnosticContext:
    """Well tilted and varied in depth, but the board never leaves the middle."""
    return context(pose_set=poses(lateral_mm=15.0))


def tiny_board() -> DiagnosticContext:
    """The board so far away that corner spacing collapses."""
    return context(pose_set=poses(distances_mm=(2600.0,), lateral_mm=200.0))


def wrong_model() -> DiagnosticContext:
    """A fisheye lens fitted with Brown-Conrady."""
    return context(
        camera=STRONG_FISHEYE,
        pose_set=poses(distances_mm=(300.0, 420.0), tilt_degrees=(25.0, 40.0),
                       lateral_mm=260.0, seed=1),
        options=RefitOptions(model="pinhole"),
    )


def right_model() -> DiagnosticContext:
    """The same fisheye lens fitted with Kannala-Brandt."""
    return context(
        camera=STRONG_FISHEYE,
        pose_set=poses(distances_mm=(300.0, 420.0), tilt_degrees=(25.0, 40.0),
                       lateral_mm=260.0, seed=1),
        options=RefitOptions(model="fisheye"),
    )


def with_bad_view(scale: float = 8.0) -> DiagnosticContext:
    """A healthy capture with one view given its own much larger noise."""
    capture = synthesise(
        WIDE_PINHOLE, BOARD, poses(), IMAGE_SIZE, noise_px=NOISE_PX, seed=5
    )
    views = list(capture.observations.views)
    rng = np.random.default_rng(0)
    spoiled = views[3]
    views[3] = ViewObservations(
        spoiled.view_id,
        spoiled.point_ids,
        spoiled.image_points + rng.normal(0.0, scale * NOISE_PX, spoiled.image_points.shape),
    )
    observations = ObservationSet(BOARD, IMAGE_SIZE, tuple(views))
    session = CalibrationSession(observations=observations)
    return DiagnosticContext(instrument(session), observations)


#: Ground truth for the hand-eye rigs: the camera's pose on the flange, and the
#: target's pose in the cell.
TRUE_FLANGE_TO_CAMERA = Pose(
    np.array([
        [np.cos(1.55), -np.sin(1.55), 0.0],
        [np.sin(1.55), np.cos(1.55), 0.0],
        [0.0, 0.0, 1.0],
    ]),
    [42.0, -17.5, 88.0],
)
TRUE_BASE_TO_TARGET = Pose(np.diag([1.0, -1.0, -1.0]), [520.0, 120.0, -300.0])


def hand_eye_session(
    mounting: str = "eye_in_hand",
    n: int = 16,
    seed: int = 0,
    noise_px: float = 0.15,
    tilt_degrees: Tuple[float, float] = (12.0, 35.0),
    roll_span_rad: float = np.pi,
    roll_only: bool = False,
    mispair: bool = False,
) -> CalibrationSession:
    """A capture with robot poses that satisfy the hand-eye constraint exactly.

    The board poses are chosen first, so every view lands in frame, and the
    robot poses are then *derived* from the constraint. Building it the other way
    round produces board poses that are behind the camera or out of frame.

    Args:
        mounting: `"eye_in_hand"` or `"eye_to_hand"`.
        n: Views to attempt.
        seed: Random seed.
        noise_px: Corner noise standard deviation.
        tilt_degrees: Range of board tilt, which sets the rotation variety the
            hand-eye solve depends on.
        roll_span_rad: Range of board roll about the optical axis. Shrinking
            both this and `tilt_degrees` is what produces a capture whose
            relative rotations are too small to determine anything.
        roll_only: Rotate the board only about the optical axis, which makes
            every relative rotation axis parallel and the solve degenerate.
        mispair: Shuffle the robot poses against the views, simulating an
            ordering or timestamp-alignment mistake.

    Returns:
        A session carrying both the detections and the robot poses.
    """
    rng = np.random.default_rng(seed)
    projector = projector_for(WIDE_PINHOLE)
    robots, views = [], []
    for index in range(n):
        board = pose_for_view(
            BOARD,
            rng.uniform(500.0, 900.0),
            tilt_rad=0.0 if roll_only else np.radians(rng.uniform(*tilt_degrees)),
            tilt_axis_rad=0.0 if roll_only else rng.uniform(0.0, 2 * np.pi),
            roll_rad=rng.uniform(-roll_span_rad, roll_span_rad),
            offset_mm=rng.uniform(-60.0, 60.0, 2),
        )
        if mounting == "eye_in_hand":
            robot = (
                TRUE_BASE_TO_TARGET.compose(board.inverse())
                .compose(TRUE_FLANGE_TO_CAMERA.inverse())
            )
        else:
            robot = TRUE_FLANGE_TO_CAMERA.compose(board).compose(
                TRUE_BASE_TO_TARGET.inverse()
            )
        points = projector.project(WIDE_PINHOLE, board, BOARD.object_points())
        points = points + rng.normal(0.0, noise_px, points.shape)
        keep = (
            (points[:, 0] >= 0)
            & (points[:, 0] < IMAGE_SIZE[0])
            & (points[:, 1] >= 0)
            & (points[:, 1] < IMAGE_SIZE[1])
        )
        if keep.sum() < 25:
            continue
        robots.append(robot)
        views.append(
            ViewObservations(
                f"v{index:02d}", np.arange(BOARD.num_points)[keep], points[keep]
            )
        )
    observations = ObservationSet(BOARD, IMAGE_SIZE, tuple(views))
    order = list(range(len(robots)))
    if mispair:
        order = list(np.random.default_rng(7).permutation(len(robots)))
    return CalibrationSession(
        observations=observations,
        robot=RobotPoses(tuple(robots[i] for i in order), observations.view_ids),
    )
