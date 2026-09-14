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

"""Pose-set sufficiency for hand-eye calibration.

Hand-eye is determined by *rotation*, and by nothing else. The constraint
between two views reduces to `R_A R_X = R_X R_B`, which says the rotation axis
of the robot's relative motion equals the camera's, carried through `R_X`. So a
robot that only translates determines nothing, a robot that rotates about one
axis determines `R_X` only up to a rotation about that axis, and two
well-separated axes are the minimum for a full answer.

The translation follows from `(R_A - I) t_X = R_X t_B - t_A`, whose conditioning
is again set by the rotation magnitudes: as `R_A` approaches the identity the
left-hand side vanishes and `t_X` runs away. That is why a cell programmed with
small, careful wrist moves produces a confident-looking hand-eye result that is
badly wrong in translation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar, Optional, Sequence, Tuple

import numpy as np

from ..core.poses import Pose
from ..diagnose.base import Finding, Severity
from ..diagnose.geometry import orientation_tensor
from ..diagnose.report import Diagnosis
from ..errors import ValidationError
from .result import HandEyeResult
from .solve import relative_motions

#: Second orientation-tensor eigenvalue below which the rotation axes are
#: effectively collinear, leaving one rotational degree of freedom free.
AXES_COLLINEAR = 0.02

#: Third eigenvalue below which the axes are confined to a plane.
AXES_COPLANAR = 0.02

#: Median relative rotation below which the translation is poorly conditioned.
ROTATION_CRITICAL_DEG = 10.0
ROTATION_WARNING_DEG = 25.0

#: Scaled condition number of the hand-eye normal equations.
CONDITION_CRITICAL = 1e6
CONDITION_WARNING = 1e4


@dataclass(frozen=True)
class HandEyeContext:
    """Everything the hand-eye diagnostics read.

    Attributes:
        result: The solved hand-eye transform.
        robot_axes: Rotation vectors of the relative robot motions.
        camera_axes: Rotation vectors of the matching relative camera motions.
    """

    result: HandEyeResult
    robot_axes: np.ndarray
    camera_axes: np.ndarray

    @classmethod
    def build(
        cls, result: HandEyeResult, robots: Sequence[Pose], boards: Sequence[Pose]
    ) -> "HandEyeContext":
        """Derive the relative motions the diagnostics need.

        Args:
            result: The solved hand-eye transform.
            robots: Flange-to-base poses, one per view.
            boards: Target-in-camera poses, one per view.

        Returns:
            A ready context.
        """
        robot_axes, camera_axes, _, _ = relative_motions(
            robots, boards, result.mounting
        )
        return cls(result, robot_axes, camera_axes)

    @property
    def angles_deg(self) -> np.ndarray:
        """Relative robot rotation angle for each pair, in degrees."""
        return np.degrees(np.linalg.norm(self.robot_axes, axis=1))

    @property
    def unit_axes(self) -> np.ndarray:
        """Unit rotation axes of the relative robot motions."""
        norms = np.linalg.norm(self.robot_axes, axis=1, keepdims=True)
        return self.robot_axes / np.where(norms > 0, norms, 1.0)


#: Relative scale error assumed for a printed target when reporting what the
#: board costs. 0.1% is the middle of the 50 to 200 micrometre range a printed
#: or laminated target holds on a 25 mm pitch; it is a stated assumption rather
#: than a measurement of the user's board, which is why the finding reports the
#: sensitivity alongside it so a reader with a certificate can substitute their
#: own figure.
TYPICAL_BOARD_SCALE_ERROR = 0.001

#: Ratio above which the note says the board is the dominant term rather than
#: merely a term. This never changes the severity, for the reason in
#: `board_scale`: the ratio grows as the capture improves, so grading on it
#: would punish good work.
BOARD_SCALE_DOMINANT = 1.0


def _finding(cause, title, severity, summary, action="", **metrics) -> Finding:
    return Finding(cause, title, severity, summary, action, metrics)


def rotation_axis_spread(context: HandEyeContext) -> Finding:
    """Do the robot's rotation axes span enough directions?"""
    eigenvalues = orientation_tensor(context.unit_axes)
    metrics = dict(
        eigenvalues=[float(v) for v in eigenvalues],
        n_pairs=int(context.unit_axes.shape[0]),
    )
    shared = (
        f"the relative rotation axes over {metrics['n_pairs']} view pairs have "
        f"orientation-tensor eigenvalues "
        f"{eigenvalues[0]:.3f}/{eigenvalues[1]:.3f}/{eigenvalues[2]:.3f}"
    )
    if eigenvalues[1] < AXES_COLLINEAR:
        return _finding(
            "hand_eye_rotation_axes", "Hand-eye rotation axes", Severity.CRITICAL,
            shared + "; the axes are effectively collinear, so the camera "
            "rotation about that axis is not determined at all",
            "Rotate the wrist about a second, clearly different axis. Hand-eye is "
            "determined by rotation and by nothing else, so no amount of "
            "translation will fix this.",
            **metrics,
        )
    if eigenvalues[2] < AXES_COPLANAR:
        return _finding(
            "hand_eye_rotation_axes", "Hand-eye rotation axes", Severity.WARNING,
            shared + "; the axes lie almost in a plane, leaving the rotation "
            "about that plane's normal weakly determined",
            "Add motions rotating about an axis out of the plane the current ones "
            "span.",
            **metrics,
        )
    return _finding(
        "hand_eye_rotation_axes", "Hand-eye rotation axes", Severity.OK,
        shared + ", which spans all three directions", **metrics,
    )


def rotation_magnitude(context: HandEyeContext) -> Finding:
    """Are the robot's relative rotations large enough to condition `t_X`?"""
    angles = context.angles_deg
    median = float(np.median(angles))
    metrics = dict(
        median_deg=median,
        max_deg=float(angles.max()),
        min_deg=float(angles.min()),
        n_pairs=int(angles.size),
    )
    shared = (
        f"relative robot rotations run {angles.min():.1f} to {angles.max():.1f} "
        f"degrees, median {median:.1f}"
    )
    action = (
        "Make larger wrist rotations between views. The translation solves "
        "through (R - I), so as the rotations shrink the offset becomes "
        "arbitrarily badly determined while the residuals stay small."
    )
    if median < ROTATION_CRITICAL_DEG:
        return _finding(
            "hand_eye_rotation_magnitude", "Hand-eye rotation magnitude",
            Severity.CRITICAL,
            shared + "; that is too small to determine the camera offset",
            action, **metrics,
        )
    if median < ROTATION_WARNING_DEG:
        return _finding(
            "hand_eye_rotation_magnitude", "Hand-eye rotation magnitude",
            Severity.WARNING, shared, action, **metrics,
        )
    return _finding(
        "hand_eye_rotation_magnitude", "Hand-eye rotation magnitude", Severity.OK,
        shared, **metrics,
    )


def conditioning(context: HandEyeContext) -> Finding:
    """Is the hand-eye system well posed once everything is combined?"""
    result = context.result
    scaled = result.spectrum.scaled_condition_number
    metrics = dict(
        scaled_condition_number=float(scaled),
        rank=int(result.spectrum.rank),
        identifiable=result.identifiable,
    )
    shared = (
        f"the hand-eye normal equations have a scaled condition number of "
        f"{scaled:.3e} at rank {result.spectrum.rank}/12"
    )
    if not result.identifiable or scaled > CONDITION_CRITICAL:
        return _finding(
            "hand_eye_conditioning", "Hand-eye conditioning", Severity.CRITICAL,
            shared + "; some combination of the twelve parameters is not "
            "determined by these motions",
            "Fix whichever of the rotation findings above applies; the "
            "conditioning is a consequence of them, not a separate problem.",
            **metrics,
        )
    if scaled > CONDITION_WARNING:
        return _finding(
            "hand_eye_conditioning", "Hand-eye conditioning", Severity.WARNING,
            shared, "Broaden the range of wrist orientations.", **metrics,
        )
    return _finding(
        "hand_eye_conditioning", "Hand-eye conditioning", Severity.OK, shared,
        **metrics,
    )


def covariance_basis(context: HandEyeContext) -> Finding:
    """Is the uncertainty below the honest one or the cheap one?

    The residual-based covariance treats the target-in-camera poses as exact
    data. They are not: they come from the calibration, and their errors are
    correlated across views because every view shares the same intrinsics, so a
    focal-length error tilts and scales all of them coherently and does not
    average down. Measured against known truth it gave 0.39 mm where the actual
    error was 1.29 mm.

    It used to be enough to label this in prose, because the number was merely
    optimistic. It is not enough now. The variance split added in item 3c asks
    which of the camera and the hand-eye to spend money on, and a residual-based
    solve answers "the camera, entirely" regardless of the truth — the share it
    hides is exactly the share being measured. A wrong number is worse than a
    loose one, so this is a finding rather than a sentence.
    """
    result = context.result
    if result.covariance_method != "residual":
        return _finding(
            "hand_eye_covariance_basis", "Hand-eye uncertainty basis", Severity.OK,
            f"the covariance comes from {result.monte_carlo_samples} resampled "
            f"calibrations, {result.optimism_factor():.1f}x wider than the "
            "residual-only estimate it replaces",
            covariance_method=result.covariance_method,
            optimism_factor=float(result.optimism_factor()),
        )
    return _finding(
        "hand_eye_covariance_basis", "Hand-eye uncertainty basis", Severity.WARNING,
        "this covariance came from the residuals alone, which treats the "
        "target-in-camera poses as exact when they came from the calibration; "
        "measured against known truth it understated the translation error "
        "threefold",
        "Re-solve with the resampling covariance, which is the default — "
        "`monte_carlo=False` was passed to get here. Note that this does not "
        "only widen the interval: a variance split taken on a residual-based "
        "solve attributes the whole error to the camera calibration and none to "
        "the hand-eye, because the share it understates is the share being "
        "measured.",
        covariance_method=result.covariance_method,
        optimism_factor=float(result.optimism_factor()),
    )


def board_scale(context: HandEyeContext) -> Finding:
    """Is a printed board's own error bigger than the interval being reported?

    This is the one error in the whole chain that nothing else can see. A target
    printed a tenth of a per cent large is a scale error on every object point;
    the fit absorbs it entirely into the poses, the focal length is untouched,
    and the reprojection residual is unchanged to the last bit. Hand-eye is
    where it surfaces, because the robot's flange poses are in true millimetres
    and the board-derived poses are not, so the translation takes the
    disagreement — at about seven times the board error.

    It is also invisible to the view-clustered covariance, which is built from
    the scatter *between* views and a board scale error moves every view
    coherently. So neither the random-error machinery nor the model-validity
    check can reach it, and this finding is the only thing that says so.

    Always a note, never a warning. The quantity it reports is the ratio of a
    systematic the tool cannot see to a random error it can, and that ratio
    *grows as the capture improves* — more views and better coverage shrink the
    denominator while leaving the numerator alone. Grading on it would mean a
    better calibration earning a louder complaint, and it would gate CI on a
    property of the user's target rather than of their work. What it is for is
    telling a reader when the number they are about to act on is limited by
    something no amount of recapturing will fix.
    """
    result = context.result
    sensitivity = float(result.board_scale_sensitivity_mm)
    random_mm = float(np.sqrt(np.diag(result.camera_covariance)[3:6]).mean())
    typical = TYPICAL_BOARD_SCALE_ERROR * sensitivity
    ratio = typical / random_mm if random_mm > 0 else float("inf")
    metrics = dict(
        board_scale_sensitivity_mm=sensitivity,
        assumed_board_scale_error=TYPICAL_BOARD_SCALE_ERROR,
        implied_translation_error_mm=typical,
        reported_translation_sd_mm=random_mm,
        ratio=float(ratio),
    )
    shared = (
        f"a board scale error moves the camera translation by "
        f"{sensitivity:.0f} mm per unit, so a target printed "
        f"{100 * TYPICAL_BOARD_SCALE_ERROR:.1f}% out would shift it "
        f"{typical:.3f} mm against the {random_mm:.3f} mm of random error "
        f"reported here"
    )
    action = (
        "calibsense treats the board as exact, so this systematic is not in any "
        "interval above and cannot be: it is identical across every view, which "
        "is precisely what a between-view estimate cannot see, and it leaves the "
        "reprojection error unchanged. Order a target with a calibration "
        "certificate and use its measured pitch, or measure your own against a "
        "gauge. A glass or ceramic target holds scale far better than a printed "
        "or laminated one."
    )
    if sensitivity <= 0:
        return _finding(
            "hand_eye_board_scale", "Board scale", Severity.NOTE,
            "the board scale sensitivity could not be measured, because the "
            "perturbed solve did not converge",
            **metrics,
        )
    dominant = ratio > BOARD_SCALE_DOMINANT
    return _finding(
        "hand_eye_board_scale", "Board scale", Severity.NOTE,
        shared + (
            f", which is {ratio:.1f}x larger — the board, not this solve, is "
            "what limits the answer"
            if dominant else
            f", or {ratio:.2f} times it"
        ),
        action, dominant=bool(dominant), **metrics,
    )


def axis_agreement(context: HandEyeContext) -> Finding:
    """Do the robot and camera relative rotations agree in magnitude?

    The two must rotate by the same angle, since `R_A` and `R_B` are conjugate
    through `R_X`. A systematic disagreement means the robot poses and the
    camera views are not the pairs they are claimed to be — usually an ordering
    or timestamp-alignment mistake, which no amount of optimisation will fix.
    """
    robot = np.linalg.norm(context.robot_axes, axis=1)
    camera = np.linalg.norm(context.camera_axes, axis=1)
    difference = np.degrees(np.abs(robot - camera))
    median = float(np.median(difference))
    worst = float(difference.max())
    metrics = dict(
        median_disagreement_deg=median,
        max_disagreement_deg=worst,
        n_pairs=int(difference.size),
    )
    shared = (
        f"robot and camera relative rotation angles disagree by a median of "
        f"{median:.3f} degrees, worst {worst:.3f}"
    )
    action = (
        "Check that each robot pose is paired with the right image. Conjugate "
        "rotations have equal angles, so a disagreement this large means the "
        "pairing is wrong rather than the calibration being noisy."
    )
    if median > 1.0:
        return _finding(
            "hand_eye_pairing", "Hand-eye pose pairing", Severity.CRITICAL,
            shared + "; these are not matched pairs", action, **metrics,
        )
    if median > 0.2:
        return _finding(
            "hand_eye_pairing", "Hand-eye pose pairing", Severity.WARNING,
            shared, action, **metrics,
        )
    return _finding(
        "hand_eye_pairing", "Hand-eye pose pairing", Severity.OK,
        shared + ", consistent with matched pairs", **metrics,
    )


#: The hand-eye diagnostics, in report order.
HAND_EYE_DIAGNOSTICS = (
    axis_agreement,
    rotation_axis_spread,
    rotation_magnitude,
    conditioning,
    covariance_basis,
    board_scale,
)


def diagnose_hand_eye(
    result: HandEyeResult, robots: Sequence[Pose], boards: Sequence[Pose]
) -> Diagnosis:
    """Run every hand-eye sufficiency check.

    Args:
        result: The solved hand-eye transform.
        robots: Flange-to-base poses, one per view.
        boards: Target-in-camera poses, one per view.

    Returns:
        The findings, in report order.

    Raises:
        ValidationError: The pose lists disagree in length.
    """
    if len(robots) != len(boards):
        raise ValidationError(
            f"{len(robots)} robot poses against {len(boards)} camera poses"
        )
    context = HandEyeContext.build(result, robots, boards)
    return Diagnosis(tuple(check(context) for check in HAND_EYE_DIAGNOSTICS))
