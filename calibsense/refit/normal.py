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

"""Assembling the normal equations, keeping the block structure.

The Jacobian of a calibration is not a dense matrix and should never be built as
one. Intrinsic columns touch every residual; each view's six pose columns touch
only that view's residuals. Storing it as `U`, a stack of `W_i`, and a stack of
`V_i` keeps memory linear in the number of views instead of quadratic, and makes
the Schur complement — the thing that actually produces the intrinsic covariance
— a short loop rather than a large inverse.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np

from ..core.camera import CameraModel
from ..core.observations import ObservationSet
from ..core.parameters import ParameterBlock, extrinsic_names
from ..core.poses import Pose
from ..errors import RefitError, ValidationError
from .linalg import DEFAULT_RCOND, SymmetricInverse, block_inverse_stack, invert_symmetric
from .projection import Projector, projector_for

#: Pose parameters per view: three for rotation, three for translation.
POSE_DIMENSION = 6


@dataclass(frozen=True)
class NormalEquations:
    """The Gauss-Newton normal equations in block form.

    Attributes:
        u: Intrinsic-intrinsic block, shape `(p, p)` over free parameters.
        u_view: Each view's own contribution to `u`, shape `(v, p, p)`, so that
            `u == u_view.sum(axis=0)`. Kept because a view's share of the
            intrinsic information is only recoverable from the per-view term,
            and that share is what "leverage" means for a calibration.
        w: Intrinsic-pose cross blocks, shape `(v, p, 6)`.
        v: Pose-pose blocks, shape `(v, 6, 6)`.
        gradient_intrinsic: `J_intrinsic.T @ residual`, shape `(p,)`.
        gradient_pose: `J_pose.T @ residual` per view, shape `(v, 6)`.
        cost: Sum of squared residuals.
        n_residuals: Number of scalar residuals, two per observed point.
        intrinsic_names: Names of the free intrinsic parameters.
        gradient_intrinsic_view: Each view's contribution to
            `gradient_intrinsic`, shape `(v, p)`. Kept for the same reason as
            `u_view`: a view-clustered covariance needs each view's own score
            and the sum cannot be taken apart again. `None` only for equations
            restored from a bundle written before calibsense stored it.
    """

    u: np.ndarray
    u_view: np.ndarray
    w: np.ndarray
    v: np.ndarray
    gradient_intrinsic: np.ndarray
    gradient_pose: np.ndarray
    cost: float
    n_residuals: int
    intrinsic_names: Tuple[str, ...]
    gradient_intrinsic_view: Optional[np.ndarray] = None

    @property
    def n_views(self) -> int:
        """Number of views."""
        return int(self.v.shape[0])

    @property
    def n_intrinsic(self) -> int:
        """Number of free intrinsic parameters."""
        return int(self.u.shape[0])

    @property
    def n_parameters(self) -> int:
        """Total free parameters, intrinsics plus six per view."""
        return self.n_intrinsic + POSE_DIMENSION * self.n_views

    @property
    def degrees_of_freedom(self) -> int:
        """Residuals minus parameters.

        Zero or negative means the model has as many knobs as the data has
        constraints, and no variance can be estimated at all.
        """
        return self.n_residuals - self.n_parameters

    @property
    def rms(self) -> float:
        """Root-mean-square reprojection error in pixels.

        Computed per point rather than per residual component, which is the
        convention `cv2.calibrateCamera` reports and therefore the number the
        engineer will compare against.
        """
        points = self.n_residuals // 2
        return float(np.sqrt(self.cost / points)) if points else 0.0

    @property
    def gradient_norm(self) -> float:
        """Norm of the full gradient.

        At a converged least-squares minimum this is near zero. A large value
        means the parameters being instrumented are not the optimum of the data
        they are being instrumented against, which is worth saying out loud when
        auditing a calibration produced elsewhere.
        """
        return float(
            np.sqrt(
                np.sum(self.gradient_intrinsic ** 2) + np.sum(self.gradient_pose ** 2)
            )
        )

    def parameter_names(self) -> Tuple[str, ...]:
        """Names of every free parameter, intrinsics first then each view's pose."""
        names: List[str] = list(self.intrinsic_names)
        for index in range(self.n_views):
            names.extend(extrinsic_names(index))
        return tuple(names)

    def schur_complement(
        self, rcond: float = DEFAULT_RCOND
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Eliminate the pose blocks, leaving the intrinsic system.

        Args:
            rcond: Relative eigenvalue cut used when inverting each pose block.

        Returns:
            A tuple `(s, v_inverse, y, singular_views)`. `s` is the Schur
            complement `U - sum_i W_i V_i^-1 W_i.T`, shape `(p, p)`. `v_inverse`
            holds each `V_i^-1`. `y` holds `Y_i = V_i^-1 W_i.T`, shape
            `(v, 6, p)`, which is what the extrinsic and cross covariance blocks
            are built from. `singular_views` flags views whose pose block was
            rank deficient.
        """
        v_inverse, singular = block_inverse_stack(self.v, rcond)
        y = np.einsum("nij,nkj->nik", v_inverse, self.w)
        s = self.u - np.einsum("nik,nkj->ij", self.w, y)
        return s, v_inverse, y, singular

    def schur_contributions(self, rcond: float = DEFAULT_RCOND) -> np.ndarray:
        """Each view's contribution to the intrinsic information.

        The Schur complement is a sum over views, `S = sum_i S_i` with
        `S_i = U_i - W_i V_i^-1 W_i.T`. Splitting it that way is what makes a
        per-view leverage well defined: `trace(S_i S^-1)` sums to `p` across
        views, so `trace(S_i S^-1) / p` is that view's share of everything the
        capture knows about the intrinsics.

        Args:
            rcond: Relative eigenvalue cut used when inverting each pose block.

        Returns:
            An `(v, p, p)` array whose sum over the first axis is the Schur
            complement.
        """
        v_inverse, _ = block_inverse_stack(self.v, rcond)
        y = np.einsum("nij,nkj->nik", v_inverse, self.w)
        return self.u_view - np.einsum("nik,nkj->nij", self.w, y)

    def view_scores(self, rcond: float = DEFAULT_RCOND) -> np.ndarray:
        """Each view's score for the intrinsics, after its own pose is removed.

        The intrinsic update solves `S dx = sum_i s_i` with

            s_i = A_i' r_i - Y_i' (B_i' r_i),   Y_i = V_i^-1 W_i'

        so `s_i` is everything view `i` still has to say about the intrinsics
        once its own six pose parameters have absorbed what they can. The
        scatter of these across views is what a view-clustered covariance is
        built from, and their sum is the reduced gradient that
        `CalibrationCovariance.newton_decrement` already forms.

        Args:
            rcond: Relative eigenvalue cut used when inverting each pose block.

        Returns:
            A `(v, p)` array whose column sums are the reduced gradient.

        Raises:
            RefitError: The per-view intrinsic gradient was not recorded. This
                happens only for equations restored from a bundle written by a
                calibsense old enough not to have stored it.
        """
        if self.gradient_intrinsic_view is None:
            raise RefitError(
                "these normal equations carry only the summed intrinsic "
                "gradient, so per-view scores cannot be recovered; re-run the "
                "refit to get a view-clustered covariance"
            )
        _, _, y, _ = self.schur_complement(rcond)
        return self.gradient_intrinsic_view - np.einsum(
            "nij,ni->nj", y, self.gradient_pose
        )


def assemble(
    camera: CameraModel,
    poses: Sequence[Pose],
    observations: ObservationSet,
    block: ParameterBlock,
    weights: Optional[Sequence[np.ndarray]] = None,
    projector: Optional[Projector] = None,
) -> Tuple[NormalEquations, List[np.ndarray]]:
    """Build the normal equations and per-view residuals at a given parameter set.

    Args:
        camera: The intrinsic model to linearise around.
        poses: One board-to-camera pose per view, in view order.
        observations: The detected points.
        block: Which intrinsic parameters are free, and any ties between them.
        weights: Optional per-point weights, one `(n_i,)` array per view. Each
            weight scales that point's squared residual contribution.
        projector: Override the projector, for tests.

    Returns:
        The normal equations and a list of `(n_i, 2)` residual arrays, one per
        view, in pixels and *not* weighted, so that residual statistics stay in
        the units an engineer reads.

    Raises:
        ValidationError: The pose count or a weight array does not match.
        RefitError: A view could not be projected.
    """
    if len(poses) != observations.n_views:
        raise ValidationError(
            f"{len(poses)} poses for {observations.n_views} views"
        )
    if block.n_full != len(camera.parameter_names()):
        raise ValidationError(
            f"parameter block covers {block.n_full} parameters but "
            f"{camera.kind} has {len(camera.parameter_names())}"
        )
    projector = projector or projector_for(camera)
    reduction = block.reduction()
    n_free = reduction.shape[1]
    if n_free == 0:
        raise ValidationError("no intrinsic parameter is free; nothing to estimate")

    u_view = np.zeros((observations.n_views, n_free, n_free))
    w = np.zeros((observations.n_views, n_free, POSE_DIMENSION))
    v = np.zeros((observations.n_views, POSE_DIMENSION, POSE_DIMENSION))
    gradient_intrinsic_view = np.zeros((observations.n_views, n_free))
    gradient_pose = np.zeros((observations.n_views, POSE_DIMENSION))
    residuals: List[np.ndarray] = []
    cost = 0.0
    n_residuals = 0

    for index, (view, pose) in enumerate(zip(observations.views, poses)):
        object_points = view.object_points(observations.target)
        try:
            projected, d_intrinsic_full, d_pose = projector.project_with_jacobian(
                camera, pose, object_points
            )
        except Exception as exc:  # OpenCV raises cv2.error, not a calibsense type
            raise RefitError(f"view {view.view_id!r} could not be projected: {exc}") from exc
        residual = projected - view.image_points
        residuals.append(residual)

        d_intrinsic = d_intrinsic_full @ reduction
        flat = residual.reshape(-1)
        if weights is not None:
            scale = _row_scale(weights[index], view.n_points, view.view_id)
            d_intrinsic = d_intrinsic * scale[:, None]
            d_pose = d_pose * scale[:, None]
            flat = flat * scale
        u_view[index] = d_intrinsic.T @ d_intrinsic
        w[index] = d_intrinsic.T @ d_pose
        v[index] = d_pose.T @ d_pose
        gradient_intrinsic_view[index] = d_intrinsic.T @ flat
        gradient_pose[index] = d_pose.T @ flat
        cost += float(flat @ flat)
        n_residuals += flat.size

    equations = NormalEquations(
        u=u_view.sum(axis=0),
        u_view=u_view,
        w=w,
        v=v,
        gradient_intrinsic=gradient_intrinsic_view.sum(axis=0),
        gradient_intrinsic_view=gradient_intrinsic_view,
        gradient_pose=gradient_pose,
        cost=cost,
        n_residuals=n_residuals,
        intrinsic_names=block.free_names(),
    )
    return equations, residuals


def _row_scale(weight: np.ndarray, n_points: int, view_id: str) -> np.ndarray:
    values = np.asarray(weight, dtype=float).reshape(-1)
    if values.size != n_points:
        raise ValidationError(
            f"view {view_id!r}: {values.size} weights for {n_points} points"
        )
    if np.any(values < 0) or not np.all(np.isfinite(values)):
        raise ValidationError(f"view {view_id!r}: weights must be finite and non-negative")
    return np.repeat(np.sqrt(values), 2)
