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

"""Drawing parameter sets from a calibration's covariance.

This is where M2's covariance stops being a diagnostic and starts being useful.
Monte Carlo propagation needs whole parameter sets sampled from the joint
distribution, not one standard deviation at a time, because the trade-offs
between parameters are exactly what determines the task-space error. Sampling
`fx` and `tz` independently from their marginals would produce a spread far
wider than reality, since the two are correlated at 0.93 and their errors
partly cancel in most tasks.

Rank deficiency is handled by construction and reported rather than hidden. An
unconstrained direction has zero variance in a pseudo-inverse, so samples never
move along it and the resulting interval is *too tight*, not too wide. Every
sample set therefore carries `bounded`, and everything downstream is required to
say so when it is `False`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Iterator, List, Optional, Sequence, Tuple

import numpy as np

from ..core.camera import CameraModel
from ..core.parameters import extrinsic_names
from ..core.poses import Pose
from ..errors import ValidationError
from ..refit.covariance import CalibrationCovariance
from ..refit.engine import parameter_block
from ..refit.normal import POSE_DIMENSION
from ..refit.result import InstrumentedFit

if TYPE_CHECKING:  # avoids a cycle: handeye imports task.sampling
    from ..handeye.result import HandEyeResult

#: Eigenvalues below this fraction of the largest are treated as zero when
#: factorising a covariance for sampling.
SAMPLING_RCOND = 1e-12


@dataclass(frozen=True)
class ParameterSample:
    """One draw from the calibration's posterior.

    Attributes:
        camera: The sampled intrinsics.
        poses: The sampled poses, in the order the view indices were requested.
        hand_eye: The sampled hand-eye camera transform, when a task needs one.
            Sampled independently of the intrinsics, because the hand-eye solve
            already folded the calibration's uncertainty into its own covariance
            and counting it twice would inflate the interval.
    """

    camera: CameraModel
    poses: Tuple[Pose, ...] = ()
    hand_eye: Optional[Pose] = None


def joint_covariance(
    covariance: CalibrationCovariance, view_indices: Sequence[int] = ()
) -> Tuple[np.ndarray, Tuple[str, ...]]:
    """Assemble the covariance over the free intrinsics and selected poses.

    Args:
        covariance: The fit's covariance.
        view_indices: Views whose poses should be included.

    Returns:
        The joint covariance and a name per row.

    Raises:
        ValidationError: A view index is out of range or repeated.
    """
    views = [int(i) for i in view_indices]
    if len(set(views)) != len(views):
        raise ValidationError(f"repeated view index in {views}")
    for index in views:
        if not 0 <= index < covariance.n_views:
            raise ValidationError(
                f"view index {index} out of range for {covariance.n_views} views"
            )
    p = covariance.n_intrinsic
    size = p + POSE_DIMENSION * len(views)
    joint = np.zeros((size, size))
    joint[:p, :p] = covariance.intrinsic
    for slot, index in enumerate(views):
        start = p + POSE_DIMENSION * slot
        stop = start + POSE_DIMENSION
        cross = covariance.cross_block(index)
        joint[:p, start:stop] = cross
        joint[start:stop, :p] = cross.T
        for other_slot, other in enumerate(views):
            other_start = p + POSE_DIMENSION * other_slot
            joint[start:stop, other_start : other_start + POSE_DIMENSION] = (
                covariance.pose_cross_block(index, other)
            )
    names: List[str] = list(covariance.intrinsic_names)
    for index in views:
        names.extend(extrinsic_names(index))
    return 0.5 * (joint + joint.T), tuple(names)


def factorise(
    matrix: np.ndarray, rcond: float = SAMPLING_RCOND
) -> Tuple[np.ndarray, int]:
    """Factorise a symmetric positive semi-definite covariance for sampling.

    Uses an eigendecomposition rather than a Cholesky factorisation, because a
    rank-deficient covariance has no Cholesky factor and a calibration that is
    not identifiable produces exactly that.

    Args:
        matrix: The covariance to factorise.
        rcond: Eigenvalues below this fraction of the largest are dropped.

    Returns:
        A factor `L` with `L @ L.T` equal to the covariance, and the rank.
    """
    symmetric = 0.5 * (np.asarray(matrix, dtype=float) + np.asarray(matrix, dtype=float).T)
    eigenvalues, vectors = np.linalg.eigh(symmetric)
    largest = float(eigenvalues[-1]) if eigenvalues.size else 0.0
    keep = eigenvalues > max(largest, 0.0) * rcond
    scaled = np.zeros_like(eigenvalues)
    scaled[keep] = np.sqrt(eigenvalues[keep])
    return vectors * scaled, int(np.count_nonzero(keep))


class CovarianceSampler:
    """Draws parameter sets from a fit's joint covariance.

    Attributes:
        fit: The calibration being sampled.
        view_indices: Views whose poses are drawn alongside the intrinsics.
        bounded: Whether the sampled distribution covers the whole parameter
            space. `False` means the fit left a direction unconstrained, so the
            samples understate the true spread and any interval derived from
            them is a lower bound.
    """

    def __init__(
        self,
        fit: InstrumentedFit,
        view_indices: Sequence[int] = (),
        seed: Optional[int] = 0,
        hand_eye: Optional["HandEyeResult"] = None,
    ):
        self.fit = fit
        self.hand_eye = hand_eye
        self._hand_eye_factor = (
            factorise(hand_eye.camera_covariance)[0] if hand_eye is not None else None
        )
        self.view_indices = tuple(int(i) for i in view_indices)
        self._rng = np.random.default_rng(seed)
        self._covariance, self.names = joint_covariance(
            fit.covariance, self.view_indices
        )
        self._factor, self.rank = factorise(self._covariance)
        self._block = parameter_block(fit.camera, fit.options)
        self._reduction = self._block.reduction()
        self._nominal_intrinsic = fit.camera.to_vector()
        self._nominal_poses = tuple(
            fit.poses[i].parameter_vector() for i in self.view_indices
        )

    @property
    def bounded(self) -> bool:
        """Whether every parameter direction carries variance."""
        return self.fit.conditioning.identifiable and self.rank == len(self.names)

    @property
    def n_free(self) -> int:
        """Number of jointly sampled parameters."""
        return len(self.names)

    def nominal(self) -> ParameterSample:
        """The fit itself, with no perturbation."""
        return ParameterSample(
            camera=self.fit.camera,
            poses=tuple(self.fit.poses[i] for i in self.view_indices),
            hand_eye=None if self.hand_eye is None else self.hand_eye.camera,
        )

    def draw(self, n_samples: int) -> List[ParameterSample]:
        """Draw parameter sets.

        Args:
            n_samples: How many to draw.

        Returns:
            One sample per draw.

        Raises:
            ValidationError: `n_samples` is not positive.
        """
        if n_samples < 1:
            raise ValidationError(f"need at least one sample, got {n_samples}")
        noise = self._rng.standard_normal((n_samples, self.n_free))
        steps = noise @ self._factor.T
        hand_eye_steps = (
            self._rng.standard_normal((n_samples, 6)) @ self._hand_eye_factor.T
            if self._hand_eye_factor is not None
            else None
        )
        p = self.fit.covariance.n_intrinsic
        model = type(self.fit.camera)
        samples: List[ParameterSample] = []
        for step in steps:
            intrinsic = self._nominal_intrinsic + self._reduction @ step[:p]
            try:
                camera = model.from_vector(intrinsic)
            except ValidationError:
                # A draw can push a focal length non-positive when the fit is
                # so uncertain that the linear model is not even locally valid.
                # Dropping it would bias the interval downwards, so the nominal
                # camera stands in and the caller sees the count.
                camera = self.fit.camera
            poses = tuple(
                Pose.from_parameter_vector(
                    nominal + step[p + POSE_DIMENSION * slot : p + POSE_DIMENSION * (slot + 1)]
                )
                for slot, nominal in enumerate(self._nominal_poses)
            )
            hand_eye = None
            if self.hand_eye is not None:
                # A right-multiplied increment keeps the perturbation in the
                # transform's own frame, which is where its covariance lives.
                hand_eye = self.hand_eye.camera.compose(
                    Pose.from_parameter_vector(hand_eye_steps[len(samples)])
                )
            samples.append(ParameterSample(camera, poses, hand_eye))
        return samples

    def marginal_std(self) -> np.ndarray:
        """Standard deviation of each jointly sampled parameter."""
        return np.sqrt(np.clip(np.diag(self._covariance), 0.0, None))
