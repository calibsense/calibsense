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

"""What a hand-eye calibration produces."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, ClassVar, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np

from ..core.poses import Pose, poses_from_array, poses_to_array
from ..errors import ValidationError
from ..refit.linalg import SymmetricInverse

#: The two mountings. Eye-in-hand puts the camera on the moving flange;
#: eye-to-hand bolts it to the cell and puts the target on the flange.
MOUNTINGS = ("eye_in_hand", "eye_to_hand")


@dataclass(frozen=True)
class HandEyeResult:
    """A hand-eye transform with its covariance.

    Two transforms come out of one solve, because the problem has two unknowns
    and neither is a nuisance parameter in practice. For eye-in-hand they are
    the camera's pose on the flange and the target's pose in the cell; for
    eye-to-hand they are the camera's pose in the cell and the target's pose on
    the flange.

    Attributes:
        mounting: `"eye_in_hand"` or `"eye_to_hand"`.
        camera: The camera transform. Flange-to-camera for eye-in-hand,
            base-to-camera for eye-to-hand.
        target: The target transform. Base-to-target for eye-in-hand,
            flange-to-target for eye-to-hand.
        covariance: Joint covariance over the twelve parameters, ordered as the
            camera transform's `[rx, ry, rz, tx, ty, tz]` then the target's.
            Rotations are in radians, translations in millimetres.
        covariance_method: `"monte_carlo"` or `"residual"`. The residual method
            treats the target-in-camera poses as exact data, which they are not
            — they come from the calibration, and their errors are correlated
            across views because every view shares the same intrinsics. Measured
            against known truth it understated the translation deviation by
            roughly a factor of three. Prefer `"monte_carlo"`.
        residual_covariance: The residual-based covariance, kept even when the
            Monte Carlo one is reported, so a report can show the gap.
        spectrum: Eigen-report of the normal equations, including the rank and
            both condition numbers.
        rotation_sigma_rad: Estimated rotational residual scale.
        translation_sigma_mm: Estimated translational residual scale.
        n_views: Views the solve used.
        residuals: Per-view residual, shape `(v, 6)`, rotation then translation.
        view_ids: The views, in residual order.
        iterations: Refinement iterations taken.
    """

    mounting: str
    camera: Pose
    target: Pose
    covariance: np.ndarray
    residual_covariance: np.ndarray
    spectrum: SymmetricInverse
    rotation_sigma_rad: float
    translation_sigma_mm: float
    n_views: int
    residuals: np.ndarray
    view_ids: Tuple[str, ...]
    covariance_method: str = "residual"
    monte_carlo_samples: int = 0
    iterations: int = 0

    PARAMETER_NAMES: ClassVar[Tuple[str, ...]] = (
        "camera.rx", "camera.ry", "camera.rz", "camera.tx", "camera.ty", "camera.tz",
        "target.rx", "target.ry", "target.rz", "target.tx", "target.ty", "target.tz",
    )

    def __post_init__(self) -> None:
        if self.mounting not in MOUNTINGS:
            raise ValidationError(
                f"unknown mounting {self.mounting!r}; expected one of {list(MOUNTINGS)}"
            )
        covariance = np.asarray(self.covariance, dtype=float)
        if covariance.shape != (12, 12):
            raise ValidationError(
                f"hand-eye covariance must be 12x12, got {covariance.shape}"
            )
        residual_covariance = np.asarray(self.residual_covariance, dtype=float)
        if residual_covariance.shape != (12, 12):
            raise ValidationError(
                f"residual covariance must be 12x12, got {residual_covariance.shape}"
            )
        if self.covariance_method not in ("residual", "monte_carlo"):
            raise ValidationError(
                f"unknown covariance method {self.covariance_method!r}"
            )
        object.__setattr__(self, "covariance", covariance)
        object.__setattr__(self, "residual_covariance", residual_covariance)
        object.__setattr__(self, "residuals", np.asarray(self.residuals, dtype=float))

    @property
    def camera_covariance(self) -> np.ndarray:
        """The `(6, 6)` covariance of the camera transform alone."""
        return self.covariance[:6, :6]

    @property
    def target_covariance(self) -> np.ndarray:
        """The `(6, 6)` covariance of the target transform alone."""
        return self.covariance[6:, 6:]

    @property
    def identifiable(self) -> bool:
        """Whether the pose set determines all twelve parameters.

        `False` means the robot motions do not exercise every degree of freedom
        — most often because the rotation axes are nearly parallel — and the
        standard deviations below describe only the subspace that was
        determined.
        """
        return not self.spectrum.rank_deficient

    def rotation_std_deg(self) -> np.ndarray:
        """Standard deviation of the camera transform's rotation, in degrees."""
        return np.degrees(np.sqrt(np.clip(np.diag(self.camera_covariance)[:3], 0.0, None)))

    def translation_std_mm(self) -> np.ndarray:
        """Standard deviation of the camera transform's translation, in millimetres."""
        return np.sqrt(np.clip(np.diag(self.camera_covariance)[3:], 0.0, None))

    def optimism_factor(self) -> float:
        """How much the residual method understates the reported deviation.

        Returns:
            The ratio of the reported translation deviation to the residual
            method's, or one when the residual method is what is reported.
        """
        reported = float(np.linalg.norm(self.translation_std_mm()))
        residual = float(
            np.linalg.norm(np.sqrt(np.clip(np.diag(self.residual_covariance)[3:6], 0.0, None)))
        )
        return reported / residual if residual > 0 else 1.0

    def rotation_rms_deg(self) -> float:
        """RMS rotational residual across views, in degrees."""
        return float(np.degrees(np.sqrt(np.mean(self.residuals[:, :3] ** 2))))

    def translation_rms_mm(self) -> float:
        """RMS translational residual across views, in millimetres."""
        return float(np.sqrt(np.mean(self.residuals[:, 3:] ** 2)))

    def summary_lines(self) -> Tuple[str, ...]:
        """A short human summary."""
        rotation = self.rotation_std_deg()
        translation = self.translation_std_mm()
        label = (
            "flange to camera" if self.mounting == "eye_in_hand" else "base to camera"
        )
        lines = [
            f"mounting     {self.mounting.replace('_', '-')} ({self.n_views} views)",
            f"{label:12s} t = "
            f"[{self.camera.translation[0]:+.2f}, {self.camera.translation[1]:+.2f}, "
            f"{self.camera.translation[2]:+.2f}] mm "
            f"+/- [{translation[0]:.2f}, {translation[1]:.2f}, {translation[2]:.2f}]",
            f"{'':12s} r = "
            f"[{np.degrees(self.camera.rvec[0]):+.3f}, "
            f"{np.degrees(self.camera.rvec[1]):+.3f}, "
            f"{np.degrees(self.camera.rvec[2]):+.3f}] deg "
            f"+/- [{rotation[0]:.3f}, {rotation[1]:.3f}, {rotation[2]:.3f}]",
            f"residual     {self.rotation_rms_deg():.4f} deg, "
            f"{self.translation_rms_mm():.3f} mm RMS over {self.n_views} views",
            f"uncertainty  from {self.covariance_method.replace('_', ' ')}"
            + (
                f" over {self.monte_carlo_samples} resampled calibrations, "
                f"{self.optimism_factor():.1f}x wider than a residual-only estimate"
                if self.covariance_method == "monte_carlo"
                else "; this treats the camera poses as exact and is optimistic"
            ),
            f"conditioning scaled condition number "
            f"{self.spectrum.scaled_condition_number:.3e}, "
            f"rank {self.spectrum.rank}/12",
        ]
        if not self.identifiable:
            lines.append(
                "IDENTIFIABILITY  the robot motions do not determine every degree "
                "of freedom; the deviations above cover only the subspace that was"
            )
        return tuple(lines)

    def to_dict(self) -> Dict[str, Any]:
        """Serialise to a plain JSON-compatible dictionary."""
        return {
            "mounting": self.mounting,
            "camera": {
                "matrix": self.camera.matrix.tolist(),
                "rvec": self.camera.rvec.tolist(),
                "translation_mm": self.camera.translation.tolist(),
            },
            "target": {
                "matrix": self.target.matrix.tolist(),
                "rvec": self.target.rvec.tolist(),
                "translation_mm": self.target.translation.tolist(),
            },
            "parameter_names": list(self.PARAMETER_NAMES),
            "covariance": self.covariance.tolist(),
            "rotation_std_deg": self.rotation_std_deg().tolist(),
            "translation_std_mm": self.translation_std_mm().tolist(),
            "rotation_rms_deg": self.rotation_rms_deg(),
            "translation_rms_mm": self.translation_rms_mm(),
            "rotation_sigma_rad": self.rotation_sigma_rad,
            "translation_sigma_mm": self.translation_sigma_mm,
            "covariance_method": self.covariance_method,
            "residual_covariance": self.residual_covariance.tolist(),
            "monte_carlo_samples": self.monte_carlo_samples,
            "optimism_factor": self.optimism_factor(),
            "identifiable": self.identifiable,
            "rank": self.spectrum.rank,
            "condition_number": self.spectrum.condition_number,
            "scaled_condition_number": self.spectrum.scaled_condition_number,
            "n_views": self.n_views,
            "view_ids": list(self.view_ids),
            "iterations": self.iterations,
        }
