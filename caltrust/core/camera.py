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

"""Camera parameter containers.

These are data, not behaviour: a `CameraModel` knows its own parameter layout,
its serialised form and how to hand its numbers to OpenCV, but it does not
project points. Projection and differentiation live in `caltrust.refit`, which
keeps this module free of an OpenCV import and therefore trivially testable.

Two models are supported, matching the two OpenCV calibration families:

* `PinholeBrownConrady` — `cv2.calibrateCamera`, 4 to 14 distortion coefficients.
* `FisheyeKannalaBrandt` — `cv2.fisheye.calibrate`, 4 coefficients plus skew.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, ClassVar, Dict, Mapping, Sequence, Tuple, Type

import numpy as np

from ..errors import ValidationError
from .parameters import ParameterBlock

_MODELS: Dict[str, Type["CameraModel"]] = {}


def register_model(cls: Type["CameraModel"]) -> Type["CameraModel"]:
    """Register a camera model so `camera_from_dict` can find it by `kind`.

    Registration is an explicit decorator rather than a module scan, so a frozen
    PyInstaller build resolves every model without dynamic discovery.

    Args:
        cls: The model class to register.

    Returns:
        `cls`, unchanged.
    """
    _MODELS[cls.kind] = cls
    return cls


@dataclass(frozen=True, eq=False)
class CameraModel(ABC):
    """Intrinsic parameters of one camera.

    Attributes:
        fx: Focal length along x, in pixels.
        fy: Focal length along y, in pixels.
        cx: Principal point x, in pixels.
        cy: Principal point y, in pixels.
        distortion: Distortion coefficients in the model's own order.
    """

    fx: float
    fy: float
    cx: float
    cy: float
    distortion: np.ndarray

    kind: ClassVar[str] = ""
    #: Distortion vector lengths this model accepts.
    valid_distortion_sizes: ClassVar[Tuple[int, ...]] = ()

    def __post_init__(self) -> None:
        distortion = np.asarray(self.distortion, dtype=float).reshape(-1)
        if distortion.size not in self.valid_distortion_sizes:
            raise ValidationError(
                f"{self.kind} takes {self.valid_distortion_sizes} distortion "
                f"coefficients, got {distortion.size}"
            )
        if not np.all(np.isfinite(distortion)):
            raise ValidationError("distortion coefficients must be finite")
        for name in ("fx", "fy", "cx", "cy"):
            value = float(getattr(self, name))
            if not np.isfinite(value):
                raise ValidationError(f"{name} must be finite, got {value}")
            object.__setattr__(self, name, value)
        if self.fx <= 0 or self.fy <= 0:
            raise ValidationError(
                f"focal lengths must be positive, got fx={self.fx}, fy={self.fy}"
            )
        object.__setattr__(self, "distortion", distortion)

    @property
    def camera_matrix(self) -> np.ndarray:
        """The 3x3 intrinsic matrix `K`, skew included where the model has one."""
        return np.array(
            [[self.fx, self.skew * self.fx, self.cx],
             [0.0, self.fy, self.cy],
             [0.0, 0.0, 1.0]]
        )

    @property
    def skew(self) -> float:
        """Skew coefficient, zero for models that do not carry one."""
        return 0.0

    @abstractmethod
    def distortion_names(self) -> Tuple[str, ...]:
        """Names of the distortion coefficients, in stored order."""

    def parameter_names(self) -> Tuple[str, ...]:
        """Names of the full intrinsic parameter vector.

        The first four entries are always `fx, fy, cx, cy`, so callers can index
        focal length and principal point identically across models.
        """
        return ("fx", "fy", "cx", "cy") + self.distortion_names()

    def to_vector(self) -> np.ndarray:
        """Flatten the intrinsics into the canonical parameter vector."""
        return np.concatenate([[self.fx, self.fy, self.cx, self.cy], self.distortion])

    @classmethod
    def from_vector(cls, vector: Sequence[float]) -> "CameraModel":
        """Rebuild a model from a canonical parameter vector.

        Args:
            vector: Values in the order given by `parameter_names`.

        Returns:
            A new model of this class.
        """
        values = np.asarray(vector, dtype=float).reshape(-1)
        if values.size < 5:
            raise ValidationError(f"parameter vector too short: {values.size}")
        return cls(values[0], values[1], values[2], values[3], values[4:])

    def free_parameters(self, fixed: Sequence[str] = (), tie_aspect: bool = False) -> ParameterBlock:
        """Describe which intrinsics are estimated.

        Args:
            fixed: Names of parameters to hold at their current value.
            tie_aspect: Hold `fy / fx` constant instead of estimating both. The
                tie is proportional, which is what a fixed aspect ratio means;
                treating `fy` as merely fixed would be a different constraint.

        Returns:
            A `ParameterBlock` over the full intrinsic vector.

        Raises:
            ValidationError: A name in `fixed` is not a parameter of this model.
        """
        names = self.parameter_names()
        unknown = set(fixed) - set(names)
        if unknown:
            raise ValidationError(
                f"unknown parameter(s) {sorted(unknown)} for {self.kind}; "
                f"expected some of {list(names)}"
            )
        free = np.array([name not in set(fixed) for name in names], dtype=bool)
        ties: Tuple[Tuple[int, int, float], ...] = ()
        if tie_aspect:
            if not (free[0] and free[1]):
                raise ValidationError("tie_aspect needs both fx and fy free")
            ties = ((1, 0, self.fy / self.fx),)
        return ParameterBlock(names, free, ties)

    def to_dict(self) -> Dict[str, Any]:
        """Serialise to a plain JSON-compatible dictionary."""
        return {
            "kind": self.kind,
            "fx": self.fx,
            "fy": self.fy,
            "cx": self.cx,
            "cy": self.cy,
            "distortion": self.distortion.tolist(),
        }

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, CameraModel) or other.kind != self.kind:
            return NotImplemented
        return (
            (self.fx, self.fy, self.cx, self.cy, self.skew)
            == (other.fx, other.fy, other.cx, other.cy, other.skew)
            and np.array_equal(self.distortion, other.distortion)
        )

    def allclose(self, other: "CameraModel", rtol: float = 1e-7, atol: float = 1e-9) -> bool:
        """Compare two models of the same kind within a tolerance.

        Args:
            other: Model to compare against.
            rtol: Relative tolerance passed to `numpy.allclose`.
            atol: Absolute tolerance passed to `numpy.allclose`.

        Returns:
            `True` if both models are the same kind and every parameter agrees.
        """
        if other.kind != self.kind or other.distortion.size != self.distortion.size:
            return False
        return bool(
            np.allclose(self.to_vector(), other.to_vector(), rtol=rtol, atol=atol)
            and np.isclose(self.skew, other.skew, rtol=rtol, atol=atol)
        )

    def __repr__(self) -> str:
        distortion = np.array2string(self.distortion, precision=5, separator=", ")
        return (
            f"{type(self).__name__}(fx={self.fx:.4f}, fy={self.fy:.4f}, "
            f"cx={self.cx:.4f}, cy={self.cy:.4f}, distortion={distortion})"
        )


_BROWN_CONRADY_NAMES = (
    "k1", "k2", "p1", "p2", "k3", "k4", "k5", "k6",
    "s1", "s2", "s3", "s4", "taux", "tauy",
)


@register_model
@dataclass(frozen=True, eq=False)
class PinholeBrownConrady(CameraModel):
    """Pinhole projection with the OpenCV Brown-Conrady distortion model.

    The distortion vector follows OpenCV's own order and may hold 4, 5, 8, 12 or
    14 coefficients; longer vectors enable the rational, thin-prism and tilted
    sensor terms in that sequence.
    """

    kind: ClassVar[str] = "pinhole_brown_conrady"
    valid_distortion_sizes: ClassVar[Tuple[int, ...]] = (4, 5, 8, 12, 14)

    def distortion_names(self) -> Tuple[str, ...]:
        """Names of the active Brown-Conrady coefficients."""
        return _BROWN_CONRADY_NAMES[: self.distortion.size]


@register_model
@dataclass(frozen=True, eq=False)
class FisheyeKannalaBrandt(CameraModel):
    """Equidistant fisheye projection with the Kannala-Brandt distortion model.

    Attributes:
        alpha: Skew coefficient. OpenCV's fisheye model places it at
            `K[0, 1] = alpha * fx`, and `cv2.fisheye.calibrate` fixes it to zero
            unless asked otherwise, which is why it defaults to zero here.
    """

    alpha: float = 0.0

    kind: ClassVar[str] = "fisheye_kannala_brandt"
    valid_distortion_sizes: ClassVar[Tuple[int, ...]] = (4,)

    def __post_init__(self) -> None:
        super().__post_init__()
        if not np.isfinite(self.alpha):
            raise ValidationError(f"alpha must be finite, got {self.alpha}")
        object.__setattr__(self, "alpha", float(self.alpha))

    @property
    def skew(self) -> float:
        """The Kannala-Brandt skew coefficient."""
        return self.alpha

    def distortion_names(self) -> Tuple[str, ...]:
        """Names of the four Kannala-Brandt coefficients."""
        return ("k1", "k2", "k3", "k4")

    def parameter_names(self) -> Tuple[str, ...]:
        """Full parameter vector, with skew last so distortion keeps its offset."""
        return super().parameter_names() + ("alpha",)

    def to_vector(self) -> np.ndarray:
        """Flatten to `[fx, fy, cx, cy, k1..k4, alpha]`."""
        return np.concatenate([super().to_vector(), [self.alpha]])

    @classmethod
    def from_vector(cls, vector: Sequence[float]) -> "FisheyeKannalaBrandt":
        """Rebuild from `[fx, fy, cx, cy, k1..k4, alpha]`.

        The trailing skew term is optional; it defaults to zero when absent.
        """
        values = np.asarray(vector, dtype=float).reshape(-1)
        if values.size not in (8, 9):
            raise ValidationError(
                f"fisheye parameter vector must hold 8 or 9 values, got {values.size}"
            )
        alpha = float(values[8]) if values.size == 9 else 0.0
        return cls(values[0], values[1], values[2], values[3], values[4:8], alpha)

    def to_dict(self) -> Dict[str, Any]:
        """Serialise including the skew term."""
        payload = super().to_dict()
        payload["alpha"] = self.alpha
        return payload


def camera_from_dict(payload: Mapping[str, Any]) -> CameraModel:
    """Rebuild a camera model from its serialised form.

    Args:
        payload: A mapping as produced by `CameraModel.to_dict`.

    Returns:
        The reconstructed model.

    Raises:
        ValidationError: `kind` is missing or names no registered model.
    """
    kind = payload.get("kind")
    if kind not in _MODELS:
        raise ValidationError(
            f"unknown camera model {kind!r}; expected one of {sorted(_MODELS)}"
        )
    cls = _MODELS[kind]
    fields = {
        "fx": payload["fx"],
        "fy": payload["fy"],
        "cx": payload["cx"],
        "cy": payload["cy"],
        "distortion": np.asarray(payload["distortion"], dtype=float),
    }
    if cls is FisheyeKannalaBrandt:
        fields["alpha"] = float(payload.get("alpha", 0.0))
    return cls(**fields)


def registered_models() -> Tuple[str, ...]:
    """Names of every registered camera model."""
    return tuple(sorted(_MODELS))
