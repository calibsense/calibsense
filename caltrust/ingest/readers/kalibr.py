"""Kalibr camera chain calibrations."""

from __future__ import annotations

import os
from typing import Any, ClassVar, Mapping, Optional

import numpy as np

from ...core.camera import FisheyeKannalaBrandt, PinholeBrownConrady
from ...core.session import CalibrationRecord
from ...errors import UnsupportedFormatError
from .base import CalibrationReader
from .ros import _load_yaml

#: Kalibr distortion model names mapped onto caltrust's two families.
_DISTORTION = {
    "radtan": "pinhole",
    "none": "pinhole",
    "equidistant": "fisheye",
    "fov": None,
}


class KalibrReader(CalibrationReader):
    """Reads a Kalibr `camchain` YAML, one camera at a time.

    A chain describes several cameras. Without a `camera` hint the first entry
    in `cam0`, `cam1`, ... order is taken, and which one was used is recorded in
    the metadata rather than left implicit.
    """

    name: ClassVar[str] = "kalibr"
    description: ClassVar[str] = "Kalibr camchain YAML"

    def __init__(self, camera: Optional[str] = None):
        self.camera = camera

    def sniff(self, path: str, head: str) -> bool:
        """Match Kalibr's characteristic `camN:` block with an `intrinsics` key."""
        if os.path.splitext(path)[1].lower() not in (".yml", ".yaml"):
            return False
        return "cam0:" in head and "intrinsics" in head

    def read(self, path: str) -> CalibrationRecord:
        """Parse one camera out of a Kalibr chain.

        Args:
            path: The file to read.

        Returns:
            The calibration for the selected camera.

        Raises:
            UnsupportedFormatError: The chain is empty, the requested camera is
                absent, or its model is one caltrust cannot represent.
        """
        payload = _load_yaml(path)
        names = sorted(k for k in payload if k.startswith("cam") and isinstance(payload[k], Mapping))
        if not names:
            raise UnsupportedFormatError(f"{path} has no camN entries")
        name = self.camera or names[0]
        if name not in payload:
            raise UnsupportedFormatError(
                f"{path} has no {name!r}; it holds {names}"
            )
        entry: Mapping[str, Any] = payload[name]

        model = str(entry.get("camera_model", "pinhole")).strip().lower()
        if model != "pinhole":
            raise UnsupportedFormatError(
                f"{path}:{name} uses camera_model {model!r}; caltrust supports "
                "'pinhole' with either radtan or equidistant distortion"
            )
        distortion_model = str(entry.get("distortion_model", "none")).strip().lower()
        family = _DISTORTION.get(distortion_model)
        if family is None:
            raise UnsupportedFormatError(
                f"{path}:{name} uses distortion_model {distortion_model!r}, which "
                "caltrust cannot represent"
            )
        intrinsics = np.asarray(entry.get("intrinsics", []), dtype=float).reshape(-1)
        if intrinsics.size != 4:
            raise UnsupportedFormatError(
                f"{path}:{name}: intrinsics must be [fu, fv, pu, pv], got "
                f"{intrinsics.size} values"
            )
        coefficients = np.asarray(entry.get("distortion_coeffs", []), dtype=float).reshape(-1)
        resolution = np.asarray(entry.get("resolution", []), dtype=int).reshape(-1)
        if resolution.size != 2:
            raise UnsupportedFormatError(
                f"{path}:{name}: resolution must be [width, height]"
            )

        fu, fv, pu, pv = intrinsics
        if family == "fisheye":
            if coefficients.size != 4:
                raise UnsupportedFormatError(
                    f"{path}:{name}: equidistant needs 4 coefficients, "
                    f"got {coefficients.size}"
                )
            camera: Any = FisheyeKannalaBrandt(fu, fv, pu, pv, coefficients)
        else:
            if distortion_model == "none":
                coefficients = np.zeros(4)
            elif coefficients.size not in (4, 5):
                raise UnsupportedFormatError(
                    f"{path}:{name}: radtan is [k1, k2, r1, r2], got "
                    f"{coefficients.size} coefficients"
                )
            camera = PinholeBrownConrady(fu, fv, pu, pv, coefficients)
        return CalibrationRecord(
            camera=camera,
            image_size=(int(resolution[0]), int(resolution[1])),
            source=f"kalibr:{os.path.basename(path)}:{name}",
            metadata={
                "declared_model": distortion_model,
                "kalibr_camera": name,
                "cameras_in_chain": names,
                "path": os.path.abspath(path),
            },
        )
