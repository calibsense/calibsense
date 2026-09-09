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

"""ROS `camera_info` calibrations."""

from __future__ import annotations

import os
from typing import Any, ClassVar, Dict, Mapping

import numpy as np
import yaml

from ...core.camera import FisheyeKannalaBrandt, PinholeBrownConrady
from ...core.session import CalibrationRecord
from ...errors import UnsupportedFormatError
from .base import CalibrationReader, looks_like_opencv_filestorage

#: ROS distortion model names mapped onto caltrust's two families.
_MODELS = {
    "plumb_bob": "pinhole",
    "rational_polynomial": "pinhole",
    "equidistant": "fisheye",
    "fisheye": "fisheye",
}


def _matrix(payload: Mapping[str, Any], key: str) -> np.ndarray:
    node = payload.get(key)
    if node is None:
        raise UnsupportedFormatError(f"camera_info is missing {key!r}")
    if isinstance(node, Mapping):
        data = np.asarray(node["data"], dtype=float)
        return data.reshape(int(node.get("rows", 1)), int(node.get("cols", data.size)))
    return np.asarray(node, dtype=float)


class RosCameraInfoReader(CalibrationReader):
    """Reads the `camera_info` YAML that `camera_calibration` writes.

    ROS records the distortion model by name, so unlike OpenCV FileStorage there
    is no ambiguity about which family the coefficients belong to.
    """

    name: ClassVar[str] = "ros"
    description: ClassVar[str] = "ROS camera_info YAML"

    def sniff(self, path: str, head: str) -> bool:
        """Match the pair of keys unique to `camera_info`, but not OpenCV's own YAML."""
        if os.path.splitext(path)[1].lower() not in (".yml", ".yaml"):
            return False
        if looks_like_opencv_filestorage(head):
            return False
        return "camera_matrix" in head and "distortion_model" in head

    def read(self, path: str) -> CalibrationRecord:
        """Parse a `camera_info` YAML file.

        Args:
            path: The file to read.

        Returns:
            The calibration it describes.

        Raises:
            UnsupportedFormatError: The YAML will not parse, a required key is
                absent, or the distortion model is one caltrust cannot represent.
        """
        payload = _load_yaml(path)
        model_name = str(payload.get("distortion_model", "")).strip().lower()
        if model_name not in _MODELS:
            raise UnsupportedFormatError(
                f"{path} declares distortion_model {model_name!r}; caltrust "
                f"supports {sorted(_MODELS)}"
            )
        matrix = _matrix(payload, "camera_matrix")
        if matrix.shape != (3, 3):
            raise UnsupportedFormatError(
                f"{path}: camera_matrix is {matrix.shape}, expected 3x3"
            )
        coefficients = _matrix(payload, "distortion_coefficients").reshape(-1)
        try:
            width = int(payload["image_width"])
            height = int(payload["image_height"])
        except KeyError as exc:
            raise UnsupportedFormatError(f"{path} is missing {exc}") from exc

        if _MODELS[model_name] == "fisheye":
            if coefficients.size < 4:
                raise UnsupportedFormatError(
                    f"{path}: equidistant model needs 4 coefficients, "
                    f"got {coefficients.size}"
                )
            camera: Any = FisheyeKannalaBrandt(
                matrix[0, 0], matrix[1, 1], matrix[0, 2], matrix[1, 2],
                coefficients[:4],
            )
        else:
            camera = PinholeBrownConrady(
                matrix[0, 0], matrix[1, 1], matrix[0, 2], matrix[1, 2], coefficients
            )
        return CalibrationRecord(
            camera=camera,
            image_size=(width, height),
            source=f"ros:{os.path.basename(path)}",
            metadata={
                "declared_model": model_name,
                "camera_name": payload.get("camera_name"),
                "path": os.path.abspath(path),
            },
        )


def _load_yaml(path: str) -> Dict[str, Any]:
    try:
        with open(path, "r", encoding="utf-8") as handle:
            payload = yaml.safe_load(handle)
    except (OSError, yaml.YAMLError) as exc:
        raise UnsupportedFormatError(f"could not parse {path} as YAML: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise UnsupportedFormatError(f"{path} is not a YAML mapping")
    return dict(payload)
