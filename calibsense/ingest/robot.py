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

"""Reading robot poses for a later hand-eye calibration.

Robot loggers export either a homogeneous matrix, a translation with a
quaternion, or a translation with a rotation vector, in either millimetres or
metres, in either hand-eye direction. All six combinations are accepted and
normalised on the way in, because getting any one of them wrong produces a
hand-eye result that is wrong rather than obviously broken.
"""

from __future__ import annotations

import csv
import json
import os
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from .. import units as units_mod
from ..core.poses import Pose, quaternion_to_matrix, rotvec_to_matrix
from ..core.session import HAND_EYE_CONVENTIONS, RobotPoses
from ..errors import UnsupportedFormatError, ValidationError

FORMAT = "calibsense.robot_poses"

_VIEW_COLUMNS = ("view_id", "view", "id", "image", "frame", "name")
_POSITION_COLUMNS = (("x", "y", "z"), ("tx", "ty", "tz"), ("px", "py", "pz"))
_QUATERNION_COLUMNS = (("qx", "qy", "qz", "qw"), ("quat_x", "quat_y", "quat_z", "quat_w"))
_ROTVEC_COLUMNS = (("rx", "ry", "rz"), ("rvec_x", "rvec_y", "rvec_z"))


def read_robot_poses(
    path: str,
    convention: str = "gripper2base",
    units: str = "mm",
    scalar_first: bool = False,
) -> RobotPoses:
    """Read robot poses from a JSON or CSV file.

    Args:
        path: The file to read.
        convention: `"gripper2base"` or `"base2gripper"`. A file may override
            this with its own `convention` key.
        units: Length unit of the translations. A file may override this too.
        scalar_first: Read quaternions as `(w, x, y, z)` rather than
            `(x, y, z, w)`.

    Returns:
        The poses, normalised to gripper-to-base and millimetres.

    Raises:
        UnsupportedFormatError: The file is missing or its shape is unrecognised.
        ValidationError: A pose is not a valid rigid transform.
    """
    if not os.path.isfile(path):
        raise UnsupportedFormatError(f"no such robot pose file: {path}")
    suffix = os.path.splitext(path)[1].lower()
    if suffix == ".json":
        view_ids, matrices, convention, units = _read_json(
            path, convention, units, scalar_first
        )
    elif suffix in (".csv", ".tsv", ".txt"):
        view_ids, matrices = _read_csv(path, units, scalar_first, suffix)
    else:
        raise UnsupportedFormatError(
            f"cannot read robot poses from {suffix or 'a file with no extension'}; "
            "expected .json or .csv"
        )
    if convention not in HAND_EYE_CONVENTIONS:
        raise ValidationError(
            f"unknown hand-eye convention {convention!r}; "
            f"expected one of {list(HAND_EYE_CONVENTIONS)}"
        )
    return RobotPoses.from_matrices(
        matrices, view_ids, convention, source=os.path.abspath(path)
    )


def write_robot_poses(poses: RobotPoses, path: str) -> str:
    """Write robot poses as calibsense JSON, in millimetres and gripper-to-base.

    Args:
        poses: The poses to write.
        path: Destination path.

    Returns:
        The path written.
    """
    payload = {
        "format": FORMAT,
        "convention": "gripper2base",
        "units": "mm",
        "poses": [
            {"view_id": view_id, "matrix": pose.matrix.tolist()}
            for view_id, pose in zip(poses.view_ids, poses.poses)
        ],
    }
    try:
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
            handle.write("\n")
    except OSError as exc:
        raise UnsupportedFormatError(f"could not write {path}: {exc}") from exc
    return path


def _matrix_from_entry(
    entry: Mapping[str, Any], scale: float, scalar_first: bool, where: str
) -> np.ndarray:
    if "matrix" in entry:
        matrix = np.asarray(entry["matrix"], dtype=float)
        if matrix.shape != (4, 4):
            raise ValidationError(f"{where}: matrix must be 4x4, got {matrix.shape}")
        matrix = matrix.copy()
        matrix[:3, 3] *= scale
        return matrix
    translation = entry.get("translation", entry.get("tvec", entry.get("position")))
    if translation is None:
        raise ValidationError(
            f"{where}: needs either 'matrix' or a translation with an orientation"
        )
    translation = np.asarray(translation, dtype=float).reshape(-1) * scale
    if "quaternion" in entry:
        rotation = quaternion_to_matrix(entry["quaternion"], scalar_first)
    elif "rvec" in entry or "rotation_vector" in entry:
        rotation = rotvec_to_matrix(entry.get("rvec", entry.get("rotation_vector")))
    elif "rotation" in entry:
        rotation = np.asarray(entry["rotation"], dtype=float).reshape(3, 3)
    else:
        raise ValidationError(
            f"{where}: needs one of 'quaternion', 'rvec' or 'rotation'"
        )
    return Pose(rotation, translation).matrix


def _read_json(
    path: str, convention: str, units: str, scalar_first: bool
) -> Tuple[List[str], List[np.ndarray], str, str]:
    try:
        with open(path, "r", encoding="utf-8") as handle:
            payload: Dict[str, Any] = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise UnsupportedFormatError(f"could not parse {path}: {exc}") from exc
    entries = payload.get("poses") if isinstance(payload, Mapping) else payload
    if not isinstance(entries, Sequence) or not entries:
        raise UnsupportedFormatError(
            f"{path} has no 'poses' list; expected an object with one, or a bare list"
        )
    if isinstance(payload, Mapping):
        convention = payload.get("convention", convention)
        units = payload.get("units", units)
    scale = units_mod.to_mm(1.0, units)
    view_ids, matrices = [], []
    for index, entry in enumerate(entries):
        where = f"{path}[{index}]"
        view_ids.append(str(entry.get("view_id", entry.get("view", f"view{index:04d}"))))
        matrices.append(_matrix_from_entry(entry, scale, scalar_first, where))
    return view_ids, matrices, convention, units


def _pick(header: Sequence[str], candidates: Sequence[Sequence[str]]) -> Optional[Sequence[str]]:
    lowered = [h.strip().lower() for h in header]
    for group in candidates:
        if all(name in lowered for name in group):
            return group
    return None


def _read_csv(
    path: str, units: str, scalar_first: bool, suffix: str
) -> Tuple[List[str], List[np.ndarray]]:
    delimiter = "\t" if suffix == ".tsv" else ","
    try:
        with open(path, "r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle, delimiter=delimiter))
    except OSError as exc:
        raise UnsupportedFormatError(f"could not read {path}: {exc}") from exc
    if not rows:
        raise UnsupportedFormatError(f"{path} has no data rows")
    header = [h for h in rows[0].keys() if h is not None]
    lowered = {h.strip().lower(): h for h in header}

    view_column = next((lowered[c] for c in _VIEW_COLUMNS if c in lowered), None)
    if view_column is None:
        raise UnsupportedFormatError(
            f"{path}: no view id column; expected one of {list(_VIEW_COLUMNS)}, "
            f"header is {header}"
        )
    position = _pick(header, _POSITION_COLUMNS)
    if position is None:
        raise UnsupportedFormatError(
            f"{path}: no position columns; expected one of "
            f"{[list(g) for g in _POSITION_COLUMNS]}, header is {header}"
        )
    quaternion = _pick(header, _QUATERNION_COLUMNS)
    rotvec = _pick(header, _ROTVEC_COLUMNS)
    if quaternion is None and rotvec is None:
        raise UnsupportedFormatError(
            f"{path}: no orientation columns; expected qx,qy,qz,qw or rx,ry,rz, "
            f"header is {header}"
        )
    scale = units_mod.to_mm(1.0, units)
    view_ids, matrices = [], []
    for index, row in enumerate(rows):
        where = f"{path}:{index + 2}"
        try:
            translation = np.array([float(row[lowered[c]]) for c in position]) * scale
            if quaternion is not None:
                rotation = quaternion_to_matrix(
                    [float(row[lowered[c]]) for c in quaternion], scalar_first
                )
            else:
                rotation = rotvec_to_matrix([float(row[lowered[c]]) for c in rotvec])
        except (TypeError, ValueError) as exc:
            raise ValidationError(f"{where}: could not parse a number: {exc}") from exc
        view_ids.append(str(row[view_column]).strip())
        matrices.append(Pose(rotation, translation).matrix)
    return view_ids, matrices
