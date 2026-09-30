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

"""Write the refined intrinsics back out in a form another tool can read.

calibsense refits every capture it audits, so it always has intrinsics in hand —
either re-estimated or the ones it was handed. Until now they only ever appeared
as a table in a report, which meant the audit could tell you your calibration
was wrong without giving you the better one it had just computed. These writers
mirror the readers in `ingest.readers`: whatever calibsense can read, it can now
write.

Every format carries the standard deviation of each free intrinsic alongside the
value, because a focal length without one is the number this whole package
exists to argue with. ROS `camera_info` is the exception — the message has
nowhere to put it — so that writer names the omission in a comment rather than
letting the figure travel alone and unqualified.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import yaml

from .._version import __version__
from ..core.camera import FisheyeKannalaBrandt
from ..core.session import CalibrationRecord
from ..errors import UnsupportedFormatError
from ..ingest.readers.native import write_calibration as write_native
from ..refit.result import InstrumentedFit

#: The formats `export_calibration` can write, in the order they are offered.
EXPORT_FORMATS: Tuple[str, ...] = ("opencv", "ros", "native")

#: Extension to format, for the cases where the extension settles it. `.yml` and
#: `.yaml` are deliberately absent: both OpenCV FileStorage and ROS
#: `camera_info` use them, so that pair is resolved by `_DEFAULT_YAML` and can be
#: overridden explicitly.
_BY_EXTENSION = {".json": "native", ".xml": "opencv"}

#: Which of the two YAML formats an unqualified `.yml` means. OpenCV, because
#: that is what `cv2.calibrateCamera` users have on disk already.
_DEFAULT_YAML = "opencv"

#: calibsense's model kinds mapped onto the names ROS `camera_info` uses.
#: Brown-Conrady with more than five coefficients is `rational_polynomial`,
#: which is the name ROS gives the 8-term rational model.
_ROS_MODEL_NAMES = {"fisheye_kannala_brandt": "equidistant"}


def resolve_format(path: str, requested: Optional[str] = None) -> str:
    """Decide which format to write, from an explicit choice or the extension.

    Args:
        path: Destination path, used only for its extension.
        requested: An explicit format name, or `None` to infer one.

    Returns:
        One of `EXPORT_FORMATS`.

    Raises:
        UnsupportedFormatError: `requested` names no known format, or the
            extension matches none and nothing was requested.
    """
    if requested is not None:
        if requested not in EXPORT_FORMATS:
            raise UnsupportedFormatError(
                f"unknown calibration format {requested!r}; "
                f"expected one of {list(EXPORT_FORMATS)}"
            )
        return requested
    suffix = os.path.splitext(path)[1].lower()
    if suffix in _BY_EXTENSION:
        return _BY_EXTENSION[suffix]
    if suffix in (".yml", ".yaml"):
        return _DEFAULT_YAML
    raise UnsupportedFormatError(
        f"cannot tell what format {path!r} should be from its extension; "
        f"pass one of {list(EXPORT_FORMATS)} explicitly"
    )


def export_calibration(
    fit: InstrumentedFit,
    path: str,
    fmt: Optional[str] = None,
    camera_name: str = "camera",
) -> str:
    """Write a fit's intrinsics to a calibration file.

    Args:
        fit: The instrumented fit whose camera to write.
        path: Destination path. Its extension picks the format when `fmt` is
            `None`; `.yml` and `.yaml` mean OpenCV FileStorage, because both it
            and ROS `camera_info` claim them.
        fmt: One of `EXPORT_FORMATS`, or `None` to infer from the extension.
        camera_name: Recorded by the formats that have a field for it.

    Returns:
        The path written.

    Raises:
        UnsupportedFormatError: The format cannot be determined or written.
    """
    chosen = resolve_format(path, fmt)
    writers = {"opencv": _write_opencv, "ros": _write_ros, "native": _write_native}
    return writers[chosen](fit, path, camera_name)


def uncertainty_table(fit: InstrumentedFit) -> Tuple[List[str], List[float]]:
    """Free intrinsic names and their standard deviations, in fit order.

    Fixed parameters are absent rather than listed with a zero, because zero
    would read as "measured to be exact" when it means "not estimated".

    Args:
        fit: The fit to describe.

    Returns:
        `(names, standard_deviations)`, both the same length.
    """
    rows = fit.parameter_table()
    return [name for name, _, _, _ in rows], [float(sd) for _, _, sd, _ in rows]


def _provenance(fit: InstrumentedFit) -> Dict[str, Any]:
    """The facts a reader of the exported file needs to judge it."""
    return {
        "produced_by": f"calibsense {__version__}",
        "calibrated": fit.created,
        "refitted": bool(fit.refitted),
        "views": int(fit.n_views),
        "points": int(fit.residuals.total_points),
        "rms_px": float(fit.rms),
        "sigma_px": float(fit.covariance.sigma),
        "identifiable": bool(fit.conditioning.identifiable),
    }


def _caveat(fit: InstrumentedFit) -> Optional[str]:
    """The one-line warning an unidentifiable fit has to travel with.

    A pseudo-inverse assigns zero variance to a direction the capture does not
    constrain, so the standard deviations in the file are narrower than the
    truth rather than wider. Exporting that silently would reproduce exactly the
    false confidence the audit exists to catch.
    """
    if fit.conditioning.identifiable:
        return None
    return (
        "This capture does not determine every intrinsic. The standard "
        "deviations below are lower bounds, not error bars."
    )


def _write_opencv(fit: InstrumentedFit, path: str, camera_name: str) -> str:
    """Write OpenCV FileStorage, with the distortion model named explicitly.

    OpenCV's own calibration samples record `K` and `D` and nothing that says
    which distortion family produced them, which leaves a four-coefficient file
    genuinely ambiguous between Brown-Conrady and Kannala-Brandt. Writing
    `distortion_model` costs one node and removes the ambiguity for good, and
    calibsense's own reader picks it up.
    """
    camera = fit.camera
    names, deviations = uncertainty_table(fit)
    width, height = fit.image_size

    storage = cv2.FileStorage(path, cv2.FILE_STORAGE_WRITE)
    if not storage.isOpened():
        raise UnsupportedFormatError(f"could not open {path} for writing")
    try:
        storage.write("camera_name", camera_name)
        storage.write("distortion_model", camera.kind)
        storage.write("image_width", int(width))
        storage.write("image_height", int(height))
        storage.write("camera_matrix", np.asarray(camera.camera_matrix, dtype=float))
        storage.write(
            "distortion_coefficients",
            np.asarray(camera.distortion, dtype=float).reshape(-1, 1),
        )
        storage.write("avg_reprojection_error", float(fit.rms))
        storage.write("residual_sigma_px", float(fit.covariance.sigma))
        if names:
            storage.write("parameter_names", " ".join(names))
            storage.write(
                "parameter_std_dev",
                np.asarray(deviations, dtype=float).reshape(-1, 1),
            )
        storage.write("produced_by", f"calibsense {__version__}")
        storage.write("calibrated", fit.created)
        storage.write("identifiable", int(fit.conditioning.identifiable))
        caveat = _caveat(fit)
        if caveat is not None:
            storage.write("warning", caveat)
    finally:
        storage.release()
    return path


def _write_ros(fit: InstrumentedFit, path: str, camera_name: str) -> str:
    """Write ROS `camera_info` YAML.

    The message has no field for parameter uncertainty, and inventing one that
    no consumer reads would be worse than leaving it out, so the standard
    deviations go in a leading comment where a human will see them and a parser
    will not trip over them.
    """
    camera = fit.camera
    matrix = np.asarray(camera.camera_matrix, dtype=float)
    coefficients = np.asarray(camera.distortion, dtype=float).reshape(-1)
    width, height = fit.image_size

    if isinstance(camera, FisheyeKannalaBrandt):
        model = _ROS_MODEL_NAMES[camera.kind]
    else:
        model = "rational_polynomial" if coefficients.size > 5 else "plumb_bob"

    projection = np.zeros((3, 4), dtype=float)
    projection[:, :3] = matrix
    payload = {
        "image_width": int(width),
        "image_height": int(height),
        "camera_name": camera_name,
        "camera_matrix": {"rows": 3, "cols": 3, "data": matrix.reshape(-1).tolist()},
        "distortion_model": model,
        "distortion_coefficients": {
            "rows": 1,
            "cols": int(coefficients.size),
            "data": coefficients.tolist(),
        },
        "rectification_matrix": {
            "rows": 3, "cols": 3, "data": np.eye(3).reshape(-1).tolist()
        },
        "projection_matrix": {
            "rows": 3, "cols": 4, "data": projection.reshape(-1).tolist()
        },
    }

    names, deviations = uncertainty_table(fit)
    comments = [f"# {key}: {value}" for key, value in _provenance(fit).items()]
    if abs(float(matrix[0, 1])) > 0:
        # camera_info has no skew term and every consumer reads K[0][1] as zero,
        # so a calibration carrying one loses it here rather than silently
        # producing a subtly different camera downstream.
        comments.append(
            f"# WARNING: this camera has a skew of {matrix[0, 1]:.6g}, which "
            "camera_info cannot represent and consumers will read as zero. "
            "Export to OpenCV FileStorage or calibsense JSON to keep it."
        )
    comments.append(
        "# camera_info has no field for parameter uncertainty, so the standard"
    )
    comments.append("# deviations calibsense measured are recorded here instead:")
    comments += [
        f"#   {name} = {value:.10g} +/- {sd:.4g}"
        for name, value, sd, _ in fit.parameter_table()
    ]
    caveat = _caveat(fit)
    if caveat is not None:
        comments.append(f"# WARNING: {caveat}")

    try:
        with open(path, "w", encoding="utf-8") as handle:
            handle.write("\n".join(comments) + "\n")
            yaml.safe_dump(payload, handle, sort_keys=False, default_flow_style=False)
    except OSError as exc:
        raise UnsupportedFormatError(f"could not write {path}: {exc}") from exc
    return path


def _write_native(fit: InstrumentedFit, path: str, camera_name: str) -> str:
    """Write calibsense's own JSON, which is the only format that loses nothing."""
    names, deviations = uncertainty_table(fit)
    metadata: Dict[str, Any] = dict(_provenance(fit))
    metadata["camera_name"] = camera_name
    metadata["std_dev"] = dict(zip(names, deviations))
    caveat = _caveat(fit)
    if caveat is not None:
        metadata["warning"] = caveat
    record = CalibrationRecord(
        camera=fit.camera,
        image_size=fit.image_size,
        source=f"calibsense {__version__}",
        reported_rms=float(fit.rms),
        metadata=metadata,
    )
    return write_native(record, path)
