"""Reading detections that were produced elsewhere.

The second ingest path: an engineer who already ran a calibration usually still
has the corner arrays that produced it. Re-using them is better than re-running
a detector, because the audit then describes the calibration they actually
shipped rather than a near-identical one.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np

from ..core.observations import DetectionSummary, ObservationSet, ViewObservations
from ..core.target import TargetSpec, target_from_dict
from ..errors import UnsupportedFormatError, ValidationError

FORMAT = "caltrust.detections"


def write_detections(observations: ObservationSet, path: str) -> str:
    """Write detections as caltrust JSON.

    Args:
        observations: The views to write.
        path: Destination path.

    Returns:
        The path written.

    Raises:
        UnsupportedFormatError: The file could not be written.
    """
    payload = {
        "format": FORMAT,
        "target": observations.target.to_dict(),
        "image_size": list(observations.image_size),
        "views": [
            {
                "view_id": view.view_id,
                "source": view.source,
                "point_ids": view.point_ids.tolist(),
                "image_points": view.image_points.tolist(),
            }
            for view in observations.views
        ],
    }
    try:
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
            handle.write("\n")
    except OSError as exc:
        raise UnsupportedFormatError(f"could not write {path}: {exc}") from exc
    return path


def read_detections(
    path: str,
    target: Optional[TargetSpec] = None,
    image_size: Optional[Tuple[int, int]] = None,
) -> ObservationSet:
    """Read detections from a JSON or NPZ file.

    Two shapes are accepted. A caltrust JSON file carries its own target and
    image size. A plain NPZ carries `image_points` of shape `(views, points, 2)`
    and optionally `point_ids`, in which case the target and image size must be
    supplied by the caller.

    Args:
        path: The file to read.
        target: The target the points belong to. Required when the file does not
            name one; overrides the file's own when both are present.
        image_size: Frame size as `(width, height)`. Same rule as `target`.

    Returns:
        The reconstructed observation set.

    Raises:
        UnsupportedFormatError: The file is missing, unparseable, or has a shape
            caltrust does not recognise.
        ValidationError: The file parsed but describes an inconsistent set.
    """
    if not os.path.isfile(path):
        raise UnsupportedFormatError(f"no such detections file: {path}")
    suffix = os.path.splitext(path)[1].lower()
    if suffix == ".json":
        return _read_json(path, target, image_size)
    if suffix in (".npz", ".npy"):
        return _read_npz(path, target, image_size)
    raise UnsupportedFormatError(
        f"cannot read detections from {suffix or 'a file with no extension'}; "
        "expected .json or .npz"
    )


def _resolve(
    from_file: Optional[Any], override: Optional[Any], what: str, path: str
) -> Any:
    value = override if override is not None else from_file
    if value is None:
        raise ValidationError(
            f"{path} does not record the {what}; pass it explicitly"
        )
    return value


def _read_json(
    path: str, target: Optional[TargetSpec], image_size: Optional[Tuple[int, int]]
) -> ObservationSet:
    try:
        with open(path, "r", encoding="utf-8") as handle:
            payload: Dict[str, Any] = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise UnsupportedFormatError(f"could not parse {path}: {exc}") from exc
    if not isinstance(payload, Mapping) or "views" not in payload:
        raise UnsupportedFormatError(
            f"{path} is not a detections file; expected an object with a 'views' list"
        )
    file_target = (
        target_from_dict(payload["target"]) if payload.get("target") else None
    )
    file_size = tuple(payload["image_size"]) if payload.get("image_size") else None
    resolved_target = _resolve(file_target, target, "target", path)
    resolved_size = _resolve(file_size, image_size, "image size", path)

    views = []
    for index, entry in enumerate(payload["views"]):
        points = np.asarray(entry["image_points"], dtype=float).reshape(-1, 2)
        ids = entry.get("point_ids")
        views.append(
            ViewObservations(
                view_id=str(entry.get("view_id", f"view{index:04d}")),
                point_ids=(
                    np.arange(points.shape[0]) if ids is None
                    else np.asarray(ids, dtype=np.int64).reshape(-1)
                ),
                image_points=points,
                source=entry.get("source"),
            )
        )
    return ObservationSet(
        target=resolved_target,
        image_size=tuple(int(v) for v in resolved_size),
        views=tuple(views),
        summary=DetectionSummary(attempted=len(views), detector="imported"),
    )


def _read_npz(
    path: str, target: Optional[TargetSpec], image_size: Optional[Tuple[int, int]]
) -> ObservationSet:
    try:
        with np.load(path, allow_pickle=False) as archive:
            arrays = {name: archive[name] for name in archive.files}
    except (OSError, ValueError) as exc:
        raise UnsupportedFormatError(f"could not read {path}: {exc}") from exc
    if "image_points" not in arrays:
        raise UnsupportedFormatError(
            f"{path} has no 'image_points' array; it holds {sorted(arrays)}"
        )
    points = np.asarray(arrays["image_points"], dtype=float)
    if points.ndim != 3 or points.shape[2] != 2:
        raise UnsupportedFormatError(
            f"{path}: image_points must be (views, points, 2), got shape {points.shape}"
        )
    n_views, n_points = points.shape[0], points.shape[1]
    ids = arrays.get("point_ids")
    if ids is None:
        ids = np.tile(np.arange(n_points, dtype=np.int64), (n_views, 1))
    else:
        ids = np.asarray(ids, dtype=np.int64)
        if ids.ndim == 1:
            ids = np.tile(ids, (n_views, 1))
        if ids.shape != (n_views, n_points):
            raise UnsupportedFormatError(
                f"{path}: point_ids is {ids.shape}, expected {(n_views, n_points)}"
            )
    resolved_target = _resolve(None, target, "target", path)
    resolved_size = _resolve(
        tuple(arrays["image_size"].tolist()) if "image_size" in arrays else None,
        image_size,
        "image size",
        path,
    )
    views = tuple(
        ViewObservations(f"view{i:04d}", ids[i], points[i]) for i in range(n_views)
    )
    return ObservationSet(
        target=resolved_target,
        image_size=tuple(int(v) for v in resolved_size),
        views=views,
        summary=DetectionSummary(attempted=n_views, detector="imported"),
    )
