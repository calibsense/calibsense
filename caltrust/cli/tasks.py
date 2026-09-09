"""Describing a measurement task on the command line.

The same shorthand idea as the target spec, for the same reason: an engineer
asking "what is my error at 800 mm" should not have to write a config file to
find out.

    length:800mm:100mm          a 100 mm feature at 800 mm
    plane:800mm:25deg           a plane at 800 mm, tilted 25 degrees
    stereo:800mm:200mm          a point at 800 mm on a 200 mm baseline
    base:800mm                  a feature at 800 mm, in the robot base frame
"""

from __future__ import annotations

import re
from typing import Any, Optional, Sequence, Tuple

import numpy as np

from .. import units as units_mod
from ..errors import ValidationError
from ..task.tasks import CameraToBase, LengthAtDepth, PlaneLocation, StereoTriangulation
from .targets import _length

_ANGLE = re.compile(r"^\s*(-?[0-9]*\.?[0-9]+)\s*(deg|degrees|rad|radians)?\s*$")

#: Shorthand names accepted for each task kind.
ALIASES = {
    "length": "length",
    "distance": "length",
    "plane": "plane",
    "stereo": "stereo",
    "triangulate": "stereo",
    "base": "base",
    "robot": "base",
}

SHORTHAND_HELP = (
    "length:DEPTH:SIZE  |  plane:DEPTH[:TILT]  |  stereo:DEPTH:BASELINE  |  "
    "base:DEPTH  (lengths may carry a unit, e.g. 800mm, 0.8m; tilt in degrees)"
)


def _angle_degrees(text: str, what: str) -> float:
    """Parse an angle, defaulting to degrees.

    Args:
        text: The angle, for example `"25deg"` or `"0.4rad"`.
        what: Field name, for the error message.

    Returns:
        The angle in degrees.

    Raises:
        ValidationError: The text is not an angle.
    """
    match = _ANGLE.match(text)
    if not match:
        raise ValidationError(f"{what}: {text!r} is not an angle, expected e.g. 25deg")
    value = float(match.group(1))
    unit = (match.group(2) or "deg").lower()
    return float(np.degrees(value)) if unit.startswith("rad") else value


def parse_task(
    text: str, hand_eye: Optional[Any] = None, flange: Optional[Any] = None
) -> Any:
    """Build a task from the colon-separated shorthand.

    Args:
        text: A shorthand string, for example `"length:800mm:100mm"`.
        hand_eye: A solved hand-eye result, required by the `base` task.
        flange: The flange pose at measurement time, for an eye-in-hand `base`
            task.

    Returns:
        The task it describes.

    Raises:
        ValidationError: The shorthand is malformed or names an unknown kind.
    """
    parts = text.split(":")
    kind = ALIASES.get(parts[0].strip().lower())
    if kind is None:
        raise ValidationError(
            f"unknown task kind {parts[0]!r}; expected one of "
            f"{sorted(set(ALIASES))}. Shorthand: {SHORTHAND_HELP}"
        )

    def depth_mm(index: int) -> float:
        value, unit = _length(parts[index], "depth")
        return units_mod.to_mm(value, unit)

    if kind == "length":
        if len(parts) != 3:
            raise ValidationError(
                f"length shorthand takes 2 fields, got {len(parts) - 1}: "
                "length:DEPTH:SIZE"
            )
        size, size_unit = _length(parts[2], "feature size")
        return LengthAtDepth(
            depth_mm=depth_mm(1), length_mm=units_mod.to_mm(size, size_unit)
        )
    if kind == "plane":
        if len(parts) not in (2, 3):
            raise ValidationError(
                f"plane shorthand takes 1 or 2 fields, got {len(parts) - 1}: "
                "plane:DEPTH[:TILT]"
            )
        tilt = _angle_degrees(parts[2], "tilt") if len(parts) == 3 else 25.0
        return PlaneLocation(depth_mm=depth_mm(1), tilt_deg=tilt)
    if kind == "stereo":
        if len(parts) != 3:
            raise ValidationError(
                f"stereo shorthand takes 2 fields, got {len(parts) - 1}: "
                "stereo:DEPTH:BASELINE"
            )
        baseline, baseline_unit = _length(parts[2], "baseline")
        return StereoTriangulation(
            baseline_mm=units_mod.to_mm(baseline, baseline_unit), depth_mm=depth_mm(1)
        )
    if len(parts) != 2:
        raise ValidationError(
            f"base shorthand takes 1 field, got {len(parts) - 1}: base:DEPTH"
        )
    if hand_eye is None:
        raise ValidationError(
            "the base-frame task needs a hand-eye solve; pass --mounting and a "
            "session that carries robot poses"
        )
    return CameraToBase(hand_eye_result=hand_eye, depth_mm=depth_mm(1), flange=flange)


def parse_tasks(
    texts: Sequence[str], hand_eye: Optional[Any] = None, flange: Optional[Any] = None
) -> Tuple[Any, ...]:
    """Build several tasks from shorthand strings.

    Args:
        texts: The shorthand strings.
        hand_eye: A solved hand-eye result, for a `base` task.
        flange: The flange pose, for an eye-in-hand `base` task.

    Returns:
        One task per string, in order.
    """
    return tuple(parse_task(text, hand_eye, flange) for text in texts)
