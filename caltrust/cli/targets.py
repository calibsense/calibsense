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

"""Describing a calibration target on the command line.

Two ways in. A YAML or JSON file is the durable one, and the shorthand is for
the common case where opening an editor to write four lines is friction:

    checkerboard:9x6:25mm
    charuco:8x11:20mm:15mm:DICT_5X5_1000
    circles:4x11:20mm:asymmetric

Units may be attached to any length, and default to millimetres.
"""

from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, Tuple

import yaml

from .. import units as units_mod
from ..core.target import CharucoBoard, Checkerboard, CircleGrid, TargetSpec, target_from_dict
from ..errors import ValidationError

_LENGTH = re.compile(r"^\s*([0-9]*\.?[0-9]+)\s*([a-zA-Z]*)\s*$")
_GRID = re.compile(r"^\s*(\d+)\s*[xX*]\s*(\d+)\s*$")

#: Shorthand names accepted for each target kind.
ALIASES = {
    "checkerboard": "checkerboard",
    "chessboard": "checkerboard",
    "checker": "checkerboard",
    "charuco": "charuco",
    "circles": "circle_grid",
    "circle_grid": "circle_grid",
    "circlegrid": "circle_grid",
}

SHORTHAND_HELP = (
    "checkerboard:COLSxROWS:SQUARE  |  "
    "charuco:SQXxSQY:SQUARE:MARKER[:DICT][:legacy]  |  "
    "circles:COLSxROWS:SPACING[:symmetric|asymmetric]  "
    "(lengths may carry a unit, e.g. 25mm, 2.5cm, 1in; default mm)"
)


def _length(text: str, what: str) -> Tuple[float, str]:
    match = _LENGTH.match(text)
    if not match:
        raise ValidationError(f"{what}: {text!r} is not a length, expected e.g. 25mm")
    value = float(match.group(1))
    unit = match.group(2) or units_mod.CANONICAL
    return value, units_mod.normalise(unit)


def _grid(text: str, what: str) -> Tuple[int, int]:
    match = _GRID.match(text)
    if not match:
        raise ValidationError(
            f"{what}: {text!r} is not a grid size, expected e.g. 9x6"
        )
    return int(match.group(1)), int(match.group(2))


def parse_shorthand(text: str) -> TargetSpec:
    """Build a target from the colon-separated shorthand.

    Args:
        text: A shorthand string, for example `"checkerboard:9x6:25mm"`.

    Returns:
        The target it describes.

    Raises:
        ValidationError: The shorthand is malformed or names an unknown kind.
    """
    parts = [p for p in text.split(":")]
    kind = ALIASES.get(parts[0].strip().lower())
    if kind is None:
        raise ValidationError(
            f"unknown target kind {parts[0]!r}; expected one of "
            f"{sorted(set(ALIASES))}. Shorthand: {SHORTHAND_HELP}"
        )
    if kind == "checkerboard":
        if len(parts) != 3:
            raise ValidationError(
                f"checkerboard shorthand takes 2 fields, got {len(parts) - 1}: "
                "checkerboard:COLSxROWS:SQUARE"
            )
        columns, rows = _grid(parts[1], "checkerboard grid")
        square, unit = _length(parts[2], "square size")
        return Checkerboard(columns, rows, square, units=unit)
    if kind == "circle_grid":
        if len(parts) not in (3, 4):
            raise ValidationError(
                f"circles shorthand takes 2 or 3 fields, got {len(parts) - 1}: "
                "circles:COLSxROWS:SPACING[:symmetric|asymmetric]"
            )
        columns, rows = _grid(parts[1], "circle grid")
        spacing, unit = _length(parts[2], "spacing")
        layout = parts[3].strip().lower() if len(parts) == 4 else "asymmetric"
        if layout not in ("symmetric", "asymmetric"):
            raise ValidationError(
                f"circle layout must be 'symmetric' or 'asymmetric', got {layout!r}"
            )
        return CircleGrid(columns, rows, spacing, layout == "asymmetric", units=unit)

    if len(parts) < 4:
        raise ValidationError(
            f"charuco shorthand takes at least 3 fields, got {len(parts) - 1}: "
            "charuco:SQXxSQY:SQUARE:MARKER[:DICT][:legacy]"
        )
    squares_x, squares_y = _grid(parts[1], "charuco grid")
    square, unit = _length(parts[2], "square size")
    marker, marker_unit = _length(parts[3], "marker size")
    if marker_unit != unit:
        marker = units_mod.from_mm(units_mod.to_mm(marker, marker_unit), unit)
    extras = [p.strip() for p in parts[4:] if p.strip()]
    legacy = any(e.lower() == "legacy" for e in extras)
    dictionaries = [e for e in extras if e.lower() != "legacy"]
    if len(dictionaries) > 1:
        raise ValidationError(f"charuco shorthand names {len(dictionaries)} dictionaries")
    dictionary = dictionaries[0] if dictionaries else "DICT_5X5_1000"
    return CharucoBoard(
        squares_x, squares_y, square, marker, dictionary, legacy, units=unit
    )


def load_target_file(path: str) -> TargetSpec:
    """Read a target description from a YAML or JSON file.

    Args:
        path: The file to read.

    Returns:
        The target it describes.

    Raises:
        ValidationError: The file will not parse, or does not describe a target.
    """
    try:
        with open(path, "r", encoding="utf-8") as handle:
            payload: Any = (
                json.load(handle)
                if path.lower().endswith(".json")
                else yaml.safe_load(handle)
            )
    except (OSError, yaml.YAMLError, json.JSONDecodeError) as exc:
        raise ValidationError(f"could not parse target file {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValidationError(f"{path} must hold a mapping describing one target")
    fields: Dict[str, Any] = dict(payload.get("target", payload))
    if "kind" in fields:
        fields["kind"] = ALIASES.get(str(fields["kind"]).lower(), fields["kind"])
    return target_from_dict(fields)


def resolve_target(text: str) -> TargetSpec:
    """Build a target from either a file path or the shorthand.

    A value that names an existing file is read as one; anything else is parsed
    as shorthand. That order means a file called `checkerboard:9x6:25mm` still
    works, and a typo in a path produces a shorthand error rather than silently
    doing nothing.

    Args:
        text: A file path or a shorthand string.

    Returns:
        The target it describes.
    """
    if os.path.isfile(text):
        return load_target_file(text)
    return parse_shorthand(text)
