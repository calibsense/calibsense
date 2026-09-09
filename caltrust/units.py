"""Length units.

caltrust works internally in millimetres. A target is described in whatever unit
the engineer measured it in, and is converted once, at ingest, so that every
downstream number — residuals, covariance, task-space error — carries the same
unit without a second conversion anywhere.
"""

from __future__ import annotations

from .errors import ValidationError

CANONICAL = "mm"

_TO_MM = {
    "mm": 1.0,
    "millimetre": 1.0,
    "millimeter": 1.0,
    "cm": 10.0,
    "centimetre": 10.0,
    "centimeter": 10.0,
    "m": 1000.0,
    "metre": 1000.0,
    "meter": 1000.0,
    "in": 25.4,
    "inch": 25.4,
    "um": 1e-3,
    "micron": 1e-3,
}


def normalise(unit: str) -> str:
    """Return the canonical spelling of a unit name.

    Args:
        unit: A unit name such as `"mm"`, `"Meter"` or `"inch"`.

    Returns:
        One of the keys of the conversion table, lowercased and stripped.

    Raises:
        ValidationError: The unit is not one caltrust knows.
    """
    key = unit.strip().lower()
    if key not in _TO_MM:
        raise ValidationError(
            f"unknown length unit {unit!r}; expected one of {sorted(_TO_MM)}"
        )
    return key


def to_mm(value: float, unit: str) -> float:
    """Convert a length to millimetres.

    Args:
        value: The length in `unit`.
        unit: The unit `value` is expressed in.

    Returns:
        The same length in millimetres.
    """
    return float(value) * _TO_MM[normalise(unit)]


def from_mm(value_mm: float, unit: str) -> float:
    """Convert a length in millimetres into `unit`.

    Args:
        value_mm: The length in millimetres.
        unit: The unit to express it in.

    Returns:
        The length in `unit`.
    """
    return float(value_mm) / _TO_MM[normalise(unit)]
