"""Target detectors, one per target kind.

The registry is a literal dictionary rather than a plugin scan so that a frozen
PyInstaller build resolves every detector without touching the filesystem.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple, Type

from ...core.target import TargetSpec
from ...errors import ValidationError
from .base import DetectorOptions, TargetDetector
from .charuco import CharucoDetector
from .checkerboard import CheckerboardDetector
from .circlegrid import CircleGridDetector

_DETECTORS: Dict[str, Type[TargetDetector]] = {
    CheckerboardDetector.kind: CheckerboardDetector,
    CircleGridDetector.kind: CircleGridDetector,
    CharucoDetector.kind: CharucoDetector,
}


def detector_for(
    target: TargetSpec, options: Optional[DetectorOptions] = None
) -> TargetDetector:
    """Build the detector that handles a target.

    Args:
        target: The target to detect.
        options: Shared detector settings; defaults are used when omitted.

    Returns:
        A ready-to-use detector.

    Raises:
        ValidationError: No detector handles this target kind.
    """
    try:
        cls = _DETECTORS[target.kind]
    except KeyError:
        raise ValidationError(
            f"no detector for target kind {target.kind!r}; "
            f"have {sorted(_DETECTORS)}"
        ) from None
    return cls(target, options)


def registered_detectors() -> Tuple[str, ...]:
    """Target kinds that can be detected."""
    return tuple(sorted(_DETECTORS))


__all__ = [
    "CharucoDetector",
    "CheckerboardDetector",
    "CircleGridDetector",
    "DetectorOptions",
    "TargetDetector",
    "detector_for",
    "registered_detectors",
]
