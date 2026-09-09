"""The detector contract.

One detector per target kind, all producing the same `ViewObservations`, so
nothing downstream of ingest branches on which board was used.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import ClassVar, Optional

import numpy as np

from ...core.observations import ViewObservations
from ...core.target import TargetSpec
from ...errors import DetectionError, ValidationError


@dataclass(frozen=True)
class DetectorOptions:
    """Knobs shared by every detector.

    Attributes:
        refine: Run sub-pixel refinement where the detector does not already do
            it internally.
        refine_window: Half-width of the `cornerSubPix` search window, in pixels.
        min_points: Fewest points a view must yield to be kept. Four is the
            algebraic minimum for a planar pose; the default is higher because a
            view at the minimum contributes noise rather than information.
        fast: Prefer speed over detection rate. Useful for a first pass over a
            large capture.
    """

    refine: bool = True
    refine_window: int = 5
    min_points: int = 8
    fast: bool = False

    def __post_init__(self) -> None:
        if self.refine_window < 1:
            raise ValidationError(f"refine_window must be >= 1, got {self.refine_window}")
        if self.min_points < 4:
            raise ValidationError(
                f"min_points must be at least 4 to constrain a pose, got {self.min_points}"
            )


class TargetDetector(ABC):
    """Finds one target's points in one image.

    Attributes:
        target: The target being looked for.
        options: Shared detector settings.
    """

    #: Matches the `kind` of the `TargetSpec` this detector handles.
    kind: ClassVar[str] = ""

    def __init__(self, target: TargetSpec, options: Optional[DetectorOptions] = None):
        if target.kind != self.kind:
            raise ValidationError(
                f"{type(self).__name__} handles {self.kind!r} targets, got {target.kind!r}"
            )
        self.target = target
        self.options = options or DetectorOptions()

    @abstractmethod
    def detect(
        self, image: np.ndarray, view_id: str, source: Optional[str] = None
    ) -> ViewObservations:
        """Find the target in one image.

        Args:
            image: A grayscale or colour image.
            view_id: Identifier to attach to the resulting view.
            source: Where the image came from, for provenance.

        Returns:
            The detected points.

        Raises:
            DetectionError: The target was not found, or too few points were.
        """

    def _check_enough(self, count: int, view_id: str) -> None:
        if count < self.options.min_points:
            raise DetectionError(
                f"{view_id}: found {count} points, below the minimum of "
                f"{self.options.min_points}"
            )
