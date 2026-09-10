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

"""The diagnostic contract.

Each cause from the problem statement gets one diagnostic, and each diagnostic
returns one `Finding` whether or not it found anything. A diagnostic that stays
silent when everything is fine is a diagnostic nobody trusts when it speaks, so
the `OK` findings are produced too and the report decides what to show.

Every diagnostic reads only from a `DiagnosticContext`. That keeps the derived
geometry — board normals, tilts, working distances — computed once and computed
the same way for all of them.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import IntEnum
from functools import cached_property
from typing import Any, ClassVar, Dict, List, Mapping, Optional, Set, Tuple

import numpy as np

from ..core.observations import ObservationSet
from ..refit.result import InstrumentedFit
from ..validate.result import CrossValidation

#: Participation in the weak subspace above which a parameter is called
#: undetermined when naming a cause.
WEAK_PARAMETER_SHARE = 0.3


class Severity(IntEnum):
    """How much a finding should worry the reader.

    Ordered so that `max()` over findings gives the overall verdict.
    """

    OK = 0
    NOTE = 1
    WARNING = 2
    CRITICAL = 3

    @property
    def label(self) -> str:
        """A short uppercase label for report output."""
        return self.name


@dataclass(frozen=True)
class Finding:
    """One diagnostic's verdict.

    Attributes:
        cause: Stable slug, for example `"depth_variation"`.
        title: Short human name.
        severity: How much it matters.
        summary: What was measured, in one sentence, with the numbers in it.
        action: What to do about it. Empty when the severity is `OK`.
        metrics: The numbers behind the summary, for machine consumption.
    """

    cause: str
    title: str
    severity: Severity
    summary: str
    action: str = ""
    metrics: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serialise to a plain JSON-compatible dictionary."""
        return {
            "cause": self.cause,
            "title": self.title,
            "severity": self.severity.label,
            "summary": self.summary,
            "action": self.action,
            "metrics": dict(self.metrics),
        }


@dataclass(frozen=True)
class DiagnosticContext:
    """Everything a diagnostic is allowed to look at.

    Attributes:
        fit: The instrumented refit.
        observations: The detections the fit was made from.
        cross_validation: Out-of-sample results, when they were computed.
    """

    fit: InstrumentedFit
    observations: ObservationSet
    cross_validation: Optional[CrossValidation] = None

    @cached_property
    def board_normals(self) -> np.ndarray:
        """Unit board normals in camera coordinates, one row per view.

        The board's own z axis, so `R[:, 2]`. Its sign is not meaningful — a
        plane has no front — so anything comparing normals folds by absolute
        value.
        """
        return np.stack([pose.rotation[:, 2] for pose in self.fit.poses])

    @cached_property
    def tilt_degrees(self) -> np.ndarray:
        """Angle between each board normal and the optical axis, in degrees.

        Folded into `[0, 90]`, so zero means the board is exactly parallel to
        the image plane and ninety means edge-on.
        """
        cosine = np.clip(np.abs(self.board_normals[:, 2]), 0.0, 1.0)
        return np.degrees(np.arccos(cosine))

    @cached_property
    def board_distances_mm(self) -> np.ndarray:
        """Distance from the camera to each board's *centre*, in millimetres.

        Measured to the centre of the point pattern rather than to the board
        frame's origin corner, which for a 200 mm board differs by over a
        hundred millimetres and would distort a depth ratio.
        """
        centre = self.observations.target.object_points().mean(axis=0).reshape(1, 3)
        return np.array(
            [float(np.linalg.norm(pose.apply(centre)[0])) for pose in self.fit.poses]
        )

    @cached_property
    def all_image_points(self) -> np.ndarray:
        """Every detected corner from every view, pooled, shape `(n, 2)`."""
        return np.concatenate([v.image_points for v in self.observations.views])

    @cached_property
    def weak_parameters(self) -> Set[str]:
        """Names of parameters the capture does not determine.

        Empty when the fit is identifiable. Diagnostics use this to say *why* a
        parameter is undetermined rather than merely that it is.
        """
        conditioning = self.fit.conditioning
        if conditioning.identifiable:
            return set()
        return {
            name
            for name, share in zip(
                self.fit.covariance.intrinsic_names, conditioning.participation
            )
            if share > WEAK_PARAMETER_SHARE
        }

    @property
    def n_views(self) -> int:
        """Number of views."""
        return self.observations.n_views

    @property
    def image_size(self) -> Tuple[int, int]:
        """Frame size as `(width, height)`."""
        return self.observations.image_size


class Diagnostic(ABC):
    """Detects and names one cause of a bad calibration."""

    #: Stable slug identifying the cause.
    cause: ClassVar[str] = ""
    #: Short human name.
    title: ClassVar[str] = ""

    @abstractmethod
    def run(self, context: DiagnosticContext) -> Finding:
        """Measure this cause.

        Args:
            context: The fit and its detections.

        Returns:
            A finding, with severity `OK` when nothing is wrong.
        """

    def _finding(
        self,
        severity: Severity,
        summary: str,
        action: str = "",
        **metrics: Any,
    ) -> Finding:
        """Build a finding for this diagnostic's cause."""
        return Finding(
            cause=self.cause,
            title=self.title,
            severity=severity,
            summary=summary,
            action=action,
            metrics=metrics,
        )


def worst(findings: Tuple[Finding, ...]) -> Severity:
    """The highest severity across findings.

    Args:
        findings: The findings to reduce.

    Returns:
        `Severity.OK` for an empty set, otherwise the worst severity present.
    """
    return max((f.severity for f in findings), default=Severity.OK)
