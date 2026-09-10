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

"""Running every diagnostic and ranking what comes back."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from ..core.observations import ObservationSet
from ..errors import ValidationError
from ..refit.result import InstrumentedFit
from ..validate.result import CrossValidation
from .base import Diagnostic, DiagnosticContext, Finding, Severity, worst
from .coverage import ImageCoverage, TargetScale
from .generalisation import OutOfSampleError
from .geometry import DepthVariation, FrontoparallelDominance, PoseDiversity
from .model import DistortionModelAdequacy
from .noise import NoiseModelValidity
from .views import OutlierViews

#: Every diagnostic, in report order. A literal tuple rather than a plugin scan,
#: so a frozen build resolves them all.
DIAGNOSTICS: Tuple[type, ...] = (
    OutOfSampleError,
    PoseDiversity,
    FrontoparallelDominance,
    DepthVariation,
    ImageCoverage,
    TargetScale,
    DistortionModelAdequacy,
    NoiseModelValidity,
    OutlierViews,
)


@dataclass(frozen=True)
class Diagnosis:
    """The result of running every diagnostic.

    Attributes:
        findings: One finding per diagnostic, in report order.
    """

    findings: Tuple[Finding, ...]

    @property
    def severity(self) -> Severity:
        """The worst severity found."""
        return worst(self.findings)

    @property
    def critical(self) -> Tuple[Finding, ...]:
        """Findings that make the calibration untrustworthy."""
        return tuple(f for f in self.findings if f.severity is Severity.CRITICAL)

    @property
    def warnings(self) -> Tuple[Finding, ...]:
        """Findings worth acting on but not disqualifying."""
        return tuple(f for f in self.findings if f.severity is Severity.WARNING)

    @property
    def notes(self) -> Tuple[Finding, ...]:
        """Findings that are informational."""
        return tuple(f for f in self.findings if f.severity is Severity.NOTE)

    @property
    def passing(self) -> Tuple[Finding, ...]:
        """Diagnostics that found nothing wrong."""
        return tuple(f for f in self.findings if f.severity is Severity.OK)

    def ranked(self) -> Tuple[Finding, ...]:
        """Findings worst first, keeping report order within a severity."""
        return tuple(
            sorted(self.findings, key=lambda f: -int(f.severity))
        )

    def by_cause(self, cause: str) -> Finding:
        """Look up one finding by its cause slug.

        Args:
            cause: The slug, for example `"depth_variation"`.

        Returns:
            That diagnostic's finding.

        Raises:
            ValidationError: No diagnostic produced that cause.
        """
        for finding in self.findings:
            if finding.cause == cause:
                return finding
        raise ValidationError(
            f"no finding for cause {cause!r}; have "
            f"{[f.cause for f in self.findings]}"
        )

    def verdict(self) -> str:
        """A one-line overall judgement.

        Deliberately names the findings rather than condemning the calibration
        wholesale. "Image coverage is critical" and "the focal length is not
        determined" are both critical and they invalidate different things; only
        the individual findings can say which.
        """
        if self.critical:
            causes = ", ".join(f.title.lower() for f in self.critical)
            return (
                f"{len(self.critical)} critical finding(s): {causes}. Each is "
                "described below with what it invalidates"
            )
        if self.warnings:
            causes = ", ".join(f.title.lower() for f in self.warnings)
            return f"{len(self.warnings)} warning(s): {causes}"
        return "no problems found in the capture geometry or the residuals"

    def summary_lines(self, include_ok: bool = False) -> Tuple[str, ...]:
        """A human summary, worst first.

        Args:
            include_ok: Also list the diagnostics that found nothing.

        Returns:
            One or more lines per finding shown.
        """
        lines: List[str] = [self.verdict(), ""]
        for finding in self.ranked():
            if finding.severity is Severity.OK and not include_ok:
                continue
            lines.append(f"[{finding.severity.label:8s}] {finding.title}")
            lines.append(f"             {finding.summary}")
            if finding.action:
                lines.append(f"             -> {finding.action}")
            lines.append("")
        return tuple(lines[:-1] if lines and lines[-1] == "" else lines)

    def to_dict(self) -> Dict[str, Any]:
        """Serialise to a plain JSON-compatible dictionary."""
        return {
            "severity": self.severity.label,
            "verdict": self.verdict(),
            "findings": [f.to_dict() for f in self.findings],
        }


def diagnose(
    fit: InstrumentedFit,
    observations: ObservationSet,
    cross_validation: Optional[CrossValidation] = None,
    diagnostics: Optional[Tuple[type, ...]] = None,
) -> Diagnosis:
    """Run every diagnostic against a fit.

    Args:
        fit: The instrumented refit.
        observations: The detections it was made from.
        cross_validation: Out-of-sample results, when available.
        diagnostics: Override the diagnostic set, for tests.

    Returns:
        Every finding, in report order.

    Raises:
        ValidationError: The fit and the observations describe different views.
    """
    if fit.view_ids != observations.view_ids:
        raise ValidationError(
            f"the fit covers {len(fit.view_ids)} views and the observations "
            f"{observations.n_views}; these are not the same capture"
        )
    context = DiagnosticContext(fit, observations, cross_validation)
    classes = diagnostics if diagnostics is not None else DIAGNOSTICS
    return Diagnosis(tuple(cls().run(context) for cls in classes))
