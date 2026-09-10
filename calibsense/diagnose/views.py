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

"""Which views the calibration actually rests on, and which ones are fighting it.

Two independent questions that a per-view RMS conflates.

*Leverage* is how much of what the capture knows about the intrinsics comes from
one view. The Schur complement is a sum over views, `S = sum_i S_i`, so
`trace(S_i S^-1)` sums to the parameter count and `trace(S_i S^-1) / p` is that
view's share of the total information. A capture where one view carries half the
information is one bad frame away from a different calibration.

*Residual* is whether a view fits. High residual on a low-leverage view is a
frame to discard; high residual on a high-leverage view is a calibration to
distrust, because the fit is being pulled by the frame it depends on most.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar, List, Tuple

import numpy as np

from .base import Diagnostic, DiagnosticContext, Finding, Severity

#: Leverage above this multiple of an even share makes a view dominant.
DOMINANT_MULTIPLE = 3.0

#: A single view holding this much of the intrinsic information is critical.
DOMINANT_SHARE = 0.5


@dataclass(frozen=True)
class ViewInfluence:
    """One view's leverage and residual, together.

    Attributes:
        view_id: The view's identifier.
        information_share: Fraction of the intrinsic information this view
            provides. Shares sum to one across views.
        even_share: What the share would be if every view contributed equally.
        rms: The view's reprojection RMS in pixels.
        robust_z: How far that RMS sits from the median view, in robust
            standard deviations.
        n_points: Points in the view.
    """

    view_id: str
    information_share: float
    even_share: float
    rms: float
    robust_z: float
    n_points: int

    @property
    def dominant(self) -> bool:
        """Whether this view carries disproportionate information."""
        return self.information_share > DOMINANT_MULTIPLE * self.even_share

    @property
    def is_outlier(self) -> bool:
        """Whether this view's residual stands out from the rest."""
        from ..refit.residuals import OUTLIER_Z

        return self.robust_z > OUTLIER_Z

    @property
    def dangerous(self) -> bool:
        """High leverage and a high residual at once, which is the bad case."""
        return self.dominant and self.is_outlier

    def to_dict(self) -> dict:
        """Serialise to a plain dictionary."""
        return {
            "view_id": self.view_id,
            "information_share": self.information_share,
            "rms": self.rms,
            "robust_z": self.robust_z,
            "n_points": self.n_points,
            "dominant": self.dominant,
            "is_outlier": self.is_outlier,
        }


def view_influences(context: DiagnosticContext) -> Tuple[ViewInfluence, ...]:
    """Compute per-view leverage and residual.

    Args:
        context: The fit and its detections.

    Returns:
        One entry per view, in view order.
    """
    equations = context.fit.equations
    options = context.fit.options
    contributions = equations.schur_contributions(options.rcond)
    schur_inverse = context.fit.covariance._schur_inverse
    parameters = max(equations.n_intrinsic, 1)
    shares = np.einsum("nij,ji->n", contributions, schur_inverse) / parameters
    # A rank-deficient system leaves the cut directions out of the inverse, so
    # the shares no longer sum to one; renormalising keeps them readable as
    # fractions of what the capture actually determined.
    total = float(shares.sum())
    if total > 0:
        shares = shares / total
    even = 1.0 / max(context.n_views, 1)
    return tuple(
        ViewInfluence(
            view_id=summary.view_id,
            information_share=float(share),
            even_share=even,
            rms=summary.rms,
            robust_z=summary.robust_z,
            n_points=summary.n_points,
        )
        for share, summary in zip(shares, context.fit.residuals.per_view)
    )


class OutlierViews(Diagnostic):
    """Per-view leverage on the objective, paired with per-view residual."""

    cause: ClassVar[str] = "outlier_views"
    title: ClassVar[str] = "Outlier and high-leverage views"

    def run(self, context: DiagnosticContext) -> Finding:
        """Rank views by leverage and residual, and flag the dangerous ones."""
        influences = view_influences(context)
        outliers = [v for v in influences if v.is_outlier]
        dominant = [v for v in influences if v.dominant]
        dangerous = [v for v in influences if v.dangerous]
        shares = np.array([v.information_share for v in influences])
        metrics = dict(
            max_information_share=float(shares.max()),
            even_share=float(1.0 / max(context.n_views, 1)),
            n_outliers=len(outliers),
            n_dominant=len(dominant),
            views=[v.to_dict() for v in influences],
        )
        shared = (
            f"the heaviest view carries {shares.max():.0%} of the intrinsic "
            f"information against an even share of {1 / max(context.n_views, 1):.0%}"
        )
        if dangerous:
            names = ", ".join(f"{v.view_id} ({v.rms:.2f} px)" for v in dangerous[:3])
            return self._finding(
                Severity.CRITICAL,
                f"{shared}; {len(dangerous)} view(s) combine high leverage with a "
                f"high residual: {names}",
                "Inspect those images before trusting this calibration. A frame "
                "that both dominates the fit and fits badly is pulling the "
                "parameters towards its own error.",
                **metrics,
            )
        if shares.max() > DOMINANT_SHARE:
            heaviest = influences[int(np.argmax(shares))]
            return self._finding(
                Severity.CRITICAL,
                f"{shared}, almost all of it from {heaviest.view_id}",
                "The calibration rests on one frame. Add views at that frame's "
                "orientation and distance so the estimate does not depend on it.",
                **metrics,
            )
        if outliers:
            names = ", ".join(
                f"{v.view_id} ({v.rms:.2f} px, z={v.robust_z:.1f})" for v in outliers[:3]
            )
            return self._finding(
                Severity.WARNING,
                f"{len(outliers)} view(s) fit worse than the rest: {names}",
                "Check those images for blur, a moving target, or a misdetection. "
                "Note that calibsense does not currently down-weight them, so they "
                "are inflating every reported uncertainty.",
                **metrics,
            )
        if dominant:
            names = ", ".join(
                f"{v.view_id} ({v.information_share:.0%})" for v in dominant[:3]
            )
            return self._finding(
                Severity.NOTE,
                f"{shared}; {len(dominant)} view(s) contribute disproportionately: "
                f"{names}",
                "Not a fault on its own, but the calibration is sensitive to those "
                "frames. Duplicating their viewpoint would make it robust.",
                **metrics,
            )
        return self._finding(Severity.OK, shared, **metrics)
